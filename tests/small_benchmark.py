#!/usr/bin/env python3
"""
OpenMythos 与原版 Transformer 在小型 HuggingFace 数据集上的
并行训练 + 基准测试（默认使用 TinyStories，流式加载）。

两个模型共享相同的微型 MLA 配置，并以相同顺序看到完全相同的批次，
因此每步训练损失和吞吐量可以直接对比。
基线模型是使用相同 TransformerBlock 基础组件（`use_moe=False`）的密集堆叠；
其唯一层深度匹配递归块的唯一参数深度（前奏层 + 1 + 尾声层），
因此总参数量处于同一数量级。注意力核心是共享的（两个模型都使用 MLA），
所以任何测量到的差异反映的是循环递归深度架构，而非核心差异。

脚本测量的内容
------------------------
1. 两个模型的每步训练损失 + tokens/秒，使用完全相同的批次。
2. 定期在独立数据集分割上进行留出评估损失（--eval-every）。
3. 训练结束后的深度外推扫描：OpenMythos 在 cfg.max_loop_iters 下训练，
   然后在 --depth-sweep 中的 n_loops 值（默认 1,2,4,8,16）下评估。
   这是递归深度架构设计要赢得的实验 —— 如果深度外推有效，
   评估损失应在超过训练深度后继续下降。
4. 汇总表，包含初始/最终/平均训练损失、总耗时、平均 tok/s
   和每步秒数。

默认参数针对笔记本 CPU 在合理时间内运行进行了调优；传入 --device cuda
并增大 --steps / --batch-size / --seq-len 以进行真实对比。

    # 默认 CPU 冒烟测试（TinyStories，1k 步，batch 32，seq 256）
    python tests/small_benchmark.py

    # 更大规模的 GPU 运行
    python tests/small_benchmark.py --steps 5000 --batch-size 64 --seq-len 512 --device cuda

    # 使用 Wikitext 替代 TinyStories
    python tests/small_benchmark.py --dataset wikitext --dataset-config wikitext-2-raw-v1

    # 激进的深度外推扫描
    python tests/small_benchmark.py --depth-sweep 1,2,4,8,16,32
"""

from __future__ import annotations

import argparse
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from open_mythos import MythosConfig, OpenMythos
from open_mythos.main import (
    RMSNorm,
    TransformerBlock,
    precompute_rope_freqs,
)


# ---------------------------------------------------------------------------
# 基线模型：密集 GQA + SwiGLU Transformer
# ---------------------------------------------------------------------------


class BaselineTransformer(nn.Module):
    """原版仅解码器 Transformer，使用密集 SwiGLU FFN。

    复用 OpenMythos 的 TransformerBlock（注意力 + FFN 核心完全相同），
    因此任何测量到的差异反映的是循环递归深度架构，而非核心差异。
    支持 attn_type="gqa" 和 "mla"。
    """

    def __init__(self, cfg: MythosConfig, n_layers: int):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.layers = nn.ModuleList(
            [TransformerBlock(cfg, use_moe=False) for _ in range(n_layers)]
        )
        self.norm = RMSNorm(cfg.dim)
        self.head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        self.head.weight = self.embed.weight  # 权重共享

        # MLA 仅对 qk_rope_head_dim 应用 RoPE；GQA 旋转完整的 head_dim。
        rope_dim = (
            cfg.qk_rope_head_dim if cfg.attn_type == "mla" else cfg.dim // cfg.n_heads
        )
        self.register_buffer(
            "freqs_cis",
            precompute_rope_freqs(rope_dim, cfg.max_seq_len, cfg.rope_theta),
            persistent=False,
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    @staticmethod
    def _causal_mask(T: int, device: torch.device) -> torch.Tensor:
        mask = torch.full((1, 1, T, T), float("-inf"), device=device)
        return torch.triu(mask, diagonal=1)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        T = input_ids.shape[1]
        x = self.embed(input_ids)
        freqs_cis = self.freqs_cis[:T]
        mask = self._causal_mask(T, x.device) if T > 1 else None
        for i, layer in enumerate(self.layers):
            x = layer(x, freqs_cis, mask, cache_key=f"layer_{i}")
        return self.head(self.norm(x))


# ---------------------------------------------------------------------------
# 数据集：一次性分词，打包为固定长度的下一个 token 对
# ---------------------------------------------------------------------------


class PackedLMDataset(Dataset):
    """将 HF 文本数据集展平为一个 token 缓冲区，切分为固定长度的对。

    同时接受映射式和流式（`IterableDataset`）HF 数据集 ——
    迭代在收集到 `max_tokens` 后停止，因此像 TinyStories 这样的大型语料库
    可以流式加载而无需下载全部数据。
    """

    def __init__(
        self,
        hf_ds,
        tokenizer,
        seq_len: int,
        max_tokens: int,
        text_field: str = "text",
    ):
        buf: list[int] = []
        for sample in hf_ds:
            text = sample[text_field]
            if not text or not text.strip():
                continue
            buf.extend(tokenizer.encode(text, add_special_tokens=False))
            if len(buf) >= max_tokens:
                break
        self.seq_len = seq_len
        n_pairs = max(1, (len(buf) - 1) // seq_len)
        buf = buf[: n_pairs * seq_len + 1]
        self.data = torch.tensor(buf, dtype=torch.long)

    def __len__(self) -> int:
        return (len(self.data) - 1) // self.seq_len

    def __getitem__(self, idx: int):
        s = idx * self.seq_len
        chunk = self.data[s : s + self.seq_len + 1]
        return chunk[:-1], chunk[1:]


# ---------------------------------------------------------------------------
# 指标
# ---------------------------------------------------------------------------


@dataclass
class Metrics:
    total_loss: float = 0.0
    total_tokens: int = 0
    total_time: float = 0.0
    steps: int = 0
    first_losses: list[float] = field(default_factory=list)
    last_losses: Deque[float] = field(default_factory=lambda: deque(maxlen=10))

    def update(self, loss: float, tokens: int, seconds: float) -> None:
        self.total_loss += loss
        self.total_tokens += tokens
        self.total_time += seconds
        self.steps += 1
        if len(self.first_losses) < 10:
            self.first_losses.append(loss)
        self.last_losses.append(loss)

    @property
    def avg_loss(self) -> float:
        return self.total_loss / max(1, self.steps)

    @property
    def tok_per_sec(self) -> float:
        return self.total_tokens / max(1e-9, self.total_time)

    @property
    def initial_loss(self) -> float:
        return sum(self.first_losses) / max(1, len(self.first_losses))

    @property
    def final_loss(self) -> float:
        return sum(self.last_losses) / max(1, len(self.last_losses))


# ---------------------------------------------------------------------------
# 训练步骤
# ---------------------------------------------------------------------------


def train_step(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    vocab_size: int,
) -> tuple[float, float]:
    """执行一步优化器更新；返回 (损失值, 耗时秒数)。"""
    t0 = time.perf_counter()
    model.train()
    optimizer.zero_grad()
    logits = model(x)
    loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    return loss.item(), time.perf_counter() - t0


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    vocab_size: int,
    max_batches: int | None = None,
    n_loops: int | None = None,
) -> float:
    """在 loader 的（最多 `max_batches` 个）批次上计算平均交叉熵。

    `n_loops` 仅传递给 OpenMythos；对于其他模块该参数会被忽略，
    因此同一函数可以统一地对基线和 mythos 进行基准测试。
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for i, (x, y) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        if isinstance(model, OpenMythos):
            logits = model(x, n_loops=n_loops)
        else:
            logits = model(x)
        # 使用 sum 归约以按 token 数量加权，而非按批次数量
        loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1), reduction="sum")
        total_loss += loss.item()
        total_tokens += y.numel()
    return total_loss / max(1, total_tokens)


# ---------------------------------------------------------------------------
# 配置 + 工具函数
# ---------------------------------------------------------------------------


def build_tiny_cfg(vocab_size: int, seq_len: int) -> MythosConfig:
    """微型共享配置，使用 MLA 注意力 —— 在 CPU 上可在合理时间内运行。

    MLA 的 LoRA 秩和头维度按 `dim=128` 缩放，而非
    2048 维大小的默认值（q_lora_rank=1536, qk_nope_head_dim=128, ...），
    否则在此规模下参数量会被这些值主导。
    """
    return MythosConfig(
        vocab_size=vocab_size,
        dim=128,
        n_heads=4,
        n_kv_heads=2,
        max_seq_len=seq_len,
        max_loop_iters=4,
        prelude_layers=1,
        coda_layers=1,
        attn_type="mla",
        kv_lora_rank=64,
        q_lora_rank=128,
        qk_rope_head_dim=16,
        qk_nope_head_dim=32,
        v_head_dim=32,
        n_experts=4,
        n_shared_experts=1,
        n_experts_per_tok=2,
        expert_dim=128,
        lora_rank=4,
        rope_theta=10000.0,
        dropout=0.0,
    )


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def fmt_count(n: float) -> str:
    for unit in ("", "K", "M", "B"):
        if abs(n) < 1000:
            return f"{n:.2f}{unit}"
        n /= 1000
    return f"{n:.2f}T"


# ---------------------------------------------------------------------------
# 主函数
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    # 默认指向 TinyStories —— 更简单的词汇表 + 更短的文档
    # 使得 dim=128 的模型能在适度时间内达到有意义的损失。
    p.add_argument("--dataset", default="roneneldan/TinyStories")
    p.add_argument(
        "--dataset-config",
        default="",
        help="对于没有配置的数据集传入 ''（例如 TinyStories）",
    )
    p.add_argument("--train-split", default="train")
    p.add_argument("--eval-split", default="validation")
    p.add_argument(
        "--train-tokens",
        type=int,
        default=5_000_000,
        help="训练缓冲区最大物化 token 数",
    )
    p.add_argument(
        "--eval-tokens",
        type=int,
        default=200_000,
        help="留出评估缓冲区最大物化 token 数",
    )
    p.add_argument("--text-field", default="text")
    p.add_argument("--tokenizer", default="gpt2")
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument(
        "--eval-every",
        type=int,
        default=200,
        help="每 N 步运行一次留出评估（0 表示禁用）",
    )
    p.add_argument("--eval-batches", type=int, default=20)
    p.add_argument(
        "--depth-sweep",
        default="1,2,4,8,16",
        help="逗号分隔的 n_loops 值，用于 OpenMythos 深度外推评估",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return p.parse_args()


def load_text_ds(name: str, config: str, split: str):
    """流式 `load_dataset`，支持可选配置（空字符串 == 无配置）。"""
    if config:
        return load_dataset(name, config, split=split, streaming=True)
    return load_dataset(name, split=split, streaming=True)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    print(
        f"[设置] device={device}  batch={args.batch_size}  "
        f"seq_len={args.seq_len}  steps={args.steps}"
    )

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    # AutoTokenizer.vocab_size 对于带有额外 token 的 BPE 分词器
    # 可能小于头部大小；使用 len(tokenizer) 更安全。
    vocab_size = len(tokenizer)
    print(f"[设置] tokenizer={args.tokenizer}  vocab_size={vocab_size:,}")

    # ------------------------------------------------------------------
    # 数据：流式训练 + 留出评估分割
    # ------------------------------------------------------------------
    print(f"[设置] dataset={args.dataset}  config={args.dataset_config or '∅'}")
    raw_train = load_text_ds(args.dataset, args.dataset_config, args.train_split)
    train_ds = PackedLMDataset(
        raw_train, tokenizer, args.seq_len, args.train_tokens, args.text_field
    )
    raw_eval = load_text_ds(args.dataset, args.dataset_config, args.eval_split)
    eval_ds = PackedLMDataset(
        raw_eval, tokenizer, args.seq_len, args.eval_tokens, args.text_field
    )
    print(
        f"[设置] 训练 tokens={train_ds.data.numel():,}  对数={len(train_ds)}  |  "
        f"评估 tokens={eval_ds.data.numel():,}  对数={len(eval_ds)}"
    )

    torch.manual_seed(args.seed)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True
    )
    eval_loader = DataLoader(
        eval_ds, batch_size=args.batch_size, shuffle=False, drop_last=False
    )

    # ------------------------------------------------------------------
    # 模型 —— 相同的初始化种子，使两者从相同的嵌入开始
    # ------------------------------------------------------------------
    cfg = build_tiny_cfg(vocab_size, args.seq_len)

    torch.manual_seed(args.seed)
    mythos = OpenMythos(cfg).to(device)

    # 参数匹配深度：前奏层 + 一个唯一的递归块 + 尾声层。
    baseline_layers = cfg.prelude_layers + 1 + cfg.coda_layers
    torch.manual_seed(args.seed)
    baseline = BaselineTransformer(cfg, n_layers=baseline_layers).to(device)

    n_m, n_b = count_params(mythos), count_params(baseline)
    print(
        f"[设置] OpenMythos 参数  = {fmt_count(n_m)}  ({n_m:,})\n"
        f"[设置] 基线模型  参数  = {fmt_count(n_b)}  ({n_b:,})  "
        f"[{baseline_layers} 层]"
    )
    print(
        f"[设置] Mythos 运行时深度 = 前奏层({cfg.prelude_layers}) + "
        f"循环({cfg.max_loop_iters}) + 尾声层({cfg.coda_layers}) = "
        f"{cfg.prelude_layers + cfg.max_loop_iters + cfg.coda_layers}"
    )

    opt_m = torch.optim.AdamW(
        mythos.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1
    )
    opt_b = torch.optim.AdamW(
        baseline.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1
    )

    mm, bm = Metrics(), Metrics()
    eval_history: list[tuple[int, float, float]] = []  # (步数, mythos评估, 基线评估)

    header = (
        f"\n{'步数':>6} | {'mythos 损失':>12} | {'基线损失':>10} | "
        f"{'mythos tok/s':>13} | {'基线 tok/s':>11}"
    )
    print(header)
    print("-" * len(header))

    # ------------------------------------------------------------------
    # 训练循环，定期进行留出评估
    # ------------------------------------------------------------------
    data_iter = iter(train_loader)
    t_total = time.perf_counter()
    for step in range(1, args.steps + 1):
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            x, y = next(data_iter)
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        tokens = x.numel()

        loss_m, dt_m = train_step(mythos, x, y, opt_m, device, vocab_size)
        loss_b, dt_b = train_step(baseline, x, y, opt_b, device, vocab_size)

        mm.update(loss_m, tokens, dt_m)
        bm.update(loss_b, tokens, dt_b)

        if step == 1 or step % args.log_every == 0:
            print(
                f"{step:>6} | {loss_m:>12.4f} | {loss_b:>10.4f} | "
                f"{tokens / dt_m:>13,.0f} | {tokens / dt_b:>11,.0f}"
            )

        if args.eval_every and step % args.eval_every == 0:
            eval_m = evaluate(
                mythos, eval_loader, device, vocab_size, args.eval_batches
            )
            eval_b = evaluate(
                baseline, eval_loader, device, vocab_size, args.eval_batches
            )
            eval_history.append((step, eval_m, eval_b))
            print(
                f"  [评估 @ 步数 {step}]  mythos {eval_m:.4f}   基线 {eval_b:.4f}   "
                f"(Δ = {eval_m - eval_b:+.4f})"
            )

    total_wall = time.perf_counter() - t_total

    # ------------------------------------------------------------------
    # 汇总
    # ------------------------------------------------------------------
    bar = "=" * 70
    print(f"\n{bar}\n汇总（{args.steps} 步，总耗时 {total_wall:.1f}s）\n{bar}")
    print(f"  {'':<24} {'OpenMythos':>16}   {'基线模型':>16}")
    print(f"  {'参数量':<24} {fmt_count(n_m):>16}   {fmt_count(n_b):>16}")
    print(
        f"  {'初始训练（前 10 步）':<24} "
        f"{mm.initial_loss:>16.4f}   {bm.initial_loss:>16.4f}"
    )
    print(
        f"  {'最终训练（后 10 步）':<24} "
        f"{mm.final_loss:>16.4f}   {bm.final_loss:>16.4f}"
    )
    print(
        f"  {'平均训练（所有步）':<24} "
        f"{mm.avg_loss:>16.4f}   {bm.avg_loss:>16.4f}"
    )
    print(
        f"  {'训练时间（秒）':<24} "
        f"{mm.total_time:>16.2f}   {bm.total_time:>16.2f}"
    )
    print(
        f"  {'平均 tok/s':<24} " f"{mm.tok_per_sec:>16,.0f}   {bm.tok_per_sec:>16,.0f}"
    )
    print(
        f"  {'秒/步':<24} "
        f"{mm.total_time / max(1, mm.steps):>16.4f}   "
        f"{bm.total_time / max(1, bm.steps):>16.4f}"
    )

    # ------------------------------------------------------------------
    # 深度外推：OpenMythos 评估损失作为 n_loops 的函数。
    # 在 cfg.max_loop_iters 下训练；我们用一组扫描值进行推理，
    # 以观察额外的循环是否持续改善（深度外推）或者
    # 模型在训练范围外是否崩溃。
    # ------------------------------------------------------------------
    loops_sweep = sorted({int(s) for s in args.depth_sweep.split(",") if s.strip()})
    print(f"\n{bar}\n深度外推（留出评估，完整评估集）\n{bar}")
    baseline_eval = evaluate(baseline, eval_loader, device, vocab_size)
    print(f"  基线（固定深度）          : 评估损失 = {baseline_eval:.4f}")
    # 先收集所有扫描损失，然后打印与训练深度的差异。
    sweep: list[tuple[int, float]] = []
    for nl in loops_sweep:
        sweep.append(
            (nl, evaluate(mythos, eval_loader, device, vocab_size, n_loops=nl))
        )
    trained_loss = next((loss for nl, loss in sweep if nl == cfg.max_loop_iters), None)
    print(f"  OpenMythos（在 n_loops={cfg.max_loop_iters} 下训练）:")
    print(f"    {'n_loops':>8}  {'评估损失':>10}  {'相对训练深度 Δ':>14}")
    for nl, loss in sweep:
        if trained_loss is None or nl == cfg.max_loop_iters:
            delta_str = ""
        else:
            delta_str = f"{loss - trained_loss:+.4f}"
        marker = " ←训练深度" if nl == cfg.max_loop_iters else ""
        print(f"    {nl:>8}  {loss:>10.4f}  {delta_str:>14}{marker}")


if __name__ == "__main__":
    main()
