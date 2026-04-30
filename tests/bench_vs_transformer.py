"""
OpenMythos 与原版 GQA+MoE Transformer 基准测试。

将 OpenMythos（前奏层 + 循环递归块 + 尾声层，含 ACT 停止机制、
LTI 稳定注入、LoRA 深度适配器）与参数匹配的原版 Transformer 进行对比。
原版 Transformer 使用相同的 GQAttention + MoEFFN 构建块以非递归方式堆叠。
基线模型复用 OpenMythos 的基础组件，因此对比隔离的是递归深度架构，
而非底层计算核心。

报告的指标：
    - 参数数量（总量、MoE 活跃参数近似值）
    - 多种序列长度下的预填充延迟 + 吞吐量
    - 带 KV 缓存的解码（自回归步骤）延迟
    - 峰值显存（仅 CUDA）
    - OpenMythos 深度缩放扫描：延迟 vs. n_loops

运行方式：
    python benchmarks/bench_vs_transformer.py                     # 小规模 CPU/GPU 冒烟测试
    python benchmarks/bench_vs_transformer.py --size 1b --device cuda
    python benchmarks/bench_vs_transformer.py --seq-lens 128,512,2048 --n-loops 1,4,8,16
"""

from __future__ import annotations

import argparse
import gc
import statistics
import time
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from open_mythos import MythosConfig, OpenMythos, mythos_1b
from open_mythos.main import (
    RMSNorm,
    TransformerBlock,
    precompute_rope_freqs,
)


# ---------------------------------------------------------------------------
# 基线模型：非循环 GQA + MoE Transformer
# ---------------------------------------------------------------------------


class BaselineTransformer(nn.Module):
    """
    原版仅解码器 Transformer，使用 GQA 注意力和 MoE FFN，
    以非递归方式堆叠。与 OpenMythos 共享 TransformerBlock / GQAttention / MoEFFN
    计算核心，因此任何速度差异都归因于递归深度架构，
    而非底层注意力/FFN 实现。
    """

    def __init__(self, cfg: MythosConfig, n_layers: int):
        super().__init__()
        self.cfg = cfg
        self.n_layers = n_layers
        self.embed = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.layers = nn.ModuleList(
            [TransformerBlock(cfg, use_moe=True) for _ in range(n_layers)]
        )
        self.norm = RMSNorm(cfg.dim)
        self.head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)

        head_dim = cfg.dim // cfg.n_heads
        self.register_buffer(
            "freqs_cis",
            precompute_rope_freqs(head_dim, cfg.max_seq_len, cfg.rope_theta),
            persistent=False,
        )

    @staticmethod
    def _causal_mask(T: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        mask = torch.full((1, 1, T, T), float("-inf"), device=device, dtype=dtype)
        return torch.triu(mask, diagonal=1)

    def forward(
        self,
        input_ids: torch.Tensor,
        kv_cache: Optional[dict] = None,
        start_pos: int = 0,
    ) -> torch.Tensor:
        T = input_ids.shape[1]
        x = self.embed(input_ids)
        freqs_cis = self.freqs_cis[start_pos : start_pos + T]
        mask = self._causal_mask(T, x.device, x.dtype) if T > 1 else None
        for i, layer in enumerate(self.layers):
            x = layer(x, freqs_cis, mask, kv_cache, cache_key=f"layer_{i}")
        return self.head(self.norm(x))


# ---------------------------------------------------------------------------
# 计时工具
# ---------------------------------------------------------------------------


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def time_fn(fn, device: torch.device, warmup: int = 2, trials: int = 5) -> float:
    """在 `warmup` 次预热后，返回 `trials` 次运行的中位数耗时（秒）。"""
    for _ in range(warmup):
        fn()
    _sync(device)
    times = []
    for _ in range(trials):
        _sync(device)
        t0 = time.perf_counter()
        fn()
        _sync(device)
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


def peak_mem_mb(device: torch.device) -> float:
    if device.type != "cuda":
        return 0.0
    return torch.cuda.max_memory_allocated() / (1024 * 1024)


def reset_mem(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


# ---------------------------------------------------------------------------
# 参数计数
# ---------------------------------------------------------------------------


@dataclass
class ParamCounts:
    total: int
    moe_active_est: int  # 每个 token 的活跃参数（共享 + top-k 路由）


def count_params(model: nn.Module, cfg: MythosConfig) -> ParamCounts:
    total = sum(p.numel() for p in model.parameters())
    # MoE 层每个 token 的粗略活跃参数计数：共享 + top-k 路由比例。
    # 为简化起见，分别报告总量和估计的激活比率。
    active_ratio = (cfg.n_shared_experts + cfg.n_experts_per_tok) / (
        cfg.n_shared_experts + cfg.n_experts
    )
    # 只有 FFN 参数在激活时缩减；注意力 + 嵌入/输出头始终活跃。
    # 这是活跃参数的粗略下界。
    ffn_params = 0
    other_params = 0
    for name, p in model.named_parameters():
        if ".ffn." in name or name.startswith("ffn.") or ".experts." in name:
            ffn_params += p.numel()
        else:
            other_params += p.numel()
    active_est = other_params + int(ffn_params * active_ratio)
    return ParamCounts(total=total, moe_active_est=active_est)


# ---------------------------------------------------------------------------
# 基准测试
# ---------------------------------------------------------------------------


def bench_prefill(
    model: nn.Module,
    vocab_size: int,
    batch: int,
    seq_len: int,
    device: torch.device,
    n_loops: Optional[int] = None,
) -> tuple[float, float]:
    """返回 (中位数秒数, tokens/秒)。"""
    ids = torch.randint(0, vocab_size, (batch, seq_len), device=device)

    if isinstance(model, OpenMythos):

        def run() -> None:
            with torch.no_grad():
                model(ids, n_loops=n_loops)

    else:

        def run() -> None:
            with torch.no_grad():
                model(ids)

    secs = time_fn(run, device)
    tps = (batch * seq_len) / secs
    return secs, tps


def bench_decode(
    model: nn.Module,
    vocab_size: int,
    batch: int,
    prompt_len: int,
    decode_steps: int,
    device: torch.device,
    n_loops: Optional[int] = None,
) -> tuple[float, float]:
    """
    预填充 `prompt_len` 长度的提示词，然后计时 `decode_steps` 个单 token 解码步骤
    （使用 KV 缓存）。返回 (每步平均秒数, 解码 tokens/秒)。
    """
    prompt = torch.randint(0, vocab_size, (batch, prompt_len), device=device)

    def one_run() -> None:
        kv_cache: dict = {}
        with torch.no_grad():
            if isinstance(model, OpenMythos):
                model(prompt, n_loops=n_loops, kv_cache=kv_cache, start_pos=0)
            else:
                model(prompt, kv_cache=kv_cache, start_pos=0)
            for i in range(decode_steps):
                next_tok = torch.randint(0, vocab_size, (batch, 1), device=device)
                if isinstance(model, OpenMythos):
                    model(
                        next_tok,
                        n_loops=n_loops,
                        kv_cache=kv_cache,
                        start_pos=prompt_len + i,
                    )
                else:
                    model(next_tok, kv_cache=kv_cache, start_pos=prompt_len + i)

    secs = time_fn(one_run, device, warmup=1, trials=3)
    per_step = secs / decode_steps
    tps = batch * decode_steps / secs
    return per_step, tps


# ---------------------------------------------------------------------------
# 配置辅助函数
# ---------------------------------------------------------------------------


def small_cfg() -> MythosConfig:
    """用于冒烟测试的微型配置 —— 在 CPU 上几秒内即可运行。"""
    return MythosConfig(
        vocab_size=1024,
        dim=256,
        n_heads=8,
        n_kv_heads=2,
        max_seq_len=1024,
        max_loop_iters=4,
        prelude_layers=1,
        coda_layers=1,
        attn_type="gqa",
        n_experts=8,
        n_shared_experts=1,
        n_experts_per_tok=2,
        expert_dim=128,
        lora_rank=4,
        dropout=0.0,
    )


def get_cfg(size: str) -> MythosConfig:
    size = size.lower()
    if size == "small":
        return small_cfg()
    if size == "1b":
        cfg = mythos_1b()
        # 使用 GQA 以进行公平对比；MLA 会改变 KV 形状语义。
        cfg.attn_type = "gqa"
        return cfg
    raise ValueError(f"未知的规模: {size!r}（请使用 'small' 或 '1b'）")


# ---------------------------------------------------------------------------
# 报告输出
# ---------------------------------------------------------------------------


def fmt_count(n: int) -> str:
    for unit in ("", "K", "M", "B", "T"):
        if abs(n) < 1000:
            return f"{n:.2f}{unit}"
        n /= 1000
    return f"{n:.2f}P"


def print_header(title: str) -> None:
    bar = "=" * 72
    print(f"\n{bar}\n{title}\n{bar}")


# ---------------------------------------------------------------------------
# 主函数
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--size", default="small", choices=["small", "1b"])
    p.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cuda", "cpu"],
    )
    p.add_argument(
        "--dtype",
        default="auto",
        choices=["auto", "fp32", "bf16", "fp16"],
        help="'auto' 在 CPU 上选择 fp32，在 CUDA 上选择 bf16",
    )
    p.add_argument("--batch", type=int, default=1)
    p.add_argument(
        "--seq-lens",
        default="128,512",
        help="逗号分隔的预填充序列长度",
    )
    p.add_argument(
        "--n-loops",
        default="1,4,8",
        help="逗号分隔的循环次数（仅 OpenMythos）",
    )
    p.add_argument(
        "--decode-steps",
        type=int,
        default=32,
        help="预填充后的自回归解码步数",
    )
    p.add_argument(
        "--decode-prompt-len",
        type=int,
        default=128,
        help="解码前的预填充长度",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    dtype_arg = args.dtype
    if dtype_arg == "auto":
        dtype_arg = "bf16" if device.type == "cuda" else "fp32"
    dtype = {
        "fp32": torch.float32,
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
    }[dtype_arg]

    seq_lens = [int(s) for s in args.seq_lens.split(",") if s.strip()]
    n_loops_sweep = [int(s) for s in args.n_loops.split(",") if s.strip()]

    cfg = get_cfg(args.size)
    print_header(f"配置: size={args.size}  device={device}  dtype={dtype_arg}")
    print(
        f"  dim={cfg.dim}  n_heads={cfg.n_heads}  n_kv_heads={cfg.n_kv_heads}  "
        f"prelude={cfg.prelude_layers}  coda={cfg.coda_layers}  "
        f"max_loop_iters={cfg.max_loop_iters}\n"
        f"  experts={cfg.n_experts}  shared={cfg.n_shared_experts}  "
        f"top_k={cfg.n_experts_per_tok}  expert_dim={cfg.expert_dim}"
    )

    # 构建模型。基线深度 = 前奏层 + 1（一个唯一的递归块）+ 尾声层
    # 以匹配 OpenMythos 的唯一参数深度（参数匹配基线）。
    baseline_n_layers = cfg.prelude_layers + 1 + cfg.coda_layers

    torch.manual_seed(0)
    mythos = OpenMythos(cfg).to(device=device, dtype=dtype).eval()
    torch.manual_seed(0)
    baseline = (
        BaselineTransformer(cfg, n_layers=baseline_n_layers)
        .to(device=device, dtype=dtype)
        .eval()
    )

    m_params = count_params(mythos, cfg)
    b_params = count_params(baseline, cfg)
    print_header(
        "参数（块匹配：基线深度 = 前奏层 + 1 递归层 + 尾声层）"
    )
    print(
        f"  OpenMythos : 总量={fmt_count(m_params.total):>10}   "
        f"活跃/token≈{fmt_count(m_params.moe_active_est):>10}"
    )
    print(
        f"  基线模型   : 总量={fmt_count(b_params.total):>10}   "
        f"活跃/token≈{fmt_count(b_params.moe_active_est):>10}"
    )
    print(
        f"  基线唯一层数 = {baseline_n_layers}  "
        f"（Mythos 在 max_loops 时的总运行时深度 = "
        f"{cfg.prelude_layers + cfg.max_loop_iters + cfg.coda_layers}）"
    )

    # ---- 预填充 ----
    print_header("预填充延迟（batch={batch}）".format(batch=args.batch))
    header = f"  {'模型':<26} {'序列':>6} {'秒':>10} {'tok/s':>12} {'峰值 MB':>10}"
    print(header)
    for seq_len in seq_lens:
        if seq_len > cfg.max_seq_len:
            print(f"  跳过 seq_len={seq_len}（> max_seq_len={cfg.max_seq_len}）")
            continue

        reset_mem(device)
        secs, tps = bench_prefill(baseline, cfg.vocab_size, args.batch, seq_len, device)
        mem = peak_mem_mb(device)
        print(
            f"  {'基线（堆叠）':<26} {seq_len:>6} "
            f"{secs*1000:>9.2f}ms {tps:>12,.0f} {mem:>10.1f}"
        )

        for nl in n_loops_sweep:
            reset_mem(device)
            secs, tps = bench_prefill(
                mythos, cfg.vocab_size, args.batch, seq_len, device, n_loops=nl
            )
            mem = peak_mem_mb(device)
            print(
                f"  {'OpenMythos (loops=' + str(nl) + ')':<26} {seq_len:>6} "
                f"{secs*1000:>9.2f}ms {tps:>12,.0f} {mem:>10.1f}"
            )

    # ---- 解码 ----
    print_header(
        f"解码延迟（预填充 {args.decode_prompt_len} tokens + "
        f"{args.decode_steps} 解码步骤，batch={args.batch}）"
    )
    print(f"  {'模型':<26} {'秒/步':>12} {'解码 tok/s':>14}")

    reset_mem(device)
    per_step, tps = bench_decode(
        baseline,
        cfg.vocab_size,
        args.batch,
        args.decode_prompt_len,
        args.decode_steps,
        device,
    )
    print(f"  {'基线（堆叠）':<26} {per_step*1000:>10.2f}ms {tps:>14,.1f}")

    for nl in n_loops_sweep:
        reset_mem(device)
        per_step, tps = bench_decode(
            mythos,
            cfg.vocab_size,
            args.batch,
            args.decode_prompt_len,
            args.decode_steps,
            device,
            n_loops=nl,
        )
        print(
            f"  {'OpenMythos (loops=' + str(nl) + ')':<26} "
            f"{per_step*1000:>10.2f}ms {tps:>14,.1f}"
        )

    # ---- 深度缩放 ----
    print_header(
        "OpenMythos 深度缩放（固定 seq={}，batch={}）".format(
            seq_lens[0], args.batch
        )
    )
    print(f"  {'n_loops':>8} {'秒':>10} {'tok/s':>12} {'相对 loops=1':>14}")
    base_secs = None
    for nl in n_loops_sweep:
        reset_mem(device)
        secs, tps = bench_prefill(
            mythos, cfg.vocab_size, args.batch, seq_lens[0], device, n_loops=nl
        )
        if base_secs is None:
            base_secs = secs
            delta = "1.00x"
        else:
            delta = f"{secs / base_secs:.2f}x"
        print(f"  {nl:>8} {secs*1000:>9.2f}ms {tps:>12,.0f} {delta:>14}")

    print("\n完成。")


if __name__ == "__main__":
    main()