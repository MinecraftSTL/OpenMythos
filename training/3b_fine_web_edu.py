#!/usr/bin/env python3
"""
OpenMythos 在 FineWeb-Edu 上使用 FSDP + AdamW 进行预训练。

单 GPU:
    python training/3b_fine_web_edu.py

多 GPU:
    torchrun --nproc_per_node=$(python -c "import torch; print(torch.cuda.device_count())") training/3b_fine_web_edu.py
"""

import os
import math
import time
import torch
import torch.nn as nn
import torch.distributed as dist
from loguru import logger
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy,
    MixedPrecision,
    FullStateDictConfig,
    StateDictType,
)
from torch.distributed.fsdp.wrap import ModuleWrapPolicy
from torch.utils.data import IterableDataset, DataLoader, get_worker_info
from contextlib import nullcontext

from datasets import load_dataset

from open_mythos import OpenMythos
from open_mythos.main import TransformerBlock, RecurrentBlock
from open_mythos.variants import mythos_3b
from open_mythos.tokenizer import MythosTokenizer


# ---------------------------------------------------------------------------
# 数据集
# ---------------------------------------------------------------------------


class FineWebEduDataset(IterableDataset):
    """
    流式 FineWeb-Edu 加载器，生成固定长度的 (输入, 目标) 对。

    FineWeb-Edu 拥有数万亿 token，因此 `streaming=True` 按需拉取分片，
    而非全部写入磁盘。分片是二维的 — `world_size` 个进程 × 每个进程
    `num_workers` 个 DataLoader 工作线程 — 每个 `(rank, worker_id)` 确定性地
    拥有全局流的一个分片。这在无需跨进程协调的情况下实现了不重叠的覆盖。

    流式数据集不可定位，因此恢复训练时会从分片开头重新进入。在预训练规模下
    这是可接受的：在训练结束前重放相同 token 的概率相对于真正可恢复加载器的
    成本来说可以忽略不计。
    """

    def __init__(self, encoding, seq_len: int, subset: str, rank: int, world_size: int):
        """
        参数:
            encoding   -- 分词器，暴露 `.encode(str) -> list[int]` 接口
            seq_len    -- 上下文长度；每个生成的对都包含这么多 token
            subset     -- FineWeb-Edu 配置名称（例如 "sample-10BT"、"default"）
            rank       -- 当前进程在分布式任务中的全局排名
            world_size -- 分布式进程总数
        """
        self.encoding = encoding
        self.seq_len = seq_len
        self.subset = subset
        self.rank = rank
        self.world_size = world_size

    def __iter__(self):
        """
        无限生成长度为 `seq_len` 的 `(input_ids, target_ids)` 张量。

        输入和目标偏移一位用于下一个 token 预测 —
        `target[i] == input[i + 1]`。文档被拼接到滚动缓冲区中，
        并切分为固定长度的块，将短文档打包在一起，长文档则被拆分。
        这使得每一步的形状保持一致，在 FSDP 下避免了因可变长度输入
        导致的重新计算，并消除了对填充感知注意力掩码的需求。
        """
        worker = get_worker_info()
        num_workers = worker.num_workers if worker else 1
        worker_id = worker.id if worker else 0

        total_shards = self.world_size * num_workers
        shard_index = self.rank * num_workers + worker_id

        ds = load_dataset(
            "HuggingFaceFW/fineweb-edu",
            name=self.subset,
            split="train",
            streaming=True,
        ).shard(num_shards=total_shards, index=shard_index)

        buf = []
        for sample in ds:
            buf.extend(self.encoding.encode(sample["text"]))
            while len(buf) >= self.seq_len + 1:
                chunk = buf[: self.seq_len + 1]
                buf = buf[self.seq_len + 1 :]
                yield (
                    torch.tensor(chunk[:-1], dtype=torch.long),
                    torch.tensor(chunk[1:], dtype=torch.long),
                )


# ---------------------------------------------------------------------------
# 学习率调度: 线性预热 → 余弦衰减
# ---------------------------------------------------------------------------


def get_lr(step: int, warmup: int, total: int, max_lr: float, min_lr: float) -> float:
    """
    线性预热 → 半余弦衰减至 `min_lr`。

    标准语言模型预训练调度。预热阶段防止 Adam 的二阶矩估计在前几步
    梯度噪声较大时崩溃为过大的学习率。余弦尾部让模型在训练末期进行
    越来越保守的小幅更新，而不是在固定步数突然降至 `min_lr`。

    各区间行为:
        step < warmup                 → 线性上升 0 → max_lr
        warmup ≤ step < total         → 余弦衰减 max_lr → min_lr
        step ≥ total                  → 钳制在 min_lr（防止训练末尾
                                        的差一错误）

    参数:
        step    -- 当前全局优化器步数（从 0 开始）
        warmup  -- 余弦衰减开始前的预热步数
        total   -- 余弦达到 `min_lr` 的步数
        max_lr  -- 预热结束时达到的峰值学习率
        min_lr  -- 在 `total` 步及之后的最低学习率

    返回:
        当前步的标量学习率。
    """
    if step < warmup:
        return max_lr * step / warmup
    if step >= total:
        return min_lr
    decay = (step - warmup) / (total - warmup)
    return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * decay))


# ---------------------------------------------------------------------------
# 检查点
# ---------------------------------------------------------------------------


def _list_ckpts(ckpt_dir: str) -> list[str]:
    """
    返回 `ckpt_dir` 中按从旧到新排序的检查点路径。

    依赖零填充的 `step_{0000000}.pt` 文件名约定，使得字典序排序
    与时间顺序一致。如果在其他地方更改文件名格式而不更新填充宽度，
    会静默破坏 `keep_last` 裁剪和启动时的恢复最新检查点功能，
    因为两者都选取此列表的最后一个元素。

    参数:
        ckpt_dir -- 要扫描的目录；目录不存在时返回 []

    返回:
        匹配的检查点文件绝对路径的排序列表。
    """
    if not os.path.isdir(ckpt_dir):
        return []
    return sorted(
        os.path.join(ckpt_dir, f)
        for f in os.listdir(ckpt_dir)
        if f.startswith("step_") and f.endswith(".pt")
    )


def save_checkpoint(
    model,
    optimizer,
    step: int,
    cfg,
    vocab_size: int,
    ckpt_dir: str,
    ddp: bool,
    master: bool,
    keep_last: int = 3,
) -> None:
    """
    收集完整的模型 + 优化器状态，原子写入，裁剪旧文件。

    在 FSDP 下，模型和优化器状态都在单个 FULL_STATE_DICT 上下文中收集，
    使得优化器状态张量绑定到完全未分片的参数上；在过去的 torch 版本中，
    混合上下文曾导致恢复时的静默发散。临时文件 + os.replace 写入意味着
    保存过程中被终止时，前一个检查点保持完整，而不会留下截断的 .pt 文件。
    非主进程参与 FSDP 收集（否则集合操作会挂起），但在接触磁盘前退出。

    参数:
        model       -- FSDP 封装（ddp=True）或原始（ddp=False）模型
        optimizer   -- 需要与模型一起往返保存的优化器
        step        -- 全局步数；零填充编码到文件名中
        cfg         -- 模型配置对象；保存后下游评估可以在不重新导入变体的情况下
                       重建模型
        vocab_size  -- 训练时的分词器词汇表大小；保存用于加载时与（可能已更新的）
                       分词器进行一致性检查
        ckpt_dir    -- 写入目录；不存在时自动创建
        ddp         -- True 表示 FSDP 路径；False 表示单 GPU / CPU
        master      -- 当前进程是否写入磁盘（仅 rank 0）
        keep_last   -- 保留的最近检查点数量；成功写入后删除更旧的检查点

    返回:
        None。在主进程上作为副作用写入磁盘。
    """
    if ddp:
        with FSDP.state_dict_type(
            model,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
        ):
            model_state = model.state_dict()
            optim_state = FSDP.optim_state_dict(model, optimizer)
    else:
        model_state = model.state_dict()
        optim_state = optimizer.state_dict()

    if not master:
        return

    os.makedirs(ckpt_dir, exist_ok=True)
    final_path = os.path.join(ckpt_dir, f"step_{step:07d}.pt")
    tmp_path = final_path + ".tmp"
    torch.save(
        {
            "step": step,
            "model": model_state,
            "optimizer": optim_state,
            "cfg": cfg,
            "vocab_size": vocab_size,
        },
        tmp_path,
    )
    os.replace(tmp_path, final_path)

    for old in _list_ckpts(ckpt_dir)[:-keep_last]:
        try:
            os.remove(old)
        except OSError as exc:
            logger.warning(f"裁剪旧检查点失败 {old}: {exc}")

    logger.success(f"检查点已保存 → {final_path}")


def load_checkpoint(model, optimizer, path: str, ddp: bool) -> int:
    """
    从磁盘恢复模型 + 优化器，返回要恢复的步数。

    每个进程都读取文件（加载时 `rank0_only=False`），使得 FSDP 在每个进程上
    都能访问完整状态 — 这是保存路径中 `rank0_only=True` 的补充。必须与保存时
    的单上下文模式一致；将模型和优化器的加载拆分到两个 `state_dict_type` 块中
    在历史上曾产生绑定到错误分片形状的优化器状态。

    `weights_only=False` 是必需的，因为检查点包含序列化的 `cfg` 数据类 —
    只有在将配置分离出来后才能切换为 `weights_only=True`。

    参数:
        model     -- 与保存时相同的 FSDP 封装或原始模型
        optimizer -- 新构建的优化器，将被就地填充
        path      -- 由 `save_checkpoint` 生成的 `step_{N:07d}.pt` 文件的绝对路径
        ddp       -- 模型是否经过 FSDP 封装；必须与保存时的运行一致

    返回:
        检查点保存时的步数；调用方从此值继续训练循环。
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=False)

    if ddp:
        with FSDP.state_dict_type(
            model,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
        ):
            model.load_state_dict(ckpt["model"])
            optim_state = FSDP.optim_state_dict_to_load(
                model=model,
                optim=optimizer,
                optim_state_dict=ckpt["optimizer"],
            )
            optimizer.load_state_dict(optim_state)
    else:
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])

    return int(ckpt["step"])


# ---------------------------------------------------------------------------
# 主函数
# ---------------------------------------------------------------------------


def main():
    """
    端到端预训练入口点。

    顺序很重要：分布式初始化必须在任何 CUDA 分配之前运行，分词器必须在模型
    构建之前存在（vocab_size 流入 cfg），FSDP 必须在优化器构建之前封装模型
    （FSDP 会重新展平参数，因此在未封装模型上构建的优化器会跟踪过时的参数对象）。
    然后恢复操作将状态就地加载到已构建的优化器中。

    生命周期:
        1. 如果在 torchrun 下启动，初始化 torch.distributed (NCCL)。
        2. 构建分词器 → 推导 vocab_size。
        3. 使用 3B 变体配置构建 OpenMythos。
        4. 使用 FULL_SHARD + bf16/fp16 混合精度封装 FSDP（多 GPU），
           或移至设备 + autocast（单 GPU）。
        5. 在（可能已分片的）参数上构建融合 AdamW。
        6. 如果 `ckpt_dir` 中存在检查点，从最新的检查点恢复。
        7. 通过梯度累积微批次流式处理 FineWeb-Edu，使用余弦学习率调度、
           逐步日志记录和定期检查点。
        8. 如果最后一次保存未对齐到 `ckpt_every`，则写入最终检查点，
           然后 barrier + 销毁进程组。

    所有超参数都是此函数中的字面常量 — 预训练运行是长期的，每次运行
    固定确切的设置；故意避免 CLI/配置层以保持文件的自审计性。
    """
    # ------------------------------------------------------------------
    # 分布式初始化
    # ------------------------------------------------------------------
    ddp = int(os.environ.get("RANK", -1)) != -1
    if ddp:
        dist.init_process_group("nccl")
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        device = f"cuda:{local_rank}"
        torch.cuda.set_device(device)
    else:
        rank = local_rank = 0
        world_size = 1
        device = "cuda" if torch.cuda.is_available() else "cpu"

    master = rank == 0

    if master:
        logger.info(
            f"GPU 数量: {torch.cuda.device_count()}  |  世界大小: {world_size}  |  设备: {device}"
        )

    # ------------------------------------------------------------------
    # 分词器
    # ------------------------------------------------------------------
    encoding = MythosTokenizer()
    vocab_size = encoding.vocab_size

    if master:
        logger.info(f"分词器: gpt-oss-20b  |  词汇表大小: {vocab_size:,}")

    # ------------------------------------------------------------------
    # 超参数
    # ------------------------------------------------------------------
    seq_len = 2048
    micro_batch = 4
    target_tokens = 30_000_000_000
    grad_accum = max(1, 256 // (world_size * micro_batch))
    global_batch_tok = world_size * micro_batch * grad_accum * seq_len
    total_steps = target_tokens // global_batch_tok
    warmup_steps = 2000
    lr = 3e-4
    wd = 0.1
    log_every = 10
    ckpt_every = 1000
    ckpt_dir = "checkpoints"
    dataset_subset = "sample-10BT"  # → sample-100BT 或 "default" 用于完整运行

    if master:
        logger.info(
            f"seq_len={seq_len} | micro_batch={micro_batch} | grad_accum={grad_accum} | "
            f"全局批次 token 数={global_batch_tok:,} | 总步数={total_steps:,}"
        )

    # ------------------------------------------------------------------
    # 模型
    # ------------------------------------------------------------------
    cfg = mythos_3b()
    cfg.vocab_size = vocab_size
    cfg.max_seq_len = seq_len

    bf16_ok = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if bf16_ok else torch.float16

    model = OpenMythos(cfg)

    if ddp:
        mp_policy = MixedPrecision(
            param_dtype=amp_dtype,
            reduce_dtype=amp_dtype,
            buffer_dtype=amp_dtype,
        )
        wrap_policy = ModuleWrapPolicy({TransformerBlock, RecurrentBlock})
        model = FSDP(
            model,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            mixed_precision=mp_policy,
            auto_wrap_policy=wrap_policy,
            device_id=local_rank,
        )
    else:
        model = model.to(device)
        amp_ctx = (
            torch.amp.autocast(device_type="cuda", dtype=amp_dtype)
            if "cuda" in device
            else nullcontext()
        )

    # FSDP 自行处理混合精度；仅单 GPU 需要 autocast
    amp_ctx = nullcontext() if ddp else amp_ctx  # type: ignore[possibly-undefined]

    if master:
        n_params = sum(p.numel() for p in model.parameters())
        logger.info(f"参数量: {n_params:,}  |  AMP 数据类型: {amp_dtype}")

    # ------------------------------------------------------------------
    # 优化器
    # ------------------------------------------------------------------
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=wd, betas=(0.9, 0.95), fused=True
    )

    # ------------------------------------------------------------------
    # 从最新检查点恢复（如果存在）
    # ------------------------------------------------------------------
    # 流式数据集不支持按位置恢复，因此接受从头重新迭代 — 在预训练规模下，
    # 丢失数据集位置相对于丢弃训练步数的成本可以忽略不计。
    start_step = 0
    existing_ckpts = _list_ckpts(ckpt_dir)
    if existing_ckpts:
        latest = existing_ckpts[-1]
        if master:
            logger.info(f"从检查点恢复: {latest}")
        start_step = load_checkpoint(model, optimizer, latest, ddp)
        if master:
            logger.success(f"已恢复至步数 {start_step}")

    # ------------------------------------------------------------------
    # 数据集 + 数据加载器
    # ------------------------------------------------------------------
    dataset = FineWebEduDataset(encoding, seq_len, dataset_subset, rank, world_size)
    loader = DataLoader(dataset, batch_size=micro_batch, num_workers=4, pin_memory=True)

    # ------------------------------------------------------------------
    # 训练循环
    # ------------------------------------------------------------------
    if master:
        os.makedirs(ckpt_dir, exist_ok=True)

    model.train()
    data_iter = iter(loader)
    t0 = time.perf_counter()
    step = start_step

    while step < total_steps:
        cur_lr = get_lr(step, warmup_steps, total_steps, lr, lr * 0.1)
        for g in optimizer.param_groups:
            g["lr"] = cur_lr

        optimizer.zero_grad()
        loss_accum = 0.0

        for micro_step in range(grad_accum):
            try:
                x, y = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                x, y = next(data_iter)

            x = x.to(device if not ddp else f"cuda:{local_rank}", non_blocking=True)
            y = y.to(device if not ddp else f"cuda:{local_rank}", non_blocking=True)

            sync = (
                nullcontext()
                if (not ddp or micro_step == grad_accum - 1)
                else model.no_sync()
            )
            with sync, amp_ctx:
                logits = model(x)
                loss = nn.functional.cross_entropy(
                    logits.view(-1, vocab_size), y.view(-1)
                )
                loss = loss / grad_accum

            loss.backward()
            loss_accum += loss.item()

        # FSDP 对参数进行分片，因此 `nn.utils.clip_grad_norm_` 会针对每个进程的
        # 本地范数进行裁剪，而遗漏跨分片的收集。
        # FSDP.clip_grad_norm_ 计算真正的全局范数并返回。
        if ddp:
            grad_norm = model.clip_grad_norm_(1.0)
        else:
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        step += 1

        if master and step % log_every == 0:
            dt = time.perf_counter() - t0
            tok_per_sec = global_batch_tok * log_every / dt
            tokens_seen = step * global_batch_tok
            logger.info(
                f"步数 {step:6d}/{total_steps} | 损失 {loss_accum:.4f} "
                f"| 梯度范数 {float(grad_norm):.2f} | 学习率 {cur_lr:.2e} "
                f"| {tok_per_sec / 1e6:.2f}M tok/s "
                f"| 已处理 {tokens_seen / 1e9:.1f}B token"
            )
            t0 = time.perf_counter()

        if step % ckpt_every == 0:
            save_checkpoint(
                model, optimizer, step, cfg, vocab_size, ckpt_dir, ddp, master
            )

    # 最终检查点 — total_steps 可能不能被 ckpt_every 整除，
    # 因此如果调度未对齐，没有这一步训练尾部就会丢失。
    if step > start_step and step % ckpt_every != 0:
        save_checkpoint(model, optimizer, step, cfg, vocab_size, ckpt_dir, ddp, master)

    if ddp:
        # 屏障确保没有进程在另一个进程仍在完成检查点收集时退出 —
        # 避免 NCCL "进程组已销毁" 的噪声。
        dist.barrier()
        dist.destroy_process_group()

    if master:
        logger.success("训练完成。")


if __name__ == "__main__":
    main()
