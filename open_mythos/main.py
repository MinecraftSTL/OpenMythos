from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from flash_attn import flash_attn_func

    _HAS_FLASH_ATTN = True
except ImportError:
    _HAS_FLASH_ATTN = False


@dataclass
class MythosConfig:
    """
    OpenMythos 超参数配置。

    核心参数:
        vocab_size      -- 词汇表大小
        dim             -- 模型隐藏维度
        n_heads         -- 查询注意力头数
        n_kv_heads      -- 键/值头数（GQA；MLA 忽略此参数）
        max_seq_len     -- RoPE 预计算的最大序列长度
        max_loop_iters  -- 推理时默认的递归循环深度 T
        prelude_layers  -- 循环前的标准 Transformer 层数
        coda_layers     -- 循环后的标准 Transformer 层数

    注意力（attn_type 在两者间选择）:
        attn_type       -- "gqa" 为分组查询注意力，"mla" 为多潜在注意力
        kv_lora_rank    -- [MLA] 缓存中存储的压缩 KV 潜在维度
        q_lora_rank     -- [MLA] 压缩 Q 潜在维度
        qk_rope_head_dim-- [MLA] 接收 RoPE 的每头维度
        qk_nope_head_dim-- [MLA] 不含位置编码的每头维度
        v_head_dim      -- [MLA] 每头值维度

    MoE FFN（在递归块内部使用）:
        n_experts       -- 路由专家 FFN 总数
        n_shared_experts-- 始终激活的共享专家数
        n_experts_per_tok-- 路由器为每个 token 选择的 Top-K 专家数
        expert_dim      -- 每个细粒度专家内部的隐藏维度

    其他:
        act_threshold   -- ACT 停止阈值（累积概率达到此值时停止循环）
        rope_theta      -- RoPE 基础频率
        lora_rank       -- 每循环深度级 LoRA 适配器的秩
    """

    vocab_size: int = 32000
    dim: int = 2048
    n_heads: int = 16
    n_kv_heads: int = 4  # GQA：KV 头数少于 Q 头数
    max_seq_len: int = 4096
    max_loop_iters: int = 16  # T — 推理时的递归深度
    prelude_layers: int = 2
    coda_layers: int = 2
    # 注意力类型: "gqa" | "mla"
    attn_type: str = "mla"
    # MLA 参数（仅在 attn_type="mla" 时使用）
    kv_lora_rank: int = 512  # 压缩 KV 潜在表示，替代完整 K/V 缓存
    q_lora_rank: int = 1536  # 压缩 Q 潜在维度
    qk_rope_head_dim: int = 64  # 接收 RoPE 的每头维度
    qk_nope_head_dim: int = 128  # 不含 RoPE 的每头维度
    v_head_dim: int = 128  # 每头值维度
    # 混合专家
    n_experts: int = 64
    n_shared_experts: int = 2
    n_experts_per_tok: int = 4  # Top-K 路由
    expert_dim: int = 512  # 细粒度: dim // (n_experts // n_experts_per_tok)
    # ACT 停止
    act_threshold: float = 0.99
    # RoPE 旋转位置编码
    rope_theta: float = 500000.0
    # LoRA 深度适配
    lora_rank: int = 16
    # 每次前向传播生成的最大 token 数
    max_output_tokens: int = 4096
    # Dropout（设为 0.0 禁用；0.1 为预训练标准值）
    dropout: float = 0.0


# ---------------------------------------------------------------------------
# RMSNorm 均方根归一化
# ---------------------------------------------------------------------------


class RMSNorm(nn.Module):
    """
    均方根层归一化（Zhang & Sennrich, 2019）。

    通过输入的均方根（而非均值+方差）进行归一化，带有可学习的逐通道
    缩放权重。无偏置项。在整个模型中替代 LayerNorm 使用，以提高
    稳定性和效率。
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        """
        参数:
            dim -- 归一化的特征维度
            eps -- sqrt 前添加的小常数，用于数值稳定性
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        参数:
            x -- 形状为 (..., dim) 的输入张量
        返回:
            相同形状的 RMS 归一化张量，经 self.weight 缩放
        """
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.weight


# ---------------------------------------------------------------------------
# RoPE 旋转位置编码
# ---------------------------------------------------------------------------


def precompute_rope_freqs(
    dim: int, max_len: int, theta: float = 500000.0
) -> torch.Tensor:
    """
    预计算位置 0..max_len-1 的复数值 RoPE 旋转矩阵。

    每个位置对每个频率对 k 获得一个复数相量 e^{i·m·θ_k}。
    存储为复数张量，使旋转操作仅需一次逐点乘法。

    参数:
        dim     -- 头维度（必须为偶数）；为 dim//2 对频率计算
        max_len -- 预计算的最大序列长度
        theta   -- RoPE 基础频率（越高 = 频率衰减越慢；500k 为 LLaMA-3 默认值）

    返回:
        形状为 (max_len, dim//2) 的 complex64 张量
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    t = torch.arange(max_len, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)


def apply_rope(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """
    将旋转位置编码应用于查询或键张量。

    将每对相邻特征解释为二维复数，并与该位置预计算的相量相乘，
    在复平面上旋转表示而不改变其范数。

    参数:
        x         -- 形状为 (B, T, H, head_dim) 的张量；head_dim 必须为偶数
        freqs_cis -- 形状为 (T, head_dim//2) 的预计算复数频率，
                     已切片到正在处理的确切位置
                     （调用者负责正确的 start_pos 偏移）

    返回:
        与 x 形状和数据类型相同的旋转后张量
    """
    xc = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    return (
        torch.view_as_real(xc * freqs_cis.unsqueeze(0).unsqueeze(2))
        .flatten(-2)
        .to(x.dtype)
    )


# ---------------------------------------------------------------------------
# 分组查询注意力（带 KV 缓存）
# ---------------------------------------------------------------------------


class GQAttention(nn.Module):
    """
    分组查询注意力（Ainslie et al., 2023），配合 Flash Attention 2（Dao et al., 2023）。

    使用比 Q 头更少的 KV 头（n_kv_heads < n_heads）。每个 KV 头在
    n_heads // n_kv_heads 个查询头之间共享，按该因子减少 KV 缓存大小，
    同时保持完整的查询表达能力。

    安装 flash-attn 时，使用原生支持 GQA 的 flash_attn_func
    （无需 KV 头扩展），且为 IO 最优。输入转换为 bfloat16 传入
    flash_attn，之后恢复原始数据类型。
    未安装 flash-attn 时回退到手动缩放点积注意力。

    RoPE 同时应用于 Q 和 K。K 和 V 在 RoPE 应用后存入 kv_cache，
    因此缓存值已包含位置编码，检索时无需重新旋转。
    """

    def __init__(self, cfg: MythosConfig):
        """
        参数:
            cfg -- MythosConfig；使用 dim、n_heads、n_kv_heads
        """
        super().__init__()
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.dim // cfg.n_heads
        self.groups = cfg.n_heads // cfg.n_kv_heads

        self.wq = nn.Linear(cfg.dim, cfg.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(cfg.dim, cfg.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(cfg.dim, cfg.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(cfg.n_heads * self.head_dim, cfg.dim, bias=False)
        self.dropout_p = cfg.dropout

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[dict] = None,
        cache_key: str = "default",
    ) -> torch.Tensor:
        """
        参数:
            x         -- 形状为 (B, T, dim) 的输入
            freqs_cis -- head_dim 的 RoPE 频率，形状 (T, head_dim//2)
            mask      -- 形状为 (1, 1, T, S) 的加性因果掩码或 None
            kv_cache  -- 原地修改的字典；按 cache_key 存储 {"k": ..., "v": ...}
            cache_key -- 在缓存字典中标识此层的唯一键

        返回:
            形状为 (B, T, dim) 的输出张量
        """
        B, T, _ = x.shape
        q = self.wq(x).view(B, T, self.n_heads, self.head_dim)
        k = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim)
        v = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim)

        q = apply_rope(q, freqs_cis)
        k = apply_rope(k, freqs_cis)

        if kv_cache is not None:
            if cache_key in kv_cache:
                k = torch.cat([kv_cache[cache_key]["k"], k], dim=1)
                v = torch.cat([kv_cache[cache_key]["v"], v], dim=1)
            kv_cache[cache_key] = {"k": k.detach(), "v": v.detach()}

        if _HAS_FLASH_ATTN:
            # flash_attn_func 期望 (B, T, H, head_dim) — 原生支持 GQA
            # （n_kv_heads < n_heads 无需 repeat_interleave）。
            # 有掩码时 causal=True（全序列预填充/训练）；
            # 单 token 解码（T=1 且 mask 为 None）时 causal=False。
            orig_dtype = q.dtype
            q = q.to(torch.bfloat16)
            k = k.to(torch.bfloat16)
            v = v.to(torch.bfloat16)
            dropout_p = self.dropout_p if self.training else 0.0
            out = flash_attn_func(
                q, k, v, dropout_p=dropout_p, causal=(mask is not None)
            )
            out = out.to(orig_dtype).contiguous().view(B, T, -1)
        else:
            # 回退：手动缩放点积注意力，显式扩展 KV 头。
            k = k.repeat_interleave(self.groups, dim=2)
            v = v.repeat_interleave(self.groups, dim=2)
            q = q.transpose(1, 2)  # (B, H, T, head_dim)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            scale = self.head_dim**-0.5
            attn = torch.matmul(q, k.transpose(-2, -1)) * scale
            if mask is not None:
                attn = attn + mask
            attn = F.dropout(
                F.softmax(attn, dim=-1), p=self.dropout_p, training=self.training
            )
            out = torch.matmul(attn, v)
            out = out.transpose(1, 2).contiguous().view(B, T, -1)

        return self.wo(out)


# ---------------------------------------------------------------------------
# 多潜在注意力（DeepSeek-V2 风格）
# ---------------------------------------------------------------------------


class MLAttention(nn.Module):
    """
    多潜在注意力（DeepSeek-V2, 2024）。

    核心思想：不缓存完整的 K 和 V 张量（每个 token 大小为
    n_heads × head_dim），MLA 通过低秩潜在表示 c_kv 压缩 KV 路径，
    仅缓存该表示加上 RoPE 键。K_nope 和 V 在每个解码步骤从 c_kv
    重建，用廉价的线性投影换取显著更小的缓存内存。

    Q 路径:
        x → q_down (dim→q_lora_rank) → q_norm
          → q_up_nope (q_lora_rank → n_heads×qk_nope_head_dim)  [无 RoPE]
          → q_up_rope (q_lora_rank → n_heads×qk_rope_head_dim)  [应用 RoPE]
        q = cat(q_nope, q_rope)  每头

    KV 路径:
        x → kv_down (dim → kv_lora_rank + qk_rope_head_dim)
          分割为 c_kv（潜在表示，缓存）和 k_rope_raw（跨头共享）
        k_rope = RoPE(expand(k_rope_raw))  — 缓存前应用
        c_kv → kv_norm → kv_up → [k_nope | v]  — 每步重建
        k = cat(k_nope, k_rope)  每头

    缓存存储: c_kv (kv_lora_rank) + k_rope (n_heads × qk_rope_head_dim)，
    对比完整 GQA 缓存: n_kv_heads × head_dim × 2。在生产规模下
    约减少 10-20 倍内存。
    """

    def __init__(self, cfg: MythosConfig):
        """
        参数:
            cfg -- MythosConfig；使用 dim、n_heads、kv_lora_rank、q_lora_rank、
                   qk_rope_head_dim、qk_nope_head_dim、v_head_dim
        """
        super().__init__()
        self.n_heads = cfg.n_heads
        self.kv_lora_rank = cfg.kv_lora_rank
        self.qk_rope_dim = cfg.qk_rope_head_dim
        self.qk_nope_dim = cfg.qk_nope_head_dim
        self.v_dim = cfg.v_head_dim
        self.q_head_dim = cfg.qk_nope_head_dim + cfg.qk_rope_head_dim

        # Q 压缩
        self.q_down = nn.Linear(cfg.dim, cfg.q_lora_rank, bias=False)
        self.q_norm = RMSNorm(cfg.q_lora_rank)
        self.q_up_nope = nn.Linear(
            cfg.q_lora_rank, cfg.n_heads * cfg.qk_nope_head_dim, bias=False
        )
        self.q_up_rope = nn.Linear(
            cfg.q_lora_rank, cfg.n_heads * cfg.qk_rope_head_dim, bias=False
        )

        # KV 压缩：输出为 [c_kv | k_rope_raw] 拼接
        self.kv_down = nn.Linear(
            cfg.dim, cfg.kv_lora_rank + cfg.qk_rope_head_dim, bias=False
        )
        self.kv_norm = RMSNorm(cfg.kv_lora_rank)
        self.kv_up = nn.Linear(
            cfg.kv_lora_rank,
            cfg.n_heads * (cfg.qk_nope_head_dim + cfg.v_head_dim),
            bias=False,
        )

        self.wo = nn.Linear(cfg.n_heads * cfg.v_head_dim, cfg.dim, bias=False)
        self.attn_drop = nn.Dropout(cfg.dropout)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[dict] = None,
        cache_key: str = "default",
    ) -> torch.Tensor:
        """
        参数:
            x         -- 形状为 (B, T, dim) 的输入
            freqs_cis -- qk_rope_head_dim 大小的 RoPE 频率，形状 (T, rope_dim//2)
            mask      -- 形状为 (1, 1, T, S) 的加性因果掩码或 None
            kv_cache  -- 原地修改的字典；存储 {"c_kv": ..., "k_rope": ...}
            cache_key -- 在缓存字典中标识此层的唯一键

        返回:
            形状为 (B, T, dim) 的输出张量
        """
        B, T, _ = x.shape

        # Q
        c_q = self.q_norm(self.q_down(x))
        q_nope = self.q_up_nope(c_q).view(B, T, self.n_heads, self.qk_nope_dim)
        q_rope = self.q_up_rope(c_q).view(B, T, self.n_heads, self.qk_rope_dim)
        q_rope = apply_rope(q_rope, freqs_cis)
        q = torch.cat([q_nope, q_rope], dim=-1)  # (B, T, H, nope+rope)

        # KV compress
        kv_raw = self.kv_down(x)
        c_kv = kv_raw[..., : self.kv_lora_rank]  # (B, T, lora_rank)  ← cached
        k_rope = kv_raw[..., self.kv_lora_rank :]  # (B, T, rope_dim)
        # 跨头扩展 rope 键并在缓存前应用 RoPE，
        # 使检索到的键已包含位置编码
        k_rope = (
            k_rope.unsqueeze(2)
            .expand(B, T, self.n_heads, self.qk_rope_dim)
            .contiguous()
        )
        k_rope = apply_rope(k_rope, freqs_cis)  # (B, T, H, rope_dim) ← cached

        if kv_cache is not None:
            if cache_key in kv_cache:
                c_kv = torch.cat([kv_cache[cache_key]["c_kv"], c_kv], dim=1)
                k_rope = torch.cat([kv_cache[cache_key]["k_rope"], k_rope], dim=1)
            kv_cache[cache_key] = {"c_kv": c_kv.detach(), "k_rope": k_rope.detach()}

        S = c_kv.shape[1]  # 包含缓存的完整序列长度

        # 从潜在表示重建 K_nope 和 V（不缓存，每步重新计算）
        kv = self.kv_up(self.kv_norm(c_kv))  # (B, S, H*(nope+v))
        kv = kv.view(B, S, self.n_heads, self.qk_nope_dim + self.v_dim)
        k_nope = kv[..., : self.qk_nope_dim]  # (B, S, H, nope)
        v = kv[..., self.qk_nope_dim :]  # (B, S, H, v_dim)
        k = torch.cat([k_nope, k_rope], dim=-1)  # (B, S, H, nope+rope)

        # 注意力计算
        q = q.transpose(1, 2)  # (B, H, T, q_head_dim)
        k = k.transpose(1, 2)  # (B, H, S, q_head_dim)
        v = v.transpose(1, 2)  # (B, H, S, v_dim)

        scale = self.q_head_dim**-0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        if mask is not None:
            attn = attn + mask
        attn = self.attn_drop(F.softmax(attn, dim=-1))
        out = torch.matmul(attn, v)  # (B, H, T, v_dim)
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.wo(out)


# ---------------------------------------------------------------------------
# DeepSeek 风格混合专家 FFN
# ---------------------------------------------------------------------------


class Expert(nn.Module):
    """
    单个 SwiGLU 前馈专家。

    实现门控线性单元变体：output = down(silu(gate(x)) * up(x))。
    既用作 MoEFFN 内部的单个路由专家，也用作前奏/尾声块中的标准
    密集 FFN（其中 expert_dim = dim * 4 // 3）。
    """

    def __init__(self, dim: int, expert_dim: int):
        """
        参数:
            dim        -- 输入和输出特征维度
            expert_dim -- 专家内部（隐藏）维度
        """
        super().__init__()
        self.gate = nn.Linear(dim, expert_dim, bias=False)
        self.up = nn.Linear(dim, expert_dim, bias=False)
        self.down = nn.Linear(expert_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        参数:
            x -- 形状为 (..., dim) 的输入
        返回:
            形状为 (..., dim) 的张量
        """
        return self.down(F.silu(self.gate(x)) * self.up(x))


class MoEFFN(nn.Module):
    """
    细粒度混合专家 FFN（DeepSeekMoE, Dai et al., 2024）。

    两类专家:
    - 路由专家：n_experts 个小型 FFN；每个 token 通过学习的路由器激活其中
      Top-K 个。路由器 logits 上的每专家偏置在训练中更新，以保持专家间
      负载均衡而不扭曲损失。
    - 共享专家：n_shared_experts 个较大的 FFN，对每个 token 始终激活，
      吸收跨领域的通用模式（语法、基础推理），否则这些模式会被多个
      路由专家冗余学习。

    每 token 激活的总参数 ≈ 路由容量的 topk/n_experts + 所有共享容量，
    保持计算稀疏的同时总参数量保持较大。
    """

    def __init__(self, cfg: MythosConfig):
        """
        参数:
            cfg -- MythosConfig；使用 n_experts、n_shared_experts、n_experts_per_tok、
                   dim、expert_dim
        """
        super().__init__()
        self.n_experts = cfg.n_experts
        self.n_shared = cfg.n_shared_experts
        self.topk = cfg.n_experts_per_tok

        self.router = nn.Linear(cfg.dim, cfg.n_experts, bias=False)
        # 训练期间外部调整的负载均衡偏置；非梯度参数
        self.register_buffer("router_bias", torch.zeros(cfg.n_experts))

        self.routed_experts = nn.ModuleList(
            [Expert(cfg.dim, cfg.expert_dim) for _ in range(cfg.n_experts)]
        )
        self.shared_experts = nn.ModuleList(
            [
                Expert(cfg.dim, cfg.expert_dim * cfg.n_experts_per_tok)
                for _ in range(self.n_shared)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        参数:
            x -- 形状为 (B, T, dim) 的输入
        返回:
            形状为 (B, T, dim) 的张量；共享专家输出叠加在加权路由专家输出之上
        """
        B, T, D = x.shape
        flat = x.view(B * T, D)

        # 无辅助损失的负载均衡（DeepSeek-V3）：偏置仅影响专家选择，
        # 使利用不足的专家被更多选中，但门控权重来自无偏的 softmax 分数，
        # 因此偏置不会出现在
        logits = self.router(flat)  # (B*T, n_experts), unbiased
        scores = F.softmax(logits, dim=-1)
        _, topk_idx = (logits + self.router_bias).topk(self.topk, dim=-1)
        topk_scores = scores.gather(-1, topk_idx)
        topk_scores = topk_scores / topk_scores.sum(dim=-1, keepdim=True)  # renorm

        # 路由专家分发（token 级散射）
        out = torch.zeros_like(flat)
        for i in range(self.topk):
            expert_ids = topk_idx[:, i]
            token_scores = topk_scores[:, i].unsqueeze(-1)
            for eid in range(self.n_experts):
                mask = expert_ids == eid
                if not mask.any():
                    continue
                out[mask] += token_scores[mask] * self.routed_experts[eid](flat[mask])

        # 共享专家对每个 token 始终激活
        for shared in self.shared_experts:
            out = out + shared(flat)

        return out.view(B, T, D)


# ---------------------------------------------------------------------------
# 循环索引 RoPE（区分递归块的不同迭代）
# ---------------------------------------------------------------------------


def loop_index_embedding(
    h: torch.Tensor, loop_t: int, loop_dim: int, theta: float = 10000.0
) -> torch.Tensor:
    """
    将正弦循环索引信号注入 h 的前 loop_dim 个通道。

    类似于序列位置的 RoPE，但作用于递归深度而非 token 位置。
    没有此信号时，共享的递归块权重必须在无法区分当前循环迭代的情况下
    同时处理早期模式匹配和后期精炼。添加循环索引使相同参数在每次
    迭代中实现功能上不同的操作。

    参数:
        h        -- 形状为 (B, T, dim) 的隐藏状态张量
        loop_t   -- 当前循环迭代索引（从 0 开始）
        loop_dim -- 接收嵌入的前导通道数（必须为偶数）
        theta    -- 正弦基础频率

    返回:
        前 loop_dim 个通道添加了正弦偏置的 h；形状不变
    """
    freqs = 1.0 / (
        theta
        ** (torch.arange(0, loop_dim, 2, device=h.device, dtype=h.dtype) / loop_dim)
    )
    angles = loop_t * freqs  # (loop_dim//2,)
    emb = torch.cat([angles.sin(), angles.cos()], dim=-1)[:loop_dim]
    emb_full = torch.zeros(h.shape[-1], device=h.device, dtype=h.dtype)
    emb_full[:loop_dim] = emb
    return h + emb_full.unsqueeze(0).unsqueeze(0)


# ---------------------------------------------------------------------------
# 深度级 LoRA 适配器（每循环迭代）
# ---------------------------------------------------------------------------


class LoRAAdapter(nn.Module):
    """
    递归块的深度级 LoRA 适配（Bae et al., 2024）。

    纯权重绑定（每次循环使用相同权重）限制了表达能力；
    每次循环使用完全独立的权重则消除了参数节省。此适配器
    介于两者之间：共享的低秩降维投影和升维矩阵 B 在所有循环间共享，
    而小型的每循环缩放向量在每个深度调整有效变换，
    不增加显著参数。

    delta(x, t) = (down(x) * scale[t]) @ B
    """

    def __init__(self, dim: int, rank: int, max_loops: int):
        """
        参数:
            dim       -- 模型隐藏维度（输入和输出大小）
            rank      -- 低秩瓶颈维度
            max_loops -- 最大循环迭代次数（决定嵌入表大小）
        """
        super().__init__()
        self.down = nn.Linear(dim, rank, bias=False)  # 共享 A: dim → rank
        self.B = nn.Parameter(torch.randn(rank, dim) * 0.02)  # 共享 B: rank → dim
        self.scale = nn.Embedding(max_loops, rank)  # 每循环的逐元素缩放

    def forward(self, x: torch.Tensor, loop_t: int) -> torch.Tensor:
        """
        参数:
            x      -- 形状为 (B, T, dim) 的输入张量
            loop_t -- 当前循环索引，用于查找每循环缩放

        返回:
            形状为 (B, T, dim) 的增量张量，添加到块输出上
        """
        # 深度外推的截断：推理时 n_loops 可能超过训练时的 max_loop_iters。
        # 超出训练范围的迭代复用最后学习的每循环缩放，而非越界索引。
        max_t = self.scale.num_embeddings - 1
        t_idx = loop_t if loop_t <= max_t else max_t
        s = self.scale(torch.tensor(t_idx, device=x.device))  # (rank,)
        down = self.down(x) * s  # (B, T, rank)
        return down @ self.B  # (B, T, dim)


# ---------------------------------------------------------------------------
# 单个 Transformer 块（在递归循环间共享）
# ---------------------------------------------------------------------------


class TransformerBlock(nn.Module):
    """
    标准预归一化 Transformer 块，支持可切换的注意力和可选的 MoE FFN。

    注意力由 cfg.attn_type 选择:
        "gqa" → GQAttention  （分组查询注意力，更少的 KV 头）
        "mla" → MLAttention  （多潜在注意力，压缩 KV 缓存）

    FFN 由 use_moe 选择:
        True  → MoEFFN  （细粒度路由 + 共享专家；用于 RecurrentBlock）
        False → Expert  （密集 SwiGLU FFN；用于前奏和尾声）
    """

    def __init__(self, cfg: MythosConfig, use_moe: bool = False):
        """
        参数:
            cfg     -- MythosConfig；attn_type 选择注意力类
            use_moe -- 为 True 时使用 MoEFFN；否则使用密集 Expert FFN
        """
        super().__init__()
        self.attn_norm = RMSNorm(cfg.dim)
        self.ffn_norm = RMSNorm(cfg.dim)
        self.attn = MLAttention(cfg) if cfg.attn_type == "mla" else GQAttention(cfg)
        self.ffn = MoEFFN(cfg) if use_moe else Expert(cfg.dim, cfg.dim * 4 // 3)
        self.resid_drop = nn.Dropout(cfg.dropout)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[dict] = None,
        cache_key: str = "default",
    ) -> torch.Tensor:
        """
        参数:
            x         -- 形状为 (B, T, dim) 的输入
            freqs_cis -- 预计算的 RoPE 频率
            mask      -- 加性因果掩码或 None
            kv_cache  -- 注意力层原地修改的缓存字典
            cache_key -- 在缓存中标识此层的键

        返回:
            形状为 (B, T, dim) 的输出张量
        """
        x = x + self.resid_drop(
            self.attn(self.attn_norm(x), freqs_cis, mask, kv_cache, cache_key)
        )
        x = x + self.resid_drop(self.ffn(self.ffn_norm(x)))
        return x


# ---------------------------------------------------------------------------
# LTI 稳定注入参数（谱半径 < 1，由构造保证）
# ---------------------------------------------------------------------------


class LTIInjection(nn.Module):
    """
    递归更新规则的稳定输入注入（Parcae, Prairie et al., 2026）。

    递归隐藏状态按以下规则演化:
        h_{t+1} = A · h_t  +  B · e  +  Transformer(h_t, e)

    其中 e 是在每个循环步骤注入的编码输入，防止漂移。
    无约束时，A 可能发展出谱半径 ≥ 1，导致隐藏状态在循环迭代中
    爆炸并使训练不稳定。

    此类通过 ZOH 离散化从构造上保证 ρ(A) < 1:
        A_continuous = Diag(-exp(log_A))       始终为负对角
        A_discrete   = exp(Δt · A_continuous)  逐元素，值 ∈ (0, 1)

    其中 log_A 和 log_dt 是可学习参数，exp 确保正值。
    这使循环模型训练对超参数选择鲁棒，即使在高学习率下也保持稳定。
    """

    def __init__(self, dim: int):
        """
        参数:
            dim -- 隐藏状态维度；A 和 B 每通道一个标量
        """
        super().__init__()
        self.log_A = nn.Parameter(torch.zeros(dim))  # A_continuous 幅度的对数
        self.log_dt = nn.Parameter(torch.zeros(1))  # 离散化步长 Δt 的对数
        self.B = nn.Parameter(torch.ones(dim) * 0.1)

    def get_A(self) -> torch.Tensor:
        """
        计算离散化对角状态矩阵 A_discrete。

        返回:
            形状为 (dim,) 的一维张量，所有值严格在 (0, 1) 内，
            无论学习参数值如何都保证 ρ(A) < 1。
        """
        # 在对数空间计算以避免 log_dt → -∞, log_A → +∞ 时 0 * inf = NaN。
        # dt * A_c = -exp(log_dt) * exp(log_A) = -exp(log_dt + log_A)
        # 截断确保乘积在 float32 中对任何梯度步长都有限。
        return torch.exp(-torch.exp((self.log_dt + self.log_A).clamp(-20, 20)))

    def forward(
        self, h: torch.Tensor, e: torch.Tensor, transformer_out: torch.Tensor
    ) -> torch.Tensor:
        """
        计算 h_{t+1} = A·h_t + B·e + transformer_out。

        参数:
            h               -- 当前隐藏状态 (B, T, dim)
            e               -- 来自前奏的编码输入，跨循环冻结 (B, T, dim)
            transformer_out -- 此步骤递归 TransformerBlock 的输出 (B, T, dim)

        返回:
            形状为 (B, T, dim) 的更新后隐藏状态
        """
        A = self.get_A()
        return A * h + self.B * e + transformer_out


# ---------------------------------------------------------------------------
# ACT 停止（自适应计算时间）
# ---------------------------------------------------------------------------


class ACTHalting(nn.Module):
    """
    自适应计算时间停止机制（Graves, 2016）。

    在每个循环迭代中学习每位置的停止概率。隐藏状态已收敛的位置
    （高累积停止概率）停止累积更新，而仍在精炼的位置继续。
    这使简单 token 提前停止，困难 token 获得更多计算，
    全部在同一批次内完成。在关于 Transformer 块表达能力的
    某些假设下，还使模型具有图灵完备性。
    """

    def __init__(self, dim: int):
        """
        参数:
            dim -- 隐藏状态维度；停止标量预测器的输入
        """
        super().__init__()
        self.halt = nn.Linear(dim, 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        从当前隐藏状态预测每位置的停止概率。

        参数:
            h -- 形状为 (B, T, dim) 的隐藏状态

        返回:
            形状为 (B, T) 的停止概率张量，值在 (0, 1) 内
        """
        return torch.sigmoid(self.halt(h)).squeeze(-1)


# ---------------------------------------------------------------------------
# 递归块（一组权重，循环 T 次）
# ---------------------------------------------------------------------------


class RecurrentBlock(nn.Module):
    """
    OpenMythos 的核心递归块 — 单个 TransformerBlock 循环 T 次。

    在每个循环迭代 t 中，隐藏状态 h 通过以下步骤更新:
        1. loop_index_embedding: 将正弦循环索引信号注入 h
        2. TransformerBlock:     在归一化的 (h + e) 上计算注意力 + MoE FFN
        3. LoRAAdapter:          对 Transformer 输出应用深度级 LoRA 增量
        4. LTIInjection:         稳定更新 h = A·h + B·e + transformer_out
        5. ACTHalting:           累积每位置停止概率；
                                  已收敛的位置停止贡献

    编码输入 e（前奏的输出）在每步注入，以在任意循环深度下保持
    原始输入信号存活，防止漂移。ACT 机制产生跨迭代的隐藏状态加权和，
    权重反映每个位置何时收敛。

    推理时更多循环迭代 = 更深的推理链，遵循循环 Transformer 的
    深度外推特性（Saunshi et al., 2025）。
    """

    def __init__(self, cfg: MythosConfig):
        """
        参数:
            cfg -- MythosConfig；使用 dim、lora_rank、max_loop_iters、act_threshold
        """
        super().__init__()
        self.cfg = cfg
        self.block = TransformerBlock(cfg, use_moe=True)
        self.injection = LTIInjection(cfg.dim)
        self.act = ACTHalting(cfg.dim)
        self.lora = LoRAAdapter(cfg.dim, cfg.lora_rank, cfg.max_loop_iters)
        self.norm = RMSNorm(cfg.dim)
        self.loop_dim = (
            cfg.dim // 8
        )  # 接收循环索引嵌入的通道比例

    def forward(
        self,
        h: torch.Tensor,
        e: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        n_loops: Optional[int] = None,
        kv_cache: Optional[dict] = None,
    ) -> torch.Tensor:
        """
        运行递归循环，最多 n_loops 次迭代，支持 ACT 提前退出。

        参数:
            h        -- 来自前奏的初始隐藏状态，形状 (B, T, dim)
            e        -- 每步注入的冻结编码输入，形状 (B, T, dim)
            freqs_cis-- 预计算的 RoPE 频率
            mask     -- 加性因果掩码或 None
            n_loops  -- 循环迭代次数；默认为 cfg.max_loop_iters。
                        推理时可增大以获得更深推理（深度外推）。
            kv_cache -- 传递给内部 TransformerBlock 的缓存字典；
                        每次循环迭代使用单独的缓存键

        返回:
            跨迭代的 ACT 加权隐藏状态和，形状 (B, T, dim)
        """
        n_loops = n_loops or self.cfg.max_loop_iters
        B, T, D = h.shape

        halted = torch.zeros(B, T, device=h.device, dtype=torch.bool)
        cumulative_p = torch.zeros(B, T, device=h.device)
        h_out = torch.zeros_like(h)

        for t in range(n_loops):
            h_loop = loop_index_embedding(h, t, self.loop_dim)
            combined = self.norm(h_loop + e)
            cache_key = f"recurrent_loop_{t}"
            trans_out = self.block(combined, freqs_cis, mask, kv_cache, cache_key)
            trans_out = trans_out + self.lora(trans_out, t)
            h = self.injection(h, e, trans_out)

            p = self.act(h)  # (B, T)
            still_running = ~halted

            # ACT 余量技巧：一旦 cumulative_p + p 超过阈值，
            # 将剩余概率质量分配为最终权重。
            # 通过 still_running 门控，使已停止的位置仅贡献一次
            # （在停止步骤），之后为零——否则 threshold<1 会留下
            # 非零余量在每步泄漏。
            remainder = (1.0 - cumulative_p).clamp(min=0)
            weight = torch.where(
                cumulative_p + p >= self.cfg.act_threshold,
                remainder,
                p,
            )
            weight = weight * still_running.float()
            h_out = h_out + weight.unsqueeze(-1) * h

            cumulative_p = cumulative_p + p * still_running.float()
            halted = halted | (cumulative_p >= self.cfg.act_threshold)

            # 仅在没有 KV 缓存需要保持一致时才短路退出。
            # 有缓存时，每次前向传播必须运行所有循环深度，
            # 以确保后续解码步骤在每个 cache_key 都能找到已填充的键。
            if halted.all() and kv_cache is None:
                break

        return h_out


# ---------------------------------------------------------------------------
# 完整模型
# ---------------------------------------------------------------------------


class OpenMythos(nn.Module):
    """
    OpenMythos — 递归深度 Transformer 语言模型。

    将假设的 Claude Mythos 架构实现为递归深度 Transformer（RDT）。
    模型将计算分为三个功能块:

        输入 token
             ↓
        [前奏]             — prelude_layers 个标准 Transformer 块，运行一次
             ↓
        [递归块]           — 一个 Transformer 块循环 T 次，带输入注入
             ↑_______↓      h_{t+1} = A·h_t + B·e + Transformer(h_t, e)
             ↓
        [尾声]             — coda_layers 个标准 Transformer 块，运行一次
             ↓
        输出 logits

    关键特性:
    - 相同权重，更多循环 → 更深推理，无参数增长
    - 深度外推：在 N 次循环上训练，在 N+k 次循环上测试（涌现特性）
    - ACT 停止：批次内每位置可变计算量
    - 递归块中的 MoE FFN：跨领域广度
    - LTI 稳定注入：谱半径 < 1 由构造保证
    - 支持 GQA 和 MLA 两种注意力（通过 cfg.attn_type 设置）
    """

    def __init__(self, cfg: MythosConfig):
        """
        参数:
            cfg -- 指定所有架构超参数的 MythosConfig
        """
        super().__init__()
        self.cfg = cfg

        self.embed = nn.Embedding(cfg.vocab_size, cfg.dim)

        # GQA 使用完整 head_dim 进行 RoPE；MLA 仅使用 qk_rope_head_dim（解耦）
        freqs = precompute_rope_freqs(
            cfg.dim // cfg.n_heads, cfg.max_seq_len, cfg.rope_theta
        )
        self.register_buffer("freqs_cis", freqs)
        freqs_mla = precompute_rope_freqs(
            cfg.qk_rope_head_dim, cfg.max_seq_len, cfg.rope_theta
        )
        self.register_buffer("freqs_cis_mla", freqs_mla)

        self.prelude = nn.ModuleList(
            [TransformerBlock(cfg, use_moe=False) for _ in range(cfg.prelude_layers)]
        )
        self.recurrent = RecurrentBlock(cfg)
        self.coda = nn.ModuleList(
            [TransformerBlock(cfg, use_moe=False) for _ in range(cfg.coda_layers)]
        )

        self.norm = RMSNorm(cfg.dim)
        self.head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        self.head.weight = self.embed.weight  # 权重绑定

        self._init_weights()

    def _init_weights(self) -> None:
        """使用 N(0, 0.02) 初始化所有线性层和嵌入层权重。"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    @staticmethod
    def _causal_mask(
        seq_len: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """
        构建加性因果掩码：对角线及以下为 0，以上为 -inf。

        参数:
            seq_len -- 序列长度
            device  -- 目标设备
            dtype   -- 张量数据类型（必须与激活数据类型匹配，以免加性掩码
                       在回退注意力路径中提升注意力 logits 的精度——例如
                       bf16 权重配合 fp32 掩码会将注意力提升为 fp32，
                       然后与 V 的 fp32-vs-bf16 矩阵乘法冲突）

        返回:
            形状为 (1, 1, seq_len, seq_len) 的张量，可广播到 (B, H, T, S)
        """
        mask = torch.full(
            (1, 1, seq_len, seq_len), float("-inf"), device=device, dtype=dtype
        )
        return torch.triu(mask, diagonal=1)

    def forward(
        self,
        input_ids: torch.Tensor,
        n_loops: Optional[int] = None,
        kv_cache: Optional[dict] = None,
        start_pos: int = 0,
    ) -> torch.Tensor:
        """
        通过 前奏 → 递归块 → 尾声 的前向传播。

        参数:
            input_ids -- 形状为 (B, T) 的 token 索引
            n_loops   -- 递归循环深度；默认为 cfg.max_loop_iters。
                         推理时增大可外推到更难的问题。
            kv_cache  -- 原地修改的自回归 KV 缓存字典；
                         传入空字典 {} 并在各解码步骤间复用
            start_pos -- input_ids 中第一个 token 在完整序列中的索引；
                         用于在增量解码时选择正确的 RoPE 频率
                         （预填充时为 0，后续每个解码步骤为 prompt_len）

        返回:
            形状为 (B, T, vocab_size) 的 logits
        """
        T = input_ids.shape[1]
        device = input_ids.device

        x = self.embed(input_ids)
        freqs_cis = (
            self.freqs_cis_mla if self.cfg.attn_type == "mla" else self.freqs_cis
        )[start_pos : start_pos + T]
        mask = self._causal_mask(T, device, x.dtype) if T > 1 else None

        for i, layer in enumerate(self.prelude):
            x = layer(x, freqs_cis, mask, kv_cache, cache_key=f"prelude_{i}")

        e = x  # 编码输入，冻结后在每次循环中注入
        x = self.recurrent(x, e, freqs_cis, mask, n_loops, kv_cache)

        for i, layer in enumerate(self.coda):
            x = layer(x, freqs_cis, mask, kv_cache, cache_key=f"coda_{i}")

        return self.head(self.norm(x))

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 64,
        n_loops: int = 8,
        temperature: float = 1.0,
        top_k: int = 50,
    ) -> torch.Tensor:
        """
        带 KV 缓存的自回归 token 生成。

        步骤 0 处理完整提示。后续步骤仅传入最后生成的 token，
        所有先前的键和值从 kv_cache 中检索。这使解码成本与每步
        单个 token 成正比，而非与不断增长的完整序列成正比。

        n_loops 可设置为高于训练值，以在推理时外推到更难的问题
        （深度外推特性）。

        参数:
            input_ids      -- 形状为 (B, T) 的提示 token 索引
            max_new_tokens -- 要生成的 token 数量
            n_loops        -- 每个解码步骤的递归循环深度
            temperature    -- softmax 温度；越低越贪婪
            top_k          -- 将采样限制在 Top-K logits（0 = 禁用）

        返回:
            形状为 (B, T + max_new_tokens) 的 token 索引
        """
        kv_cache: dict = {}
        prompt_len = input_ids.shape[1]
        for step in range(max_new_tokens):
            if step == 0:
                cur_ids = input_ids
                start_pos = 0
            else:
                cur_ids = input_ids[:, -1:]
                start_pos = prompt_len + step - 1
            logits = self.forward(
                cur_ids, n_loops=n_loops, kv_cache=kv_cache, start_pos=start_pos
            )
            logits = logits[:, -1, :] / temperature
            if top_k > 0:
                v, _ = logits.topk(top_k)
                logits[logits < v[:, -1:]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_tok], dim=1)
        return input_ids
