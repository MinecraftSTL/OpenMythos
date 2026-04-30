"""
混合深度注意力 (MoDA) + DeepSeek 混合专家 FFN
======================================================================
论文（注意力）:   "Mixture-of-Depths Attention"   arXiv 2603.15619
论文（MoE）:     "DeepSeekMoE: Towards Ultimate Expert Specialization
                  in Mixture-of-Experts Language Models" arXiv 2401.06066
参考实现 (V3):   https://github.com/deepseek-ai/DeepSeek-V3

架构
----
本文件融合了两项独立的架构改进:

  1. **MoDA** — 每个注意力头在单个 softmax 下同时关注当前层的序列
     KV 对（因果）*以及*来自所有前序层在相同 token 位置的深度 KV 对。

  2. **DeepSeek MoE**（替换每个块中的密集 SwiGLU FFN）:
       * K_s 个*共享专家* — 始终激活，捕获通用知识。
       * N_r 个*路由专家* — 稀疏；每 token 激活 Top-K 个。
       * 细粒度专家分割：每个专家具有较小的隐藏维度
         (≈ dense_hidden / m)，使激活更多专家时 FLOPs 保持不变，
         同时提高专业化程度。
       * 专家级负载均衡损失防止路由崩溃。

门控路由（忠实于 DeepSeek-V3 model.py）
----------------------------------------
  scores       = softmax(x W^T)          # 或 V3 风格的 sigmoid
  original     = scores                  # 保存用于权重计算
  [可选]       scores += bias            # V3 无辅助损失路由
  [可选]       分组限制掩码             # V3 设备组路由
  indices      = topk(scores, K)
  weights      = original[indices]       # 无偏的原始分数
  [sigmoid]    weights /= sum(weights)   # sigmoid 门控的重新归一化
  weights     *= route_scale

负载均衡损失（DeepSeekMoE §3.3，在不使用 V3 偏置路由训练时使用）
  L_ExpBal = Σ_i  f_i · P_i
  f_i = (N_r / (K · T)) · #{路由到 i 的 token 数}   （归一化频率）
  P_i = (1/T) Σ_t s_{i,t}                            （平均软门控分数）

内存说明
--------
MoDA 的统一注意力具有 O(T·L) 的组合 KV 长度。对于长序列
请使用 https://github.com/hustvl/MoDA 的 Triton 内核。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MoDAConfig:
    """MoDA + DeepSeek-MoE 纯解码器语言模型的配置。

    注意力 (MoDA)
    -------------
    vocab_size:        词汇表大小。
    d_model:           隐藏维度（必须等于 n_heads_q * head_dim）。
    n_layers:          Transformer 层数。
    n_heads_q:         查询头数。
    n_heads_kv:        GQA 的键/值头数（必须整除 n_heads_q）。
    head_dim:          每头维度（通常为 d_model // n_heads_q）。
    max_seq_len:       RoPE 缓存的最大序列长度。
    rope_base:         RoPE 频率基数。
    attn_dropout:      注意力 dropout（推理时为 0）。
    norm_eps:          RMSNorm epsilon。

    MoE FFN（DeepSeekMoE / DeepSeek-V3 风格）
    ------------------------------------------
    n_shared_experts:     始终激活的共享专家数 (K_s)。捕获通用知识；
                          不参与路由和负载均衡损失。
    n_routed_experts:     路由专家总池 (N_r)。
    n_activated_experts:  每 token 从路由专家中选择的 Top-K 数 (K')。
    expert_hidden_dim:    每专家中间维度。
                          设为 dense_ffn_hidden / m，其中 m 为细粒度
                          分割因子，使总激活 FLOPs 匹配密集 FFN:
                          (n_shared + n_activated) × expert_hidden ≈ dense_hidden
    moe_balance_alpha:    专家级负载均衡损失权重。设为 0.0 禁用
                          （例如使用 V3 偏置路由时）。
    moe_score_func:       "softmax"（DeepSeekMoE / V2）或 "sigmoid"（V3）。
    moe_n_groups:         分组限制路由的专家组数
                          （V3 使用 8；设为 1 禁用，默认值）。
    moe_topk_groups:      每 token 可路由到的组数
                          （V3 使用 3；设为 1 禁用，默认值）。
    moe_route_scale:      归一化后乘以所选门控权重的标量
                          （V3 使用 2.5446；默认 1.0）。

    默认值近似 DeepSeekMoE 2B 配置，缩放到 d_model = 2048，
    保持每 token FLOPs 等于隐藏维度为 5 632（≈ 8/3 × 2048）的密集 SwiGLU:
        (n_shared + n_activated) × expert_hidden = (2+6) × 704 = 5 632。
    """

    # ---- Transformer / MoDA ----
    vocab_size: int = 32_000
    d_model: int = 2048
    n_layers: int = 24
    n_heads_q: int = 16
    n_heads_kv: int = 8
    head_dim: int = 128
    max_seq_len: int = 4_096
    rope_base: float = 10_000.0
    attn_dropout: float = 0.0
    norm_eps: float = 1e-6

    # ---- DeepSeek MoE FFN ----
    n_shared_experts: int = 2  # K_s
    n_routed_experts: int = 64  # N_r
    n_activated_experts: int = 6  # K' 从路由池中选择的 Top-K
    expert_hidden_dim: int = 704  # 每专家中间维度
    moe_balance_alpha: float = 0.001  # 负载均衡损失权重（0 = 禁用）
    moe_score_func: str = "softmax"  # "softmax" | "sigmoid"
    moe_n_groups: int = 1  # 专家组数（1 = 不分组）
    moe_topk_groups: int = 1  # 可路由到的组数（1 = 无限制）
    moe_route_scale: float = 1.0  # 门控权重缩放因子


# ---------------------------------------------------------------------------
# 基础组件
# ---------------------------------------------------------------------------


class RMSNorm(nn.Module):
    """均方根层归一化（无偏置，无均值减法）。"""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        """创建 RMSNorm 层。

        参数:
            dim: 归一化的特征维度（输入的最后一个轴）。
            eps: 在倒数平方根前添加的稳定性常数，
                 防止 RMS 接近零时除以零。
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """通过均方根归一化 *x* 并应用可学习缩放。

        参数:
            x: 任意形状 ``[..., dim]`` 的输入张量。

        返回:
            归一化后的张量，形状与 *x* 相同。
        """
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.weight


class RotaryEmbedding(nn.Module):
    """旋转位置编码 (RoPE)，带惰性缓存扩展。

    参数:
        dim:         每头维度 (head_dim)。
        max_seq_len: 初始缓存大小。
        base:        频率基数（默认 10 000）。
    """

    def __init__(
        self, dim: int, max_seq_len: int = 8_192, base: float = 10_000.0
    ) -> None:
        """初始化 RoPE 并预计算 cos/sin 缓存。

        参数:
            dim:         每头维度。必须为偶数。
            max_seq_len: 构造时缓存的位置数。当遇到更长序列时
                         缓存自动翻倍。
            base:        频率基数 θ。值越高旋转速率越慢，
                         有效上下文长度越长。
        """
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int) -> None:
        """预计算并注册 ``_cos`` / ``_sin`` 缓冲区，最多到 *seq_len*。

        在初始化时调用一次，当 ``forward`` 请求的序列长于当前缓存时
        再次调用（容量翻倍）。

        参数:
            seq_len: 要预计算的位置数。
        """
        t = torch.arange(
            seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype
        )
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)  # [T, dim/2]
        emb = torch.cat([freqs, freqs], dim=-1)  # [T, dim]
        self.register_buffer("_cos", emb.cos()[None, None], persistent=False)
        self.register_buffer("_sin", emb.sin()[None, None], persistent=False)

    def forward(self, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """返回前 *seq_len* 个位置的缓存 (cos, sin) 表。

        参数:
            seq_len: 所需的位置数。

        返回:
            ``(cos, sin)`` 元组，每个形状为 ``[1, 1, seq_len, dim]``，
            可广播到 ``[B, H, T, dim]`` 的查询/键张量。
        """
        if seq_len > self._cos.shape[2]:
            self._build_cache(seq_len * 2)
        return self._cos[:, :, :seq_len], self._sin[:, :, :seq_len]


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """返回 *x* 最后一维分半并取反交换的结果。

    给定 ``x = [x₁, x₂]``（最后一维的各半），返回
    ``[-x₂, x₁]``。与 :func:`apply_rotary_emb` 中的 cos/sin 乘法
    结合，实现定义 RoPE 的二维旋转矩阵。

    参数:
        x: 最后一维为偶数大小的张量。

    返回:
        与 *x* 形状相同的旋转后张量。
    """
    half = x.shape[-1] // 2
    return torch.cat([-x[..., half:], x[..., :half]], dim=-1)


def apply_rotary_emb(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """将旋转位置编码应用于查询或键张量。

    实现 ``x_rot = x * cos + rotate_half(x) * sin``，等价于将每对
    连续维度乘以一个二维旋转矩阵，其角度取决于序列位置和维度频率。

    参数:
        x:   查询或键张量，形状 ``[B, H, T, d]``。
        cos: 预计算的余弦值，形状 ``[1, 1, T, d]``。
        sin: 预计算的正弦值，形状 ``[1, 1, T, d]``。

    返回:
        与 *x* 形状和数据类型相同的旋转后张量。
    """
    return x * cos + _rotate_half(x) * sin


# ---------------------------------------------------------------------------
# DeepSeek MoE FFN
# ---------------------------------------------------------------------------


class DeepSeekExpert(nn.Module):
    """单个细粒度 SwiGLU 专家。

    忠实于 DeepSeek-V3 ``Expert``:
        output = w2( SiLU(w1(x)) ⊙ w3(x) )

    其中 w1 是门控投影，w3 是上投影，w2 是下投影 —
    与较小隐藏维度的 SwiGLU FFN 完全相同。

    参数:
        d_model:    输入/输出维度。
        hidden_dim: 专家中间维度（远小于密集 FFN hidden_dim）。
    """

    def __init__(self, d_model: int, hidden_dim: int) -> None:
        """创建单个细粒度 SwiGLU 专家。

        参数:
            d_model:    token 隐藏维度（输入和输出大小）。
            hidden_dim: 专家中间维度。通常远小于密集 FFN 隐藏维度 —
                        设为 ``dense_hidden / m``，其中 *m* 为细粒度
                        分割因子，使总激活 FLOPs 匹配密集基线。
        """
        super().__init__()
        self.w1 = nn.Linear(d_model, hidden_dim, bias=False)  # 门控
        self.w3 = nn.Linear(d_model, hidden_dim, bias=False)  # 上投影
        self.w2 = nn.Linear(hidden_dim, d_model, bias=False)  # 下投影

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """计算 ``w2( SiLU(w1(x)) ⊙ w3(x) )``。

        参数:
            x: 分配给此专家的 token 特征，形状
               ``[num_assigned_tokens, d_model]``。

        返回:
            专家输出，形状 ``[num_assigned_tokens, d_model]``。
        """
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class DeepSeekGate(nn.Module):
    """Token 到专家的路由门控。

    忠实于 DeepSeek-V3 ``Gate``（去除分布式分片）。

    路由算法
    ~~~~~~~~
    1.  ``scores = softmax(x W^T)``  或  ``sigmoid(x W^T)``
    2.  ``original_scores = scores``  （保存 — 将用于门控权重）
    3.  [可选] ``scores += bias``  （V3 无辅助损失偏置路由）
    4.  [可选] 分组限制掩码:
            - 将 scores 重塑为 [T, n_groups, experts_per_group]
            - 仅保留每 token 的 top-``topk_groups`` 个组
            - 将其余掩码为 −∞
    5.  ``indices = topk(scores, K')``          （路由决策）
    6.  ``weights = original_scores[indices]``  （无偏权重）
    7.  [仅 sigmoid] ``weights /= sum(weights)``  （重新归一化）
    8.  ``weights *= route_scale``

    ``original_scores``（完整分布，偏置/掩码前）也会返回，
    以便 MoE 层计算专家级负载均衡损失。

    参数:
        d_model:           token 隐藏维度。
        n_routed_experts:  路由专家总池大小 (N_r)。
        n_activated:       选择的 Top-K 专家数 (K')。
        score_func:        ``"softmax"`` 或 ``"sigmoid"``。
        n_groups:          专家组数（1 = 禁用）。
        topk_groups:       每 token 可路由到的组数（1 = 禁用）。
        route_scale:       应用于最终门控权重的标量。
        use_bias:          若为 True，添加仅用于路由决策的可学习
                           每专家偏置（V3 无辅助损失方案）。
    """

    def __init__(
        self,
        d_model: int,
        n_routed_experts: int,
        n_activated: int,
        score_func: str = "softmax",
        n_groups: int = 1,
        topk_groups: int = 1,
        route_scale: float = 1.0,
        use_bias: bool = False,
    ) -> None:
        """创建路由门控。

        参数:
            d_model:          token 隐藏维度。
            n_routed_experts: 路由专家池中的专家总数 (N_r)。
            n_activated:      每 token 选择的专家数 (K')。
            score_func:       亲和函数 — ``"softmax"``（原始
                              DeepSeekMoE / V2）或 ``"sigmoid"``（V3）。
            n_groups:         设备限制路由的专家组数。
                              设为 1 禁用（默认）。
            topk_groups:      每 token 可路由到的组数。
                              设为 1 禁用（默认）。
            route_scale:      可选 sigmoid 归一化后乘以门控权重的标量
                              （V3 使用 2.5446；默认 1.0 不改变权重）。
            use_bias:         若为 ``True``，初始化仅添加到路由分数
                              （非门控权重）的可学习每专家 float32 偏置。
                              启用 V3 无辅助损失负载均衡方案，其中偏置
                              在优化器外部通过监控专家负载来调整。
        """
        super().__init__()
        self.n_routed_experts = n_routed_experts
        self.n_activated = n_activated
        self.score_func = score_func
        self.n_groups = n_groups
        self.topk_groups = topk_groups
        self.route_scale = route_scale

        # 门控投影: [N_r, D]
        self.weight = nn.Parameter(torch.empty(n_routed_experts, d_model))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        # 可选的每专家路由偏置（V3 无辅助损失负载均衡）。
        # 通过监控专家负载在优化器外部更新 — 不通过负载均衡损失训练。
        # 初始化为零。
        self.bias: Optional[nn.Parameter] = (
            nn.Parameter(torch.zeros(n_routed_experts, dtype=torch.float32))
            if use_bias
            else None
        )

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """计算路由权重和专家索引。

        参数:
            x: ``[num_tokens, d_model]``（展平的 B × T）。

        返回:
            weights:        ``[num_tokens, K']``  门控权重（dtype = x.dtype）。
            indices:        ``[num_tokens, K']``  选中的专家索引（int64）。
            original_scores: ``[num_tokens, N_r]``  完整软分数（float32），
                             由 :class:`DeepSeekMoE` 用于负载均衡损失。
        """
        # 亲和 logits
        logits = F.linear(x, self.weight)  # [T, N_r]

        if self.score_func == "softmax":
            scores = logits.softmax(dim=-1, dtype=torch.float32)
        else:  # sigmoid (V3)
            scores = logits.sigmoid().to(torch.float32)

        original_scores = scores  # 无偏；用于权重 + 负载均衡损失

        # 路由分数（如果偏置激活，可能与 original_scores 不同）
        routing = scores
        if self.bias is not None:
            routing = routing + self.bias

        # 分组限制路由（V3 设备组约束）
        if self.n_groups > 1:
            # [T, n_groups, experts_per_group]
            g = routing.view(x.size(0), self.n_groups, -1)
            if self.bias is None:
                group_scores = g.amax(dim=-1)  # [T, G]
            else:
                # 每组 Top-2 求和（V3 启发式）
                group_scores = g.topk(2, dim=-1)[0].sum(dim=-1)
            _, top_groups = group_scores.topk(self.topk_groups, dim=-1)  # [T, topk_g]
            mask = torch.ones(
                x.size(0), self.n_groups, dtype=torch.bool, device=x.device
            ).scatter_(
                1, top_groups, False
            )  # True = 被掩码
            routing = g.masked_fill(mask.unsqueeze(-1), float("-inf")).flatten(1)

        # Top-K 选择（基于可能包含偏置/组掩码的路由分数）
        _, indices = routing.topk(self.n_activated, dim=-1)  # [T, K']

        # 门控权重来自原始（无偏）分数
        weights = original_scores.gather(1, indices)  # [T, K']

        if self.score_func == "sigmoid":
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp(min=1e-9)

        weights = (weights * self.route_scale).to(x.dtype)
        return weights, indices, original_scores


class DeepSeekMoE(nn.Module):
    """DeepSeek 混合专家层 — 密集 FFN 的直接替代品。

    组合共享专家（始终激活）和路由专家（稀疏 Top-K），
    完全按照 DeepSeek-V3 ``MoE`` 实现，适配单设备训练
    （无 ColumnParallel/RowParallel，无 all_reduce）。

    前向传播
    ~~~~~~~~
    ::

        x_flat = x.view(-1, D)                         # [B*T, D]

        # 共享路径（始终执行）
        z = shared_experts(x_flat)                     # [B*T, D]

        # 路由路径（稀疏）
        weights, indices, scores = gate(x_flat)        # [B*T, K'], [B*T, K'], [B*T, N_r]
        y = zeros_like(x_flat)
        for 每个专家 i:
            toks = 选择了专家 i 的 token
            y[toks] += experts[i](x_flat[toks]) * weights[toks, rank_of_i]

        output = (y + z).view(B, T, D)

        # 训练时: 专家级负载均衡损失（DeepSeekMoE §3.3）
        L_ExpBal = Σ_i  f_i · P_i
          f_i = (N_r / (K' · T)) · #{token → 专家 i}
          P_i = mean_t(scores_{t,i})

    参数:
        cfg: :class:`MoDAConfig` 实例。
    """

    def __init__(self, cfg: MoDAConfig) -> None:
        """从 :class:`MoDAConfig` 构建 MoE 层。

        构造:
          * ``shared_experts`` — 一个隐藏维度为
            ``n_shared_experts × expert_hidden_dim`` 的密集 SwiGLU FFN。
          * ``gate``           — 用于 Top-K 路由的 :class:`DeepSeekGate`。
          * ``experts``        — ``n_routed_experts`` 个
            :class:`DeepSeekExpert` 实例的 ``nn.ModuleList``，
            每个具有 ``expert_hidden_dim`` 个中间单元。

        参数:
            cfg: 模型配置。相关字段为
                 ``n_shared_experts``、``n_routed_experts``、
                 ``n_activated_experts``、``expert_hidden_dim``、
                 ``moe_balance_alpha``、``moe_score_func``、
                 ``moe_n_groups``、``moe_topk_groups`` 和
                 ``moe_route_scale``。
        """
        super().__init__()
        self.d_model = cfg.d_model
        self.n_routed_experts = cfg.n_routed_experts
        self.n_activated_experts = cfg.n_activated_experts
        self.moe_balance_alpha = cfg.moe_balance_alpha

        # 共享专家: 单个密集 SwiGLU，隐藏维度 = K_s × expert_hidden
        # （匹配 DeepSeek-V3 的 ``MLP(dim, n_shared_experts * moe_inter_dim)``）
        shared_hidden = cfg.n_shared_experts * cfg.expert_hidden_dim
        self.shared_experts = _SharedFFN(cfg.d_model, shared_hidden)

        # 路由门控
        self.gate = DeepSeekGate(
            d_model=cfg.d_model,
            n_routed_experts=cfg.n_routed_experts,
            n_activated=cfg.n_activated_experts,
            score_func=cfg.moe_score_func,
            n_groups=cfg.moe_n_groups,
            topk_groups=cfg.moe_topk_groups,
            route_scale=cfg.moe_route_scale,
            use_bias=False,
        )

        # 路由专家池
        self.experts = nn.ModuleList(
            [
                DeepSeekExpert(cfg.d_model, cfg.expert_hidden_dim)
                for _ in range(cfg.n_routed_experts)
            ]
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """运行 MoE 层。

        参数:
            x: ``[B, T, D]`` 隐藏状态。

        返回:
            output:        ``[B, T, D]``  更新后的隐藏状态。
            balance_loss:  训练时的标量专家级负载均衡损失，
                           推理时为 ``None``。
        """
        shape = x.shape
        x_flat = x.view(-1, self.d_model)  # [T_tot, D]
        n_tokens = x_flat.size(0)

        # ---- 共享专家（所有 token）---------------------------------
        z = self.shared_experts(x_flat)  # [T_tot, D]

        # ---- 路由专家（稀疏）-------------------------------------
        weights, indices, scores = self.gate(x_flat)
        # weights: [T_tot, K'], indices: [T_tot, K'], scores: [T_tot, N_r]

        y = torch.zeros_like(x_flat)

        # 分发: 对每个专家计算其分配到的 token
        # （token 主循环匹配 DeepSeek-V3 的参考实现）
        counts = torch.bincount(indices.flatten(), minlength=self.n_routed_experts)
        for i, expert in enumerate(self.experts):
            if counts[i].item() == 0:
                continue
            tok_idx, rank_in_k = torch.where(
                indices == i
            )  # 哪些 token 和哪个 K 槽位
            y[tok_idx] += expert(x_flat[tok_idx]) * weights[tok_idx, rank_in_k, None]

        output = (y + z).view(shape)

        # ---- 专家级负载均衡损失（DeepSeekMoE §3.3）----------------
        balance_loss: Optional[torch.Tensor] = None
        if self.training and self.moe_balance_alpha > 0.0:
            balance_loss = self._balance_loss(indices, scores, n_tokens)

        return output, balance_loss

    def _balance_loss(
        self,
        indices: torch.Tensor,  # [T, K']  int64
        scores: torch.Tensor,  # [T, N_r] float32（完整分布）
        n_tokens: int,
    ) -> torch.Tensor:
        """计算专家级负载均衡损失（DeepSeekMoE §3.3）。

        通过鼓励模型将 token 均匀分布到各专家来惩罚路由不平衡。
        只有软分数项 ``P_i`` 接收梯度；硬计数项 ``f_i`` 不可微分，
        作为固定加权系数。

        ::

            f_i = (N_r / (K' × T)) × #{路由到专家 i 的 token 数}
            P_i = (1/T) Σ_t scores[t, i]
            L   = Σ_i  f_i · P_i

        完美平衡时 ``f_i = 1``（对所有 *i*）且 ``L = Σ P_i = 1``
        （softmax）或某个常数（sigmoid）。过载的专家产生较大的 ``f_i``，
        通过梯度推高其平均分数 ``P_i``，从而吸引更多 token —
        在训练过程中稳定负载。

        参数:
            indices:  ``[T, K']`` int64 — 每 token 选择的专家索引。
            scores:   ``[T, N_r]`` float32 — 来自门控的完整软分布
                      （Top-K 选择前），用于 ``P_i``。
            n_tokens: 批次中的 token 总数 (``B × T``)。

        返回:
            标量负载均衡损失张量。
        """
        Nr, K = self.n_routed_experts, self.n_activated_experts

        # 每专家路由计数（不可微分）
        counts = torch.zeros(Nr, dtype=torch.float32, device=indices.device)
        counts.scatter_add_(
            0,
            indices.flatten(),
            torch.ones(indices.numel(), dtype=torch.float32, device=indices.device),
        )
        f = counts * (Nr / (K * n_tokens))  # 归一化频率 [N_r]

        # 每专家平均软门控分数（通过 softmax/sigmoid 可微分）
        P = scores.mean(dim=0)  # [N_r]

        # f 来自硬 Top-K → 无梯度；梯度仅通过 P 流动
        return (f * P).sum()


class _SharedFFN(nn.Module):
    """用于始终激活的共享专家的密集 SwiGLU FFN。

    与 :class:`DeepSeekExpert` 结构相同，但是一个更大的单一 MLP，
    其 ``hidden_dim`` 等于 ``n_shared_experts × expert_hidden_dim``。
    这匹配 DeepSeek-V3 的 ``MLP(dim, n_shared_experts * moe_inter_dim)``。

    不属于公共 API — 仅由 :class:`DeepSeekMoE` 实例化。
    """

    def __init__(self, d_model: int, hidden_dim: int) -> None:
        """创建共享专家 FFN。

        参数:
            d_model:    token 隐藏维度（输入和输出）。
            hidden_dim: 所有共享专家的组合中间维度
                        (``n_shared_experts × expert_hidden_dim``)。
        """
        super().__init__()
        self.w1 = nn.Linear(d_model, hidden_dim, bias=False)
        self.w3 = nn.Linear(d_model, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """对每个 token 应用共享 SwiGLU FFN。

        参数:
            x: 展平的 token 特征，形状 ``[B*T, d_model]``。

        返回:
            变换后的特征，形状 ``[B*T, d_model]``。
        """
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


# ---------------------------------------------------------------------------
# MoDA 注意力（与基础文件相同）
# ---------------------------------------------------------------------------


class MoDAAttention(nn.Module):
    """混合深度注意力 — 读取端。

    每个查询在单个 softmax 下同时关注:
      * 当前层的序列 KV（因果 GQA）。
      * 来自所有前序层在*相同* token 位置的深度 KV。

    深度缓存条目由 :class:`MoDABlock` 从完整块输出 X_l^out
    （MoE FFN 之后）外部写入。

    参数:
        cfg: :class:`MoDAConfig` 实例。
    """

    def __init__(self, cfg: MoDAConfig) -> None:
        """构建 MoDA 注意力模块。

        创建四个投影矩阵（Q、K、V、O），大小适配 GQA，
        并存储注意力缩放因子和 dropout 率。

        参数:
            cfg: 模型配置。必须满足
                 ``n_heads_q % n_heads_kv == 0``（GQA 要求）。

        异常:
            ValueError: 当 ``n_heads_q`` 不能被 ``n_heads_kv`` 整除时。
        """
        super().__init__()
        if cfg.n_heads_q % cfg.n_heads_kv != 0:
            raise ValueError(
                f"n_heads_q ({cfg.n_heads_q}) 必须能被 "
                f"n_heads_kv ({cfg.n_heads_kv}) 整除以支持 GQA。"
            )

        self.n_heads_q = cfg.n_heads_q
        self.n_heads_kv = cfg.n_heads_kv
        self.head_dim = cfg.head_dim
        self.gqa_group = cfg.n_heads_q // cfg.n_heads_kv
        self.scale = cfg.head_dim**-0.5
        self.dropout = cfg.attn_dropout

        inner_q = cfg.n_heads_q * cfg.head_dim
        inner_kv = cfg.n_heads_kv * cfg.head_dim

        self.q_proj = nn.Linear(cfg.d_model, inner_q, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, inner_kv, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, inner_kv, bias=False)
        self.o_proj = nn.Linear(inner_q, cfg.d_model, bias=False)

    def _expand_kv(self, kv: torch.Tensor) -> torch.Tensor:
        """沿 dim 1 重复 KV 头以匹配查询头数。

        GQA 组大小为 G 时，每个 KV 头被 G 个查询头共享。
        ``repeat_interleave(G, dim=1)`` 产生正确的交错扩展，
        使查询头 ``h`` 与 KV 头 ``h // G`` 配对。

        参数:
            kv: 键或值张量，dim 1 为 KV 头轴。
                支持的形状: ``[B, Hk, T, d]``（序列）和
                ``[B, Hk, T, L, d]``（深度堆栈）。

        返回:
            dim 1 从 ``Hk`` 扩展到 ``Hq = Hk × G`` 的张量。
            当 ``gqa_group == 1`` 时返回 *kv* 不变。
        """
        if self.gqa_group == 1:
            return kv
        return kv.repeat_interleave(self.gqa_group, dim=1)

    def forward(
        self,
        x: torch.Tensor,
        depth_k_cache: List[torch.Tensor],
        depth_v_cache: List[torch.Tensor],
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        """计算 MoDA 注意力输出。

        参数:
            x:             ``[B, T, D]`` 输入隐藏状态。
            depth_k_cache: ``L`` 个张量，每个 ``[B, Hk, T, d]`` — 深度键。
            depth_v_cache: 匹配的深度值。
            cos/sin:       RoPE 表 ``[1, 1, T, d]``。

        返回:
            ``[B, T, D]`` 输出隐藏状态。
        """
        B, T, D = x.shape
        Hq, Hk, d = self.n_heads_q, self.n_heads_kv, self.head_dim

        Q = self.q_proj(x).view(B, T, Hq, d).transpose(1, 2)
        K = self.k_proj(x).view(B, T, Hk, d).transpose(1, 2)
        V = self.v_proj(x).view(B, T, Hk, d).transpose(1, 2)

        Q = apply_rotary_emb(Q, cos, sin)
        K = apply_rotary_emb(K, cos, sin)

        K_e = self._expand_kv(K)
        V_e = self._expand_kv(V)

        L = len(depth_k_cache)

        if L == 0:
            out = F.scaled_dot_product_attention(
                Q,
                K_e,
                V_e,
                is_causal=True,
                dropout_p=self.dropout if self.training else 0.0,
                scale=self.scale,
            )
        else:
            # 序列 logits [B, Hq, T, T]，带因果掩码
            seq_logits = torch.matmul(Q, K_e.transpose(-2, -1)) * self.scale
            causal_mask = torch.triu(
                torch.full((T, T), float("-inf"), device=x.device, dtype=Q.dtype),
                diagonal=1,
            )
            seq_logits = seq_logits + causal_mask

            # 深度 KV: [B, Hk, L, T, d] → [B, Hk, T, L, d]
            K_depth = torch.stack(depth_k_cache, dim=2).permute(0, 1, 3, 2, 4)
            V_depth = torch.stack(depth_v_cache, dim=2).permute(0, 1, 3, 2, 4)
            K_depth_e = self._expand_kv(K_depth)
            V_depth_e = self._expand_kv(V_depth)

            # 深度 logits [B, Hq, T, L]
            depth_logits = torch.einsum("bhid,bhild->bhil", Q, K_depth_e) * self.scale

            # 在 T + L 个位置上的统一 softmax
            combined = torch.cat([seq_logits, depth_logits], dim=-1)
            weights = F.softmax(combined, dim=-1)
            if self.training and self.dropout > 0.0:
                weights = F.dropout(weights, p=self.dropout)

            seq_contrib = torch.matmul(weights[:, :, :, :T], V_e)
            depth_contrib = torch.einsum(
                "bhil,bhild->bhid", weights[:, :, :, T:], V_depth_e
            )
            out = seq_contrib + depth_contrib

        out = out.transpose(1, 2).reshape(B, T, Hq * d)
        return self.o_proj(out)


# ---------------------------------------------------------------------------
# MoDA Transformer 块
# ---------------------------------------------------------------------------


class MoDABlock(nn.Module):
    """单个 MoDA + DeepSeek-MoE Transformer 块。

    结构（后归一化，按 MoDA 论文推荐）:

    .. code-block::

        x  ──► 注意力 ──► + ──► RMSNorm ──► x_mid
        x                 ↑ (残差)
        x_mid ──► MoE  ──► + ──► RMSNorm ──► x_out
        x_mid              ↑ (残差)
        x_out ──► W_K^W ──► k_write  }  追加到 MoDA 深度 KV 缓存
              └─► W_V^W ──► v_write  }  由 MoDAModel 传递给下一层

    MoE 层还返回一个可选的专家级负载均衡损失标量，
    该标量被传播到 :class:`MoDAModel` 以包含在总训练损失中。

    参数:
        cfg: :class:`MoDAConfig` 实例。
    """

    def __init__(self, cfg: MoDAConfig) -> None:
        """构建一个 MoDA + MoE Transformer 块。

        构造并连接:
          * ``attn``      — :class:`MoDAAttention`（深度感知 GQA）。
          * ``moe``       — :class:`DeepSeekMoE`（共享 + 路由专家）。
          * ``norm_attn`` / ``norm_ffn`` — 子层后 :class:`RMSNorm`。
          * ``k_write`` / ``v_write`` — 深度缓存写入投影
            ``D → n_heads_kv × head_dim``。

        参数:
            cfg: 模型配置。
        """
        super().__init__()
        inner_kv = cfg.n_heads_kv * cfg.head_dim

        self.attn = MoDAAttention(cfg)
        self.moe = DeepSeekMoE(cfg)
        self.norm_attn = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.norm_ffn = RMSNorm(cfg.d_model, cfg.norm_eps)

        # MoDA 深度缓存写入投影: K_l = X_l^out W_K^W, V_l = X_l^out W_V^W
        self.k_write = nn.Linear(cfg.d_model, inner_kv, bias=False)
        self.v_write = nn.Linear(cfg.d_model, inner_kv, bias=False)

        self._n_heads_kv = cfg.n_heads_kv
        self._head_dim = cfg.head_dim

    def forward(
        self,
        x: torch.Tensor,
        depth_k_cache: List[torch.Tensor],
        depth_v_cache: List[torch.Tensor],
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """运行一个 MoDA + MoE Transformer 块。

        参数:
            x:             ``[B, T, D]`` 输入隐藏状态。
            depth_k_cache: 来自所有前序层的深度键，每个 ``[B, Hk, T, d]``。
            depth_v_cache: 匹配的深度值。
            cos/sin:       RoPE 表 ``[1, 1, T, d]``。

        返回:
            x_out:        ``[B, T, D]`` 更新后的隐藏状态。
            k_write:      ``[B, Hk, T, d]`` 本层的深度缓存键。
            v_write:      ``[B, Hk, T, d]`` 本层的深度缓存值。
            balance_loss: 标量专家级负载均衡损失，推理时为 ``None``。
        """
        B, T, _ = x.shape

        # 后归一化注意力子层
        x = self.norm_attn(x + self.attn(x, depth_k_cache, depth_v_cache, cos, sin))

        # 后归一化 MoE 子层
        moe_out, balance_loss = self.moe(x)
        x = self.norm_ffn(x + moe_out)

        # 从 X_l^out（完整块输出，MoE 之后）进行深度写入投影
        k_write = (
            self.k_write(x).view(B, T, self._n_heads_kv, self._head_dim).transpose(1, 2)
        )
        v_write = (
            self.v_write(x).view(B, T, self._n_heads_kv, self._head_dim).transpose(1, 2)
        )

        # 对 k_write 应用 RoPE 以保持未来深度读取时的位置一致性
        k_write = apply_rotary_emb(k_write, cos, sin)

        return x, k_write, v_write, balance_loss


# ---------------------------------------------------------------------------
# 完整 MoDA + MoE 语言模型
# ---------------------------------------------------------------------------


class MoDAModel(nn.Module):
    """带混合深度注意力和 DeepSeek MoE FFN 的纯解码器语言模型。

    损失 = LM 交叉熵 + moe_balance_alpha × mean(每层负载均衡损失)

    深度 KV 缓存是 :meth:`forward` 内部的局部列表 — 不存储在
    ``self`` 上，因此自动微分在独立的前向调用之间是干净的。

    参数:
        cfg: 指定完整模型的 :class:`MoDAConfig`。
    """

    def __init__(self, cfg: MoDAConfig) -> None:
        """构建完整的 MoDA + MoE 语言模型。

        构造 token 嵌入、RoPE、所有 Transformer 块、最终 RMSNorm
        和语言模型头。嵌入和 LM 头权重绑定，共享同一参数。

        参数:
            cfg: 完整指定模型的 :class:`MoDAConfig`。
        """
        super().__init__()
        self.cfg = cfg

        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.rope = RotaryEmbedding(cfg.head_dim, cfg.max_seq_len, cfg.rope_base)
        self.blocks = nn.ModuleList([MoDABlock(cfg) for _ in range(cfg.n_layers)])
        self.norm_out = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        self.lm_head.weight = self.embed.weight  # 权重绑定

        self._init_weights()

    def _init_weights(self) -> None:
        """对每个子模块应用 GPT 风格的权重初始化。

        * :class:`nn.Linear` 和 :class:`nn.Embedding` 权重从
          ``Normal(0, 0.02)`` 采样 — GPT-2 及后续大多数 Transformer
          实现使用的标准初始化。
        * :class:`DeepSeekGate` 权重矩阵用 ``kaiming_uniform``（fan-in）
          重新初始化，以匹配默认的 ``nn.Linear`` 初始化，避免正态分布
          对于不带后续非线性的矩阵过于狭窄。

        在 :meth:`__init__` 结束时自动调用。
        """
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
            elif isinstance(m, DeepSeekGate):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """运行完整的 MoDA + MoE 语言模型。

        参数:
            input_ids: ``[B, T]`` token 索引。
            labels:    ``[B, T]`` LM 损失的目标；-100 位置被忽略。

        返回:
            logits:    ``[B, T, vocab_size]``。
            loss:      若提供 labels 则为 ``lm_loss + balance_loss``，否则为 ``None``。
        """
        B, T = input_ids.shape
        if T > self.cfg.max_seq_len:
            raise ValueError(
                f"序列长度 {T} 超过 max_seq_len={self.cfg.max_seq_len}。"
            )

        x = self.embed(input_ids)
        cos, sin = self.rope(T)

        depth_k_cache: List[torch.Tensor] = []
        depth_v_cache: List[torch.Tensor] = []
        balance_losses: List[torch.Tensor] = []

        for block in self.blocks:
            x, k_write, v_write, bal = block(x, depth_k_cache, depth_v_cache, cos, sin)
            depth_k_cache.append(k_write)
            depth_v_cache.append(v_write)
            if bal is not None:
                balance_losses.append(bal)

        x = self.norm_out(x)
        logits = self.lm_head(x)

        loss: Optional[torch.Tensor] = None
        if labels is not None:
            lm_loss = F.cross_entropy(
                logits.view(-1, self.cfg.vocab_size),
                labels.view(-1),
                ignore_index=-100,
            )
            if balance_losses and self.cfg.moe_balance_alpha > 0.0:
                avg_balance = torch.stack(balance_losses).mean()
                loss = lm_loss + self.cfg.moe_balance_alpha * avg_balance
            else:
                loss = lm_loss

        return logits, loss

    def num_parameters(self, trainable_only: bool = False) -> int:
        """统计模型中标量参数的总数。

        参数:
            trainable_only: 若为 ``True``，仅统计 ``requires_grad=True``
                            的参数，排除冻结层。

        返回:
            整数参数计数。
        """
        params = (
            self.parameters()
            if not trainable_only
            else (p for p in self.parameters() if p.requires_grad)
        )
        return sum(p.numel() for p in params)

    def extra_repr(self) -> str:
        """返回显示在 ``repr(model)`` 中的单行摘要字符串。

        由 PyTorch 的 ``__repr__`` 在类名之后、子模块树之前直接显示。

        返回:
            列出关键模型维度和总参数数的人类可读字符串。
        """
        c = self.cfg
        return (
            f"vocab={c.vocab_size}, d_model={c.d_model}, layers={c.n_layers}, "
            f"heads={c.n_heads_q}/{c.n_heads_kv} (GQA), "
            f"experts=共享×{c.n_shared_experts}+路由×{c.n_routed_experts}"
            f"(top-{c.n_activated_experts}), "
            f"参数量={self.num_parameters():,}"
        )
