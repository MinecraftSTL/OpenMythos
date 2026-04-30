# `OpenMythos` — 类参考

**模块：** `open_mythos.main`  
**基类：** `torch.nn.Module`

---

## 概述

`OpenMythos` 是实现递归深度 Transformer（RDT）架构的顶层模型类，架构描述见 [OpenMythos 假说](../README.md)。它将三个功能阶段——**前奏（Prelude）**、**递归块（Recurrent Block）**和**尾声（Coda）**——组装成一个完整的自回归语言模型。

```
输入 token ID  (B, T)
        ↓
   [嵌入层]            token 索引 → dim 维向量
        ↓
   [前奏]              prelude_layers × 标准 TransformerBlock（运行一次）
        ↓
   [递归块]            一个 TransformerBlock 循环 T 次
        ↑___________↓   h_{t+1} = A·h_t + B·e + Transformer(h_t, e)
        ↓
   [尾声]              coda_layers × 标准 TransformerBlock（运行一次）
        ↓
   [RMSNorm → LM 头]
        ↓
输出 logits  (B, T, vocab_size)
```

`OpenMythos` 中的每个架构选择都可以通过构造时传入的单个 [`MythosConfig`](#mythosconfig) 数据类进行配置。

---

## `MythosConfig`

```python
@dataclass
class MythosConfig
```

模型的所有超参数都存储在这个单一的冻结式数据类中。将实例传递给 `OpenMythos.__init__`。

### 核心字段

| 字段 | 类型 | 默认值 | 描述 |
|---|---|---|---|
| `vocab_size` | `int` | `32000` | 词汇表大小；决定嵌入层和 LM 头的维度 |
| `dim` | `int` | `2048` | 模型隐藏维度——整个残差流的宽度 |
| `n_heads` | `int` | `16` | 查询注意力头数 |
| `n_kv_heads` | `int` | `4` | 键/值头数（仅 GQA）；每 `n_heads // n_kv_heads` 个 Q 头共享一个 KV 对 |
| `max_seq_len` | `int` | `4096` | 最大序列长度；RoPE 频率预计算到此长度 |
| `max_loop_iters` | `int` | `16` | 推理时默认的递归循环深度 T。可在每次调用时覆盖 |
| `prelude_layers` | `int` | `2` | 递归循环前运行一次的标准 Transformer 块数量 |
| `coda_layers` | `int` | `2` | 递归循环后运行一次的标准 Transformer 块数量 |

### 注意力字段

`attn_type` 在两种完整的注意力实现之间选择。其他注意力字段是特定实现所需的。

| 字段 | 类型 | 默认值 | 描述 |
|---|---|---|---|
| `attn_type` | `str` | `"mla"` | `"gqa"` 为分组查询注意力；`"mla"` 为多潜在注意力 |
| `kv_lora_rank` | `int` | `512` | **[仅 MLA]** 缓存中存储的压缩 KV 潜在秩，替代完整的 K 和 V |
| `q_lora_rank` | `int` | `1536` | **[仅 MLA]** 压缩 Q 潜在秩 |
| `qk_rope_head_dim` | `int` | `64` | **[仅 MLA]** 接收 RoPE 位置编码的每头维度 |
| `qk_nope_head_dim` | `int` | `128` | **[仅 MLA]** 不含位置编码的每头维度 |
| `v_head_dim` | `int` | `128` | **[仅 MLA]** 每头值维度 |

**GQA vs MLA：** GQA 通过使 KV 头数少于 Q 头数来减少 KV 缓存（减少 `n_heads / n_kv_heads` 倍）。MLA 通过缓存低秩 KV 潜在表示（`kv_lora_rank`）和 RoPE 键（`n_heads × qk_rope_head_dim`），然后即时重建完整的 K 和 V，实现更大幅度的缩减。在生产规模下，MLA 的 KV 缓存大约比标准注意力小 10-20 倍。

### MoE FFN 字段

混合专家 FFN 仅在递归块内部使用。前奏和尾声使用密集 SwiGLU FFN。

| 字段 | 类型 | 默认值 | 描述 |
|---|---|---|---|
| `n_experts` | `int` | `64` | 路由专家 FFN 的总数 |
| `n_shared_experts` | `int` | `2` | 始终激活的共享专家；吸收跨领域的通用模式 |
| `n_experts_per_tok` | `int` | `4` | 路由器为每个 token 选择的 Top-K 路由专家数 |
| `expert_dim` | `int` | `512` | 每个细粒度路由专家内部的隐藏维度 |

每个 token 大约激活 `n_experts_per_tok / n_experts = 6.25%` 的路由专家参数，加上所有共享专家参数。

### 稳定性和适配字段

| 字段 | 类型 | 默认值 | 描述 |
|---|---|---|---|
| `act_threshold` | `float` | `0.99` | ACT 累积停止阈值；当超过此值时，该位置退出循环 |
| `rope_theta` | `float` | `500000.0` | RoPE 基础频率（LLaMA-3 默认值；越高 = 频率在序列位置上衰减越慢） |
| `lora_rank` | `int` | `16` | 每次循环迭代中应用的深度级 LoRA 适配器的秩 |

---

## 构造函数

```python
OpenMythos(cfg: MythosConfig)
```

构建所有子模块，预计算 RoPE 频率缓冲区，并执行权重初始化。

**内部执行流程：**

1. `nn.Embedding(vocab_size, dim)` — token 嵌入表，与 LM 头权重绑定。
2. RoPE 缓冲区 — `freqs_cis`（用于 GQA，dim = `dim // n_heads`）和 `freqs_cis_mla`（用于 MLA，dim = `qk_rope_head_dim`）预计算一次并注册为非参数缓冲区。前向传播时根据 `cfg.attn_type` 选择正确的缓冲区。
3. `prelude` — 包含 `prelude_layers` 个 `TransformerBlock` 实例的 `nn.ModuleList`，使用密集 SwiGLU FFN。
4. `recurrent` — 单个 `RecurrentBlock`，包含一个 `TransformerBlock`（使用 MoE FFN）、`LTIInjection`、`ACTHalting` 和 `LoRAAdapter`。
5. `coda` — 包含 `coda_layers` 个 `TransformerBlock` 实例的 `nn.ModuleList`，使用密集 SwiGLU FFN。
6. `RMSNorm(dim)` 在 LM 头之前应用。
7. `nn.Linear(dim, vocab_size, bias=False)` LM 头，权重与嵌入层绑定。
8. 所有 `nn.Linear` 和 `nn.Embedding` 权重从 N(0, 0.02) 初始化。

**示例：**

```python
from open_mythos.main import OpenMythos, MythosConfig

cfg = MythosConfig(
    vocab_size=32000,
    dim=2048,
    n_heads=16,
    n_kv_heads=4,
    max_loop_iters=16,
    attn_type="mla",
)
model = OpenMythos(cfg)
print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")
```

---

## `forward`

```python
def forward(
    self,
    input_ids: torch.Tensor,
    n_loops: Optional[int] = None,
    kv_cache: Optional[dict] = None,
) -> torch.Tensor
```

通过完整的 前奏 → 递归块 → 尾声 流水线的单次前向传播。

### 参数

| 参数 | 类型 | 描述 |
|---|---|---|
| `input_ids` | `Tensor (B, T)` | token 索引序列的批次。`B` = 批大小，`T` = 序列长度 |
| `n_loops` | `int \| None` | 本次调用的递归循环深度。默认为 `cfg.max_loop_iters`。在推理时传入更大的值可外推到更难的问题（深度外推特性）。 |
| `kv_cache` | `dict \| None` | 如果提供，键和值将在此处累积用于自回归解码。在第一个解码步骤传入 `{}`，并在各步骤间复用同一字典。训练或全上下文推理时传入 `None`。 |

### 返回值

`Tensor (B, T, vocab_size)` — 每个位置上词汇表的原始（未归一化）logits。

### 行为详解

```
1. 嵌入:     x = embedding(input_ids)              # (B, T, dim)
2. 选择 RoPE 缓冲区:
     if attn_type == "mla": 使用 freqs_cis_mla[:T]
     else:                   使用 freqs_cis[:T]
3. 构建因果掩码（上三角 -inf）:
     if T > 1: mask = _causal_mask(T, device)
     else:     mask = None  （单 token 解码步骤）
4. 前奏:
     for i, layer in prelude:
         x = layer(x, freqs_cis, mask, kv_cache, f"prelude_{i}")
5. 冻结编码输入:
     e = x                                          # (B, T, dim)
6. 递归循环:
     x = recurrent(x, e, freqs_cis, mask, n_loops, kv_cache)
7. 尾声:
     for i, layer in coda:
         x = layer(x, freqs_cis, mask, kv_cache, f"coda_{i}")
8. 投影:   logits = lm_head(norm(x))             # (B, T, vocab_size)
```

**步骤 5（冻结 `e`）**是关键的架构不变量：编码输入 `e` 在前奏之后被捕获，并在*每次*循环迭代中不变地注入。这防止了隐藏状态在任何循环深度下偏离原始输入信号。

### 训练示例

```python
import torch
from open_mythos.main import OpenMythos, MythosConfig

model = OpenMythos(MythosConfig()).cuda()
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

input_ids = torch.randint(0, 32000, (2, 512)).cuda()
labels    = torch.randint(0, 32000, (2, 512)).cuda()

logits = model(input_ids)                    # (2, 512, 32000)
loss   = torch.nn.functional.cross_entropy(
    logits.view(-1, 32000),
    labels.view(-1),
)
loss.backward()
optimizer.step()
```

### 推理时深度外推

在 `N` 次循环上训练的循环 Transformer 可以在 `N + k` 次循环上评估，通常在困难的多跳问题上获得更高质量。在推理时传入 `n_loops`：

```python
# 使用 max_loop_iters=16 训练 — 在测试时尝试更深的推理
logits_deep = model(input_ids, n_loops=32)
```

---

## `generate`

```python
@torch.no_grad()
def generate(
    self,
    input_ids: torch.Tensor,
    max_new_tokens: int = 64,
    n_loops: int = 8,
    temperature: float = 1.0,
    top_k: int = 50,
) -> torch.Tensor
```

带 KV 缓存的自回归 token 生成。在步骤 0 处理完整提示，然后使用累积的缓存逐个 token 解码。

### 参数

| 参数 | 类型 | 默认值 | 描述 |
|---|---|---|---|
| `input_ids` | `Tensor (B, T)` | — | 提示 token 索引 |
| `max_new_tokens` | `int` | `64` | 要生成的新 token 数量 |
| `n_loops` | `int` | `8` | 每个解码步骤的递归循环深度。对于更难的提示可以高于训练值（深度外推） |
| `temperature` | `float` | `1.0` | 采样前应用于 logits 的 softmax 温度。值 < 1 使分布更尖锐（更少随机）；值 > 1 使分布更平坦 |
| `top_k` | `int` | `50` | 将每步采样限制在概率最高的 Top-K 个 token。`0` 禁用过滤（全词汇表采样） |

### 返回值

`Tensor (B, T + max_new_tokens)` — 原始提示与生成的 token 索引拼接。

### KV 缓存机制

在步骤 0，完整提示 `(B, T)` 被传入，所有层的键/值都填充到 `kv_cache` 中。在步骤 1…N 中，只传入最近的单个 token `(B, 1)`；注意力层从缓存中读取所有先前的 K/V。这使得解码成本与每步单个 token 成正比，而非与不断增长的完整序列成正比。

每层使用确定性字符串键进行缓存（`"prelude_0"`、`"recurrent_loop_3"`、`"coda_1"` 等），因此不同层的缓存永远不会冲突。

### 采样策略

```
logits = forward(cur_ids, n_loops, kv_cache)[:, -1, :] / temperature

if top_k > 0:
    threshold = logits.topk(top_k).values[:, -1:]
    logits[logits < threshold] = -inf

probs    = softmax(logits)
next_tok = multinomial(probs, num_samples=1)
```

### 生成示例

```python
import torch
from open_mythos.main import OpenMythos, MythosConfig

model = OpenMythos(MythosConfig()).eval()

# 分词后的提示（使用你选择的分词器）
prompt = torch.tensor([[1, 450, 3118, 310, 278]])   # (1, 5)

output = model.generate(
    prompt,
    max_new_tokens=128,
    n_loops=16,        # 更深的推理
    temperature=0.8,
    top_k=40,
)
# output.shape == (1, 133)
```

---

## 内部组件

以下子模块在 `OpenMythos` 内部组装。通常不直接调用，但理解它们有助于澄清模型的行为。

### `RecurrentBlock`

架构的核心。单个 `TransformerBlock`（使用 MoE FFN）在循环中运行最多 `n_loops` 次迭代，每次迭代的流水线如下：

```
h_loop = loop_index_embedding(h, t, loop_dim)   # 注入正弦循环索引信号
combined = RMSNorm(h_loop + e)                   # 加上冻结的编码输入
trans_out = TransformerBlock(combined, ...)       # 注意力 + MoE FFN
trans_out = trans_out + LoRAAdapter(trans_out, t) # 深度级 LoRA 增量
h = LTIInjection(h, e, trans_out)               # 稳定更新: A·h + B·e + trans_out
p = ACTHalting(h)                                # 每位置停止概率
```

当某位置的累积停止概率超过 `cfg.act_threshold` 时，该位置提前退出循环。如果所有位置都已停止，循环在 `n_loops` 之前退出。最终输出是 `h` 在各迭代间的 ACT 加权和。

### `LTIInjection`

实现稳定的递归更新规则 `h_{t+1} = A·h_t + B·e + transformer_out`。对角矩阵 `A` 的参数化方式：

```
A_continuous = Diag(-exp(log_A))     # 始终为负对角
A_discrete   = exp(Δt · A_continuous) # ZOH 离散化，值 ∈ (0, 1)
```

这从构造上保证谱半径 `ρ(A) < 1`，使循环模型无条件稳定，不受学习率或批次噪声影响。理论基础见 [Parcae (Prairie et al., 2026)](https://arxiv.org/abs/2604.12946)。

### `ACTHalting`

单个线性层将 `(B, T, dim) → (B, T)` 映射后接 sigmoid。在每个循环步骤中，每位置的标量停止概率被累积。当累积和超过 `cfg.act_threshold` 时，ACT 余量技巧将剩余概率质量分配为最终权重，该位置停止贡献。实现了 Graves (2016) ACT。

### `LoRAAdapter`

深度级低秩适配器，包含三个组件：

- `down`：共享的 `Linear(dim, rank)` — 将 Transformer 输出降维
- `B`：共享参数矩阵 `(rank, dim)` — 升维回完整维度
- `scale`：`Embedding(max_loops, rank)` — 每循环的逐元素缩放

每次迭代的增量为 `(down(x) * scale[t]) @ B`。弥合了纯权重绑定和完全独立逐层权重之间的表达能力差距。基于 [松弛递归 Transformer (Bae et al., 2024)](https://arxiv.org/pdf/2410.20672)。

### `TransformerBlock`

预归一化 Transformer 块，支持可切换的注意力和 FFN：

- **注意力：** `MLAttention`（MLA）或 `GQAttention`（GQA），由 `cfg.attn_type` 选择
- **FFN：** `MoEFFN`（当 `use_moe=True` 时，在 `RecurrentBlock` 内部）或密集 `Expert`（前奏、尾声）
- 通过 `RMSNorm` 对注意力输入和 FFN 输入进行预归一化

### `MLAttention`

多潜在注意力（[DeepSeek-V2, 2024](https://arxiv.org/abs/2405.04434)）。缓存仅存储压缩的 KV 潜在表示 `c_kv`（秩 `kv_lora_rank`）加上 RoPE 编码的键。在每个解码步骤中，`K_nope` 和 `V` 通过共享的上投影从 `c_kv` 廉价重建，用快速线性乘法换取显著更小的 KV 内存占用。

每层每 token 的缓存大小：`kv_lora_rank + n_heads × qk_rope_head_dim`，对比完整 GQA 缓存的 `n_kv_heads × head_dim × 2`。

### `GQAttention`

分组查询注意力（[Ainslie et al., 2023](https://arxiv.org/abs/2305.13245)）。`n_kv_heads` 个 KV 对在每 `n_heads // n_kv_heads` 个查询头之间共享，在保持完整查询表达能力的同时按该因子减少 KV 缓存。

### `MoEFFN`

细粒度混合专家 FFN（[DeepSeekMoE, Dai et al., 2024](https://arxiv.org/abs/2401.06066)）：

- **路由专家：** `n_experts` 个小型 SwiGLU FFN。每个 token 的路由器通过对学习的 logits 进行 softmax 选择 top-`n_experts_per_tok` 个。每专家偏置 `router_bias`（无梯度，外部更新）保持负载均衡。
- **共享专家：** `n_shared_experts` 个始终激活的 FFN，宽度为 `expert_dim × n_experts_per_tok`，吸收跨领域模式。

每 token 激活的总参数：路由容量的 `(n_experts_per_tok / n_experts)` + 所有共享容量。

### `Expert`

单个 SwiGLU 前馈单元：`down(silu(gate(x)) * up(x))`。既用作 `MoEFFN` 内部的单个路由专家，也用作前奏/尾声块中的密集 FFN。

### `RMSNorm`

均方根层归一化（[Zhang & Sennrich, 2019](https://arxiv.org/abs/1910.07467)）。通过 `x / rms(x)` 归一化，带有可学习的逐通道缩放权重。无偏置，无均值减除。在整个模型中替代标准 LayerNorm 使用。

---

## 工具函数

### `precompute_rope_freqs(dim, max_len, theta)`

预计算复数值 RoPE 旋转矩阵，作为 `(max_len, dim//2)` 的 complex64 张量。在 `__init__` 中调用一次并存储为缓冲区。

### `apply_rope(x, freqs_cis)`

将预计算的 RoPE 频率应用于查询或键张量，方法是将相邻特征对视为复数，并与位置相量逐点相乘。

### `loop_index_embedding(h, loop_t, loop_dim, theta)`

将正弦循环索引信号注入隐藏状态的前 `loop_dim` 个通道，类似于 RoPE 但作用于递归深度而非序列位置。允许共享的递归块权重在不同循环迭代中表现不同。

---

## 关键设计特性

| 特性 | 机制 | 优势 |
|---|---|---|
| 深度外推 | 使用循环相同权重的递归块 | 在 N 次循环上训练，在 N+k 上测试——无需重新训练即可解决更难的问题 |
| 参数效率 | 所有循环迭代间的权重共享 | k 层模型达到 kL 层模型的质量；参数 ≈ k，计算 ∝ L |
| 自适应计算 | 每位置 ACT 停止 | 简单 token 提前退出；困难 token 获得完整循环深度——在同一批次内 |
| 稳定训练 | ZOH 约束 A 的 LTI 注入（ρ(A) < 1） | 无残差爆炸；对高学习率鲁棒 |
| 领域广度 | 递归块中的 MoE FFN | 每个循环深度可路由到不同的专家子集 |
| 循环区分 | 循环索引正弦嵌入 | 相同权重在每次迭代中实现功能上不同的阶段 |
| 高效 KV 内存 | MLA（默认）或 GQA | MLA：在生产规模下比标准注意力小 10-20 倍的缓存 |
| 深度级适配 | 每循环迭代的 LoRA 适配器 | 超越纯权重绑定的表达能力；最小参数开销 |

---

## 完整配置参考

默认的 `MythosConfig()` 面向中等规模的研究模型。以下是用于快速实验的最小配置：

```python
from open_mythos.main import OpenMythos, MythosConfig

# 用于快速迭代/单元测试的最小配置
small_cfg = MythosConfig(
    vocab_size=8192,
    dim=256,
    n_heads=4,
    n_kv_heads=2,
    max_seq_len=512,
    max_loop_iters=4,
    prelude_layers=1,
    coda_layers=1,
    attn_type="gqa",
    n_experts=8,
    n_shared_experts=1,
    n_experts_per_tok=2,
    expert_dim=64,
    lora_rank=4,
)
model = OpenMythos(small_cfg)
```

以及匹配默认超参数的面向生产的 MLA 配置：

```python
# 默认 MLA 配置（匹配 MythosConfig() 默认值）
prod_cfg = MythosConfig(
    vocab_size=32000,
    dim=2048,
    n_heads=16,
    n_kv_heads=4,
    max_seq_len=4096,
    max_loop_iters=16,
    prelude_layers=2,
    coda_layers=2,
    attn_type="mla",           # 多潜在注意力
    kv_lora_rank=512,
    q_lora_rank=1536,
    qk_rope_head_dim=64,
    qk_nope_head_dim=128,
    v_head_dim=128,
    n_experts=64,
    n_shared_experts=2,
    n_experts_per_tok=4,
    expert_dim=512,
    act_threshold=0.99,
    rope_theta=500000.0,
    lora_rank=16,
)
model = OpenMythos(prod_cfg)
```

---

## 参考文献

| 组件 | 论文 |
|---|---|
| 递归深度 Transformer | [循环、思考与泛化 (2025)](https://arxiv.org/pdf/2604.07822) |
| LTI 稳定注入（Parcae） | [稳定循环语言模型的缩放定律 (Prairie et al., 2026)](https://arxiv.org/abs/2604.12946) |
| 循环 Transformer 推理 | [用潜在思维推理 (Saunshi et al., 2025)](https://arxiv.org/abs/2502.17416) |
| 多潜在注意力 | [DeepSeek-V2 (2024)](https://arxiv.org/abs/2405.04434) |
| 分组查询注意力 | [Ainslie et al., 2023](https://arxiv.org/abs/2305.13245) |
| 混合专家 FFN | [DeepSeekMoE (Dai et al., 2024)](https://arxiv.org/abs/2401.06066) |
| 自适应计算时间 | [Graves, 2016](https://arxiv.org/abs/1603.08983) |
| 深度级 LoRA | [松弛递归 Transformer (Bae et al., 2024)](https://arxiv.org/pdf/2410.20672) |
| RMSNorm | [Zhang & Sennrich, 2019](https://arxiv.org/abs/1910.07467) |
| RoPE | [Su et al., 2021](https://arxiv.org/abs/2104.09864) |
| Flash Attention 2 | [Dao, 2023](https://arxiv.org/abs/2307.08691) |
