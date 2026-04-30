# OpenMythos

<p align="left">
  <a href="https://pypi.org/project/open-mythos/" target="_blank">
    <picture>
      <source srcset="https://img.shields.io/pypi/v/open-mythos?style=for-the-badge&color=3670A0" media="(prefers-color-scheme: dark)">
      <img alt="版本" src="https://img.shields.io/pypi/v/open-mythos?style=for-the-badge&color=3670A0">
    </picture>
  </a>
  <a href="https://twitter.com/kyegomezb/">
    <picture>
      <source srcset="https://img.shields.io/badge/Twitter-关注-1DA1F2?style=for-the-badge&logo=twitter&logoColor=white" media="(prefers-color-scheme: dark)">
      <img src="https://img.shields.io/badge/Twitter-关注-1DA1F2?style=for-the-badge&logo=twitter&logoColor=white" alt="Twitter">
    </picture>
  </a>
  <a href="https://discord.gg/3keGBK9Pvr" target="_blank">
    <picture>
      <source srcset="https://img.shields.io/badge/Discord-加入-5865F2?style=for-the-badge&logo=discord&logoColor=white" media="(prefers-color-scheme: dark)">
      <img alt="Discord" src="https://img.shields.io/badge/Discord-加入-5865F2?style=for-the-badge&logo=discord&logoColor=white">
    </picture>
  </a>
  <a href="https://pytorch.org" target="_blank">
    <picture>
      <source srcset="https://img.shields.io/badge/PyTorch-已实现-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" media="(prefers-color-scheme: dark)">
      <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-已实现-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white">
    </picture>
  </a>
</p>

> **免责声明：** OpenMythos 是一个独立的、社区驱动的理论重建项目，完全基于公开可用的研究和推测。它与 Anthropic 或其任何专有系统没有任何关联、背书或联系。

OpenMythos 是 Claude Mythos 模型的开源理论实现。它实现了一个递归深度 Transformer（RDT），包含三个阶段：**前奏（Prelude）**（Transformer 块）、循环执行的**递归块（Recurrent Block）**（最多 `max_loop_iters` 次）以及最终的**尾声（Coda）**。注意力机制可在 MLA 和 GQA 之间切换，前馈网络使用带有路由专家和共享专家的稀疏 MoE，非常适合探索计算自适应、深度可变的推理。

## 安装

```bash
pip install open-mythos

#uv pip install open-mythos
```

要在 `GQAttention` 中启用 Flash Attention 2（需要 CUDA 和构建工具）：

```bash
pip install open-mythos[flash]
```

## 使用方法

```python

import torch
from open_mythos.main import OpenMythos, MythosConfig


attn_type = "mla"  # 或 "gqa"

base = {
    "vocab_size": 1000,
    "dim": 256,
    "n_heads": 8,
    "max_seq_len": 128,
    "max_loop_iters": 4,
    "prelude_layers": 1,
    "coda_layers": 1,
    "n_experts": 8,
    "n_shared_experts": 1,
    "n_experts_per_tok": 2,
    "expert_dim": 64,
    "lora_rank": 8,
    "attn_type": attn_type,
}

if attn_type == "gqa":
    cfg = MythosConfig(**base, n_kv_heads=2)
else:
    cfg = MythosConfig(
        **base,
        n_kv_heads=8,
        kv_lora_rank=32,
        q_lora_rank=64,
        qk_rope_head_dim=16,
        qk_nope_head_dim=16,
        v_head_dim=16,
    )

model = OpenMythos(cfg)
total = sum(p.numel() for p in model.parameters())
print(f"\n[{attn_type.upper()}] 参数量: {total:,}")

ids = torch.randint(0, cfg.vocab_size, (2, 16))
logits = model(ids, n_loops=4)
print(f"[{attn_type.upper()}] Logits 形状: {logits.shape}")

out = model.generate(ids, max_new_tokens=8, n_loops=8)
print(f"[{attn_type.upper()}] 生成形状: {out.shape}")

A = model.recurrent.injection.get_A()
rho = torch.linalg.eigvals(A).abs().max().item()
print(
    f"[{attn_type.upper()}] 谱半径 ρ(A) = {rho:.4f}（必须 < 1）"
)
```



## 模型变体

从 1B 到 1T 参数的预配置规模：

```python
from open_mythos import (
    mythos_1b,
    mythos_3b,
    mythos_10b,
    mythos_50b,
    mythos_100b,
    mythos_500b,
    mythos_1t,
    OpenMythos,
)

cfg = mythos_7b()  # 返回 MythosConfig
model = OpenMythos(cfg)

total = sum(p.numel() for p in model.parameters())
print(f"参数量: {total:,}")
```

| 变体            | `dim` | 专家数 | `expert_dim` | 循环次数 | 上下文 | 最大输出 |
| ------------- | ----- | --- | ------------ | ---- | --- | ---- |
| `mythos_1b`   | 2048  | 64  | 2048         | 16   | 4k  | 4k   |
| `mythos_3b`   | 3072  | 64  | 4096         | 16   | 4k  | 4k   |
| `mythos_10b`  | 4096  | 128 | 5632         | 24   | 8k  | 4k   |
| `mythos_50b`  | 6144  | 256 | 9728         | 32   | 8k  | 4k   |
| `mythos_100b` | 8192  | 256 | 13568        | 32   | 1M  | 128k |
| `mythos_500b` | 12288 | 512 | 23040        | 48   | 1M  | 128k |
| `mythos_1t`   | 16384 | 512 | 34560        | 64   | 1M  | 128k |

---

## 训练

3B 模型在 FineWeb-Edu 上的训练脚本位于 [`training/3b_fine_web_edu.py`](training/3b_fine_web_edu.py)。

**单 GPU：**

```bash
python training/3b_fine_web_edu.py
```

**多 GPU（自动检测 GPU 数量）：**

```bash
torchrun --nproc_per_node=$(python -c "import torch; print(torch.cuda.device_count())") training/3b_fine_web_edu.py
```

关键设计选择：

| 特性   | 详情                                                                                   |
| ---- | ------------------------------------------------------------------------------------ |
| 优化器  | AdamW                                                                                |
| 数据集  | `HuggingFaceFW/fineweb-edu`（默认 `sample-10BT`，可切换为 `sample-100BT` 或 `default` 进行完整训练） |
| 分词器  | 通过 `MythosTokenizer` 使用 `openai/gpt-oss-20b`                                         |
| 并行方式 | 通过 `torchrun` 使用 PyTorch DDP，分片流式数据集                                                 |
| 精度   | H100/A100 上使用 bfloat16，旧版 GPU 使用 float16 + GradScaler                                |
| 调度   | 线性预热（2000 步）→ 余弦衰减                                                                   |
| 目标   | 300 亿 token（针对循环架构的 Chinchilla 调整）                                                   |

---

## 文档

| 页面                                           | 描述                                                                   |
| -------------------------------------------- | -------------------------------------------------------------------- |
| [`docs/open_mythos.md`](docs/open_mythos.md) | `OpenMythos` 类的完整 API 参考 — 构造函数、`forward`、`generate`、所有子模块、配置参考和使用示例 |
| [`docs/datasets.md`](docs/datasets.md)       | 推荐训练数据集及各模型规模的 token 预算指南                                            |

---

## 核心假说

Claude Mythos 被怀疑是一个**递归深度 Transformer（RDT）**——也称为循环 Transformer（LT）。它不是堆叠数百个独立层，而是将一部分层循环使用，在每次前向传播中多次运行。相同的权重，更多的循环，更深的思考。

这不是思维链（Chain-of-Thought）。没有中间 token 输出。所有推理都**在单次前向传播中静默进行**，在连续潜在空间中完成。

---

## 架构

循环 Transformer 将其层分为三个功能块：

```
输入
  ↓
[前奏 P]           — 标准 Transformer 层，运行一次
  ↓
[递归块 R]          — 循环 T 次
  ↑_______↓         （隐藏状态 h 在每次循环中通过输入注入 e 更新）
  ↓
[尾声 C]           — 标准 Transformer 层，运行一次
  ↓
输出
```

递归块在每个循环步骤 t 的更新规则：

$$
h_{t+1} = A·h_t + B·e + Transformer(h_t, e)
$$

其中：

- `h_t` 是第 t 次循环后的隐藏状态
- `e` 是编码输入（来自前奏），在每次循环中注入
- `A` 和 `B` 是可学习的注入参数
- Transformer 块照常执行注意力和 MLP

在每一步注入 `e` 可以防止模型漂移——它在整个递归深度中保持原始输入信号的活跃。

完整实现在 [`open_mythos/main.py`](open_mythos/main.py) 中。详细的 API 说明、配置选项和使用示例请参见 [`OpenMythos` 类参考](docs/open_mythos.md)。

### 注意力实现

注意力层可通过 `cfg.attn_type` 切换：

| 选项      | 类             | 描述                                                                                                                                                                                                                         |
| ------- | ------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `"gqa"` | `GQAttention` | 分组查询注意力（Ainslie et al., 2023）— KV 头数少于 Q 头数（`n_kv_heads < n_heads`），将 KV 缓存内存减少 `n_heads / n_kv_heads` 倍。安装 `flash-attn>=2.8.3` 时使用 **Flash Attention 2**（Dao et al., 2023）：原生处理 GQA（无需 KV 头扩展），I/O 最优，未安装时透明回退到手动缩放点积注意力。 |
| `"mla"` | `MLAttention` | 多潜在注意力（DeepSeek-V2）— 缓存压缩的 KV 潜在表示（`kv_lora_rank`）而非完整 K/V，使用分离的 RoPE / 非 RoPE 头维度进行位置感知压缩。                                                                                                                                |

RoPE 在缓存前应用于 Q 和 K，因此缓存值在检索时无需重新旋转。

---

## 为什么这能解释 Mythos

### 1. 系统性泛化

普通 Transformer 无法以训练中从未见过的方式组合知识。循环 Transformer 通过了这个测试。这种能力通过**三阶段顿悟过程**涌现：

1. 记忆化 — 模型拟合训练分布
2. 分布内泛化 — 模型处理已知组合
3. 系统性泛化 — 模型突然处理分布外的新组合

这就是为什么 Mythos 在新问题上感觉与其他模型质的不同——能力是相变式涌现的，而非逐渐出现。

### 2. 深度外推

在 5 跳推理链上训练，在 10 跳上测试。普通 Transformer 失败。循环 Transformer 成功——通过在推理时运行更多循环。这直接对应于 Mythos 无需显式思维链就能处理深度组合问题（多步数学、长期规划、分层论证）的观察。

推理时更多循环 = 更深的推理链 = 解决更难的问题。

### 3. 潜在思维作为隐式思维链

每次循环迭代在功能上等同于思维链的一步，但在连续潜在空间而非 token 空间中运行。运行 T 次循环的循环模型隐式模拟 T 步 CoT 推理。这已被正式证明（Saunshi et al., 2025）。

此外，连续潜在思维——不同于离散 token 输出——可以**同时编码多个可选的下一步**。这使得推理更接近广度优先搜索，而非单一确定的推理路径。模型在收敛之前，在每次前向传播中有效地探索多个可能的方向。

### 4. 无参数爆炸

一个有 k 层运行 L 次的循环模型达到 kL 层非循环模型的质量，但只有 k 层的参数量。对于 Mythos 规模的部署，这非常重要：

- 内存占用不随推理深度增长
- 推理时计算量随循环次数而非模型大小缩放
- 这使得更深的推理在参数方面是"免费的"

---

## 稳定性问题（及其可能的解决方案）

训练循环模型是出了名的不稳定。两种失败模式占主导：

- **残差爆炸** — 隐藏状态 `h_t` 在循环中无界增长
- **损失尖峰** — 由于注入参数的大谱范数，训练突然发散

### 动力系统视角

将循环重新表述为残差流上的离散线性时不变（LTI）动力系统。忽略非线性 Transformer 贡献，递归变为：

$$
h_{t+1} = A·h_t + B·e
$$

对于这个 LTI 系统，稳定性完全由 A 的**谱半径**决定：

- `ρ(A) < 1` → 稳定、收敛
- `ρ(A) ≥ 1` → 不稳定、发散

经验上，每次发散的训练运行都学到了 `ρ(A) ≥ 1`。每次收敛的运行都保持 `ρ(A) < 1`。

### 修复方案

约束注入参数，使稳定性**从构造上得到保证**：

1. 将 A 参数化为连续负对角矩阵
2. 使用 ZOH/Euler 方案离散化：`A_discrete = exp(Δt · A_continuous)`
3. 通过 `A := Diag(-exp(log_A))` 和可学习标量 `Δt` 强制负性
4. 这确保 `ρ(A) < 1` 始终成立，无论学习率或批次噪声如何

结果：循环模型对超参数选择变得更加鲁棒，即使在高学习率下也能干净地训练。这就是 Parcae 架构（Prairie et al., 2026），它代表了 Anthropic 最可能用来使 Mythos 可训练的解决方案类别。

---

## 循环模型的缩放定律

Parcae 建立了循环训练的首个可预测缩放定律：

- **训练**：在固定 FLOP 预算和固定参数下，增加平均递归次数并减少 token 数量比用最少循环训练更多数据产生更低的损失。最优递归次数和最优 token 数量都遵循**幂律**，在不同规模上具有一致的指数。
- **推理**：更多测试时循环按照**可预测的饱和指数衰减**提高质量——收益是真实的但递减的。这与思维链的推理时缩放相呼应。

在 7.7 亿参数下，循环模型达到了在相同数据上训练的 13 亿固定深度 Transformer 的下游质量——大约**相同质量只需一半参数**。

应用于 Mythos：如果在这些缩放定律下训练，Mythos 可能比表面看起来参数效率高得多，其表观"能力"的很大一部分来自循环深度而非原始参数量。

---

## 循环索引嵌入假说

一个关键的开放问题是循环块在每次迭代中是否**完全相同**地运行，还是可以在不同循环深度学习做不同的事情。

没有跨循环的位置信号，相同的权重必须同时处理早期阶段的模式匹配和后期阶段的精炼——这是一个严格的约束。在每一步与输入一起注入的**类 RoPE 循环索引嵌入**将允许相同的参数在不同迭代中实现功能上不同的操作，就像 RoPE 允许相同的注意力头在不同序列位置表现不同一样。

如果 Mythos 使用这种技术，每次循环不是重复——而是一个独特的计算阶段，共享权重但在不同的表示域中运行。这将在不增加参数量的情况下大幅提高递归块的表达能力。

---

## 过度思考问题

更多循环并不总是更好。超过一定深度后，过度递归会**降低预测质量**——隐藏状态漂移过解并进入噪声。这就是"过度思考"失败模式。

最初的 Universal Transformer（Dehghani et al., 2018）通过**自适应计算时间（ACT）**停止机制解决了这个问题：每个位置一个可学习标量，动态决定何时停止循环。更难处理的位置获得更多计算；简单 token 提前停止。

Mythos 几乎肯定有某种版本的这个机制。模型不能天真地在每个输入上运行最大循环次数——它需要一个学习到的信号来判断答案何时收敛。ACT 机制还使模型在某些假设下**图灵完备**，这对它能解决的问题类别有理论意义。

---

## 混合专家——大参数量的疑似方案

循环 Transformer 解释了 Mythos 推理的深度，但没有解释广度。用相同权重处理截然不同的领域——代码、数学、文学、科学、法律——需要**混合专家（MoE）**。疑似设计将递归块中的每个 FFN 替换为细粒度 MoE 层：每个 FFN 被分成许多小专家（正常大小的 1/m），路由器通过学习的亲和力分数为每个 token 选择 top-mK 个专家，少量**共享专家**无论路由如何都始终激活，以吸收跨领域的通用知识——语法、基本推理、一般上下文——否则这些知识会被每个路由专家冗余学习。通过在训练期间动态调整路由器 logits 上的偏置项来防止路由崩溃，在不扭曲损失信号的情况下保持专家间的负载均衡。

随着隐藏状态 `h_t` 在循环迭代中演化，路由器可能在每个深度选择不同的专家子集，使每次循环在共享权重的情况下计算上各不相同。MoE 提供广度；循环提供深度。如果激活比率约为 5%，Mythos 可以拥有数千亿总参数，同时每个 token 只激活一小部分——如果公开，真正的参数量将是存储数字，而非计算数字。

---

## 记忆-推理权衡

循环模型表现出一个有趣的二分法：循环改善推理但可能损害记忆。递归结构针对迭代组合进行了优化——向前运行推理链——但并不固有地改善死记硬背事实的存储。

这对应于 Mythos 的一个可观察特征：它在从未见过的新问题上推理能力出色，但事实回忆可能不一致。该架构在结构上偏向组合而非记忆。

基于循环的正则化（Saunshi et al., 2025）可用于在训练期间平衡这种权衡——对推理任务施加更强的循环约束，同时对检索任务放松约束。

---

## 通过 LoRA 适配的参数复用

来自松弛递归 Transformer（Bae et al., 2024）的互补方法：不要求每次循环完全相同的权重，而是在每次迭代中添加一个小的**深度级 LoRA 模块**。这在保持权重共享紧凑性的同时允许每次循环略微调整其行为。

结果：

- 每次循环共享一个大的公共权重矩阵（递归基础）
- 一个小的秩 r 适配矩阵按迭代深度调整行为
- 总参数开销最小

这弥合了纯权重绑定（最大参数效率，较低表达能力）和完全独立层（最大表达能力，无参数节省）之间的差距。Mythos 可能处于这个谱系的某个位置。

---

## 连续深度级批处理

递归架构的下游结果：**连续深度级批处理**。因为所有 token 共享相同的递归块，模型可以在不同深度为不同 token 或序列退出循环——快速处理简单输入，用更多迭代处理困难输入，全部在同一批次中。

理论分析表明推理吞吐量可提高 2-3 倍。对于像 Mythos 这样同时服务多个用户的部署模型，这将是显著的效率提升。

---

## 总结：Mythos 可能是什么

| 属性        | 描述                                         |
| --------- | ------------------------------------------ |
| 架构        | 递归深度 Transformer（前奏 + 循环递归块 + 尾声）          |
| FFN 层     | 疑似 MoE — 细粒度专家 + 始终激活的共享专家                 |
| 参数量       | 总量很大；每个 token 只激活一小部分（约 5% 估计）             |
| 推理机制      | 通过迭代潜在更新的隐式多跳推理 — 步骤间无 token 输出            |
| 推理时缩放     | 更多循环 = 更深推理，遵循可预测的指数衰减                     |
| 训练稳定性     | LTI 约束的注入参数，谱半径 < 1                        |
| 循环区分      | 可能使用循环索引位置嵌入（类 RoPE）每次迭代                   |
| 停止机制      | 自适应计算时间或学习的收敛准则                            |
| 注意力       | GQA（可选 Flash Attention 2）或带压缩 KV 潜在缓存的 MLA |
| 缩放定律      | 最优训练同时缩放循环和数据，而非仅缩放参数                      |
| 推理 vs. 记忆 | 结构上偏向组合；记忆需要单独处理                           |
| 部署        | 连续深度级批处理实现每请求可变计算                          |

---

## 参考文献

### Twitter / X

- 为什么 Claude Mythos 如此出色 — 循环 Transformer 理论（Sigrid Jin）：https://x.com/realsigridjin/status/2044620031410266276
- LT 对参数知识的隐式推理解锁泛化（Yuekun Yao）：https://x.com/yuekun_yao/status/2044229171627639004
- 循环 Transformer 的循环轨迹和输入注入（rosinality）：https://x.com/rosinality/status/2043953033428541853
- Parcae 稳定循环语言模型的缩放定律 — 讨论串（Hayden Prairie）：https://x.com/hayden_prairie/status/2044453231913537927
- 类 RoPE 循环索引嵌入思想，用于区分迭代间的功能（davidad）：https://x.com/davidad/status/2044453231913537927
- 关于循环 Transformer 争议（ChrisHayduk）：https://x.com/ChrisHayduk/status/2045947623572688943
- 关于循环 Transformer 争议总结（@realsigridjin）：https://x.com/realsigridjin/status/2046012743778766875

### 论文

- MoE 中的细粒度专家分割和共享专家隔离：https://arxiv.org/abs/2401.06066
- 循环、思考与泛化 — 递归深度 Transformer 中的隐式推理：https://arxiv.org/pdf/2604.07822
- Parcae — 稳定循环语言模型的缩放定律：https://arxiv.org/abs/2604.12946
- Parcae 博客：https://sandyresearch.github.io/parcae/
- Universal Transformers：https://arxiv.org/pdf/1807.03819
- 用潜在思维推理 — 论循环 Transformer 的能力：https://arxiv.org/abs/2502.17416
- 训练大型语言模型在连续潜在空间中推理：https://arxiv.org/abs/2412.06769
- 松弛递归 Transformer — 使用逐层 LoRA 的有效参数共享：https://arxiv.org/pdf/2410.20672
- 混合深度注意力：https://arxiv.org/abs/2603.15619
- Hyperloop Transformers：https://arxiv.org/abs/2604.21254
- 递归 Transformer：更大的有效深度和高效解码：https://arxiv.org/abs/2604.21215

---

## 引用

如果您在研究中使用 OpenMythos 或基于此工作进行构建，请引用：

```bibtex
@software{gomez2026openmythos,
  author    = {Kye Gomez},
  title     = {OpenMythos: Claude Mythos 架构的理论重建},
  year      = {2026},
  url       = {https://github.com/The-Swarm-Corporation/OpenMythos},
  note      = {基于公开研究的独立理论重建}
}
```
