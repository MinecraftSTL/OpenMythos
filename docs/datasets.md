# 推荐训练数据集

| 数据集 | HuggingFace | Token 数量 | 许可证 | 用途 |
|---|---|---|---|---|
| FineWeb-Edu | [HuggingFaceFW/fineweb-edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) | 1.3T | Apache 2.0 | 主要预训练 |
| OpenHermes 2.5 | [teknium/OpenHermes-2.5](https://huggingface.co/datasets/teknium/OpenHermes-2.5) | ~100万样本 | Apache 2.0 | 指令微调（~5% 混合） |
| OpenWebMath | [open-web-math/open-web-math](https://huggingface.co/datasets/open-web-math/open-web-math) | 14.7B | ODC-By | 数学/推理增强 |

---

## 主要预训练

### FineWeb-Edu
- **HuggingFace：** `HuggingFaceFW/fineweb-edu`
- **规模：** 1.3T tokens
- **许可证：** Apache 2.0
- **选择理由：** 经过教育质量过滤的网页文本。在下游基准测试中优于 The Pile、C4 和 RefinedWeb。已完成去重和清洗。
- **建议起步：** 先用 `sample-10BT` 验证你的训练流水线，然后使用 `sample-100BT` 或完整语料库进行正式训练。

## 补充数据集

### OpenHermes 2.5
- **HuggingFace：** `teknium/OpenHermes-2.5`
- **规模：** ~100万条指令样本
- **许可证：** Apache 2.0
- **选择理由：** 高质量的指令遵循数据。在 FineWeb-Edu 基础上按 token 数量混入约 5%，可提升指令遵循能力而不降低通用能力。

### OpenWebMath
- **HuggingFace：** `open-web-math/open-web-math`
- **规模：** ~14.7B tokens
- **许可证：** ODC-By
- **选择理由：** 数学导向的网页文本。如果你希望增强定量和符号推理能力，可以添加此数据集。对于 10B+ 变体尤其有用，因为推理深度在这些规模下更为重要。

## Token 预算建议

| 变体 | Chinchilla 最优 | 推荐（循环架构） |
|---|---|---|
| 1B | ~20B tokens | ~10–15B tokens |
| 3B | ~60B tokens | ~30–40B tokens |
| 10B | ~200B tokens | ~100–150B tokens |
| 50B+ | ~1T+ tokens | ~500B+ tokens |

循环架构比标准 Transformer 具有更高的样本效率——由于收敛更快，用更少的 token 即可达到相同的验证损失。"推荐（循环架构）"列反映了这一点，基于 Tiny Shakespeare 实验结果，OpenMythos 达到等效损失的速度比 nanoGPT 快约 2.5 倍。