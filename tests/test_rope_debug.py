"""
独立的 RoPE 调试测试 ── 记录张量输出和中间计算结果，
以便直观验证 precompute_rope_freqs 和 apply_rope 的正确性。
"""

import torch
from open_mythos.main import apply_rope, precompute_rope_freqs

DIM = 8
MAX_LEN = 6
THETA = 500000.0


def section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print("=" * 60)


def log(label: str, value) -> None:
    print(f"\n[{label}]")
    print(value)


# ---------------------------------------------------------------------------
# 1. precompute_rope_freqs —— 原始频率表
# ---------------------------------------------------------------------------

section("1. precompute_rope_freqs")

freqs = precompute_rope_freqs(dim=DIM, max_len=MAX_LEN, theta=THETA)
log("freqs 形状", freqs.shape)
log("freqs（复数）", freqs)
log("freqs.real", freqs.real)
log("freqs.imag", freqs.imag)
log("freqs 幅值（应全为 1.0）", freqs.abs())
log("freqs 角度（弧度）", freqs.angle())

print("\n预期：基础频率（每个维度对一个）")
base = 1.0 / (THETA ** (torch.arange(0, DIM, 2, dtype=torch.float32) / DIM))
log("基础频率", base)

print("\n预期：freqs[t, k].angle() == t * base[k]")
for t in range(MAX_LEN):
    expected_angles = t * base
    actual_angles = freqs[t].angle()
    match = torch.allclose(actual_angles, expected_angles, atol=1e-5)
    print(f"  t={t}: 角度匹配 = {match}  实际={actual_angles.tolist()}")

# ---------------------------------------------------------------------------
# 2. 位置 0 是恒等变换（freqs[0] == 1+0j）
# ---------------------------------------------------------------------------

section("2. freqs[0] 是恒等相量（1+0j）")
log("freqs[0]", freqs[0])
print(f"  所有幅值=1: {torch.allclose(freqs[0].abs(), torch.ones(DIM // 2))}")
print(f"  所有角度=0: {torch.allclose(freqs[0].angle(), torch.zeros(DIM // 2))}")

# ---------------------------------------------------------------------------
# 3. apply_rope —— 形状和数据类型
# ---------------------------------------------------------------------------

section("3. apply_rope —— 形状和数据类型")

B, T, H = 2, MAX_LEN, 3
x = torch.randn(B, T, H, DIM)
log("输入 x 形状", x.shape)

out = apply_rope(x, freqs)
log("输出形状", out.shape)
print(f"  形状保持不变: {out.shape == x.shape}")

# 数据类型 float16
x_half = x.half()
out_half = apply_rope(x_half, freqs)
print(f"  float16 数据类型保持不变: {out_half.dtype == torch.float16}")

# ---------------------------------------------------------------------------
# 4. apply_rope —— 等距性（范数保持）
# ---------------------------------------------------------------------------

section("4. apply_rope —— 范数保持（等距性）")

norms_in = x.norm(dim=-1)
norms_out = out.norm(dim=-1)
log("输入范数（第一个批次项）", norms_in[0])
log("输出范数（第一个批次项）", norms_out[0])
print(
    f"  最大绝对范数差异: {(norms_in - norms_out).abs().max().item():.2e}"
)
print(
    f"  范数保持（atol=1e-5）: {torch.allclose(norms_in, norms_out, atol=1e-5)}"
)

# ---------------------------------------------------------------------------
# 5. 位置 0 是恒等变换
# ---------------------------------------------------------------------------

section("5. 位置 0 是恒等变换")

x1 = torch.randn(1, 1, 2, DIM)
out1 = apply_rope(x1, freqs[:1])
log("输入  x[:,0]", x1[0, 0])
log("输出  x[:,0]", out1[0, 0])
log("差异（应约为 0）", (x1 - out1).abs())
print(f"  位置 0 恒等: {torch.allclose(x1, out1, atol=1e-6)}")

# ---------------------------------------------------------------------------
# 6. 不同位置产生不同的旋转
# ---------------------------------------------------------------------------

section("6. 不同位置产生不同的旋转")

x2 = torch.ones(1, MAX_LEN, 1, DIM)
out2 = apply_rope(x2, freqs)
print("每个位置的输出（所有输入=1.0）:")
for t in range(MAX_LEN):
    print(f"  pos={t}: {out2[0, t, 0].tolist()}")

# ---------------------------------------------------------------------------
# 7. 逆旋转恢复原始值
# ---------------------------------------------------------------------------

section("7. 逆旋转恢复原始值")

x3 = torch.randn(2, T, H, DIM)
rotated = apply_rope(x3, freqs)
xc = torch.view_as_complex(rotated.float().reshape(*rotated.shape[:-1], -1, 2))
inv_freqs = freqs.conj()
recovered = (
    torch.view_as_real(xc * inv_freqs.unsqueeze(0).unsqueeze(2))
    .flatten(-2)
    .to(x3.dtype)
)
diff = (x3 - recovered).abs()
log("最大恢复误差", diff.max().item())
print(f"  恢复成功（atol=1e-5）: {torch.allclose(x3, recovered, atol=1e-5)}")

# ---------------------------------------------------------------------------
# 8. start_pos 正确性 —— 核心生成 bug 修复
# ---------------------------------------------------------------------------

section("8. start_pos 正确性（生成 bug）")

print(
    "模拟：长度为 4 的提示词，然后解码步骤在位置 4 接收 token。\n"
    "旧的有 bug 的代码：freqs[:1] → 始终是位置 0。\n"
    "修复后的代码：freqs[4:5] → 位置 4。"
)

prompt_len = 4
decode_token = torch.randn(1, 1, 1, DIM)

out_buggy = apply_rope(decode_token, freqs[:1])  # 错误：始终是位置 0
out_fixed = apply_rope(
    decode_token, freqs[prompt_len : prompt_len + 1]
)  # 正确：位置 4

log("freqs[0]（位置 0）", freqs[0])
log(f"freqs[{prompt_len}]（位置 {prompt_len}）", freqs[prompt_len])
log("有 bug 的输出（位置 0 编码）", out_buggy[0, 0, 0])
log("修复后的输出（位置 4 编码）", out_fixed[0, 0, 0])
print(f"  输出不同（应该不同）: {not torch.allclose(out_buggy, out_fixed)}")

# ---------------------------------------------------------------------------
# 9. 相对位置属性
# ---------------------------------------------------------------------------

section("9. 相对位置属性: <RoPE(q,m), RoPE(k,n)> 仅取决于 (n-m)")

dim = 16
max_len = 32
freqs_big = precompute_rope_freqs(dim=dim, max_len=max_len, theta=THETA)
torch.manual_seed(42)
q = torch.randn(1, 1, 1, dim)
k = torch.randn(1, 1, 1, dim)


def rope_at(tensor, pos):
    seq = torch.zeros(1, pos + 1, 1, dim)
    seq[0, pos] = tensor[0, 0]
    return apply_rope(seq, freqs_big[: pos + 1])[:, pos : pos + 1]


dot_3_9 = (rope_at(q, 3) * rope_at(k, 9)).sum()
dot_1_7 = (rope_at(q, 1) * rope_at(k, 7)).sum()
dot_0_6 = (rope_at(q, 0) * rope_at(k, 6)).sum()
print(f"  <RoPE(q,3), RoPE(k,9)>: {dot_3_9.item():.6f}")
print(f"  <RoPE(q,1), RoPE(k,7)>: {dot_1_7.item():.6f}")
print(f"  <RoPE(q,0), RoPE(k,6)>: {dot_0_6.item():.6f}")
print(
    f"  全部相等（偏移=6）: {torch.allclose(dot_3_9, dot_1_7, atol=1e-5) and torch.allclose(dot_1_7, dot_0_6, atol=1e-5)}"
)

section("完成 —— 所有检查已完成")