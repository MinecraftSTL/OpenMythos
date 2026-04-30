import torch
from open_mythos.moda import MoDAConfig, MoDAModel


# ---------------------------------------------------------------------------
# 冒烟测试
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    # 微型配置: 4 层, 8 个路由专家, top-2
    cfg = MoDAConfig(
        vocab_size=512,
        d_model=128,
        n_layers=4,
        n_heads_q=4,
        n_heads_kv=2,
        head_dim=32,
        max_seq_len=64,
        # MoE: 2 个共享 + 8 个路由专家, 激活 top-2
        # (2+2)*64 = 256 ≈ 等效于密集 SwiGLU hidden~256
        n_shared_experts=2,
        n_routed_experts=8,
        n_activated_experts=2,
        expert_hidden_dim=64,
        moe_balance_alpha=0.01,
        moe_score_func="softmax",
    )

    model = MoDAModel(cfg).to(device)
    print(f"参数量: {model.num_parameters():,}")
    print(model)

    B, T = 2, 32
    input_ids = torch.randint(0, cfg.vocab_size, (B, T), device=device)
    labels = torch.randint(0, cfg.vocab_size, (B, T), device=device)

    logits, loss = model(input_ids, labels)
    assert logits.shape == (B, T, cfg.vocab_size)
    print(f"Logits 形状 : {logits.shape}")
    print(f"损失 (LM + 均衡): {loss.item():.4f}")

    loss.backward()

    # 验证梯度
    last_writes = {
        f"blocks.{cfg.n_layers - 1}.k_write.weight",
        f"blocks.{cfg.n_layers - 1}.v_write.weight",
    }
    missing = [
        name
        for name, p in model.named_parameters()
        if p.grad is None and name not in last_writes
    ]
    if missing:
        print(f"警告 — 意外缺失梯度的参数: {missing}")
    else:
        print("所有参数均已接收梯度（排除最后一层的写入投影）。")

    # 抽查: MoE 门控权重必须接收梯度（通过均衡损失 P_i）
    gate0_grad = model.blocks[0].moe.gate.weight.grad
    assert gate0_grad is not None, "blocks[0].moe.gate.weight 没有梯度！"
    print(f"blocks[0].moe.gate.weight 梯度范数 : {gate0_grad.norm().item():.6f}")

    # 抽查: 深度写入投影的梯度从第 ≥ 1 层的深度读取流回
    k0_grad = model.blocks[0].k_write.weight.grad
    assert k0_grad is not None, "blocks[0].k_write.weight 没有梯度！"
    print(f"blocks[0].k_write.weight 梯度范数  : {k0_grad.norm().item():.6f}")

    print("冒烟测试通过。")