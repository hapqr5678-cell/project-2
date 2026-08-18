"""v2_ddae_fsce_moe 公平性檢查。

驗證項目：
1. 同 seed 下 baseline MLPAE 與 MoEMLPAE 的 encoder/decoder 初始參數逐個完全一致
2. 初始輸出一致（因 residual experts zero-init）
3. train/val split 一致
4. 1 epoch smoke test：loss backward、gradient、shape、gates 加總
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(f"{os.path.dirname(__file__)}/../.."))
from ae import MLPAE, MoEMLPAE, Patches, corrupt, poisson_nll  # noqa: E402
from ae import build_fsce_graph, fsce_loss  # noqa: E402
from config.dataset import ensure_patches, PATCHES, N_CAT  # noqa: E402

import numpy as np  # noqa: E402

SEED = 0
LATENT_DIM = 2
BATCH = 256
NOISE_P = 0.3
NOISE_MODE = "thinning"
VAL_FRAC = 0.1
N_NEIGHBORS = 15
GRAPH_METRIC = "euclidean"
EDGE_BATCH = 256
LAMBDA_FSCE = 0.01
WARMUP_EPOCHS = 200
LR = 1e-3
WEIGHT_DECAY = 1e-6

passed = 0
failed = 0


def check(name, condition):
    global passed, failed
    if condition:
        print(f"  [PASS] {name}")
        passed += 1
    else:
        print(f"  [FAIL] {name}")
        failed += 1


def main():
    global passed, failed

    # ================================================================
    # 1. 同 seed 下 encoder/decoder 初始參數逐個完全一致
    # ================================================================
    print("=" * 60)
    print("1. Encoder/Decoder 初始參數一致性檢查")
    print("=" * 60)

    torch.manual_seed(SEED)
    baseline = MLPAE(LATENT_DIM)

    torch.manual_seed(SEED)
    moe = MoEMLPAE(LATENT_DIM)

    # 比較 encoder 的每一層參數
    for i, (bp, mp) in enumerate(zip(baseline.encoder.parameters(),
                                      moe.encoder.parameters())):
        name = f"encoder param {i} (shape={list(bp.shape)})"
        try:
            torch.testing.assert_close(bp, mp)
            check(name, True)
        except AssertionError:
            check(name, False)

    # 比較 decoder 的每一層參數
    for i, (bp, mp) in enumerate(zip(baseline.decoder.parameters(),
                                      moe.decoder.parameters())):
        name = f"decoder param {i} (shape={list(bp.shape)})"
        try:
            torch.testing.assert_close(bp, mp)
            check(name, True)
        except AssertionError:
            check(name, False)

    # 確認參數數量一致
    n_enc_base = sum(1 for _ in baseline.encoder.parameters())
    n_enc_moe = sum(1 for _ in moe.encoder.parameters())
    check(f"encoder param count match ({n_enc_base} vs {n_enc_moe})",
          n_enc_base == n_enc_moe)

    n_dec_base = sum(1 for _ in baseline.decoder.parameters())
    n_dec_moe = sum(1 for _ in moe.decoder.parameters())
    check(f"decoder param count match ({n_dec_base} vs {n_dec_moe})",
          n_dec_base == n_dec_moe)

    # ================================================================
    # 2. 初始輸出一致性（zero-init residual → 輸出應完全相同）
    # ================================================================
    print()
    print("=" * 60)
    print("2. 初始輸出一致性檢查（zero-init residual）")
    print("=" * 60)

    baseline.eval()
    moe.eval()
    x_test = torch.randn(16, N_CAT).abs()  # 模擬 count 向量

    with torch.no_grad():
        z_base, log_lam_base = baseline(x_test)
        z_moe, log_lam_moe = moe(x_test)

    try:
        torch.testing.assert_close(z_base, z_moe)
        check("z (latent) 一致", True)
    except AssertionError:
        check("z (latent) 一致", False)

    try:
        torch.testing.assert_close(log_lam_base, log_lam_moe)
        check("log_lam 一致", True)
    except AssertionError:
        check("log_lam 一致", False)

    # 也測 forward_with_gates
    with torch.no_grad():
        z_moe2, log_lam_moe2, gates = moe.forward_with_gates(x_test)

    try:
        torch.testing.assert_close(log_lam_moe, log_lam_moe2)
        check("forward vs forward_with_gates 一致", True)
    except AssertionError:
        check("forward vs forward_with_gates 一致", False)

    # ================================================================
    # 3. Train/Val split 一致性
    # ================================================================
    print()
    print("=" * 60)
    print("3. Train/Val split 一致性檢查")
    print("=" * 60)

    ensure_patches()
    data = Patches(PATCHES)

    g1 = torch.Generator().manual_seed(SEED)
    perm1 = torch.randperm(data.n, generator=g1)
    n_val1 = int(data.n * VAL_FRAC)
    val1, train1 = perm1[:n_val1], perm1[n_val1:]

    g2 = torch.Generator().manual_seed(SEED)
    perm2 = torch.randperm(data.n, generator=g2)
    n_val2 = int(data.n * VAL_FRAC)
    val2, train2 = perm2[:n_val2], perm2[n_val2:]

    check("train indices 一致", torch.equal(train1, train2))
    check("val indices 一致", torch.equal(val1, val2))
    check(f"train size = {len(train1)}", len(train1) == data.n - n_val1)
    check(f"val size = {len(val1)}", len(val1) == n_val1)

    # ================================================================
    # 4. Smoke test：1 epoch 訓練
    # ================================================================
    print()
    print("=" * 60)
    print("4. Smoke test（1 epoch 訓練）")
    print("=" * 60)

    torch.manual_seed(SEED)
    model = MoEMLPAE(LATENT_DIM)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    train_idx = train1
    val_idx = val1

    x_all = np.log1p(data.agg(torch.arange(data.n)).numpy())
    edge_i, edge_j, edge_w, a, b = build_fsce_graph(
        x_all, n_neighbors=N_NEIGHBORS, metric=GRAPH_METRIC)

    n_edges = len(edge_i)
    epoch = 0
    lam_t = LAMBDA_FSCE * min(1.0, (epoch + 1) / WARMUP_EPOCHS)
    g = torch.Generator().manual_seed(SEED + epoch)
    perm = train_idx[torch.randperm(len(train_idx), generator=g)]

    model.train()
    batch = perm[:BATCH]
    x = data.agg(batch)
    x_in = corrupt(x, NOISE_P, NOISE_MODE, generator=g)
    _, log_lam, gates = model.forward_with_gates(x_in)
    recon = poisson_nll(log_lam, x).mean()

    eg = torch.randint(0, n_edges, (EDGE_BATCH,), generator=g)
    pi, pj, pw = edge_i[eg], edge_j[eg], edge_w[eg]
    ni = torch.randint(0, data.n, (EDGE_BATCH,), generator=g)
    nj = torch.randint(0, data.n, (EDGE_BATCH,), generator=g)
    xi = corrupt(data.agg(torch.cat([pi, ni])), NOISE_P,
                 NOISE_MODE, generator=g)
    xj = corrupt(data.agg(torch.cat([pj, nj])), NOISE_P,
                 NOISE_MODE, generator=g)
    w = torch.cat([pw, torch.zeros(EDGE_BATCH)])
    zi, zj = model.encode(xi), model.encode(xj)
    fsce_val = fsce_loss(zi, zj, w, a, b).mean()

    loss = recon + lam_t * fsce_val
    opt.zero_grad()
    loss.backward()

    # 檢查 loss 有限
    check(f"loss is finite ({loss.item():.5f})", torch.isfinite(loss).item())

    # 檢查 router 有 gradient（注意：zero-init experts 讓 MoE residual=0，
    # 但 router 的 gradient 仍可能透過 softmax 的 log_lam 路徑傳回來。
    # 這裡只檢查 grad 存在且非 None，不要求 > 0。）
    router_has_grad = all(
        p.grad is not None
        for p in model.router.parameters()
    )
    check("router has gradient (not None)", router_has_grad)

    # 檢查 experts 有 gradient（zero-init 的最後一層 weight/bias grad 可能為 0，
    # 但前面幾層應有 gradient。這裡只檢查 grad 存在。）
    expert_has_grad = all(
        p.grad is not None
        for expert in model.residual_experts
        for p in expert.parameters()
    )
    check("residual experts have gradient (not None)", expert_has_grad)

    # 檢查沒有 NaN
    has_nan = any(
        torch.isnan(p.grad).any() for p in model.parameters()
        if p.grad is not None
    )
    check("no NaN in gradients", not has_nan)

    # 檢查 shape
    check(f"latent shape = (B, 2): {list(zi.shape)}", zi.shape == (EDGE_BATCH * 2, 2))
    check(f"log_lam shape = (B, N_CAT): {list(log_lam.shape)}",
          log_lam.shape == (len(batch), N_CAT))
    check(f"gates shape = (B, 2): {list(gates.shape)}",
          gates.shape == (len(batch), 2))

    # 檢查 gates 加總約等於 1
    gate_sums = gates.sum(dim=1)
    gates_sum_ok = torch.allclose(gate_sums, torch.ones_like(gate_sums), atol=1e-5)
    check(f"gates sum ~= 1 (max deviation: {(gate_sums - 1).abs().max():.2e})",
          gates_sum_ok)

    opt.step()
    check("optimizer step succeeded", True)

    # ================================================================
    # 總結
    # ================================================================
    print()
    print("=" * 60)
    total = passed + failed
    print(f"結果：{passed}/{total} 通過，{failed}/{total} 失敗")
    if failed == 0:
        print("All fairness checks passed!")
    else:
        print("Some checks FAILED!")
    print("=" * 60)


if __name__ == "__main__":
    main()
