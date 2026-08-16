"""訓練 v2_dvae_fsce：Dirichlet VAE + FSCE。

跟 v2_dvae 的 train.py 相比多了兩件事——訓練前先在高維 count 向量上建一次
FSCE 用的 fuzzy graph；訓練迴圈裡每個 step 除了原本的 reconstruction + KL
batch，額外抽一批正樣本邊（來自 fuzzy graph）跟等量的隨機負樣本邊，對
theta 算 FSCE loss，加權後跟 (Poisson NLL + BETA·dirichlet_kl) 加在一起、
共用同一次 backward/opt.step()。

FSCE 權重用線性 warm-up：前 FSCE_WARMUP_EPOCHS 個 epoch 是 0，之後花同樣
長度線性升到 LAMBDA_FSCE，先讓 encoder 學出有意義的 count -> theta 映射，
再逐步加拓撲約束，避免兩個 loss 一開始就互相打架。這一段跟 v2_vae_fsce
完全相同。

EPOCHS 刻意留 1000 而不是跟 v2_vae_fsce 一樣的 500：這一版要回答的問題是
「在 v2_dvae 上加 FSCE 會怎樣」，跟 v2_dvae 的 A/B 只能有 FSCE 一個變因，
訓練長度不能一起改。代價是不能直接拿 v2_vae_fsce 的數字並排比。

以下是 v2_dvae 原本的說明，其餘設定完全沿用。
---
單段訓練，loss = Poisson NLL + BETA·dirichlet_kl，沒有 v2_gvae 那種
暖身 + GMM 初始化的兩段式流程——因為 Dirichlet 的先驗參數是固定的
（對稱 alpha），不需要先把 latent 養出形狀再去 fit，這本身就是換這個
先驗的好處之一。

BETA 沿用 v2_vae / v2_gvae 的 0.01，讓重建那一路是同一把尺，三版之間
唯一的差別只有 KL 的先驗形狀：
    v2_vae   單峰高斯 N(0,I)          latent 是不受限的實數向量
    v2_gvae  K 個高斯峰的混合          latent 仍是實數向量，額外有硬分群
    v2_dvae  Dir(alpha)               latent 是 K 維的成分比例（simplex）

K=3 不是隨便挑的：simplex 只有 K-1 個自由度，要跟 v2_vae / v2_gvae 的
latent_dim=2 在同一個容量上比較，K 就必須是 3（K=2 只剩 1 個自由度，
等於把模型砍成一維）。K=3 的 simplex 是一個三角形，正好可以直接畫成
三角座標圖，不需要任何投影，analyze 的兩張圖都是這樣畫的。

存進 latents.npz：
  z         (N,K) 的 theta，softmax 後的成分比例（每列總和 1）
  mu        (N,K) logit 空間的後驗均值，theta = softmax(mu)
  logvar    (N,K) 檢查 posterior collapse
  cluster   (N,) 硬分群結果，argmax theta（主 archetype）
  conf      (N,) max theta，主 archetype 的佔比，低 = 混得很均勻
  log_p_z   (N,) Dir(alpha) 下的 log 密度（方向見 ae.log_p_theta 的說明）
  err       (N,) Poisson deviance
"""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(f"{os.path.dirname(__file__)}/../.."))
from ae import (DirVAE, Patches, build_fsce_graph, dirichlet_kl,  # noqa: E402
                fsce_loss, log_p_theta, poisson_deviance, poisson_nll)
from config.dataset import ensure_patches, PATCHES, result  # noqa: E402

VERSION = "v2_dvae_fsce"
OUT = result(VERSION, "latents.npz")
CKPT = result(VERSION, "ae.pt")

K = 3            # archetype 數；simplex 自由度 K-1=2，對齊 v2_vae 的 latent_dim
ALPHA = 1.0 / K  # 對稱 Dirichlet 的濃度；<1 稀疏、=1 在 simplex 上均勻
EPOCHS = 1000
BATCH = 256
LR = 1e-3
VAL_FRAC = 0.1
SEED = 0
BETA = 0.01      # KL 的權重，沿用 v2_vae 的 sweep 結果

N_NEIGHBORS = 15         # 建高維 fuzzy graph 的 kNN 數，跟 v2_vae_fsce 一致
GRAPH_METRIC = "cosine"  # 對整包 count 向量的組成比例敏感、對總量不敏感
EDGE_BATCH = 256         # 每個 step 抽的正樣本邊數，負樣本抽等量
LAMBDA_FSCE = 0.5        # warm-up 結束後 FSCE loss 的權重，沿用 v2_vae_fsce
FSCE_WARMUP_EPOCHS = 80  # 前這麼多個 epoch 權重是 0，之後花同樣長度線性升到 LAMBDA_FSCE

device = ("mps" if torch.backends.mps.is_available()
          else "cuda" if torch.cuda.is_available() else "cpu")


def fsce_weight(epoch):
    """epoch 是目前的 epoch 編號（0-indexed），回傳這個 epoch 的 FSCE loss 權重：
    前 FSCE_WARMUP_EPOCHS 個 epoch 是 0，之後花同樣長度線性升到 LAMBDA_FSCE。
    """
    t = (epoch - FSCE_WARMUP_EPOCHS) / FSCE_WARMUP_EPOCHS
    return LAMBDA_FSCE * min(1.0, max(0.0, t))


def evaluate(model, data, idx):
    """在 idx 這批 patch 上算 (NLL, KL, deviance) 三個平均值。"""
    model.eval()
    vr, vk, vd = [], [], []
    with torch.no_grad():
        for i in range(0, len(idx), BATCH):
            x = data.agg(idx[i:i + BATCH]).to(device)
            mu, logvar, theta, log_lam = model(x)
            vr.append(poisson_nll(log_lam, x))
            vk.append(dirichlet_kl(model, mu, logvar))
            vd.append(poisson_deviance(log_lam, x))
    return (torch.cat(vr).mean().item(), torch.cat(vk).mean().item(),
            torch.cat(vd).mean().item())


def run(data, train_idx, val_idx, edge_i, edge_j, edge_w, a, b):
    """訓練並回傳 (mu, logvar, theta, log_p_z, err, model)，前五個是全體
    patch 的 ndarray，形狀依序 (N,K)、(N,K)、(N,K)、(N,)、(N,)。

    edge_i / edge_j / edge_w / a / b：build_fsce_graph() 的回傳值，整個
    訓練共用同一張 fuzzy graph。
    """
    torch.manual_seed(SEED)
    model = DirVAE(K, ALPHA).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    n_edges = len(edge_i)

    for epoch in range(EPOCHS):
        model.train()
        lam_t = fsce_weight(epoch)
        g = torch.Generator().manual_seed(SEED + epoch)
        perm = train_idx[torch.randperm(len(train_idx), generator=g)]
        total_recon, total_kl, total_fsce = 0.0, 0.0, 0.0
        for i in range(0, len(perm), BATCH):
            batch = perm[i:i + BATCH]
            x = data.agg(batch).to(device)
            mu, logvar, theta, log_lam = model(x)
            recon = poisson_nll(log_lam, x).mean()
            kl = dirichlet_kl(model, mu, logvar).mean()

            eg = torch.randint(0, n_edges, (EDGE_BATCH,), generator=g)
            pi, pj, pw = edge_i[eg], edge_j[eg], edge_w[eg]
            ni = torch.randint(0, data.n, (EDGE_BATCH,), generator=g)
            nj = torch.randint(0, data.n, (EDGE_BATCH,), generator=g)
            xi = data.agg(torch.cat([pi, ni])).to(device)
            xj = data.agg(torch.cat([pj, nj])).to(device)
            w = torch.cat([pw, torch.zeros(EDGE_BATCH)]).to(device)
            ti, tj = model.encode(xi), model.encode(xj)
            fsce = fsce_loss(ti, tj, w, a, b).mean()

            loss = recon + BETA * kl + lam_t * fsce
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_recon += recon.item() * len(batch)
            total_kl += kl.item() * len(batch)
            total_fsce += fsce.item() * len(batch)

        if (epoch + 1) % 20 == 0 or epoch == 0:
            val_recon, val_kl, dev = evaluate(model, data, val_idx)
            print(f"  epoch {epoch + 1:4d}  "
                  f"train NLL {total_recon / len(perm):.5f}  "
                  f"train KL {total_kl / len(perm):.5f}  "
                  f"train FSCE {total_fsce / len(perm):.5f}  "
                  f"lambda {lam_t:.3f}  "
                  f"val NLL {val_recon:.5f}  val KL {val_kl:.5f}  "
                  f"val deviance {dev:.5f}")

    torch.save(model.state_dict(), CKPT)

    # 推論用 eval 模式(reparameterize 回傳 mu)，theta = softmax(mu) 唯一
    model.eval()
    mus, logvars, thetas, lps, errs = [], [], [], [], []
    with torch.no_grad():
        for i in range(0, data.n, BATCH):
            idx = torch.arange(i, min(i + BATCH, data.n))
            x = data.agg(idx).to(device)
            mu, logvar, theta, log_lam = model(x)
            mus.append(mu.cpu())
            logvars.append(logvar.cpu())
            thetas.append(theta.cpu())
            lps.append(log_p_theta(model, theta).cpu())
            errs.append(poisson_deviance(log_lam, x).cpu())
    return (torch.cat(mus).numpy(), torch.cat(logvars).numpy(),
            torch.cat(thetas).numpy(), torch.cat(lps).numpy(),
            torch.cat(errs).numpy(), model)


def report_posterior(logvar):
    """印出 logit 空間 posterior std 的統計量，跟先驗的 std 對照看有沒有
    collapse。std 普遍逼近 sqrt(var_p) 就是塌回先驗的訊號。
    """
    std = np.exp(0.5 * logvar)
    prior_std = np.sqrt((1.0 / ALPHA) * (1.0 - 1.0 / K))
    print(f"\nposterior std（logit 空間）：平均 {std.mean():.3f}  "
          f"中位數 {np.median(std):.3f}  先驗 std {prior_std:.3f}")


def report_archetypes(theta, n_poi):
    """印出每個 archetype 的平均 theta、硬分群佔比與 POI 數中位數。

    theta：(N,K) 的成分比例。n_poi：(N,) 每個 patch 的 POI 總數，用來看
    archetype 有沒有只是在分密度。沒有回傳值，只印診斷。
    """
    hard = theta.argmax(1)
    print(f"\narchetype 使用情況（K={K}，alpha={ALPHA:.3f}；"
          f"平均 theta 接近 0 的分量等於沒被用到）")
    for c in range(K):
        m = hard == c
        share = theta[:, c].mean()
        if not m.any():
            print(f"  a{c}  主 archetype n=0        平均 theta={share:.3f}  (空)")
            continue
        print(f"  a{c}  主 archetype n={m.sum():<5d} {m.mean() * 100:5.1f}%  "
              f"平均 theta={share:.3f}  "
              f"組內平均 theta={theta[m, c].mean():.3f}  "
              f"POI 數中位數={np.median(n_poi[m]):.0f}")
    conf = theta.max(1)
    print(f"  主 archetype 佔比：中位數 {np.median(conf):.3f}，"
          f"< 0.5 的有 {(conf < 0.5).sum()} 個（混得比較均勻的 patch）")
    print(f"  有效稀疏度：平均每個 patch 用到 "
          f"{(theta > 0.05).sum(1).mean():.2f} 個 archetype（theta > 0.05）")


def main():
    ensure_patches()
    data = Patches(PATCHES)
    print(f"{data.n} 個 patch，device={device}，K={K}，"
          f"ALPHA={ALPHA:.3f}，BETA={BETA}，LAMBDA_FSCE={LAMBDA_FSCE}")

    x_all = np.log1p(data.agg(torch.arange(data.n)).numpy())
    edge_i, edge_j, edge_w, a, b = build_fsce_graph(
        x_all, n_neighbors=N_NEIGHBORS, metric=GRAPH_METRIC)
    q_far = 1.0 / (1.0 + a * 2.0 ** b)   # simplex 上最遠的一對點能到的 q
    print(f"FSCE graph：{len(edge_i)} 條邊，a={a:.4f} b={b:.4f}；"
          f"simplex 直徑處 q={q_far:.3f}（壓不到 0，見 ae.py 的風險說明）")

    g = torch.Generator().manual_seed(SEED)
    perm = torch.randperm(data.n, generator=g)
    n_val = int(data.n * VAL_FRAC)
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    mu, logvar, theta, log_p_z, err, model = run(
        data, train_idx, val_idx, edge_i, edge_j, edge_w, a, b)
    np.savez(OUT, n_poi=data.n_poi, lat=data.lat, lon=data.lon,
             z=theta, mu=mu, logvar=logvar, err=err,
             cluster=theta.argmax(1), conf=theta.max(1), log_p_z=log_p_z,
             alpha=np.float32(ALPHA), k=np.int32(K))
    report_posterior(logvar)
    report_archetypes(theta, data.n_poi)
    print(f"\n已存 {OUT}")


main()
