"""訓練 v2_vae_tanh_fsce：跟 v2_vae_tanh 的 train.py 相比，多了兩件事——訓練前
先在高維 count 向量上建一次 FSCE 用的 fuzzy graph，訓練迴圈裡每個 step 除了
原本的 reconstruction + KL batch，額外抽一批正樣本邊（來自 fuzzy graph）跟
等量的隨機負樣本邊，對 mu 算 FSCE loss，加權後跟 (Poisson NLL + BETA*KL)
加在一起、共用同一次 backward/opt.step()。

FSCE 權重用線性 warm-up：前 FSCE_WARMUP_EPOCHS 個 epoch 是 0，之後花同樣長度
線性升到 LAMBDA_FSCE，理由跟 v2_ae_tanh_fsce 一樣——先讓 encoder 學出有意義
的 count -> mu 映射，再逐步加拓撲約束，避免兩個 loss 一開始就互相打架。
"""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(f"{os.path.dirname(__file__)}/../.."))
from ae import (VAE, Patches, build_fsce_graph, fsce_loss,  # noqa: E402
                 kl_divergence, poisson_deviance, poisson_nll)
from config.dataset import ensure_patches, PATCHES, result  # noqa: E402

VERSION = "v2_vae_tanh_fsce"
OUT = result(VERSION, "latents.npz")
CKPT = result(VERSION, "ae.pt")

LATENT_DIM = 2
EPOCHS = 500
BATCH = 256
LR = 1e-3
VAL_FRAC = 0.1
SEED = 0
BETA = 0.01

N_NEIGHBORS = 15         # 建高維 fuzzy graph 的 kNN 數，跟 data/patch/umap_grid.py 一致
GRAPH_METRIC = "cosine"  # 對整包 count 向量的組成比例敏感、對總量不敏感
EDGE_BATCH = 256          # 每個 step 抽的正樣本邊數，負樣本抽等量
LAMBDA_FSCE = 0.5         # warm-up 結束後 FSCE loss 的權重
FSCE_WARMUP_EPOCHS = 80   # 前這麼多個 epoch 權重是 0，之後花同樣長度線性升到 LAMBDA_FSCE

device = ("mps" if torch.backends.mps.is_available()
          else "cuda" if torch.cuda.is_available() else "cpu")


def fsce_weight(epoch):
    """epoch 是目前的 epoch 編號（0-indexed），回傳這個 epoch 的 FSCE loss 權重：
    前 FSCE_WARMUP_EPOCHS 個 epoch 是 0，之後花同樣長度線性升到 LAMBDA_FSCE。
    """
    t = (epoch - FSCE_WARMUP_EPOCHS) / FSCE_WARMUP_EPOCHS
    return LAMBDA_FSCE * min(1.0, max(0.0, t))


def run(data, train_idx, val_idx, edge_i, edge_j, edge_w, a, b):
    torch.manual_seed(SEED)
    model = VAE(LATENT_DIM).to(device)
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
            mu, logvar, _, log_lam = model(x)
            recon = poisson_nll(log_lam, x).mean()
            kl = kl_divergence(mu, logvar).mean()

            eg = torch.randint(0, n_edges, (EDGE_BATCH,), generator=g)
            pi, pj, pw = edge_i[eg], edge_j[eg], edge_w[eg]
            ni = torch.randint(0, data.n, (EDGE_BATCH,), generator=g)
            nj = torch.randint(0, data.n, (EDGE_BATCH,), generator=g)
            xi = data.agg(torch.cat([pi, ni])).to(device)
            xj = data.agg(torch.cat([pj, nj])).to(device)
            w = torch.cat([pw, torch.zeros(EDGE_BATCH)]).to(device)
            zi, zj = model.encode(xi), model.encode(xj)
            fsce = fsce_loss(zi, zj, w, a, b).mean()

            loss = recon + BETA * kl + lam_t * fsce
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_recon += recon.item() * len(batch)
            total_kl += kl.item() * len(batch)
            total_fsce += fsce.item() * len(batch)

        if (epoch + 1) % 20 == 0 or epoch == 0:
            model.eval()
            with torch.no_grad():
                vr, vk, vd = [], [], []
                for i in range(0, len(val_idx), BATCH):
                    batch = val_idx[i:i + BATCH]
                    x = data.agg(batch).to(device)
                    mu, logvar, _, log_lam = model(x)
                    vr.append(poisson_nll(log_lam, x))
                    vk.append(kl_divergence(mu, logvar))
                    vd.append(poisson_deviance(log_lam, x))
                val_recon = torch.cat(vr).mean().item()
                val_kl = torch.cat(vk).mean().item()
                dev = torch.cat(vd).mean().item()
            print(f"  epoch {epoch + 1:3d}  "
                  f"train NLL {total_recon / len(perm):.5f}  train KL {total_kl / len(perm):.5f}  "
                  f"train FSCE {total_fsce / len(perm):.5f}  lambda {lam_t:.3f}  "
                  f"val NLL {val_recon:.5f}  val KL {val_kl:.5f}  val deviance {dev:.5f}")

    torch.save(model.state_dict(), CKPT)

    model.eval()
    mus, logvars, errs = [], [], []
    with torch.no_grad():
        for i in range(0, data.n, BATCH):
            idx = torch.arange(i, min(i + BATCH, data.n))
            x = data.agg(idx).to(device)
            mu, logvar, z, log_lam = model(x)
            mus.append(mu.cpu())
            logvars.append(logvar.cpu())
            errs.append(poisson_deviance(log_lam, x).cpu())
    return torch.cat(mus).numpy(), torch.cat(logvars).numpy(), torch.cat(errs).numpy()


def report_posterior(logvar):
    std = np.exp(0.5 * logvar)
    print(f"\nposterior std：平均 {std.mean():.3f}  中位數 {np.median(std):.3f}")
    for d in range(std.shape[1]):
        print(f"  z{d + 1} std 平均 {std[:, d].mean():.3f}")


def main():
    ensure_patches()
    data = Patches(PATCHES)
    print(f"{data.n} 個 patch，device={device}，BETA={BETA}")

    x_all = np.log1p(data.agg(torch.arange(data.n)).numpy())
    edge_i, edge_j, edge_w, a, b = build_fsce_graph(
        x_all, n_neighbors=N_NEIGHBORS, metric=GRAPH_METRIC)
    print(f"FSCE graph：{len(edge_i)} 條邊，a={a:.4f} b={b:.4f}")

    g = torch.Generator().manual_seed(SEED)
    perm = torch.randperm(data.n, generator=g)
    n_val = int(data.n * VAL_FRAC)
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    mu, logvar, err = run(data, train_idx, val_idx, edge_i, edge_j, edge_w, a, b)
    np.savez(OUT, n_poi=data.n_poi, lat=data.lat, lon=data.lon,
             z=mu, logvar=logvar, err=err)
    report_posterior(logvar)
    print(f"\n已存 {OUT}")


main()
