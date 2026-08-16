"""訓練 v2_ae_tanh_fsce：跟 v2_ae_tanh 的 train.py 相比，多了兩件事——訓練前
先在高維 count 向量上建一次 FSCE 用的 fuzzy graph，訓練迴圈裡每個 step 除了
原本的 reconstruction batch，額外抽一批正樣本邊（來自 fuzzy graph）跟等量的
隨機負樣本邊，把 FSCE loss 加權後跟 Poisson NLL 加在一起、共用同一次
backward/opt.step()。

FSCE 權重固定用一個很小的常數 LAMBDA_FSCE，全程不做 warm-up。
"""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(f"{os.path.dirname(__file__)}/../.."))
from ae import (MLPAE, Patches, build_fsce_graph, fsce_loss,  # noqa: E402
                 poisson_deviance, poisson_nll)
from config.dataset import ensure_patches, PATCHES, result  # noqa: E402

VERSION = "v2_ae_tanh_fsce"
OUT = result(VERSION, "latents.npz")
CKPT = result(VERSION, "ae.pt")

LATENT_DIM = 2
EPOCHS = 300
BATCH = 256
LR = 1e-3
WEIGHT_DECAY = 1e-4
VAL_FRAC = 0.1
SEED = 0

N_NEIGHBORS = 15         # 建高維 fuzzy graph 的 kNN 數，跟 data/patch/umap_grid.py 一致
GRAPH_METRIC = "cosine"  # 對整包 count 向量的組成比例敏感、對總量不敏感
EDGE_BATCH = 256          # 每個 step 抽的正樣本邊數，負樣本抽等量
LAMBDA_FSCE = 0.05        # FSCE loss 的權重，固定值、全程都很小，不做 warm-up

device = ("mps" if torch.backends.mps.is_available()
          else "cuda" if torch.cuda.is_available() else "cpu")


def run(data, train_idx, val_idx, edge_i, edge_j, edge_w, a, b):
    torch.manual_seed(SEED)
    model = MLPAE(LATENT_DIM).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    n_edges = len(edge_i)

    for epoch in range(EPOCHS):
        model.train()
        lam_t = LAMBDA_FSCE
        g = torch.Generator().manual_seed(SEED + epoch)
        perm = train_idx[torch.randperm(len(train_idx), generator=g)]
        total, total_fsce = 0.0, 0.0
        for i in range(0, len(perm), BATCH):
            batch = perm[i:i + BATCH]
            x = data.agg(batch).to(device)
            _, log_lam = model(x)
            recon = poisson_nll(log_lam, x).mean()

            eg = torch.randint(0, n_edges, (EDGE_BATCH,), generator=g)
            pi, pj, pw = edge_i[eg], edge_j[eg], edge_w[eg]
            ni = torch.randint(0, data.n, (EDGE_BATCH,), generator=g)
            nj = torch.randint(0, data.n, (EDGE_BATCH,), generator=g)
            xi = data.agg(torch.cat([pi, ni])).to(device)
            xj = data.agg(torch.cat([pj, nj])).to(device)
            w = torch.cat([pw, torch.zeros(EDGE_BATCH)]).to(device)
            zi, zj = model.encode(xi), model.encode(xj)
            fsce = fsce_loss(zi, zj, w, a, b).mean()

            loss = recon + lam_t * fsce
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += recon.item() * len(batch)
            total_fsce += fsce.item() * len(batch)

        if (epoch + 1) % 20 == 0 or epoch == 0:
            model.eval()
            with torch.no_grad():
                vl, vd = [], []
                for i in range(0, len(val_idx), BATCH):
                    batch = val_idx[i:i + BATCH]
                    x = data.agg(batch).to(device)
                    _, log_lam = model(x)
                    vl.append(poisson_nll(log_lam, x))
                    vd.append(poisson_deviance(log_lam, x))
                val = torch.cat(vl).mean().item()
                dev = torch.cat(vd).mean().item()
            print(f"  epoch {epoch + 1:3d}  train NLL {total / len(perm):.5f}  "
                  f"train FSCE {total_fsce / len(perm):.5f}  lambda {lam_t:.3f}  "
                  f"val NLL {val:.5f}  val deviance {dev:.5f}")

    torch.save(model.state_dict(), CKPT)

    model.eval()
    zs, errs = [], []
    with torch.no_grad():
        for i in range(0, data.n, BATCH):
            idx = torch.arange(i, min(i + BATCH, data.n))
            x = data.agg(idx).to(device)
            z, log_lam = model(x)
            zs.append(z.cpu())
            errs.append(poisson_deviance(log_lam, x).cpu())
    return torch.cat(zs).numpy(), torch.cat(errs).numpy()


def main():
    ensure_patches()
    data = Patches(PATCHES)
    print(f"{data.n} 個 patch，device={device}")

    x_all = np.log1p(data.agg(torch.arange(data.n)).numpy())
    edge_i, edge_j, edge_w, a, b = build_fsce_graph(
        x_all, n_neighbors=N_NEIGHBORS, metric=GRAPH_METRIC)
    print(f"FSCE graph：{len(edge_i)} 條邊，a={a:.4f} b={b:.4f}")

    g = torch.Generator().manual_seed(SEED)
    perm = torch.randperm(data.n, generator=g)
    n_val = int(data.n * VAL_FRAC)
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    z, err = run(data, train_idx, val_idx, edge_i, edge_j, edge_w, a, b)
    np.savez(OUT, n_poi=data.n_poi, lat=data.lat, lon=data.lon, z=z, err=err)
    print(f"已存 {OUT}")


main()
