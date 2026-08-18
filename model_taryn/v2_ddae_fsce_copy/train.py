"""訓練 v2_ddae_fsce_copy：v2_ddae_fsce 的無 validation leakage 副本。

模型仍是 v2_dae_fsce 的加深變體（encoder/decoder
各 4 層隱藏層），其餘（超參數、FSCE 那組、NOISE_P/NOISE_MODE）全部跟
v2_dae_fsce 對齊，這樣兩邊的 latent 才是在同一組資料/loss 下比較，
差別只有模型容量這件事。

所有餵進 encoder 的輸入（reconstruction batch 跟 FSCE 的 pair 兩邊都算）
都先過 corrupt()，Poisson NLL 的目標仍然是乾淨的原始 count；fuzzy graph
用乾淨 count 建（鄰接關係是資料的性質，不是噪聲的性質）。

驗證與最後輸出 latent 的推論階段一律不加噪。破壞是每個 step 重新抽的
（generator 綁 SEED+epoch，可重現）。
"""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(f"{os.path.dirname(__file__)}/../.."))
from taryn_model.v2_ddae_fsce_copy.ae import (MLPAE, Patches, build_fsce_graph, corrupt,  # noqa: E402
                fsce_loss, poisson_deviance, poisson_nll)
from config.dataset import ensure_patches, PATCHES, result  # noqa: E402

VERSION = "v2_ddae_fsce_copy"
OUT = result(VERSION, "latents.npz")
CKPT = result(VERSION, "ae.pt")

LATENT_DIM = 2
EPOCHS = 1000
BATCH = 256
LR = 1e-3
WEIGHT_DECAY = 1e-4
VAL_FRAC = 0.1
SEED = 0

N_NEIGHBORS = 15         # 建高維 fuzzy graph 的 kNN 數，跟 data/patch/umap_grid.py 一致
GRAPH_METRIC = "cosine"  # 對整包 count 向量的組成比例敏感、對總量不敏感
EDGE_BATCH = 256          # 每個 step 抽的正樣本邊數，負樣本抽等量
LAMBDA_FSCE = 0.5        # FSCE loss 的權重，warm-up 結束後的最終值
WARMUP_EPOCHS = 500        # lambda 從 0 線性升到 LAMBDA_FSCE 所花的 epoch 數

NOISE_P = 0.3            # 破壞強度：thinning 是每個 POI 被丟掉的機率
NOISE_MODE = "thinning"  # "thinning"（逐 POI 丟）或 "mask"（整類歸零）

device = ("mps" if torch.backends.mps.is_available()
          else "cuda" if torch.cuda.is_available() else "cpu")


def run(data, train_idx, val_idx, edge_i, edge_j, edge_w, a, b):
    """訓練並回傳全體 patch 的 (z, err)：z 是 (N,LATENT_DIM) 的 latent 座標，
    err 是 (N,) 的 Poisson deviance，兩者都用乾淨輸入、eval 模式算出來。
    data 是 Patches；train_idx / val_idx 是 patch 編號的 LongTensor；
    edge_i / edge_j / edge_w / a / b 是 build_fsce_graph() 的回傳值。
    """
    torch.manual_seed(SEED)
    model = MLPAE(LATENT_DIM).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    n_edges = len(edge_i)
    is_train = torch.zeros(data.n, dtype=torch.bool)
    is_train[train_idx] = True
    if not (is_train[edge_i].all() and is_train[edge_j].all()):
        raise RuntimeError("FSCE graph contains a validation node")

    for epoch in range(EPOCHS):
        model.train()
        lam_t = LAMBDA_FSCE * min(1.0, (epoch + 1) / WARMUP_EPOCHS)
        g = torch.Generator().manual_seed(SEED + epoch)
        perm = train_idx[torch.randperm(len(train_idx), generator=g)]
        total, total_fsce = 0.0, 0.0
        for i in range(0, len(perm), BATCH):
            batch = perm[i:i + BATCH]
            x = data.agg(batch)                                  # 乾淨目標（CPU）
            x_in = corrupt(x, NOISE_P, NOISE_MODE, generator=g)  # 加噪輸入
            x, x_in = x.to(device), x_in.to(device)
            _, log_lam = model(x_in)
            recon = poisson_nll(log_lam, x).mean()   # 目標永遠是乾淨的 x

            eg = torch.randint(0, n_edges, (EDGE_BATCH,), generator=g)
            pi, pj, pw = edge_i[eg], edge_j[eg], edge_w[eg]
            ni = train_idx[
                torch.randint(0, len(train_idx), (EDGE_BATCH,), generator=g)
            ]
            nj = train_idx[
                torch.randint(0, len(train_idx), (EDGE_BATCH,), generator=g)
            ]
            # pair 也加噪，encoder 訓練時看到的輸入分布才跟 recon 那一路一致
            xi = corrupt(data.agg(torch.cat([pi, ni])), NOISE_P,
                         NOISE_MODE, generator=g).to(device)
            xj = corrupt(data.agg(torch.cat([pj, nj])), NOISE_P,
                         NOISE_MODE, generator=g).to(device)
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
                    x = data.agg(batch).to(device)   # 驗證不加噪
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
            x = data.agg(idx).to(device)   # 推論不加噪
            z, log_lam = model(x)
            zs.append(z.cpu())
            errs.append(poisson_deviance(log_lam, x).cpu())
    return torch.cat(zs).numpy(), torch.cat(errs).numpy()


def main():
    ensure_patches()
    data = Patches(PATCHES)
    print(f"{data.n} 個 patch，device={device}，"
          f"噪聲 {NOISE_MODE} p={NOISE_P}")

    g = torch.Generator().manual_seed(SEED)
    perm = torch.randperm(data.n, generator=g)
    n_val = int(data.n * VAL_FRAC)
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    # 先切分，再只用 training clean count 建 graph。這些 edge 初始是
    # x_train 的區域索引，必須映射回全體 patch 索引才能交給 data.agg()。
    x_train = np.log1p(data.agg(train_idx).numpy())
    local_i, local_j, edge_w, a, b = build_fsce_graph(
        x_train, n_neighbors=N_NEIGHBORS, metric=GRAPH_METRIC)
    edge_i, edge_j = train_idx[local_i], train_idx[local_j]
    print(f"FSCE graph：{len(edge_i)} 條 training-only 邊，"
          f"a={a:.4f} b={b:.4f}")

    z, err = run(data, train_idx, val_idx, edge_i, edge_j, edge_w, a, b)
    np.savez(OUT, n_poi=data.n_poi, lat=data.lat, lon=data.lon, z=z, err=err)
    print(f"已存 {OUT}")


main()
