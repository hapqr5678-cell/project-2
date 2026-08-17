"""訓練 v2_ddae_fsce_ms：v2_ddae_fsce_copy 加上多尺度環狀 context 的變體。

唯一的變因是 decoder 多吃一組 build_rings() 算出來的周邊環特徵（100/200/
400/800m 四個同心方環，各 N_CAT 維 count 取 log1p，一律扣掉 patch 自己那
HALF_WIDTH 的窗）。其餘——模型容量、超參數、FSCE 那組、NOISE_P/NOISE_MODE、
train/val 切分——全部跟 _copy 對齊，兩邊的 latent 才是在同一組條件下比較。

環特徵當 decoder 的觀測協變量、不進 bottleneck：它的訊噪比比類別組成高一個
數量級，串進 encoder 輸入會主導距離、把 latent 變成一張密度圖。走 decoder
這條路，latent 就不必浪費維度去記街區密度，被迫只表達「在這個街區脈絡下，
這個 patch 有什麼特別」。理由詳見 ae.py 的模組 docstring。

環特徵的標準化統計量只從 training patch 算，再套用到全體。

FSCE graph 一樣只用 training 邊、負樣本也只從 train_idx 抽，而且建圖用的是
乾淨 count 的 log1p + cosine，不使用環特徵——encoder 看不到環，用環建圖等於
逼 encoder 去擬合它拿不到的資訊。

所有餵進 encoder 的輸入（reconstruction batch 跟 FSCE 的 pair 兩邊都算）都先
過 corrupt()，Poisson NLL 的目標仍然是乾淨的原始 count。驗證與最後輸出 latent
的推論階段一律不加噪。破壞是每個 step 重新抽的（generator 綁 SEED+epoch）。

已知限制：切分沿用隨機切分（為了跟 lab/eval_latents.py 表上的其他版本橫比），
但 800m 的環讓相距 CENTER_STEP 的 train/val patch 幾乎共用整個 context，
val NLL / val deviance 會偏樂觀，不能當成泛化能力的絕對估計。
"""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(f"{os.path.dirname(__file__)}/../.."))
from ae import (MLPAE, Patches, build_fsce_graph, build_rings,  # noqa: E402
                corrupt, fsce_loss, poisson_deviance, poisson_nll)
from config.dataset import ensure_patches, PATCHES, result  # noqa: E402

VERSION = "v2_ddae_fsce_ms"
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

# 環的外邊界（公尺）；內邊界是 HALF_WIDTH，所以 context 不含 patch 自身的 POI
RING_RADII = (100.0, 200.0, 400.0, 800.0)
RING_MODE = "total"      # "total"：每環一個總數（密度輪廓，4 維）
                         # "count"：每環完整 N_CAT 維類別 count（40 維）
                         # 用 "count" 會嚴重過擬合，理由見 ae.build_rings()

device = ("mps" if torch.backends.mps.is_available()
          else "cuda" if torch.cuda.is_available() else "cpu")


def standardize(s, train_idx):
    """對環特徵做 z-score，統計量只從 training patch 算。

    s：(N,D) 的環特徵（build_rings() 的回傳值）。
    train_idx：training patch 編號的 LongTensor。

    回傳 (N,D) 的 float32 tensor（CPU）。std 為 0 的欄墊 1，避免除以 0。
    """
    tr = s[train_idx.numpy()]
    mu, sd = tr.mean(axis=0), tr.std(axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    return torch.from_numpy(((s - mu) / sd).astype(np.float32))


def run(data, ctx, train_idx, val_idx, edge_i, edge_j, edge_w, a, b):
    """訓練並回傳全體 patch 的 (z, err)：z 是 (N,LATENT_DIM) 的 latent 座標，
    err 是 (N,) 的 Poisson deviance，兩者都用乾淨輸入、eval 模式算出來。
    data 是 Patches；ctx 是 (N,D) 標準化後的環特徵 FloatTensor（CPU）；
    train_idx / val_idx 是 patch 編號的 LongTensor；
    edge_i / edge_j / edge_w / a / b 是 build_fsce_graph() 的回傳值。
    """
    torch.manual_seed(SEED)
    model = MLPAE(LATENT_DIM, ctx_dim=ctx.shape[1]).to(device)
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
            s = ctx[batch].to(device)                # 環是觀測值，不加噪
            _, log_lam = model(x_in, s)
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
            # FSCE 只跑 encoder，跟 context 無關
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
                    _, log_lam = model(x, ctx[batch].to(device))
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
            z, log_lam = model(x, ctx[idx].to(device))
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

    rings = build_rings(data.cx, data.cy, RING_RADII, mode=RING_MODE)
    ctx = standardize(rings, train_idx)
    print(f"環狀 context：{len(RING_RADII)} 環 {RING_RADII}（內邊界 "
          f"HALF_WIDTH），mode={RING_MODE}，{ctx.shape[1]} 維")

    # 先切分，再只用 training clean count 建 graph。這些 edge 初始是
    # x_train 的區域索引，必須映射回全體 patch 索引才能交給 data.agg()。
    x_train = np.log1p(data.agg(train_idx).numpy())
    local_i, local_j, edge_w, a, b = build_fsce_graph(
        x_train, n_neighbors=N_NEIGHBORS, metric=GRAPH_METRIC)
    edge_i, edge_j = train_idx[local_i], train_idx[local_j]
    print(f"FSCE graph：{len(edge_i)} 條 training-only 邊，"
          f"a={a:.4f} b={b:.4f}")

    z, err = run(data, ctx, train_idx, val_idx, edge_i, edge_j, edge_w, a, b)
    np.savez(OUT, n_poi=data.n_poi, lat=data.lat, lon=data.lon, z=z, err=err)
    print(f"已存 {OUT}")


main()
