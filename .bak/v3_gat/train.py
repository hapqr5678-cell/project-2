"""訓練 v3_gat：在 v2_ddae_fsce 基礎上加一條 OD-GAT 分支的雙分支模型。

跟 v2_ddae_fsce 保持一致的部分：SEED、VAL_FRAC 與切分方式、BATCH、LR、
EPOCHS、WEIGHT_DECAY、破壞方式、Poisson NLL、FSCE loss、explained
deviance、latents.npz 的欄位。差別只有 encoder 多了一條 GAT 分支跟
CrossAttentionFusion 融合機制（見 ae.py）。

OD 矩陣是獨立於 count 的觀測，不經過 corrupt()：recon batch 跟 FSCE pair
兩處都是「count 過 corrupt()、OD 原樣用」。FSCE 的 fuzzy graph 只用乾淨
count 的 log1p 建（跟 v2_ddae_fsce 一樣，鄰接關係是資料組成的性質），OD
不參與建圖。

result.log 每 5 個 epoch 多印一個 alpha 欄位：CrossAttentionFusion 學到的
GAT 分支混合比例，純觀測，恆小於 ALPHA_CAP。
"""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(f"{os.path.dirname(__file__)}/../.."))
from ae import (GATAE, ODMatrices, Patches, build_fsce_graph, corrupt,  # noqa: E402
                fsce_loss, poisson_deviance, poisson_nll)
from config.dataset import OD, PATCHES, ensure_od, result  # noqa: E402
from config.train_log import open_log  # noqa: E402

VERSION = "v3_gat"
OUT = result(VERSION, "latents.npz")
CKPT = result(VERSION, "ae.pt")

LATENT_DIM = 2
EPOCHS = 1000
BATCH = 256
LR = 1e-3
WEIGHT_DECAY = 1e-6
VAL_FRAC = 0.1
SEED = 0

N_NEIGHBORS = 15         # 建高維 fuzzy graph 的 kNN 數，跟 data/patch/umap_grid.py 一致
GRAPH_METRIC = "euclidean"
EDGE_BATCH = 256          # 每個 step 抽的正樣本邊數，負樣本抽等量
LAMBDA_FSCE = 0.01        # FSCE loss 的權重，warm-up 結束後的最終值
WARMUP_EPOCHS = 200        # lambda 從 0 線性升到 LAMBDA_FSCE 所花的 epoch 數
FREEZE_GAT_EPOCH = 300     # 從這個 epoch 起凍結 GAT 分支參數，之後不再更新

NOISE_P = 0.3            # 破壞強度：thinning 是每個 POI 被丟掉的機率
NOISE_MODE = "thinning"  # "thinning"（逐 POI 丟）或 "mask"（整類歸零）

GAT_HIDDEN = 32
CAT_EMB_DIM = 8
N_GAT_LAYERS = 2
GAT_HEADS = 4
READOUT_DIM = 8
FUSION_HEADS = 4
ALPHA_CAP = 0.1

device = ("cpu" if torch.backends.mps.is_available()
          else "cuda" if torch.cuda.is_available() else "cpu")


def run(data, od, train_idx, val_idx, edge_i, edge_j, edge_w, a, b, log):
    """訓練並回傳全體 patch 的 (z, err)：z 是 (N,LATENT_DIM) 的 latent 座標，
    err 是 (N,) 的 Poisson deviance，兩者都用乾淨輸入、eval 模式算出來。
    data 是 Patches；od 是已搬上 device 的 ODMatrices；train_idx / val_idx
    是 patch 編號的 LongTensor；edge_i / edge_j / edge_w / a / b 是
    build_fsce_graph() 的回傳值。log 是 config.train_log.open_log() 回傳的
    函式，訓練過程的訊息都灌進去。訓練中按 Ctrl-C 會提前跳出迴圈，用當下的
    模型狀態存 checkpoint 跟 latent，不會整個丟掉重來。
    """
    torch.manual_seed(SEED)
    model = GATAE(LATENT_DIM).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    n_edges = len(edge_i)

    # 空模型（λ=訓練集全域平均）的 log_lam，當 explained deviance 的分母，
    # 用訓練集估、驗證集算 deviance，不然分母本身就偷看了驗證集
    log_lam_null = data.agg(train_idx).mean(dim=0, keepdim=True) \
        .clamp_min(1e-8).log().to(device)

    for epoch in range(EPOCHS):
        try:
            if epoch == FREEZE_GAT_EPOCH:
                for p in model.gat.parameters():
                    p.requires_grad_(False)
            model.train()
            lam_t = LAMBDA_FSCE * min(1.0, (epoch + 1) / WARMUP_EPOCHS)
            g = torch.Generator().manual_seed(SEED + epoch)
            perm = train_idx[torch.randperm(len(train_idx), generator=g)]
            total, total_fsce, alpha_sum, n_seen = 0.0, 0.0, 0.0, 0
            for i in range(0, len(perm), BATCH):
                batch = perm[i:i + BATCH]
                x = data.agg(batch)                                  # 乾淨目標（CPU）
                x_in = corrupt(x, NOISE_P, NOISE_MODE, generator=g)  # 加噪輸入
                x, x_in = x.to(device), x_in.to(device)
                od_batch = od.get(batch)                             # OD 不加噪
                _, log_lam, alpha = model(x_in, od_batch)
                recon = poisson_nll(log_lam, x).mean()   # 目標永遠是乾淨的 x

                eg = torch.randint(0, n_edges, (EDGE_BATCH,), generator=g)
                pi, pj, pw = edge_i[eg], edge_j[eg], edge_w[eg]
                ni = torch.randint(0, data.n, (EDGE_BATCH,), generator=g)
                nj = torch.randint(0, data.n, (EDGE_BATCH,), generator=g)
                idx_i, idx_j = torch.cat([pi, ni]), torch.cat([pj, nj])
                # pair 也加噪，encoder 訓練時看到的輸入分布才跟 recon 那一路一致
                xi = corrupt(data.agg(idx_i), NOISE_P, NOISE_MODE,
                             generator=g).to(device)
                xj = corrupt(data.agg(idx_j), NOISE_P, NOISE_MODE,
                             generator=g).to(device)
                odi, odj = od.get(idx_i), od.get(idx_j)
                w = torch.cat([pw, torch.zeros(EDGE_BATCH)]).to(device)
                zi, _ = model.encode(xi, odi)
                zj, _ = model.encode(xj, odj)
                fsce = fsce_loss(zi, zj, w, a, b).mean()

                loss = recon + lam_t * fsce
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += recon.item() * len(batch)
                total_fsce += fsce.item() * len(batch)
                alpha_sum += alpha.item() * len(batch)
                n_seen += len(batch)

            if (epoch + 1) % 5 == 0 or epoch == 0:
                model.eval()
                with torch.no_grad():
                    vd, vdn = [], []
                    for i in range(0, len(val_idx), BATCH):
                        batch = val_idx[i:i + BATCH]
                        x = data.agg(batch).to(device)   # 驗證不加噪
                        od_batch = od.get(batch)
                        _, log_lam, _ = model(x, od_batch)
                        vd.append(poisson_deviance(log_lam, x))
                        vdn.append(poisson_deviance(log_lam_null.expand(len(batch), -1), x))
                    dev = torch.cat(vd).mean().item()
                    dev_null = torch.cat(vdn).mean().item()
                    expl = 1 - dev / dev_null
                log(f"epoch {epoch + 1:4d} | train_nll {total / len(perm):.5f} | "
                    f"train_fsce {total_fsce / len(perm):.5f} | val_dev {dev:.5f} | "
                    f"expl_dev {expl:.5f} | alpha {alpha_sum / n_seen:.4f}")
        except KeyboardInterrupt:
            log(f"\n[中斷] epoch {epoch + 1} 收到 Ctrl-C，用目前模型存 checkpoint 跟 latent")
            break

    torch.save(model.state_dict(), CKPT)

    model.eval()
    zs, errs = [], []
    with torch.no_grad():
        for i in range(0, data.n, BATCH):
            idx = torch.arange(i, min(i + BATCH, data.n))
            x = data.agg(idx).to(device)   # 推論不加噪
            od_batch = od.get(idx)
            z, log_lam, _ = model(x, od_batch)
            zs.append(z.cpu())
            errs.append(poisson_deviance(log_lam, x).cpu())
    return torch.cat(zs).numpy(), torch.cat(errs).numpy()


def main():
    log = open_log(VERSION, {
        "LATENT_DIM": LATENT_DIM, "EPOCHS": EPOCHS, "BATCH": BATCH, "LR": LR,
        "WEIGHT_DECAY": WEIGHT_DECAY, "VAL_FRAC": VAL_FRAC, "SEED": SEED,
        "N_NEIGHBORS": N_NEIGHBORS, "GRAPH_METRIC": GRAPH_METRIC,
        "EDGE_BATCH": EDGE_BATCH, "LAMBDA_FSCE": LAMBDA_FSCE,
        "WARMUP_EPOCHS": WARMUP_EPOCHS,
        "NOISE_P": NOISE_P, "NOISE_MODE": NOISE_MODE,
        "GAT_HIDDEN": GAT_HIDDEN, "CAT_EMB_DIM": CAT_EMB_DIM,
        "N_GAT_LAYERS": N_GAT_LAYERS, "GAT_HEADS": GAT_HEADS,
        "READOUT_DIM": READOUT_DIM, "FUSION_HEADS": FUSION_HEADS,
        "ALPHA_CAP": ALPHA_CAP,
    })

    ensure_od()   # 內含 ensure_patches()
    data = Patches(PATCHES)
    od = ODMatrices(OD).to(device)
    assert od.n == data.n, f"od.npz 有 {od.n} 個 patch，patches.npz 有 {data.n} 個"
    log(f"{data.n} 個 patch，device={device}，噪聲 {NOISE_MODE} p={NOISE_P}")

    # fuzzy graph 用乾淨 count 建：鄰接關係是資料的性質，不是噪聲的性質；
    # OD 不參與建圖
    x_all = np.log1p(data.agg(torch.arange(data.n)).numpy())
    edge_i, edge_j, edge_w, a, b = build_fsce_graph(
        x_all, n_neighbors=N_NEIGHBORS, metric=GRAPH_METRIC)
    log(f"FSCE graph：{len(edge_i)} 條邊，a={a:.4f} b={b:.4f}")

    g = torch.Generator().manual_seed(SEED)
    perm = torch.randperm(data.n, generator=g)
    n_val = int(data.n * VAL_FRAC)
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    z, err = run(data, od, train_idx, val_idx, edge_i, edge_j, edge_w, a, b, log)
    np.savez(OUT, n_poi=data.n_poi, lat=data.lat, lon=data.lon, z=z, err=err)
    log(f"已存 {OUT}")


main()
