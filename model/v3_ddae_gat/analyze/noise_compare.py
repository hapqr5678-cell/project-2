"""比較 v2_ddae_fsce 在不同 NOISE_P（thinning 破壞強度）下的重建效果。

每個 NOISE_P 獨立訓練一個全新的模型（EPOCHS 縮短到 700，其餘超參數
跟 train.py 完全一致，fuzzy graph／train-val 切分共用同一份，只有
NOISE_P 在變），全程每隔 EVAL_EVERY 個 epoch 記一次驗證集 Poisson
deviance，畫成一張圖：x 軸 epoch、y 軸 val_dev，每個 NOISE_P 一條線
疊在一起比較訓練過程的變化，不只看終點。不寫入、也不影響 train.py
產出的正式 checkpoint（ae.pt）與 latents.npz——這是獨立的比較實驗，
輸出另外存成 noise_compare.log／noise_compare.png。
"""

import os
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(f"{os.path.dirname(__file__)}/.."))
sys.path.insert(0, os.path.abspath(f"{os.path.dirname(__file__)}/../../.."))
from ae import (MLPAE, Patches, build_fsce_graph, corrupt,  # noqa: E402
                fsce_loss, poisson_deviance, poisson_nll)
from config.dataset import ensure_patches, PATCHES, result  # noqa: E402

VERSION = "v2_ddae_fsce"

LATENT_DIM = 2
EPOCHS = 700              # 正式 train.py 是 1000，這裡縮短先看趨勢
EVAL_EVERY = 25           # 每隔這麼多 epoch 記一次驗證集 deviance，畫成曲線
BATCH = 256
LR = 1e-3
WEIGHT_DECAY = 1e-6
VAL_FRAC = 0.1
SEED = 0

N_NEIGHBORS = 15
GRAPH_METRIC = "euclidean"
EDGE_BATCH = 256
LAMBDA_FSCE = 0.01
WARMUP_EPOCHS = 200

NOISE_MODE = "thinning"
NOISE_PS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

mpl.rcParams["font.family"] = ["Heiti TC"]
mpl.rcParams["axes.unicode_minus"] = False
mpl.rcParams["figure.dpi"] = 130

device = ("mps" if torch.backends.mps.is_available()
          else "cuda" if torch.cuda.is_available() else "cpu")


def train_one(noise_p, data, train_idx, val_idx, edge_i, edge_j, edge_w, a, b,
              log_lam_null, log):
    """用給定的 NOISE_P 訓練一個全新模型 EPOCHS 輪。

    noise_p：這次訓練用的 thinning 強度，餵進 corrupt()。
    data：Patches；train_idx／val_idx：patch 編號的 LongTensor；
    edge_i／edge_j／edge_w／a／b：build_fsce_graph() 的回傳值，所有
    noise_p 共用同一張圖（用乾淨 count 建，跟噪聲無關）。
    log_lam_null：訓練集全域平均 log λ，(1,N_CAT)，當 explained
    deviance 的分母。
    log：可呼叫的印字函式，訓練過程灌進去。

    回傳 (history, dev, expl)：history 是 [(epoch, val_dev, expl_dev), ...]
    的 list，每 EVAL_EVERY 個 epoch 記一筆；dev／expl 是最後一個 epoch的
    數字，兩者都用乾淨輸入、eval 模式算出來。
    """
    torch.manual_seed(SEED)
    model = MLPAE(LATENT_DIM).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    n_edges = len(edge_i)

    history = []
    dev, expl = None, None
    for epoch in range(EPOCHS):
        model.train()
        lam_t = LAMBDA_FSCE * min(1.0, (epoch + 1) / WARMUP_EPOCHS)
        g = torch.Generator().manual_seed(SEED + epoch)
        perm = train_idx[torch.randperm(len(train_idx), generator=g)]
        for i in range(0, len(perm), BATCH):
            batch = perm[i:i + BATCH]
            x = data.agg(batch)
            x_in = corrupt(x, noise_p, NOISE_MODE, generator=g)
            x, x_in = x.to(device), x_in.to(device)
            _, log_lam = model(x_in)
            recon = poisson_nll(log_lam, x).mean()

            eg = torch.randint(0, n_edges, (EDGE_BATCH,), generator=g)
            pi, pj, pw = edge_i[eg], edge_j[eg], edge_w[eg]
            ni = torch.randint(0, data.n, (EDGE_BATCH,), generator=g)
            nj = torch.randint(0, data.n, (EDGE_BATCH,), generator=g)
            xi = corrupt(data.agg(torch.cat([pi, ni])), noise_p,
                         NOISE_MODE, generator=g).to(device)
            xj = corrupt(data.agg(torch.cat([pj, nj])), noise_p,
                         NOISE_MODE, generator=g).to(device)
            w = torch.cat([pw, torch.zeros(EDGE_BATCH)]).to(device)
            zi, zj = model.encode(xi), model.encode(xj)
            fsce = fsce_loss(zi, zj, w, a, b).mean()

            loss = recon + lam_t * fsce
            opt.zero_grad()
            loss.backward()
            opt.step()

        if (epoch + 1) % EVAL_EVERY == 0 or epoch == EPOCHS - 1:
            model.eval()
            with torch.no_grad():
                vd, vdn = [], []
                for i in range(0, len(val_idx), BATCH):
                    batch = val_idx[i:i + BATCH]
                    x = data.agg(batch).to(device)
                    _, log_lam = model(x)
                    vd.append(poisson_deviance(log_lam, x))
                    vdn.append(poisson_deviance(log_lam_null.expand(len(batch), -1), x))
                dev = torch.cat(vd).mean().item()
                dev_null = torch.cat(vdn).mean().item()
                expl = 1 - dev / dev_null
            history.append((epoch + 1, dev, expl))
            log(f"  NOISE_P={noise_p:.2f} | epoch {epoch + 1:4d} | "
                f"val_dev {dev:.5f} | expl_dev {expl:.5f}")

    return history, dev, expl


def main():
    log_path = result(VERSION, "noise_compare.log")
    f = open(log_path, "w")

    def log(msg):
        print(msg)
        f.write(msg + "\n")
        f.flush()

    log(f"[noise_compare] {VERSION}，EPOCHS={EPOCHS}，NOISE_MODE={NOISE_MODE}，"
        f"NOISE_PS={NOISE_PS}")

    ensure_patches()
    data = Patches(PATCHES)
    log(f"{data.n} 個 patch，device={device}")

    x_all = np.log1p(data.agg(torch.arange(data.n)).numpy())
    edge_i, edge_j, edge_w, a, b = build_fsce_graph(
        x_all, n_neighbors=N_NEIGHBORS, metric=GRAPH_METRIC)
    log(f"FSCE graph：{len(edge_i)} 條邊，a={a:.4f} b={b:.4f}")

    g = torch.Generator().manual_seed(SEED)
    perm = torch.randperm(data.n, generator=g)
    n_val = int(data.n * VAL_FRAC)
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    log_lam_null = data.agg(train_idx).mean(dim=0, keepdim=True) \
        .clamp_min(1e-8).log().to(device)

    rows = []
    histories = {}
    for noise_p in NOISE_PS:
        log(f"\n=== NOISE_P={noise_p:.2f} ===")
        history, dev, expl = train_one(noise_p, data, train_idx, val_idx,
                                        edge_i, edge_j, edge_w, a, b,
                                        log_lam_null, log)
        histories[noise_p] = history
        rows.append((noise_p, dev, expl))

    log("\n=== 總表（EPOCHS=%d，NOISE_MODE=%s）===" % (EPOCHS, NOISE_MODE))
    log(f"{'NOISE_P':>8} | {'val_dev':>10} | {'expl_dev':>10}")
    for noise_p, dev, expl in rows:
        log(f"{noise_p:8.2f} | {dev:10.5f} | {expl:10.5f}")
    log(f"\n已存 {log_path}")
    f.close()

    cmap = plt.get_cmap("viridis")
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for i, noise_p in enumerate(NOISE_PS):
        h = histories[noise_p]
        epochs = [r[0] for r in h]
        devs = [r[1] for r in h]
        ax.plot(epochs, devs, marker="o", markersize=3,
                color=cmap(i / max(1, len(NOISE_PS) - 1)),
                label=f"NOISE_P={noise_p:.2f}")
    ax.set_xlabel("epoch")
    ax.set_ylabel("val_dev（越小越好）")
    ax.set_title(f"{VERSION}：不同 NOISE_P（{NOISE_MODE}）的驗證集 "
                 f"Poisson deviance 訓練曲線（EPOCHS={EPOCHS}）", fontsize=10)
    ax.grid(alpha=0.2, linewidth=0.5)
    ax.legend(fontsize=9, framealpha=0.9)
    for s in ax.spines.values():
        s.set_alpha(0.3)

    fig.tight_layout()
    out_png = result(VERSION, "noise_compare.png")
    fig.savefig(out_png, bbox_inches="tight")
    print(f"已存 {out_png}")


main()
