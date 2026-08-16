"""訓練 v2_denoise_vae_fsce：v2_denoise_vae（VAE + 加噪）再加上 FSCE，
loss = Poisson NLL + BETA * KL + lambda(t) * FSCE。

跟 v2_denoise_vae 的 train.py 相比多了兩件事——訓練前先在**乾淨** count
向量上建一次 FSCE 用的 fuzzy graph（鄰接關係是資料的性質，不是這次抽到的
噪聲的性質），訓練迴圈裡每個 step 除了原本的 reconstruction + KL batch，
額外抽一批正樣本邊（來自 fuzzy graph）跟等量的隨機負樣本邊，對 mu 算
FSCE loss，加權後跟 (Poisson NLL + BETA*KL) 共用同一次 backward/opt.step()。

所有餵進 encoder 的輸入（reconstruction batch 跟 FSCE 的 pair 兩邊都算）
都先過 corrupt()，Poisson NLL 的目標仍然是乾淨的原始 count。FSCE 的 pair
也要加噪的理由跟 v2_dae_fsce 一樣：訓練時 encoder 只該看到同一種輸入分布，
只加噪一邊的話 encoder 等於被要求同時服務兩種尺度不同的輸入，latent 幾何
會被這個不一致污染。

FSCE 權重用線性 warm-up：前 FSCE_WARMUP_EPOCHS 個 epoch 是 0，之後花同樣
長度線性升到 LAMBDA_FSCE。這裡的 warm-up 比 v2_dae_fsce（固定小權重、不
warm-up）更必要——KL 把 mu 往原點收、FSCE 的 negative sampling 把 mu 往外
推，方向相反，先讓 encoder 學出有意義的 count -> mu 映射再加拓撲約束，
避免兩者一開始就對撞。

驗證與最後輸出 latent 的推論階段一律不加噪、eval 模式下 reparameterize()
直接回傳 mu，latent 決定性、每個 patch 唯一。破壞是每個 step 重新抽的
（generator 綁 SEED+epoch，可重現）。

超參數分別對齊三個來源，這樣任兩個版本之間的差異都只有一項：
LATENT_DIM/EPOCHS/BATCH/LR/VAL_FRAC/SEED/BETA、NOISE_P/NOISE_MODE 跟
v2_denoise_vae 一致，FSCE 那一組（N_NEIGHBORS/GRAPH_METRIC/EDGE_BATCH/
LAMBDA_FSCE/FSCE_WARMUP_EPOCHS）跟 v2_vae_fsce 一致。

存進 latents.npz 的 z 是 mu，另外存 logvar 讓 analyze 可以檢查有沒有
posterior collapse（見 report_posterior）。
"""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(f"{os.path.dirname(__file__)}/../.."))
from ae import (VAE, Patches, build_fsce_graph, corrupt,  # noqa: E402
                fsce_loss, kl_divergence, poisson_deviance, poisson_nll)
from config.dataset import ensure_patches, PATCHES, result  # noqa: E402

VERSION = "v2_denoise_vae_fsce"
OUT = result(VERSION, "latents.npz")
CKPT = result(VERSION, "ae.pt")

LATENT_DIM = 2
EPOCHS = 1000
BATCH = 256
LR = 1e-3
VAL_FRAC = 0.1
SEED = 0
BETA = 0.01              # KL 的權重；見 v2_vae/ae.py docstring 的 sweep 結果

N_NEIGHBORS = 15         # 建高維 fuzzy graph 的 kNN 數，跟 data/patch/umap_grid.py 一致
GRAPH_METRIC = "cosine"  # 對整包 count 向量的組成比例敏感、對總量不敏感
EDGE_BATCH = 256          # 每個 step 抽的正樣本邊數，負樣本抽等量
LAMBDA_FSCE = 1         # warm-up 結束後 FSCE loss 的權重
FSCE_WARMUP_EPOCHS = 80   # 前這麼多個 epoch 權重是 0，之後花同樣長度線性升到 LAMBDA_FSCE

NOISE_P = 0.3            # 破壞強度：thinning 是每個 POI 被丟掉的機率
NOISE_MODE = "thinning"  # "thinning"（逐 POI 丟）或 "mask"（整類歸零）

device = ("mps" if torch.backends.mps.is_available()
          else "cuda" if torch.cuda.is_available() else "cpu")


def fsce_weight(epoch):
    """epoch 是目前的 epoch 編號（0-indexed），回傳這個 epoch 的 FSCE loss 權重：
    前 FSCE_WARMUP_EPOCHS 個 epoch 是 0，之後花同樣長度線性升到 LAMBDA_FSCE。
    """
    t = (epoch - FSCE_WARMUP_EPOCHS) / FSCE_WARMUP_EPOCHS
    return LAMBDA_FSCE * min(1.0, max(0.0, t))


def run(data, train_idx, val_idx, edge_i, edge_j, edge_w, a, b):
    """訓練並回傳全體 patch 的 (mu, logvar, err)：mu 是 (N,LATENT_DIM) 的
    latent 座標、logvar 是 (N,LATENT_DIM) 的 posterior log 變異數、
    err 是 (N,) 的 Poisson deviance，三者都用乾淨輸入、eval 模式算出來。
    data 是 Patches；train_idx / val_idx 是 patch 編號的 LongTensor；
    edge_i / edge_j / edge_w / a / b 是 build_fsce_graph() 的回傳值。
    """
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
            x = data.agg(batch)                                  # 乾淨目標（CPU）
            x_in = corrupt(x, NOISE_P, NOISE_MODE, generator=g)  # 加噪輸入
            x, x_in = x.to(device), x_in.to(device)
            mu, logvar, _, log_lam = model(x_in)
            recon = poisson_nll(log_lam, x).mean()   # 目標永遠是乾淨的 x
            kl = kl_divergence(mu, logvar).mean()

            eg = torch.randint(0, n_edges, (EDGE_BATCH,), generator=g)
            pi, pj, pw = edge_i[eg], edge_j[eg], edge_w[eg]
            ni = torch.randint(0, data.n, (EDGE_BATCH,), generator=g)
            nj = torch.randint(0, data.n, (EDGE_BATCH,), generator=g)
            # pair 也加噪，encoder 訓練時看到的輸入分布才跟 recon 那一路一致
            xi = corrupt(data.agg(torch.cat([pi, ni])), NOISE_P,
                         NOISE_MODE, generator=g).to(device)
            xj = corrupt(data.agg(torch.cat([pj, nj])), NOISE_P,
                         NOISE_MODE, generator=g).to(device)
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
                    x = data.agg(batch).to(device)   # 驗證不加噪
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

    # 推論用 eval 模式(reparameterize 回傳 mu)、且不加噪，latent 唯一
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
    """logvar 是 (N,LATENT_DIM) 的 numpy 陣列；印出 posterior std 的統計量，
    檢查有沒有 collapse 到先驗（std -> 1）。沒有回傳值。
    """
    std = np.exp(0.5 * logvar)
    print(f"\nposterior std：平均 {std.mean():.3f}  中位數 {np.median(std):.3f}  "
          f"（逼近 1 代表 q(z|x) 塌縮成先驗 N(0,I)，latent 對輸入不敏感）")
    for d in range(std.shape[1]):
        print(f"  z{d + 1} std 平均 {std[:, d].mean():.3f}")


def main():
    ensure_patches()
    data = Patches(PATCHES)
    print(f"{data.n} 個 patch，device={device}，BETA={BETA}，"
          f"噪聲 {NOISE_MODE} p={NOISE_P}")

    # fuzzy graph 用乾淨 count 建：鄰接關係是資料的性質，不是噪聲的性質
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
