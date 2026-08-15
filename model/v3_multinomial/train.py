"""訓練 v3_multinomial，輸出每個 patch 的 latent 與 Poisson deviance。

超參數（latent 2 維、lr 1e-3、300 epoch、val 10%）刻意跟 v2_ae 一字不差，
唯一的變因就是 ae.py 裡的 multinomial 分解，這樣兩邊的 latent 才是在
同一組資料與訓練條件下比較。

訓練結束會多印一段對照：latent 的兩個軸各自對 log(POI 數) 的相關係數。
v2_ae 在同一份 overture 資料上是 +0.284 / +0.906（latent 有一整維在
編碼密度），v3 若成功把總量移出去，這兩個數字應該明顯往 0 靠。
完整的對照（含 kNN R² 與噪聲下限）跑 analyze/compare_v2.py。
"""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(f"{os.path.dirname(__file__)}/../.."))
from ae import MultinomialAE, Patches, poisson_deviance, poisson_nll  # noqa: E402
from config.dataset import PATCHES, result  # noqa: E402

OUT = result("v3_multinomial", "latents.npz")
CKPT = result("v3_multinomial", "ae.pt")

LATENT_DIM = 2
EPOCHS = 300
BATCH = 256
LR = 1e-3
VAL_FRAC = 0.1
SEED = 0

device = ("mps" if torch.backends.mps.is_available()
          else "cuda" if torch.cuda.is_available() else "cpu")


def run(data, train_idx, val_idx):
    torch.manual_seed(SEED)
    model = MultinomialAE(LATENT_DIM).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        model.train()
        g = torch.Generator().manual_seed(SEED + epoch)
        perm = train_idx[torch.randperm(len(train_idx), generator=g)]
        total = 0.0
        for i in range(0, len(perm), BATCH):
            batch = perm[i:i + BATCH]
            x = data.agg(batch).to(device)
            _, log_lam = model(x)
            loss = poisson_nll(log_lam, x).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(batch)

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
    data = Patches(PATCHES)
    print(f"{data.n} 個 patch，device={device}")

    g = torch.Generator().manual_seed(SEED)
    perm = torch.randperm(data.n, generator=g)
    n_val = int(data.n * VAL_FRAC)
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    z, err = run(data, train_idx, val_idx)
    np.savez(OUT, n_poi=data.n_poi, lat=data.lat, lon=data.lon, z=z, err=err)
    print(f"已存 {OUT}")

    # latent 還剩多少密度資訊？這是這一版存在的理由，訓練完直接印出來
    log_n = np.log(data.n_poi.astype(float))
    print("\nlatent 各軸對 log(POI 數) 的相關（v2_ae 是 +0.60 / -0.59）：")
    for d in range(z.shape[1]):
        print(f"  z[{d}]  corr = {np.corrcoef(z[:, d], log_n)[0, 1]:+.3f}")


main()
