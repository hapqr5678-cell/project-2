"""訓練 v0_l32_sizefactor 的 ConvAE，輸出每個 patch 的 latent 與 Poisson deviance。

跟 v0_sizefactor 只差 latent 維度（2 -> 32），其他（lr、切分、不旋轉、
size factor 正規化）完全相同，差異才能歸因給容量。

EPOCHS 從 30 加到 60 是唯一的例外：v0_sizefactor 跑到第 30 epoch 時
val NLL 還在降（0.11136 -> 0.11095），32 維的參數更多、收斂更慢，
用 30 epoch 收的話「組成 R² 沒上來」會分不清是容量不夠還是沒訓練完。
log 每 5 epoch 印一次，所以第 30 epoch 的數字仍在裡面，要跟
v0_sizefactor 做同 epoch 的對照時直接看那一行。

latents.npz 比其他版本多存一個 s（= 圓內 POI 總數，等於 n_poi）。
存它是為了 analyze/saturation.py 好讀，也順便當作 render 沒把點弄丟的檢查。
"""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(f"{os.path.dirname(__file__)}/../.."))
from ae import ConvAE, Patches, poisson_deviance, poisson_nll  # noqa: E402
from config.dataset import ensure_patches, PATCHES, result  # noqa: E402

OUT = result("v0_l32_sizefactor", "latents.npz")
CKPT = result("v0_l32_sizefactor", "ae.pt")

LATENT_DIM = 32
EPOCHS = 60
BATCH = 256
LR = 1e-3
VAL_FRAC = 0.1
SEED = 0
ROTATE = False  # 暫時關掉隨機旋轉

device = ("mps" if torch.backends.mps.is_available()
          else "cuda" if torch.cuda.is_available() else "cpu")


def run(data, train_idx, val_idx):
    torch.manual_seed(SEED)
    model = ConvAE(LATENT_DIM).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        model.train()
        g = torch.Generator().manual_seed(SEED + epoch)
        perm = train_idx[torch.randperm(len(train_idx), generator=g)]
        total = 0.0
        for i in range(0, len(perm), BATCH):
            batch = perm[i:i + BATCH]
            x = data.render(batch, rotate=ROTATE).to(device)
            _, log_lam = model(x)
            loss = poisson_nll(log_lam, x).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(batch)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            model.eval()
            with torch.no_grad():
                vl, vd = [], []
                for i in range(0, len(val_idx), BATCH):
                    batch = val_idx[i:i + BATCH]
                    x = data.render(batch, rotate=ROTATE).to(device)
                    _, log_lam = model(x)
                    vl.append(poisson_nll(log_lam, x))
                    vd.append(poisson_deviance(log_lam, x))
                val = torch.cat(vl).mean().item()
                dev = torch.cat(vd).mean().item()
            print(f"  epoch {epoch + 1:3d}  train NLL {total / len(perm):.5f}  "
                  f"val NLL {val:.5f}  val deviance {dev:.5f}")

    torch.save(model.state_dict(), CKPT)

    # 推論用固定朝向(不旋轉)，讓每個 patch 的 latent 是唯一的
    model.eval()
    zs, errs, ss = [], [], []
    with torch.no_grad():
        for i in range(0, data.n, BATCH):
            idx = torch.arange(i, min(i + BATCH, data.n))
            x = data.render(idx, rotate=False).to(device)
            z, log_lam = model(x)
            zs.append(z.cpu())
            errs.append(poisson_deviance(log_lam, x).cpu())
            ss.append(x.sum(dim=(1, 2, 3)).cpu())
    return (torch.cat(zs).numpy(), torch.cat(errs).numpy(),
            torch.cat(ss).numpy())


def main():
    ensure_patches()
    data = Patches(PATCHES)
    print(f"{data.n} 個 patch，device={device}")

    g = torch.Generator().manual_seed(SEED)
    perm = torch.randperm(data.n, generator=g)
    n_val = int(data.n * VAL_FRAC)
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    z, err, s = run(data, train_idx, val_idx)
    bad = int((s != data.n_poi).sum())
    print(f"size factor 與 n_poi 不一致的 patch：{bad} 個（應為 0）")

    np.savez(OUT, n_poi=data.n_poi, lat=data.lat, lon=data.lon,
             z=z, err=err, s=s)
    print(f"已存 {OUT}")


main()
