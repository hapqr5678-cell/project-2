"""訓練 v0 的 in = out ConvAE，輸出每個 patch 的 latent 與 MSE 重建誤差。

patch 在 batch 內即時 binning；ROTATE=True 時會做隨機旋轉，避免 latent 被
「街區朝向」吃掉（目前暫時關閉）。
"""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(f"{os.path.dirname(__file__)}/../.."))
from ae import ConvAE, Patches, mse_loss  # noqa: E402
from config.dataset import ensure_patches, PATCHES, result  # noqa: E402

OUT = result("v0_l32", "latents.npz")
CKPT = result("v0_l32", "ae.pt")

LATENT_DIM = 32
EPOCHS = 100
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
            _, recon = model(x)
            loss = mse_loss(recon, x).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(batch)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            model.eval()
            with torch.no_grad():
                vl = []
                for i in range(0, len(val_idx), BATCH):
                    batch = val_idx[i:i + BATCH]
                    x = data.render(batch, rotate=ROTATE).to(device)
                    _, recon = model(x)
                    vl.append(mse_loss(recon, x))
                val = torch.cat(vl).mean().item()
            print(f"  epoch {epoch + 1:3d}  "
                  f"train {total / len(perm):.5f}  val {val:.5f}")

    torch.save(model.state_dict(), CKPT)

    # 推論用固定朝向(不旋轉)，讓每個 patch 的 latent 是唯一的
    model.eval()
    zs, errs = [], []
    with torch.no_grad():
        for i in range(0, data.n, BATCH):
            idx = torch.arange(i, min(i + BATCH, data.n))
            x = data.render(idx, rotate=False).to(device)
            z, recon = model(x)
            zs.append(z.cpu())
            errs.append(mse_loss(recon, x).cpu())
    return torch.cat(zs).numpy(), torch.cat(errs).numpy()


def main():
    ensure_patches()
    data = Patches(PATCHES)
    print(f"{data.n} 個 patch，device={device}")

    g = torch.Generator().manual_seed(SEED)
    perm = torch.randperm(data.n, generator=g)
    n_val = int(data.n * VAL_FRAC)
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    z, err = run(data, train_idx, val_idx)
    np.savez(OUT, n_poi=data.n_poi, lat=data.lat, lon=data.lon, z=z, err=err)
    print(f"已存 {OUT}")


main()
