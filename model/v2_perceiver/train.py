"""訓練 v2_perceiver：encoder 吃原始 POI 類別 token 集合，decoder/loss 跟 v2_ae 對齊。

超參數對齊 v2_ae（latent 2 維、lr 1e-3），reconstruction target 一律是
data.agg() 算出的聚合向量，這樣兩邊的 latent 才能直接比較，唯一的
變因是 encoder 怎麼把「一包 POI」讀成 2 維 latent。
"""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(f"{os.path.dirname(__file__)}/../.."))
from ae import PerceiverAE, Patches, poisson_deviance, poisson_nll  # noqa: E402
from config.dataset import ensure_patches, PATCHES, result  # noqa: E402

OUT = result("v2_perceiver", "latents.npz")
CKPT = result("v2_perceiver", "ae.pt")

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
    model = PerceiverAE(LATENT_DIM).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        model.train()
        g = torch.Generator().manual_seed(SEED + epoch)
        perm = train_idx[torch.randperm(len(train_idx), generator=g)]
        total = 0.0
        for i in range(0, len(perm), BATCH):
            batch = perm[i:i + BATCH]
            tok, pad_mask = data.tokens(batch)
            x = data.agg(batch).to(device)
            _, log_lam = model(tok.to(device), pad_mask.to(device))
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
                    tok, pad_mask = data.tokens(batch)
                    x = data.agg(batch).to(device)
                    _, log_lam = model(tok.to(device), pad_mask.to(device))
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
            tok, pad_mask = data.tokens(idx)
            x = data.agg(idx).to(device)
            z, log_lam = model(tok.to(device), pad_mask.to(device))
            zs.append(z.cpu())
            errs.append(poisson_deviance(log_lam, x).cpu())
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
