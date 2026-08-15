"""訓練 v1_masked 的 ConvAE，輸出每個 patch 的 latent 與（遮罩後的）重建誤差。

跟 v1_embeding 的流程完全一樣，唯一的差別是 loss 只算有 POI 的格子。

訓練完會印出學到的類別 embedding 兩兩 cosine。可學習 embedding 加上遮罩後，
「所有類別轉成同一個方向、decoder 輸出常數」可以讓 loss 精確等於 0，
所以 cosine 逼近 1 而且 loss 掉到接近 0，就是塌縮發生的訊號。
"""

import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(f"{os.path.dirname(__file__)}/../.."))
from ae import ConvAE, Patches, mse_loss  # noqa: E402
from config.dataset import CAT_ZH, ensure_patches, N_CAT, PATCHES, result  # noqa: E402

OUT = result("v1_masked", "latents.npz")
CKPT = result("v1_masked", "ae.pt")

LATENT_DIM = 2
EPOCHS = 30
BATCH = 256
LR = 1e-3
VAL_FRAC = 0.1
SEED = 0

device = ("mps" if torch.backends.mps.is_available()
          else "cuda" if torch.cuda.is_available() else "cpu")


def run(data, train_idx, val_idx):
    torch.manual_seed(SEED)
    model = ConvAE(LATENT_DIM).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        model.train()
        gen = torch.Generator().manual_seed(SEED + epoch)
        perm = train_idx[torch.randperm(len(train_idx), generator=gen)]
        total = 0.0
        for i in range(0, len(perm), BATCH):
            batch = perm[i:i + BATCH]
            g = data.render(batch).to(device)
            x, _, recon = model(g)
            loss = mse_loss(recon, x, g).mean()
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
                    g = data.render(batch).to(device)
                    x, _, recon = model(g)
                    vl.append(mse_loss(recon, x, g))
                val = torch.cat(vl).mean().item()
            print(f"  epoch {epoch + 1:3d}  "
                  f"train {total / len(perm):.5f}  val {val:.5f}")

    torch.save(model.state_dict(), CKPT)

    model.eval()
    zs, errs = [], []
    with torch.no_grad():
        for i in range(0, data.n, BATCH):
            idx = torch.arange(i, min(i + BATCH, data.n))
            g = data.render(idx).to(device)
            x, z, recon = model(g)
            zs.append(z.cpu())
            errs.append(mse_loss(recon, x, g).cpu())
        emb = F.normalize(model.emb.weight, dim=1).cpu().numpy()
    return torch.cat(zs).numpy(), torch.cat(errs).numpy(), emb


def report_emb(emb):
    """印出類別 embedding 的兩兩 cosine，檢查有沒有全部塌成同一個方向。"""
    cos = emb @ emb.T
    off = cos[~np.eye(N_CAT, dtype=bool)]
    print(f"\n類別 embedding 兩兩 cosine："
          f"平均 {off.mean():+.3f}  最小 {off.min():+.3f}  最大 {off.max():+.3f}")
    print("      " + "".join(f"{z[:2]:>6}" for z in CAT_ZH))
    for i, z in enumerate(CAT_ZH):
        print(f"{z[:2]:>6}" + "".join(f"{cos[i, j]:>6.2f}" for j in range(N_CAT)))


def main():
    ensure_patches()
    data = Patches(PATCHES)
    occ = data.n_occupied
    print(f"{data.n} 個 patch，device={device}")
    print(f"每 patch 佔用格數 中位數 {np.median(occ):.0f}  "
          f"（POI 數中位數 {np.median(data.n_poi):.0f}，"
          f"重疊而被丟掉 {1 - occ.sum() / data.n_poi.sum():.1%}）")

    g = torch.Generator().manual_seed(SEED)
    perm = torch.randperm(data.n, generator=g)
    n_val = int(data.n * VAL_FRAC)
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    z, err, emb = run(data, train_idx, val_idx)
    np.savez(OUT, n_poi=data.n_poi, n_occupied=occ,
             lat=data.lat, lon=data.lon, z=z, err=err, emb=emb)
    report_emb(emb)
    print(f"\n已存 {OUT}")


main()
