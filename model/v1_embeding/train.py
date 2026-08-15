"""訓練 v1_embeding 的 in = out ConvAE，輸出每個 patch 的 latent 與 MSE 重建誤差。

跟 v0 的流程一樣，只有兩點不同：
  1. render() 給的是類別編號矩陣，embedding 查表在模型裡面，梯度會流到 embedding
  2. 重疊格的代表在載入時就抽定了，沒有隨機旋轉

訓練完會印出學到的類別 embedding 兩兩 cosine。in = out 的設定下模型有動機
把所有類別轉成同一個方向（輸入變成單純的「有沒有東西」，比較好壓），
cosine 若普遍逼近 1 就代表發生了這件事，要另外加多樣性懲罰才救得回來。
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

OUT = result("v1_embeding", "latents.npz")
CKPT = result("v1_embeding", "ae.pt")

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
        g = torch.Generator().manual_seed(SEED + epoch)
        perm = train_idx[torch.randperm(len(train_idx), generator=g)]
        total = 0.0
        for i in range(0, len(perm), BATCH):
            batch = perm[i:i + BATCH]
            x, _, recon = model(data.render(batch).to(device))
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
                    x, _, recon = model(data.render(batch).to(device))
                    vl.append(mse_loss(recon, x))
                val = torch.cat(vl).mean().item()
            print(f"  epoch {epoch + 1:3d}  "
                  f"train {total / len(perm):.5f}  val {val:.5f}")

    torch.save(model.state_dict(), CKPT)

    model.eval()
    zs, errs = [], []
    with torch.no_grad():
        for i in range(0, data.n, BATCH):
            idx = torch.arange(i, min(i + BATCH, data.n))
            x, z, recon = model(data.render(idx).to(device))
            zs.append(z.cpu())
            errs.append(mse_loss(recon, x).cpu())
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
