"""訓練 v0_nb 的 ConvAE，輸出每個 patch 的 latent 與 NB deviance。

超參數刻意跟 v0_poisson_nll 完全相同（latent 2 維、30 epoch、lr 1e-3、
不旋轉、同一組切分），差別只有 loss 從 Poisson 換成 NB，
這樣兩邊的 latent 才是可比較的。

結尾會印出每個類別學到的 theta。要看的是它有多大：
theta 越大代表 NB 越接近 Poisson，也就是模型自己說「不需要 overdispersion」。
Var/mean = 1 + mu/theta，以 mu ~ 0.01 的典型格子來說，theta=1 也只帶來
1% 的額外變異——所以 theta 要小到 0.1 以下才算真的有在用這個自由度。
"""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(f"{os.path.dirname(__file__)}/../.."))
from ae import ConvAE, Patches, nb_deviance, nb_nll  # noqa: E402
from config.dataset import CAT_ZH, ensure_patches, PATCHES, result  # noqa: E402

OUT = result("v0_nb", "latents.npz")
CKPT = result("v0_nb", "ae.pt")

LATENT_DIM = 2
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
            _, log_mu = model(x)
            loss = nb_nll(log_mu, model.log_theta_map(), x).mean()
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
                    _, log_mu = model(x)
                    vl.append(nb_nll(log_mu, model.log_theta_map(), x))
                    vd.append(nb_deviance(log_mu, model.log_theta_map(), x))
                val = torch.cat(vl).mean().item()
                dev = torch.cat(vd).mean().item()
            print(f"  epoch {epoch + 1:3d}  train NLL {total / len(perm):.5f}  "
                  f"val NLL {val:.5f}  val deviance {dev:.5f}")

    torch.save(model.state_dict(), CKPT)

    print("\n各類別學到的 dispersion（theta 越大越接近 Poisson）")
    th = torch.exp(model.log_theta.detach().cpu()).numpy()
    for k, t in enumerate(th):
        print(f"  {CAT_ZH[k]:<6} theta = {t:9.3f}   "
              f"mu=0.01 時 var/mean = {1 + 0.01 / t:.4f}")

    # 推論用固定朝向(不旋轉)，讓每個 patch 的 latent 是唯一的
    model.eval()
    zs, errs = [], []
    with torch.no_grad():
        for i in range(0, data.n, BATCH):
            idx = torch.arange(i, min(i + BATCH, data.n))
            x = data.render(idx, rotate=False).to(device)
            z, log_mu = model(x)
            zs.append(z.cpu())
            errs.append(nb_deviance(log_mu, model.log_theta_map(), x).cpu())
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
