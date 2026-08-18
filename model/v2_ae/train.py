"""訓練 v2_ae 的 in = out MLP AutoEncoder，輸出每個 patch 的 latent 與 Poisson deviance。

輸入是整包 patch 的 (N_CAT,) 聚合 count 向量，沒有空間結構，
所以沒有 v0/v0_poisson_nll 那種「隨機旋轉」的概念——一個 patch 的
聚合向量本來就唯一，訓練跟推論看到的是同一份東西。

超參數（latent 2 維、lr 1e-3）留給 v2_perceiver、v2_vae 對齊，
這樣三個版本的 latent 才是在同一組資料/loss 下比較，差別只在 encoder。
"""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(f"{os.path.dirname(__file__)}/../.."))
from ae import MLPAE, Patches, poisson_deviance, poisson_nll  # noqa: E402
from config.dataset import ensure_patches, PATCHES, result  # noqa: E402
from config.train_log import open_log  # noqa: E402

OUT = result("v2_ae", "latents.npz")
CKPT = result("v2_ae", "ae.pt")

LATENT_DIM = 2
EPOCHS = 1500
BATCH = 256
LR = 1e-3
VAL_FRAC = 0.1
SEED = 0

device = ("mps" if torch.backends.mps.is_available()
          else "cuda" if torch.cuda.is_available() else "cpu")


def run(data, train_idx, val_idx, log):
    torch.manual_seed(SEED)
    model = MLPAE(LATENT_DIM).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    # 空模型（λ=訓練集全域平均）的 log_lam，當 explained deviance 的分母，
    # 用訓練集估、驗證集算 deviance，不然分母本身就偷看了驗證集
    log_lam_null = data.agg(train_idx).mean(dim=0, keepdim=True) \
        .clamp_min(1e-8).log().to(device)

    for epoch in range(EPOCHS):
        try:
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

            if (epoch + 1) % 5 == 0 or epoch == 0:
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
                log(f"epoch {epoch + 1:4d} | train_nll {total / len(perm):.5f} | "
                    f"val_dev {dev:.5f} | expl_dev {expl:.5f}")
        except KeyboardInterrupt:
            log(f"\n[中斷] epoch {epoch + 1} 收到 Ctrl-C，用目前模型存 checkpoint 跟 latent")
            break

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
    log = open_log("v2_ae", {
        "LATENT_DIM": LATENT_DIM, "EPOCHS": EPOCHS, "BATCH": BATCH, "LR": LR,
        "VAL_FRAC": VAL_FRAC, "SEED": SEED,
    })

    ensure_patches()
    data = Patches(PATCHES)
    log(f"{data.n} 個 patch，device={device}")

    g = torch.Generator().manual_seed(SEED)
    perm = torch.randperm(data.n, generator=g)
    n_val = int(data.n * VAL_FRAC)
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    z, err = run(data, train_idx, val_idx, log)
    np.savez(OUT, n_poi=data.n_poi, lat=data.lat, lon=data.lon, z=z, err=err)
    log(f"已存 {OUT}")


main()
