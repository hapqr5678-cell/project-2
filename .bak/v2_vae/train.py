"""訓練 v2_vae：encoder 輸出 mu/logvar，loss = Poisson NLL + BETA * KL。

超參數對齊 v2_ae（latent 2 維、lr 1e-3、Poisson NLL 那一項完全一樣），
唯一的變因是 encoder 多了機率化的 head 與 KL 正規化項。

存進 latents.npz 的 z 是 mu（eval 模式下 reparameterize() 直接回傳 mu，
決定性、每個 patch 唯一），另外存 logvar 讓 analyze 可以檢查有沒有
posterior collapse（見 report_posterior）。
"""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(f"{os.path.dirname(__file__)}/../.."))
from ae import VAE, Patches, kl_divergence, poisson_deviance, poisson_nll  # noqa: E402
from config.dataset import ensure_patches, PATCHES, result  # noqa: E402

OUT = result("v2_vae", "latents.npz")
CKPT = result("v2_vae", "ae.pt")

LATENT_DIM = 2
EPOCHS = 1000
BATCH = 256
LR = 1e-3
VAL_FRAC = 0.1
SEED = 0
BETA = 0.01      # KL 的權重；見 ae.py docstring 的 sweep 結果

device = ("mps" if torch.backends.mps.is_available()
          else "cuda" if torch.cuda.is_available() else "cpu")


def run(data, train_idx, val_idx):
    torch.manual_seed(SEED)
    model = VAE(LATENT_DIM).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        model.train()
        g = torch.Generator().manual_seed(SEED + epoch)
        perm = train_idx[torch.randperm(len(train_idx), generator=g)]
        total_recon, total_kl = 0.0, 0.0
        for i in range(0, len(perm), BATCH):
            batch = perm[i:i + BATCH]
            x = data.agg(batch).to(device)
            mu, logvar, _, log_lam = model(x)
            recon = poisson_nll(log_lam, x).mean()
            kl = kl_divergence(mu, logvar).mean()
            loss = recon + BETA * kl
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_recon += recon.item() * len(batch)
            total_kl += kl.item() * len(batch)

        if (epoch + 1) % 20 == 0 or epoch == 0:
            model.eval()
            with torch.no_grad():
                vr, vk, vd = [], [], []
                for i in range(0, len(val_idx), BATCH):
                    batch = val_idx[i:i + BATCH]
                    x = data.agg(batch).to(device)
                    mu, logvar, _, log_lam = model(x)
                    vr.append(poisson_nll(log_lam, x))
                    vk.append(kl_divergence(mu, logvar))
                    vd.append(poisson_deviance(log_lam, x))
                val_recon = torch.cat(vr).mean().item()
                val_kl = torch.cat(vk).mean().item()
                dev = torch.cat(vd).mean().item()
            print(f"  epoch {epoch + 1:3d}  "
                  f"train NLL {total_recon / len(perm):.5f}  train KL {total_kl / len(perm):.5f}  "
                  f"val NLL {val_recon:.5f}  val KL {val_kl:.5f}  val deviance {dev:.5f}")

    torch.save(model.state_dict(), CKPT)

    # 推論用 eval 模式(reparameterize 回傳 mu)，latent 唯一
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
    """印出 posterior std 的統計量，檢查有沒有 collapse 到先驗（std -> 1）。"""
    std = np.exp(0.5 * logvar)
    print(f"\nposterior std：平均 {std.mean():.3f}  中位數 {np.median(std):.3f}  "
          f"（逼近 1 代表 q(z|x) 塌縮成先驗 N(0,I)，latent 對輸入不敏感）")
    for d in range(std.shape[1]):
        print(f"  z{d + 1} std 平均 {std[:, d].mean():.3f}")


def main():
    ensure_patches()
    data = Patches(PATCHES)
    print(f"{data.n} 個 patch，device={device}，BETA={BETA}")

    g = torch.Generator().manual_seed(SEED)
    perm = torch.randperm(data.n, generator=g)
    n_val = int(data.n * VAL_FRAC)
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    mu, logvar, err = run(data, train_idx, val_idx)
    np.savez(OUT, n_poi=data.n_poi, lat=data.lat, lon=data.lon,
             z=mu, logvar=logvar, err=err)
    report_posterior(logvar)
    print(f"\n已存 {OUT}")


main()
