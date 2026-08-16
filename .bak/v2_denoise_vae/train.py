"""訓練 v2_denoise_vae：v2_vae 加上 v2_dae 的加噪，loss = Poisson NLL + BETA * KL。

跟 v2_vae 的 train.py 相比只多一件事：訓練迴圈裡，餵給 encoder 的是
corrupt() 破壞過的 count 向量，Poisson NLL 仍然拿乾淨的原始 count 當目標，
KL 則是算在加噪輸入導出的 q(z|x_tilde) 上。驗證與最後輸出 latent 的推論
階段一律不加噪——噪聲是正則化手段，不是資料本身的性質。

破壞是每個 step 重新抽的（generator 綁 SEED+epoch，可重現），同一個 patch
在不同 epoch 會看到不同的殘缺版本。

超參數（LATENT_DIM、EPOCHS、BATCH、LR、VAL_FRAC、SEED、BETA）全部跟 v2_vae
對齊，NOISE_P / NOISE_MODE 跟 v2_dae 對齊，這樣這一版相對兩邊各自的差異
都只有一項。

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
from ae import (VAE, Patches, corrupt, kl_divergence,  # noqa: E402
                poisson_deviance, poisson_nll)
from config.dataset import ensure_patches, PATCHES, result  # noqa: E402

VERSION = "v2_denoise_vae"
OUT = result(VERSION, "latents.npz")
CKPT = result(VERSION, "ae.pt")

LATENT_DIM = 2
EPOCHS = 1500
BATCH = 256
LR = 1e-3
VAL_FRAC = 0.1
SEED = 0
BETA = 0.01              # KL 的權重；見 v2_vae/ae.py docstring 的 sweep 結果

NOISE_P = 0.3            # 破壞強度：thinning 是每個 POI 被丟掉的機率
NOISE_MODE = "thinning"  # "thinning"（逐 POI 丟）或 "mask"（整類歸零）

device = ("mps" if torch.backends.mps.is_available()
          else "cuda" if torch.cuda.is_available() else "cpu")


def run(data, train_idx, val_idx):
    """訓練並回傳全體 patch 的 (mu, logvar, err)：mu 是 (N,LATENT_DIM) 的
    latent 座標、logvar 是 (N,LATENT_DIM) 的 posterior log 變異數、
    err 是 (N,) 的 Poisson deviance，三者都用乾淨輸入、eval 模式算出來。
    data 是 Patches，train_idx / val_idx 是 patch 編號的 LongTensor。
    """
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
            x = data.agg(batch)                                  # 乾淨目標（CPU）
            x_in = corrupt(x, NOISE_P, NOISE_MODE, generator=g)  # 加噪輸入
            x, x_in = x.to(device), x_in.to(device)
            mu, logvar, _, log_lam = model(x_in)
            recon = poisson_nll(log_lam, x).mean()   # 目標永遠是乾淨的 x
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
