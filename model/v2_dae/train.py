"""訓練 v2_dae 的 Denoising AutoEncoder，輸出每個 patch 的 latent 與 Poisson deviance。

沿用 v2_ae 的架構與 Loss (Poisson NLL)，但在輸入加入 Binomial thinning noise，
要求模型從帶噪的輸入還原乾淨的 POI count。
"""

import os
import sys

import numpy as np
import torch
from torch.distributions import Binomial

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(f"{os.path.dirname(__file__)}/../.."))
from ae import MLPAE, Patches, poisson_deviance, poisson_nll  # noqa: E402
from config.dataset import ensure_patches, PATCHES, result  # noqa: E402

OUT = result("v2_dae", "latents.npz")
CKPT = result("v2_dae", "dae.pt")

LATENT_DIM = 2
EPOCHS = 1000
BATCH = 256
LR = 1e-3
VAL_FRAC = 0.1
SEED = 0
KEEP_PROB = 0.8

device = ("mps" if torch.backends.mps.is_available()
          else "cuda" if torch.cuda.is_available() else "cpu")


def binomial_thinning(x, keep_prob):
    """
    使用 Binomial thinning 製造帶噪輸入。
    x: 乾淨的 POI count 向量
    keep_prob: 每個 POI 被保留的機率
    """
    try:
        m = Binomial(total_count=x, probs=torch.tensor(keep_prob, device=x.device, dtype=torch.float32))
        return m.sample()
    except Exception:
        # MPS 可能不支援 Binomial，退回 CPU 抽樣
        m = Binomial(total_count=x.cpu(), probs=torch.tensor(keep_prob, device='cpu', dtype=torch.float32))
        return m.sample().to(x.device)


def run(data, train_idx, val_idx):
    torch.manual_seed(SEED)
    model = MLPAE(LATENT_DIM).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        model.train()
        g = torch.Generator().manual_seed(SEED + epoch)
        perm = train_idx[torch.randperm(len(train_idx), generator=g)]
        total = 0.0
        for i in range(0, len(perm), BATCH):
            batch = perm[i:i + BATCH]
            x_clean = data.agg(batch).to(device)
            
            # DAE 擾動
            x_noisy = binomial_thinning(x_clean, KEEP_PROB)
            
            # 模型輸入為帶噪的 x_noisy，但 target 必須是乾淨的 x_clean
            _, log_lam = model(x_noisy)
            loss = poisson_nll(log_lam, x_clean).mean()
            
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
            x_clean = data.agg(idx).to(device)
            # 訓練完成後，使用乾淨輸入進行推論
            z, log_lam = model(x_clean)
            zs.append(z.cpu())
            errs.append(poisson_deviance(log_lam, x_clean).cpu())
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


if __name__ == "__main__":
    main()
