"""比較 v0 AE 的重建 MSE 跟兩個沒有學習能力的 baseline。

三個數字都在同一個 log1p 空間、同一套 train/val 切分下算：
  1. AE       模型重建的 MSE
  2. zeros    永遠輸出全 0（= 輸入的平方平均，代表「什麼都不做」的成本）
  3. mean     永遠輸出訓練集的平均矩陣（每格每類的平均強度）

AE 至少要贏 mean 才代表它真的有學到「這個 patch 長什麼樣」，
只贏 zeros 的話只是學會了整體的平均密度。
"""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(f"{os.path.dirname(__file__)}/../../.."))
from ae import ConvAE, Patches, mse_loss  # noqa: E402
from config.dataset import PATCHES, result  # noqa: E402

CKPT = result("v0", "ae.pt")

LATENT_DIM = 2
BATCH = 256
VAL_FRAC = 0.1
SEED = 0   # 跟 train.py 一致，才切出同一組 val

device = ("mps" if torch.backends.mps.is_available()
          else "cuda" if torch.cuda.is_available() else "cpu")


def mean_matrix(data, idx):
    """訓練集上每格每類的平均 log1p 強度，shape (1,10,40,40)。"""
    acc = None
    for i in range(0, len(idx), BATCH):
        x = data.render(idx[i:i + BATCH], rotate=False)
        s = x.sum(dim=0, keepdim=True)
        acc = s if acc is None else acc + s
    return acc / len(idx)


def evaluate(data, idx, model, mu):
    """回傳這組 patch 的三種 per-patch MSE。"""
    ae, zeros, mean = [], [], []
    mu = mu.to(device)
    with torch.no_grad():
        for i in range(0, len(idx), BATCH):
            x = data.render(idx[i:i + BATCH], rotate=False).to(device)
            _, recon = model(x)
            ae.append(mse_loss(recon, x).cpu())
            zeros.append(mse_loss(torch.zeros_like(x), x).cpu())
            mean.append(mse_loss(mu.expand_as(x), x).cpu())
    return (torch.cat(ae).numpy(), torch.cat(zeros).numpy(),
            torch.cat(mean).numpy())


def report(name, res):
    ae, zeros, mean = res
    print(f"\n{name}（{len(ae)} 個 patch）")
    print(f"  {'方法':<8}{'平均 MSE':>12}{'中位數':>12}{'相對 zeros':>12}")
    for label, v in (("AE", ae), ("zeros", zeros), ("mean", mean)):
        print(f"  {label:<8}{v.mean():>12.6f}{np.median(v):>12.6f}"
              f"{v.mean() / zeros.mean():>12.3f}")
    print(f"  AE 比 zeros 好 {(1 - ae.mean() / zeros.mean()) * 100:5.1f}%"
          f"，比 mean 好 {(1 - ae.mean() / mean.mean()) * 100:5.1f}%")
    print(f"  AE 贏 mean 的 patch 比例：{(ae < mean).mean() * 100:.1f}%")


def main():
    data = Patches(PATCHES)
    g = torch.Generator().manual_seed(SEED)
    perm = torch.randperm(data.n, generator=g)
    n_val = int(data.n * VAL_FRAC)
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    model = ConvAE(LATENT_DIM).to(device)
    model.load_state_dict(torch.load(CKPT, map_location=device))
    model.eval()

    mu = mean_matrix(data, train_idx)
    print(f"{data.n} 個 patch，device={device}")
    print(f"訓練集平均矩陣：非零格佔 {(mu > 0).float().mean() * 100:.1f}%，"
          f"最大 {mu.max():.4f}")

    report("train", evaluate(data, train_idx, model, mu))
    report("val", evaluate(data, val_idx, model, mu))


main()
