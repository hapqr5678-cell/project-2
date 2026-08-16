"""v2_vae_tanh：跟 v2_vae 完全同架構，唯一差異是 fc_mu 的輸出多接一個
Tanh()，把 mu（eval 時就是 z）在訓練時就強制夾在 (-1,1) 之間。logvar
不動——只有代表位置的 mu 需要被夾住，跟「限制 latent 座標範圍」的實驗
目的對齊；logvar 是控制取樣噪聲大小的分支，跟座標範圍無關。
"""

import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(f"{os.path.dirname(__file__)}/../.."))
from config.dataset import N_CAT, HALF_WIDTH  # noqa: E402,F401

HIDDEN = 64


class Patches:
    """稀疏點列表；agg() 把整個 patch 聚合成一個 (N_CAT,) 的 count 向量。"""

    def __init__(self, path):
        d = np.load(path)
        self.cat = torch.from_numpy(d["cat"].astype(np.int64))
        self.offsets = torch.from_numpy(d["offsets"])
        self.n_poi = d["n_poi"]
        self.lat = d["center_lat"]
        self.lon = d["center_lon"]
        self.n = len(self.n_poi)

    def agg(self, idx):
        b = len(idx)
        starts, ends = self.offsets[idx], self.offsets[idx + 1]
        lens = ends - starts
        pos = torch.repeat_interleave(starts, lens) + \
            torch.arange(int(lens.sum())) - torch.repeat_interleave(
                torch.cumsum(lens, 0) - lens, lens)
        owner = torch.repeat_interleave(torch.arange(b), lens)
        cat = self.cat[pos]

        flat = owner * N_CAT + cat
        counts = torch.bincount(flat, minlength=b * N_CAT)
        return counts.view(b, N_CAT).float()


class VAE(nn.Module):
    def __init__(self, latent_dim=2):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(N_CAT, HIDDEN),
            nn.GELU(),
            nn.Linear(HIDDEN, HIDDEN),
            nn.GELU(),
        )
        self.fc_mu = nn.Linear(HIDDEN, latent_dim)
        self.fc_logvar = nn.Linear(HIDDEN, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, HIDDEN),
            nn.GELU(),
            nn.Linear(HIDDEN, HIDDEN),
            nn.GELU(),
            nn.Linear(HIDDEN, N_CAT),   # 輸出是 log λ
        )

    def reparameterize(self, mu, logvar):
        """訓練時取樣、eval 時直接回傳 mu，讓推論的 latent 唯一。"""
        if not self.training:
            return mu
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def forward(self, x):
        h = self.trunk(x)
        mu = torch.tanh(self.fc_mu(h))   # <- 跟 v2_vae 唯一的差異：mu 夾在 (-1,1)
        logvar = self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        return mu, logvar, z, self.decoder(z)


def poisson_nll(log_lam, x):
    """Poisson NLL，省略跟模型無關的 log(y!) 常數項，回傳每個 patch 一個數字。"""
    cell = torch.exp(log_lam) - x * log_lam
    return cell.mean(dim=1)


def poisson_deviance(log_lam, x):
    """Poisson deviance：2·(y·log(y/λ) - (y - λ))，越小越好、有下界，可跨 patch 比較。"""
    lam = torch.exp(log_lam)
    cell = 2 * (torch.xlogy(x, x) - x * log_lam - x + lam)
    return cell.mean(dim=1)


def kl_divergence(mu, logvar):
    """KL(N(mu, diag(exp(logvar))) || N(0,I))，回傳每個 patch 一個數字。"""
    return -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1)
