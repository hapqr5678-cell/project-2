"""v2_vae：encoder 換成機率化的 VAE head，decoder/loss 跟 v2_ae 共用同一套設計。

跟 v2_ae 的差別只在 encoder 出口：不是直接輸出一個 2 維向量，而是輸出
mu、logvar 兩組 2 維向量，描述 q(z|x) = N(mu, diag(exp(logvar)))，
訓練時用 reparameterization trick 取樣 z 餵給 decoder，eval 時
（model.eval() 把 self.training 設成 False）reparameterize() 直接
回傳 mu，讓推論階段的 latent 跟其他版本一樣是決定性的、每個 patch 唯一。

loss 多一項 KL(q(z|x) || N(0,I))，用 BETA 加權：這是把 latent 往一個
連續、以原點為中心的先驗拉的正規化力，動機跟這個題目直接相關——
「離群 = latent 上的距離」這件事在一個被 KL 規範過、connected 的空間裡
更站得住腳，不必擔心 latent 出現訓練資料沒覆蓋到的斷裂區域。

已知風險（VAE 的老問題，這裡不例外）：
  * posterior collapse：如果 BETA 太大，KL 會把 logvar 一路推向 0、
    mu 推向 0，q(z|x) 塌縮成先驗本身，latent 對輸入不再敏感。
    train.py 的 BETA=0.01 是拿 [0.01, 0.003, 0.001, 0.0003, 0.0001]
    掃過一輪選出來的：這個值下 posterior std 平均 ~0.17（沒有塌縮到
    1.0），deviance 也沒有比幾乎不加 KL 的版本差多少，是重建品質
    與正規化力道的折衷。train.py 最後會印出 logvar 的統計量，
    std 普遍逼近 1 就是塌縮的訊號，要調的話這是第一個該動的地方。
  * mu/logvar 共用同一段 MLP trunk 特徵（分頭只有最後兩個 Linear不同），
    這是刻意省參數，不是說兩者一定要高度相關。
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
        """idx 是 patch 編號的 tensor，回傳 (B,N_CAT) 的整包類別 count 向量。"""
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
        mu, logvar = self.fc_mu(h), self.fc_logvar(h)
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
