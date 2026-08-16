"""v2_ae：v2 系列的基準組。輸入不再是空間矩陣，而是整個 patch 聚合成一個
(N_CAT,) 的類別 count 向量，MLP AutoEncoder + Poisson NLL。

v0/v1 系列把 patch 內的 POI 依 (x,y) binning 成 GRID×GRID 的矩陣，
「格子」本身帶有空間資訊，latent 有機會偷學到形狀/朝向。v2 系列刻意拿掉
這個概念：輸入只是半徑 HALF_WIDTH 圓內「每一類有幾個」的一維向量，
例如 [餐飲:3, 零售:2, 其他:0, ...]——沒有 x,y、沒有 cell、沒有卷積。
latent 因此只能從純粹的類別組成比例與總量去判斷飽和度，跟這個題目
「城市 POI 飽和度」的假設更貼近：我們關心的是「這裡的類別組合合不合理」，
不是「這裡的形狀合不合理」。

decoder 輸出的是 log λ（跟 v0_poisson_nll 一樣的理由：λ 必須為正、
對 log λ 的梯度是 λ-y，數值溫和），loss 是每個 patch N_CAT 維的
Poisson NLL，不再有「圓內/圓外」的區別——本來就沒有格子可以區分。

v2_ae / v2_perceiver / v2_vae 三個版本的 decoder、loss 完全共用同一份定義；
差別只在 encoder 怎麼把「一包 POI」壓成 2 維 latent：
  v2_ae        直接吃已經算好的 (N_CAT,) 聚合向量，MLP
  v2_perceiver 吃還沒聚合的原始 POI 類別 token 集合，靠 cross-attention 自己聚合
  v2_vae       跟 v2_ae 一樣吃聚合向量，但 encoder 出口機率化(mu/logvar)
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


class MLPAE(nn.Module):
    """in = out 的純 MLP AutoEncoder：(N_CAT,) 向量進，(N_CAT,) 的 log λ 出。"""

    def __init__(self, latent_dim=2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(N_CAT, HIDDEN),
            nn.GELU(),
            nn.Linear(HIDDEN, HIDDEN),
            nn.GELU(),
            nn.Linear(HIDDEN, latent_dim),   # latent 前不接激活，離群值才不會被壓回來
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, HIDDEN),
            nn.GELU(),
            nn.Linear(HIDDEN, HIDDEN),
            nn.GELU(),
            nn.Linear(HIDDEN, N_CAT),   # 輸出是 log λ
        )

    def forward(self, x):
        z = self.encoder(x)
        return z, self.decoder(z)


def poisson_nll(log_lam, x):
    """Poisson NLL，省略跟模型無關的 log(y!) 常數項，回傳每個 patch 一個數字。"""
    cell = torch.exp(log_lam) - x * log_lam
    return cell.mean(dim=1)


def poisson_deviance(log_lam, x):
    """Poisson deviance：2·(y·log(y/λ) - (y - λ))，越小越好、有下界，可跨 patch 比較。"""
    lam = torch.exp(log_lam)
    cell = 2 * (torch.xlogy(x, x) - x * log_lam - x + lam)
    return cell.mean(dim=1)
