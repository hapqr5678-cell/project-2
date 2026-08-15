"""v3_multinomial：把「總量」從 latent 結構性移除，latent 只負責「類別組成」。

## 為什麼要有這一版

v2 系列（v2_ae / v2_perceiver / v2_vae）的 latent 幾乎被密度佔滿：
在 overture 9770 個 patch 上實測，v2_ae 的 latent 對 log(POI 總數) 的
kNN R² 高達 +0.972（z1 單軸相關就有 +0.906），2 維 latent 裡實質有
一整維在編碼「這裡有幾個 POI」，剩下的容量才輪到 13 類的組成比例。
但這個題目要判斷的是「這裡的類別組合合不合理」——總量本來就該是
另一個獨立的軸，不該跟組成擠在同一個 2 維空間裡搶容量。

做法沿用 v0_sizefactor 的思路，但改成 multinomial 分解的形式：

    P(x) = Poisson(n | 總量) × Multinomial(x | n, p(z))

也就是把「一個 patch 有幾個 POI」跟「這些 POI 怎麼分到各類別」拆成兩件事，
模型只學後者。實作上是兩處對稱的修改：

  encoder 端：輸入不是 raw count，而是 log1p(x / n × REF_SIZE)。
              除以 n 之後總量資訊就不在輸入裡了，encoder 想編碼密度也
              沒東西可編。乘回 REF_SIZE 只是把數值拉回跟 raw count
              同一個量級，避免第一層權重要處理 0~1 的小數。

  decoder 端：輸出接 log_softmax 得到 log p（保證 Σp = 1），再加上
              log n 這個 size factor 當 offset：log λ = log p + log n。
              於是 Σλ = n 恆成立——模型完全不必花容量去猜總量，
              總量是「給」它的。

## 跟 v2 的可比較性

loss 與評估指標都沿用 v2 的 poisson_nll / poisson_deviance，定義一字不差，
所以 v3 的 deviance 可以直接跟 v2_ae 對照（同一份 overture 資料上
v2_ae = 0.907、v3 = 0.737）。但要注意這個比較對 v2 是不公平的
（v2 的 decoder 得自己猜總量，v3 是白拿 log n），deviance 變好不算成就；
真正要看的是 analyze/latent_plot.py 印的「latent → log(POI 數) 的 kNN R²」
有沒有從 v2 的水準掉下來——實測 0.972 → 0.174，密度確實被移出去了。

## 已知風險

  * 除以 n 是有損的：兩個 patch 只要類別比例一樣，不管是 15 個 POI
    還是 85 個 POI，encoder 看到的輸入完全相同。這是刻意的，但也意味著
    「小 patch 的比例估計比大 patch 更嘈雜」這件事模型看不到——
    n=10 的 patch 算出的比例，本質上是 10 次抽樣的結果，變異很大，
    模型會把這些噪聲當成真實的組成差異去擬合。REF_SIZE 的選擇不能
    修正這件事，只有加大 HALF_WIDTH（提高每個 patch 的 POI 數）可以。
  * 飽和度的判定不能再用 robust distance：latent 不含密度，距離只代表
    「組成怪」。改用 analyze/saturation.py 的條件式 z-score，
    跟 v0_sizefactor 同一套邏輯。
"""

import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(f"{os.path.dirname(__file__)}/../.."))
from config.dataset import N_CAT, HALF_WIDTH  # noqa: E402,F401

HIDDEN = 64
REF_SIZE = 15.0   # 參考 patch 大小，本資料集的 POI 數中位數；只影響輸入的數值量級


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


def to_composition(x):
    """(B,N_CAT) raw count -> encoder 輸入 log1p(比例 × REF_SIZE)，總量資訊已移除。"""
    n = x.sum(dim=1, keepdim=True).clamp_min(1.0)
    return torch.log1p(x / n * REF_SIZE)


class MultinomialAE(nn.Module):
    """encoder 吃組成比例、decoder 出 log p，總量由外部的 size factor 補回來。"""

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
            nn.Linear(HIDDEN, N_CAT),   # 輸出是 logit，log_softmax 之後才是 log p
        )

    def forward(self, x):
        """x 是 raw count；回傳 (z, log_lam)，log_lam = log p + log n。"""
        n = x.sum(dim=1, keepdim=True).clamp_min(1.0)
        z = self.encoder(to_composition(x))
        log_p = F.log_softmax(self.decoder(z), dim=1)
        return z, log_p + torch.log(n)


def poisson_nll(log_lam, x):
    """Poisson NLL，省略跟模型無關的 log(y!) 常數項，回傳每個 patch 一個數字。

    Σλ = n 已由 log_softmax + log n 保證，所以 exp(log_lam).sum() 是常數 n，
    這個 loss 對 z 的梯度等價於 multinomial NLL 的梯度——用 Poisson 的形式
    寫只是為了讓 v2 / v3 的數字能放在同一張表上比。
    """
    cell = torch.exp(log_lam) - x * log_lam
    return cell.mean(dim=1)


def poisson_deviance(log_lam, x):
    """Poisson deviance：2·(y·log(y/λ) - (y - λ))，越小越好、有下界，可跨 patch 比較。"""
    lam = torch.exp(log_lam)
    cell = 2 * (torch.xlogy(x, x) - x * log_lam - x + lam)
    return cell.mean(dim=1)
