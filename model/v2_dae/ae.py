"""v2_dae：v2_ae 的 denoising 變體。模型架構、decoder、loss 跟 v2_ae 完全一樣，
唯一差別是訓練時餵給 encoder 的不是乾淨的 (N_CAT,) count 向量，而是被隨機
破壞過的版本，重建目標仍然是乾淨的原始 count。

為什麼要加噪：v2_ae 這種 in = out 的 AE 只要 latent 維度夠、資料夠簡單，
就有機會把輸入近似「抄」過去——尤其這裡的輸入是 10 維左右的 count 向量，
總量本身就是一個很強、很好抄的訊號，latent 很容易退化成「密度計」。
DAE 的作法是強迫模型從一份殘缺的觀測去猜完整的分布：能做到這件事，
latent 就必須抓住「這一區的類別組成長什麼樣」這種跨類別的結構，而不是
單看某一維的數字。這對「POI 飽和度」的假設是加分的——真實世界的 POI
資料本來就是不完整的抽樣（有些店沒被收錄、有些類別覆蓋率低），
DAE 的破壞過程剛好模擬了這件事。

破壞方式有兩種，用 NOISE_MODE 切換：
  "thinning"  binomial thinning：每個 POI 以 1-NOISE_P 的機率被保留，
              x_tilde ~ Binomial(x, 1-NOISE_P)。這是 count 資料最自然的
              破壞方式（Poisson 被 thinning 之後還是 Poisson），等同於
              「這一區的 POI 只被收錄到一部分」。
  "mask"      整個類別歸零：每一類以 NOISE_P 的機率整維被抹成 0，
              等同於「某一類的資料整批缺失」，逼模型從其他類別去推。

兩種都會把破壞後的向量除以 1-NOISE_P 再送進 encoder（跟 dropout 的
inverted scaling 同一個道理）：不補這個尺度，訓練時 encoder 看到的
平均總量只有目標的 1-NOISE_P 倍，模型會學成「輸入乘上 1/(1-NOISE_P)」，
推論時餵乾淨資料就會系統性高估 λ。補回來之後訓練/推論的輸入尺度一致，
噪聲只留下它該留下的東西——變異，而不是偏移。
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


def corrupt(x, p, mode="thinning", generator=None):
    """把乾淨的 count 向量破壞成 DAE 的輸入。

    x：(B,N_CAT) 的乾淨 count 向量（float，值是非負整數）。
    p：破壞強度 ∈[0,1)。thinning 是每個 POI 被丟掉的機率，mask 是每一類
       整維被抹成 0 的機率。p=0 時直接回傳 x（等價於退化成 v2_ae）。
    mode："thinning" 或 "mask"，意義見模組 docstring。
    generator：torch.Generator，給定就用它抽亂數，讓每個 epoch 的破壞可重現。

    回傳跟 x 同 shape 的 (B,N_CAT) tensor，已經除以 1-p 做過尺度補償，
    期望值等於 x（thinning 嚴格成立，mask 對每一維的期望也成立），
    所以可以直接餵進跟推論時同一個 encoder。
    """
    if p <= 0:
        return x
    keep = 1.0 - p
    if mode == "thinning":
        # binomial 沒有吃 generator 的版本，用 x 個 Bernoulli 的和等價實作：
        # 每個 patch 的每一類最多 max_c 個 POI，各自擲一次銅板再依真實 count 遮掉
        max_c = int(x.max().item())
        if max_c == 0:
            return x
        coin = torch.rand(x.shape + (max_c,), generator=generator,
                          device=x.device) < keep
        alive = torch.arange(max_c, device=x.device) < x.unsqueeze(-1)
        noisy = (coin & alive).sum(dim=-1).float()
    elif mode == "mask":
        m = (torch.rand(x.shape, generator=generator, device=x.device) < keep)
        noisy = x * m.float()
    else:
        raise ValueError(f"未知的 mode：{mode}")
    return noisy / keep


class MLPAE(nn.Module):
    """in = out 的純 MLP AutoEncoder：(N_CAT,) 向量進，(N_CAT,) 的 log λ 出。

    跟 v2_ae 的 MLPAE 同架構、同權重形狀（checkpoint 可互換），
    denoising 完全發生在資料端（見 corrupt()），模型本身不需要知道。
    """

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
        """x 是 (B,N_CAT) 的 count 向量（訓練時是加噪版、推論時是乾淨版），
        回傳 (z, log_lam)：z 是 (B,latent_dim) 的 latent，log_lam 是 (B,N_CAT)。
        """
        z = self.encoder(x)
        return z, self.decoder(z)

    def encode(self, x):
        """x 是 (B,N_CAT) 的 count 向量，回傳 (B,latent_dim) 的 z——只跑 encoder。"""
        return self.encoder(x)


def poisson_nll(log_lam, x):
    """Poisson NLL，省略跟模型無關的 log(y!) 常數項，回傳每個 patch 一個數字。"""
    cell = torch.exp(log_lam) - x * log_lam
    return cell.mean(dim=1)


def poisson_deviance(log_lam, x):
    """Poisson deviance：2·(y·log(y/λ) - (y - λ))，越小越好、有下界，可跨 patch 比較。"""
    lam = torch.exp(log_lam)
    cell = 2 * (torch.xlogy(x, x) - x * log_lam - x + lam)
    return cell.mean(dim=1)
