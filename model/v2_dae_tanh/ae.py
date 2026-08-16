"""v2_dae_tanh：v2_dae 的 tanh 變體，等於「denoising + latent 被 tanh 夾住」
兩個變因同時開。模型架構、decoder、loss 跟 v2_dae 一樣，唯一差異是 encoder
最後一層接 Tanh()，訓練時就把 latent 夾在 (-1,1)；破壞輸入的做法完全沿用
v2_dae 的 corrupt()。

為什麼要有這一版：v2_dae 的加噪是在「輸入端」逼 encoder 學跨類別結構，
v2_ae_tanh 的 tanh 是在「latent 端」限制表達範圍，兩者是不同層面的正則化。
單獨看 v2_dae vs v2_ae、v2_ae_tanh vs v2_ae 各自的效果之後，這一版用來看
兩個一起加會不會互相抵銷——tanh 的飽和梯度本來就會壓縮離群 patch 的訊號，
而 DAE 又刻意把輸入弄殘缺，兩邊都在削減可用資訊，latent 有可能直接塌成
一團密度計；反過來也可能 DAE 逼出來的類別組成結構剛好把 (-1,1)² 這個
有限的盒子填得比 v2_ae_tanh 更均勻。這是要實測的。

破壞方式有兩種，用 NOISE_MODE 切換（意義同 v2_dae）：
  "thinning"  binomial thinning：每個 POI 以 1-NOISE_P 的機率被保留，
              x_tilde ~ Binomial(x, 1-NOISE_P)，等同於「這一區的 POI
              只被收錄到一部分」。
  "mask"      整個類別歸零：每一類以 NOISE_P 的機率整維被抹成 0，
              等同於「某一類的資料整批缺失」，逼模型從其他類別去推。

兩種都會把破壞後的向量除以 1-NOISE_P 再送進 encoder（inverted scaling），
讓訓練/推論的輸入尺度一致，噪聲只留下變異、不留下偏移。
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
       整維被抹成 0 的機率。p=0 時直接回傳 x（等價於退化成 v2_ae_tanh）。
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
    """跟 v2_dae 的 MLPAE 唯一差異：latent 前多接一個 Tanh()，把 z 夾在 (-1,1)。

    denoising 完全發生在資料端（見 corrupt()），模型本身不需要知道，
    所以權重形狀跟 v2_ae_tanh 完全一致（checkpoint 可互換）。
    """

    def __init__(self, latent_dim=2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(N_CAT, HIDDEN),
            nn.GELU(),
            nn.Linear(HIDDEN, HIDDEN),
            nn.GELU(),
            nn.Linear(HIDDEN, latent_dim),
            nn.Tanh(),   # <- 跟 v2_dae 唯一的差異：訓練時就把 latent 夾在 (-1,1)
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
