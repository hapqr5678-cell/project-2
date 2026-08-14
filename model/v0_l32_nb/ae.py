"""v0_l32_nb：把 v0_poisson_nll 的 Poisson 換成 Negative Binomial，其他完全不動。

這一版存在的目的是「排除法」，不是期待它變好。

動機來自實測（scratch 的 count 診斷）：把圓內每格每類的 count 拿出來，
按 v0_poisson_nll 給的 λ 分桶，觀測的 var/mean 是

    λ<0.002 (50% 樣本)  1.03      λ<0.15  1.22
    λ<0.016 (90% 樣本)  1.07      λ<0.35  1.48      λ>=0.43  1.57

也就是說 overdispersion 確實存在，但集中在 λ 最大的 0.1% 格子
（一棟樓塞 20 幾間同類店那種），承載 99% 樣本的低密度區幾乎就是 Poisson。
NB 多一個 dispersion 參數，理論上該接住那條尾巴。

同一份診斷也顯示 **不需要 zero-inflation**：低 λ 桶的觀測零比例
（99.914%）比 Poisson 預測的（99.923%）還低，是 zero-deflation。
高 λ 桶的多餘零是 overdispersion 的副產品，NB 自己就會產生。
所以這裡只做 NB 不做 ZINB——DCA 那個 ZI 分支是為了 scRNA-seq 的技術性
dropout（基因有表達卻沒被捕捉），POI 的 0 是「那裡真的沒有店」，沒有這個機制。

參數化：每個類別一個 dispersion（共 N_CAT 個 scalar），不隨位置變。
    y ~ NB(mu, theta),  Var = mu + mu^2/theta,  theta -> inf 時退回 Poisson
theta 用 log 參數化保證為正，初始 log_theta=0（theta=1，相當寬鬆），
訓練後印出來看它跑去哪：如果各類 theta 都被推到很大，等於模型自己說
「Poisson 就夠了」，那 v0_poisson_nll 的結論不用改。

已知風險：
  * 極稀疏（99.2% 的格子是 0）時 theta 幾乎沒有 likelihood 訊號可以擬合，
    梯度主要來自那 0.1% 的高密度格子。theta 的估計會很不穩，換 seed 可能差很多。
  * 存進 latents.npz 的 err 是 NB deviance，跟 v0 的 MSE、
    v0_poisson_nll 的 Poisson deviance 都不能比大小，只能各自比排序。
"""

import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(f"{os.path.dirname(__file__)}/../.."))
from config.dataset import CELL, GRID, N_CAT, HALF_WIDTH  # noqa: E402,F401

IN_CH = N_CAT



class Patches:
    """稀疏點列表；render() 才展開成稠密矩陣。跟 v0_poisson_nll 相同。"""

    def __init__(self, path):
        d = np.load(path)
        self.dx = torch.from_numpy(d["dx"])
        self.dy = torch.from_numpy(d["dy"])
        self.cat = torch.from_numpy(d["cat"].astype(np.int64))
        self.offsets = torch.from_numpy(d["offsets"])
        self.n_poi = d["n_poi"]
        self.lat = d["center_lat"]
        self.lon = d["center_lon"]
        self.n = len(self.n_poi)

    def render(self, idx, rotate):
        """idx 是 patch 編號的 tensor，回傳 (B,10,40,40) 的原始 count 矩陣。"""
        b = len(idx)
        starts, ends = self.offsets[idx], self.offsets[idx + 1]
        lens = ends - starts
        pos = torch.repeat_interleave(starts, lens) + \
            torch.arange(int(lens.sum())) - torch.repeat_interleave(
                torch.cumsum(lens, 0) - lens, lens)
        owner = torch.repeat_interleave(torch.arange(b), lens)

        dx, dy, cat = self.dx[pos], self.dy[pos], self.cat[pos]
        if rotate:
            theta = torch.rand(b) * (2 * torch.pi)
            cos, sin = torch.cos(theta)[owner], torch.sin(theta)[owner]
            dx, dy = dx * cos - dy * sin, dx * sin + dy * cos
            flip = (torch.rand(b) < 0.5)[owner]
            dx = torch.where(flip, -dx, dx)

        ix = (dx / CELL + GRID / 2).floor().long().clamp(0, GRID - 1)
        iy = (dy / CELL + GRID / 2).floor().long().clamp(0, GRID - 1)

        flat = ((owner * N_CAT + cat) * GRID + iy) * GRID + ix
        counts = torch.bincount(flat, minlength=b * N_CAT * GRID * GRID)
        return counts.view(b, N_CAT, GRID, GRID).float()


def block(cin, cout, transpose=False):
    conv = (nn.ConvTranspose2d(cin, cout, 4, 2, 1) if transpose
            else nn.Conv2d(cin, cout, 3, 2, 1))
    return nn.Sequential(conv, nn.GroupNorm(8, cout), nn.GELU())


class ConvAE(nn.Module):
    """架構跟 v0_poisson_nll 相同，多一組 per-category 的 log theta。

    theta 不由 decoder 產生而是獨立的 nn.Parameter：它描述的是
    「這一類 POI 在空間上的聚集傾向」這種全域性質，不該隨 patch 變動，
    也避免 decoder 用 theta 去記憶個別 patch。
    """

    def __init__(self, latent_dim=32):
        super().__init__()
        s1 = (GRID + 2*1 - 3) // 2 + 1
        s2 = (s1 + 2*1 - 3) // 2 + 1
        s3 = (s2 + 2*1 - 3) // 2 + 1
        self.enc_size = s3
        self.log_theta = nn.Parameter(torch.zeros(N_CAT))
        self.encoder = nn.Sequential(
            block(IN_CH, 32),   # 40 -> 20
            block(32, 64),      # 20 -> 10
            block(64, 128),     # 10 -> 5
            nn.Flatten(),
            nn.Linear(128 * s3 * s3, 256),
            nn.GELU(),
            nn.Linear(256, latent_dim),   # latent 前不接激活，離群值才不會被壓回來
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.GELU(),
            nn.Linear(256, 128 * s3 * s3),
            nn.Unflatten(1, (128, s3, s3)),
            block(128, 64, transpose=True),   # 5 -> 10
            block(64, 32, transpose=True),    # 10 -> 20
            nn.ConvTranspose2d(32, IN_CH, 4, 2, 1),   # 20 -> 40，輸出是 log mu
        )

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        if out.shape[-1] != x.shape[-1] or out.shape[-2] != x.shape[-2]:
            import torch.nn.functional as F
            out = F.interpolate(out, size=(x.shape[-2], x.shape[-1]), mode='bilinear', align_corners=False)
        return z, out

    def log_theta_map(self):
        """(1,N_CAT,1,1)，broadcast 到整張圖。"""
        return self.log_theta.view(1, -1, 1, 1)


def nb_nll(log_mu, log_theta, x):
    """NB negative log-likelihood，省略跟模型無關的 -lgamma(y+1) 常數項。

    每格 = -[lgamma(y+th) - lgamma(th) + th*(log th - lse) + y*(log mu - lse)]
    其中 lse = log(th + mu)，用 logaddexp 算以免 mu 或 th 很大時溢位。
    只加總圓內的格子再除以 N_VALID，所以 patch 之間可以直接比較。
    """
    th = torch.exp(log_theta)
    lse = torch.logaddexp(log_theta, log_mu)
    ll = (torch.lgamma(x + th) - torch.lgamma(th)
          + th * (log_theta - lse) + x * (log_mu - lse))
    return -ll.mean(dim=(1, 2, 3))


def nb_deviance(log_mu, log_theta, x):
    """NB deviance：2*[y*log(y/mu) - (y+th)*log((y+th)/(mu+th))]，y=0 時第一項為 0。

    跟 Poisson deviance 同樣的用意：減掉 saturated model（mu=y）的
    log-likelihood，所以完美重建為 0、永遠非負，後面 robust distance
    與各張圖的語義不用改。y=0 時整式退化成 2*th*(lse - log th) > 0。
    """
    th = torch.exp(log_theta)
    lse = torch.logaddexp(log_theta, log_mu)
    cell = 2 * (torch.xlogy(x, x) - x * log_mu
                - (x + th) * (torch.log(x + th) - lse))
    return cell.mean(dim=(1, 2, 3))
