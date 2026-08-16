"""v2_denoise_vae：v2_vae 的 denoising 變體。模型架構（trunk + mu/logvar head +
decoder）、Poisson NLL、KL 全部跟 v2_vae 一模一樣，唯一差別是訓練時餵給
encoder 的不是乾淨的 (N_CAT,) count 向量，而是被 corrupt() 破壞過的版本，
重建目標仍然是乾淨的原始 count——也就是把 v2_dae 的加噪手法搬到 VAE 上。

為什麼把兩者疊在一起：VAE 的 KL 和 DAE 的加噪都是正則化，但管的事情不同。
KL 管的是 latent 的「形狀」——把 q(z|x) 往連續、以原點為中心的先驗拉，
讓 latent 空間沒有訓練資料沒覆蓋到的斷裂區域，「離群 = latent 上的距離」
才站得住腳。加噪管的是 latent 的「內容」——強迫模型從一份殘缺的觀測去猜
完整的分布，latent 就不能只當個「密度計」（總量本身是很好抄的訊號），
必須抓住跨類別的組成結構。這一版問的問題是：兩種正則化是互補還是打架。

要注意的是加噪之後 q(z|x_tilde) 的隨機性有兩個來源：破壞過程本身，加上
reparameterization 取樣。同一個 patch 在不同 epoch 會看到不同的殘缺版本，
encoder 學到的等於是「對破壞取期望後的 posterior」，這會讓 posterior
比 v2_vae 寬一些（logvar 偏大）是預期內的，不要跟 posterior collapse 搞混
——collapse 的訊號是 std 普遍逼近 1 而且 mu 縮到 0，train.py 最後會印。

破壞方式有兩種，用 NOISE_MODE 切換（意義同 v2_dae）：
  "thinning"  binomial thinning：每個 POI 以 1-NOISE_P 的機率被保留，
              x_tilde ~ Binomial(x, 1-NOISE_P)。這是 count 資料最自然的
              破壞方式，等同於「這一區的 POI 只被收錄到一部分」。
  "mask"      整個類別歸零：每一類以 NOISE_P 的機率整維被抹成 0，
              等同於「某一類的資料整批缺失」，逼模型從其他類別去推。
兩種都會把破壞後的向量除以 1-NOISE_P 再送進 encoder（inverted scaling），
讓訓練/推論的輸入尺度一致，噪聲只留下變異、不留下偏移。

推論階段（model.eval()）不加噪、reparameterize() 直接回傳 mu，latent 是
決定性的、每個 patch 唯一，跟其他版本可以直接比較。
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
    """把乾淨的 count 向量破壞成 denoising 的輸入。

    x：(B,N_CAT) 的乾淨 count 向量（float，值是非負整數）。
    p：破壞強度 ∈[0,1)。thinning 是每個 POI 被丟掉的機率，mask 是每一類
       整維被抹成 0 的機率。p=0 時直接回傳 x（等價於退化成 v2_vae）。
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


class VAE(nn.Module):
    """跟 v2_vae 的 VAE 同架構、同權重形狀（checkpoint 可互換），
    denoising 完全發生在資料端（見 corrupt()），模型本身不需要知道。
    """

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
        """mu、logvar 都是 (B,latent_dim)；訓練時回傳取樣的 z，
        eval 時直接回傳 mu，讓推論的 latent 唯一。
        """
        if not self.training:
            return mu
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def forward(self, x):
        """x 是 (B,N_CAT) 的 count 向量（訓練時是加噪版、推論時是乾淨版），
        回傳 (mu, logvar, z, log_lam)：前三個是 (B,latent_dim)，
        log_lam 是 (B,N_CAT) 的 log λ。
        """
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
