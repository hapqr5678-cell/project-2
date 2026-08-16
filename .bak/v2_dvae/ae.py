"""v2_dvae：Dirichlet VAE，把 v2_vae 的 latent 先驗 N(0,I) 換成
Dir(alpha)，其餘（encoder trunk、decoder、Poisson NLL）跟 v2_vae 完全
一樣，唯一的變因是 latent 的先驗形狀。

動機：v2_vae 的 latent 是不受限的實數向量，兩個維度本身沒有意義，
只能靠事後畫圖去猜；v2_gvae 用 K 個高斯峰做硬分群，一個 patch 只能屬於
一個峰。Dirichlet 先驗給的是第三種東西——latent 直接是一個**成分比例**

    theta ~ Dir(alpha)          theta_k >= 0, sum_k theta_k = 1
    x     ~ Poisson(exp(decoder(theta)))

theta 的第 k 維就是「這個 patch 有幾成像第 k 種 archetype」，是軟性的
混合權重而不是硬標籤。alpha < 1 時先驗把質量壓在 simplex 的角落，
鼓勵每個 patch 只用少數幾個 archetype（跟 LDA 的稀疏主題分布同一個道理）。

實作上的關鍵問題是 Dirichlet 沒有像高斯那樣簡單的可微分取樣。這裡用
Srivastava & Sutton (2017, ProdLDA) 的 **Laplace 近似**：在 softmax 基底
下把 Dir(alpha) 近似成一個對角高斯，於是

  encoder 出 (mu, logvar) -> 高斯 reparameterize -> softmax -> theta

取樣照樣可微，KL 也退回熟悉的高斯對高斯閉式解，只是目標從 N(0,I) 換成
Laplace 近似出來的 N(mu_p, diag(var_p))。對稱 alpha 下

    mu_p_k  = 0
    var_p_k = (1/alpha)·(1 - 1/K)

alpha 越小 var_p 越大，logit 空間的先驗越寬，softmax 後就越靠近角落
（越稀疏）。這就是「稀疏先驗」在這個參數化下的具體長相。

已知風險：
  * archetype collapse：alpha 太小或 BETA 太小時，有些 archetype 會完全
    沒有 patch 用到（平均 theta_k 接近 0）。train.py 最後會印每個
    archetype 的平均 theta 與硬分群佔比，這是第一個該看的診斷。
  * theta 在 simplex 上只有 K-1 個自由度，而且總和恆為 1，「這個 patch
    有多少 POI」這件事沒辦法直接寫在 latent 的長度上（v2_vae 可以）。
    密度資訊只能靠 decoder 從比例的細微差異裡重建出來，重建品質因此
    輸給同自由度的 v2_vae——K=3 時 Poisson deviance 0.705 對 v2_vae 的
    0.628，差約 12%。這是換先驗要付的代價，不是 bug。
    也因為自由度是 K-1，要跟 latent_dim=2 的 v2_vae / v2_gvae 公平比較，
    K 必須設 3（K=2 只剩一個自由度，等於一維 latent）。
  * Laplace 近似在 alpha 很小（<0.05）時會失準，softmax 前的高斯尾巴
    跟真正的 Dirichlet 差很多。要更小的 alpha 就得換 Gamma/Weibull 那套
    reparameterization，不是這一版的範圍。
  * decoder 保持跟 v2_vae 同一個 MLP，是為了「只改先驗」這個對照條件。
    如果要 archetype 直接可讀（log lam = log(theta @ B)），那是 ProdLDA
    式的線性 decoder，屬於下一個變體。
"""

import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(f"{os.path.dirname(__file__)}/../.."))
from config.dataset import N_CAT, HALF_WIDTH  # noqa: E402,F401

HIDDEN = 64
LOGVAR_MIN, LOGVAR_MAX = -8.0, 4.0   # encoder logvar 的夾擠範圍，避免 exp 爆掉


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


class DirVAE(nn.Module):
    """encoder trunk / decoder 跟 v2_vae 相同，差別在 encoder 出口的
    (mu, logvar) 是「softmax 前的 logit 空間」，經過 softmax 後才是餵給
    decoder 的 theta；先驗也從 N(0,I) 換成 Dir(alpha) 的 Laplace 近似。

    k：archetype 數，也就是 simplex 的維度（有效自由度是 k-1，所以要跟
       latent_dim=2 的 v2_vae 比就取 k=3）。
    alpha：對稱 Dirichlet 的濃度參數，<1 為稀疏、=1 為 simplex 上均勻。
    """

    def __init__(self, k=3, alpha=1.0 / 3):
        super().__init__()
        self.k = k
        self.alpha = float(alpha)
        self.trunk = nn.Sequential(
            nn.Linear(N_CAT, HIDDEN),
            nn.GELU(),
            nn.Linear(HIDDEN, HIDDEN),
            nn.GELU(),
        )
        self.fc_mu = nn.Linear(HIDDEN, k)
        self.fc_logvar = nn.Linear(HIDDEN, k)
        self.decoder = nn.Sequential(
            nn.Linear(k, HIDDEN),
            nn.GELU(),
            nn.Linear(HIDDEN, HIDDEN),
            nn.GELU(),
            nn.Linear(HIDDEN, N_CAT),   # 輸出是 log λ
        )
        # Dir(alpha) 在 softmax 基底下的 Laplace 近似，對稱 alpha 的閉式解。
        # 存成 buffer：跟著 .to(device) 走，但不是要學的參數（先驗是固定的）。
        mu_p = torch.zeros(k)
        var_p = torch.full((k,), (1.0 / self.alpha) * (1.0 - 1.0 / k))
        self.register_buffer("mu_p", mu_p)
        self.register_buffer("var_p", var_p)

    def reparameterize(self, mu, logvar):
        """訓練時在 logit 空間取樣、eval 時直接回傳 mu，讓推論的 theta 唯一。
        回傳 (B,K) 的 logit，還沒過 softmax。
        """
        if not self.training:
            return mu
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def forward(self, x):
        """x：(B,N_CAT) 的 count 向量。

        回傳 (mu, logvar, theta, log_lam)，形狀依序 (B,K)、(B,K)、(B,K)、
        (B,N_CAT)。mu/logvar 是 logit 空間的後驗參數，theta 是 softmax 後
        落在 simplex 上的成分比例（每列總和 1），log_lam 是 decoder 輸出。
        """
        h = self.trunk(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h).clamp(LOGVAR_MIN, LOGVAR_MAX)
        theta = torch.softmax(self.reparameterize(mu, logvar), dim=1)
        return mu, logvar, theta, self.decoder(theta)


def poisson_nll(log_lam, x):
    """Poisson NLL，省略跟模型無關的 log(y!) 常數項，回傳每個 patch 一個數字。"""
    cell = torch.exp(log_lam) - x * log_lam
    return cell.mean(dim=1)


def poisson_deviance(log_lam, x):
    """Poisson deviance：2·(y·log(y/λ) - (y - λ))，越小越好、有下界，可跨 patch 比較。"""
    lam = torch.exp(log_lam)
    cell = 2 * (torch.xlogy(x, x) - x * log_lam - x + lam)
    return cell.mean(dim=1)


def dirichlet_kl(model, mu, logvar):
    """KL(q(z|x) || Laplace近似的 Dir(alpha))，兩邊都是 logit 空間的對角
    高斯，所以是標準的高斯對高斯閉式解，回傳 (B,)。

    model：DirVAE，提供 mu_p / var_p 兩個先驗 buffer。
    mu / logvar：encoder 輸出的 (B,K)。

    KL = 0.5·sum_k [ (sigma_k^2 + (mu_k - mu_p_k)^2)/var_p_k - 1
                     + log(var_p_k) - logvar_k ]
    """
    d = mu - model.mu_p.unsqueeze(0)                    # (B,K)
    return 0.5 * ((logvar.exp() + d.pow(2)) / model.var_p
                  - 1.0 + model.var_p.log() - logvar).sum(dim=1)


def log_p_theta(model, theta, eps=1e-8):
    """theta 在真正的 Dir(alpha) 先驗下的 log 密度，回傳 (B,)。

    model：DirVAE，提供 k / alpha。
    theta：(B,K) 的 simplex 向量。
    eps：log 前的下夾，避免 theta 有 0 分量時變成 -inf。

    log Dir = logGamma(K·alpha) - K·logGamma(alpha) + (alpha-1)·sum_k log theta_k

    注意方向：alpha < 1 時密度的高峰在 simplex 的角落，所以「log 密度低」
    代表的是「這個 patch 混得很均勻、沒有明顯的主 archetype」，而不是
    v2_gvae 那種「離所有 cluster 都遠」。當離群分數用的時候要記得這件事。
    """
    a = model.alpha
    const = (float(torch.lgamma(torch.tensor(model.k * a)))
             - model.k * float(torch.lgamma(torch.tensor(a))))
    return const + (a - 1.0) * theta.clamp_min(eps).log().sum(dim=1)
