"""v2_dvae_fsce：在 v2_dvae 的架構跟 (Poisson NLL + Dirichlet KL) 之外，
額外加一個 FSCE（fuzzy set cross entropy，UMAP 訓練用的那個 loss）項，
作用在 **theta** 上。其餘跟 v2_dvae 一字不差，唯一的變因是多了 FSCE。

為什麼作用在 theta 而不是 mu（其他 fsce 版都是作用在 mu）：softmax 對
「所有 logit 同加一個常數」是不變的，mu 跟 mu + c·1 會給出完全一樣的
theta。所以 logit 空間的歐氏距離有一整個方向是模型看不見的，FSCE 拿它
當座標會有一部分推力推在空轉的維度上。theta 才是 eval 時唯一代表位置的
座標（也是三角圖上畫的東西），FSCE 該作用在它身上。

已知風險（FSCE × simplex 特有，其他 fsce 版沒有）：
  * **UMAP 核函數在 simplex 上會飽和。** find_ab_params(spread=1.0,
    min_dist=0.1) 校出的 a≈1.577、b≈0.895，是給無界嵌入空間用的；
    但 theta 被關在三角形裡，兩點最遠只有 sqrt(2)（兩個頂點之間）。
    代入 q = 1/(1+a·d^(2b))，即使是最遠的一對點也只降到 q≈0.25，
    負樣本項 -log(1-q) 永遠壓不到 0。後果是 FSCE 的斥力會持續存在、
    無法被滿足，唯一能降低它的方向就是把所有點推向三角形的角落。
    這跟稀疏 Dirichlet 先驗想要的方向剛好一致，所以 FSCE 在這一版
    可能不只是加拓撲約束，還會順便放大先驗的稀疏效果——兩件事混在
    一起，解讀結果時不能只歸因於其中一個。
    （v2_vae_fsce 的 docstring 提過同一個張力，那邊是拿 tanh 版當對照；
    這裡沒有「無界版」可以當對照，因為 simplex 本質上就是有界的。）
  * theta 的座標尺度固定（三角形直徑 sqrt(2)），LAMBDA_FSCE 不能直接
    沿用 v2_vae_fsce 的 0.5 就假設力道相當。train.py 會印 FSCE 那一項
    的絕對值，跟 recon 的量級對照著看。

以下是 v2_dvae 原本的說明，架構部分完全沿用。
---
v2_dvae：Dirichlet VAE，把 v2_vae 的 latent 先驗 N(0,I) 換成
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
from sklearn.neighbors import NearestNeighbors
from umap.umap_ import find_ab_params, fuzzy_simplicial_set

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

    def encode(self, x):
        """x 是 (B,N_CAT) 的 count 向量，回傳 (B,K) 的 theta = softmax(mu)——
        跟 forward() 算出來的 theta 走同一條路徑，只是不取樣、不接 decoder。
        給 FSCE loss 用：不取樣是刻意的，取樣噪聲交給 logvar 負責，跟座標
        範圍/拓撲關係無關。
        """
        return torch.softmax(self.fc_mu(self.trunk(x)), dim=1)

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


def build_fsce_graph(x, n_neighbors=15, metric="cosine"):
    """x 是高維空間的特徵矩陣 (N,D)（這裡傳 log1p(count)），在上面建一次 UMAP
    的 fuzzy simplicial set。回傳 edge_i、edge_j：邊兩端的 patch 編號 (E,)
    LongTensor；edge_w：這條邊在高維空間的模糊隸屬度 (E,)∈(0,1] FloatTensor，
    當作 FSCE loss 裡的正樣本權重；a、b：UMAP 低維核函數 1/(1+a·d^(2b)) 的形狀
    參數，由 find_ab_params(spread=1.0, min_dist=0.1) 算出。

    跟 v2_vae_fsce 用同一組設定（n_neighbors=15、cosine），圖只建一次、
    整個訓練共用，所以兩版之間高維鄰接關係是同一張。
    """
    knn = NearestNeighbors(n_neighbors=n_neighbors, metric=metric).fit(x)
    knn_dists, knn_idx = knn.kneighbors(x)
    graph, _, _ = fuzzy_simplicial_set(
        x, n_neighbors=n_neighbors, random_state=0, metric=metric,
        knn_indices=knn_idx, knn_dists=knn_dists,
    )
    graph = graph.tocoo()
    edge_i = torch.from_numpy(graph.row).long()
    edge_j = torch.from_numpy(graph.col).long()
    edge_w = torch.from_numpy(graph.data).float()
    a, b = find_ab_params(spread=1.0, min_dist=0.1)
    return edge_i, edge_j, edge_w, a, b


def fsce_loss(z_i, z_j, w, a, b, eps=1e-4):
    """FSCE（fuzzy set cross entropy）：z_i、z_j 是一批 pair 的 theta 座標
    (P,K)，w 是這對點在高維空間的模糊隸屬度 (P,)——正樣本填 build_fsce_graph()
    算出的 edge_w，負樣本填 0。a、b 是 UMAP 核函數參數。
    回傳每個 pair 一個數字：q（低維距離換算出的相似度）跟 w 差越多，這個數字
    越大，逼著 encoder 把 w 大的點在 simplex 上拉近、w=0 的點推遠。

    d2 用 eps 墊底：b<1 時 d2^b 在 d2=0 處的梯度是無限大，兩個不同 patch 剛好
    被 encoder 映到同一點（或負樣本剛好抽到 i==j）並不無法排除，沒墊底會讓
    backward 炸成 NaN。

    注意 simplex 上 d2 最大只有 2（兩個頂點之間），所以 q 的下界是
    1/(1+a·2^b)≈0.25，負樣本項壓不到 0——見模組 docstring 的風險說明。
    """
    d2 = (z_i - z_j).pow(2).sum(dim=1).clamp(min=eps)
    q = (1.0 + a * d2.pow(b)).reciprocal().clamp(eps, 1 - eps)
    return -(w * q.log() + (1 - w) * (1 - q).log())
