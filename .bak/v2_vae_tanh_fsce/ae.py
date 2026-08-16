"""v2_vae_tanh_fsce：在 v2_vae_tanh 的架構跟 (Poisson NLL + KL) 之外，額外加一
個 FSCE（fuzzy set cross entropy，UMAP 訓練用的那個 loss）項，作用在 mu 上
（不是取樣後的 z）——mu 是 eval 時唯一代表位置的座標，取樣噪聲交給 logvar
負責，跟座標範圍/拓撲關係無關，這跟 v2_vae_tanh 筆記裡「只有 mu 需要被夾住」
的邏輯一致。目的是實測 UMAP 的 loss 能不能把高維 kNN 鄰接關係直接拉進
mu 的距離，緩解 tanh 飽和把離群 patch 訊號壓縮的問題。

FSCE 的高維鄰接關係是用「整包 count 向量的 log1p、cosine 距離」kNN 建的
fuzzy simplicial set（跟 data/patch/umap_grid.py 的可行性檢查用同一種 metric），
只建一次、整個訓練共用同一張圖。
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
    """跟 v2_vae_tanh 的 VAE 完全同架構，唯一差異是多一個 encode()，讓 FSCE
    loss 可以只算到 mu、不用取樣也不用經過 decoder。
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

    def encode(self, x):
        """x 是 (B,N_CAT) 的 count 向量，回傳 (B,latent_dim) 的 mu——跟 forward()
        算出來的 mu 是同一條路徑，只是不接 logvar/取樣/decoder。
        """
        h = self.trunk(x)
        return torch.tanh(self.fc_mu(h))

    def reparameterize(self, mu, logvar):
        """訓練時取樣、eval 時直接回傳 mu，讓推論的 latent 唯一。"""
        if not self.training:
            return mu
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def forward(self, x):
        h = self.trunk(x)
        mu = torch.tanh(self.fc_mu(h))   # mu 訓練時就夾在 (-1,1)
        # logvar 沒有任何界限的話，用目前規模的 patch count 向量（單格可到幾千）
        # 初始化後第一個 batch 就可能讓 exp(0.5*logvar) 溢位、整個 forward 變成
        # inf/nan，這在這份資料上實測會發生（跟 FSCE 無關，v2_vae_tanh 原架構
        # 也一樣）。clamp 在合理範圍內，只是數值穩定用，不影響 tanh(mu) 這條
        # FSCE 真正在乎的座標。
        logvar = self.fc_logvar(h).clamp(-10.0, 10.0)
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


def build_fsce_graph(x, n_neighbors=15, metric="cosine"):
    """x 是高維空間的特徵矩陣 (N,D)（這裡傳 log1p(count)），在上面建一次 UMAP
    的 fuzzy simplicial set。回傳 edge_i、edge_j：邊兩端的 patch 編號 (E,)
    LongTensor；edge_w：這條邊在高維空間的模糊隸屬度 (E,)∈(0,1] FloatTensor，
    當作 FSCE loss 裡的正樣本權重；a、b：UMAP 低維核函數 1/(1+a·d^(2b)) 的形狀
    參數，由 find_ab_params(spread=1.0, min_dist=0.1) 算出。
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
    """FSCE（fuzzy set cross entropy）：z_i、z_j 是一批 pair 的 mu 座標
    (P,latent_dim)，w 是這對點在高維空間的模糊隸屬度 (P,)——正樣本填
    build_fsce_graph() 算出的 edge_w，負樣本填 0。a、b 是 UMAP 核函數參數。
    回傳每個 pair 一個數字：q（低維距離換算出的相似度）跟 w 差越多，這個數字
    越大，逼著 encoder 把 w 大的點拉近、w=0 的點推遠。

    d2 用 eps 墊底：b<1 時 d2^b 在 d2=0 處的梯度是無限大，tanh 飽和常讓不同
    patch 被壓到同一個角落（或負樣本剛好抽到 i==j），d2 剛好等於 0 並不罕見，
    沒墊底會直接讓 backward 炸成 NaN。
    """
    d2 = (z_i - z_j).pow(2).sum(dim=1).clamp(min=eps)
    q = (1.0 + a * d2.pow(b)).reciprocal().clamp(eps, 1 - eps)
    return -(w * q.log() + (1 - w) * (1 - q).log())
