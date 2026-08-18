"""v3_gat：在 v2_ddae_fsce 的 count-only Poisson DAE 基礎上，加一條處理
OD（起訖點）矩陣的 GAT 分支。

GAT 分支的節點是 (s,d)——「起點類別 s、終點類別 d」這個 pair，而不是類別
本身。節點 (s,d) 的鄰居是「終點為 s 的點」∪「起點為 d 的點」（能接續成一條
移動鏈的 pair），這個鄰接關係只由類別 index 決定、跟 patch 無關，是全域
固定的圖結構，所有 patch 共用同一張 mask。

count 分支輸出的 h_count 跟 GAT 分支輸出的 h_gat 會做一次 multi-head
cross-attention 讓兩者互相參考對方資訊，更新後的兩個向量再用一個上限鎖在
30%（ALPHA_CAP）的 alpha 做凸組合融合進 z，避免資料量遠比 count 稀疏的
OD 分支主導 latent。

破壞方式、Poisson NLL/deviance、FSCE loss 完全沿用 v2_ddae_fsce，OD 矩陣
本身不加噪（是獨立於 count 的觀測）。
"""

import os
import sys

import numpy as np
import torch
import torch.nn as nn
from sklearn.neighbors import NearestNeighbors
from umap.umap_ import find_ab_params, fuzzy_simplicial_set

sys.path.insert(0, os.path.abspath(f"{os.path.dirname(__file__)}/../.."))
from config.dataset import N_CAT  # noqa: E402

HIDDEN = 64

GAT_HIDDEN = 32
CAT_EMB_DIM = 8
N_GAT_LAYERS = 2
GAT_HEADS = 4
READOUT_DIM = 8
FUSION_HEADS = 4
ALPHA_CAP = 0.1


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


class ODMatrices:
    """每個 patch 的 (N_CAT,N_CAT) OD 密集矩陣，從 od.npz 的 CSR 攤平格式
    （od_cell = s*N_CAT+d，見 data/od/build_od.py）展開成 dense 後整包放
    記憶體/device——規模很小（patch 數 * N_CAT^2 * 4 bytes 不到 1MB）。

    path：od.npz 的路徑。
    """

    def __init__(self, path):
        d = np.load(path)
        n = len(d["od_offsets"]) - 1
        owner = np.repeat(np.arange(n), np.diff(d["od_offsets"]))
        dense = np.zeros((n, N_CAT * N_CAT), dtype=np.float32)
        np.add.at(dense, (owner, d["od_cell"].astype(np.int64)),
                   d["od_value"].astype(np.float32))
        self.mat = torch.from_numpy(dense).view(n, N_CAT, N_CAT)
        self.n = n

    def to(self, device):
        """把 OD 矩陣搬到 device 並回傳自己。"""
        self.mat = self.mat.to(device)
        return self

    def get(self, idx):
        """idx：patch 編號的 LongTensor（可在任何 device）。
        回傳 (B,N_CAT,N_CAT) float32，落在 self.mat 所在的 device。"""
        return self.mat[idx.to(self.mat.device)]


def corrupt(x, p, mode="thinning", generator=None):
    """把乾淨的 count 向量破壞成 DAE 的輸入。

    x：(B,N_CAT) 的乾淨 count 向量（float，值是非負整數）。
    p：破壞強度 ∈[0,1)。thinning 是每個 POI 被丟掉的機率，mask 是每一類
       整維被抹成 0 的機率。p=0 時直接回傳 x。
    mode："thinning" 或 "mask"。
    generator：torch.Generator，給定就用它抽亂數，讓每個 epoch 的破壞可重現。

    回傳跟 x 同 shape 的 (B,N_CAT) tensor，已經除以 1-p 做過尺度補償。
    OD 不走這個函式：它是另一份獨立觀測。
    """
    if p <= 0:
        return x
    keep = 1.0 - p
    if mode == "thinning":
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


def _od_adjacency():
    """建 (N_CAT^2,N_CAT^2) 的 bool 鄰接矩陣：節點 k=s*N_CAT+d，節點 i=(s_i,d_i)
    的鄰居是「終點為 s_i 的點」（即所有 (x,s_i)）∪「起點為 d_i 的點」（即所有
    (d_i,y)），再顯式補上全體 self-loop（讓每個節點都能自由權衡自己的原始
    計數與鄰居，不因 s==d 與否而有結構性差異）。

    回傳 (N_CAT^2,N_CAT^2) BoolTensor，True 代表兩節點相鄰。
    """
    idx = torch.arange(N_CAT * N_CAT)
    s, d = idx // N_CAT, idx % N_CAT
    adj = (d.view(1, -1) == s.view(-1, 1)) | (s.view(1, -1) == d.view(-1, 1))
    return adj | torch.eye(N_CAT * N_CAT, dtype=torch.bool)


class GATLayer(nn.Module):
    """N_CAT^2 個節點的 masked multi-head attention block：attention 只在
    _od_adjacency() 允許的邊上計算，其餘標準 post-norm Transformer block
    結構（residual + LayerNorm + feedforward）不變。

    dim：節點特徵維度。n_heads：attention 頭數。
    """

    def __init__(self, dim, n_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, h, attn_mask):
        """h：(B,N_CAT^2,dim) 節點表示。attn_mask：(N_CAT^2,N_CAT^2) bool，
        True 代表不允許 attend（nn.MultiheadAttention 的 mask 語意）。
        回傳跟 h 同 shape 的更新後節點表示。
        """
        a, _ = self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)
        h = self.norm1(h + a)
        return self.norm2(h + self.ff(h))


class GATBranch(nn.Module):
    """OD 圖分支：節點是 (s,d) pair，特徵是該 patch 這個 pair 的 log1p(筆數)
    加上起點/終點類別的可學習 embedding，經過幾層 masked GAT 後，把每個
    節點各自壓縮再攤平做 readout，得到一個跟 count 分支同尺寸的 h_gat。

    節點集合固定且每個位置語意固定（永遠是同一個 (s,d)），readout 用
    「逐節點共享權重壓縮＋攤平」而非 mean/sum pool：OD 很稀疏（大多數
    pair 是 0），對稱平均會被大量零值稀釋掉少數有效節點的訊號。
    """

    def __init__(self, gat_hidden=GAT_HIDDEN, cat_emb_dim=CAT_EMB_DIM,
                 n_layers=N_GAT_LAYERS, n_heads=GAT_HEADS,
                 readout_dim=READOUT_DIM):
        super().__init__()
        self.register_buffer("attn_mask", ~_od_adjacency())
        self.origin_emb = nn.Embedding(N_CAT, cat_emb_dim)
        self.dest_emb = nn.Embedding(N_CAT, cat_emb_dim)
        idx = torch.arange(N_CAT * N_CAT)
        self.register_buffer("node_s", idx // N_CAT)
        self.register_buffer("node_d", idx % N_CAT)
        self.node_in = nn.Sequential(
            nn.Linear(1 + 2 * cat_emb_dim, gat_hidden), nn.GELU())
        self.layers = nn.ModuleList(
            [GATLayer(gat_hidden, n_heads) for _ in range(n_layers)])
        self.node_squeeze = nn.Sequential(
            nn.Linear(gat_hidden, readout_dim), nn.GELU())
        self.readout = nn.Linear(N_CAT * N_CAT * readout_dim, HIDDEN)

    def forward(self, od):
        """od：(B,N_CAT,N_CAT) 該 patch 的原始 OD 筆數。回傳 h_gat：(B,HIDDEN)。"""
        b = od.shape[0]
        x = torch.log1p(od).view(b, N_CAT * N_CAT, 1)
        emb = torch.cat([self.origin_emb(self.node_s),
                         self.dest_emb(self.node_d)], dim=-1)
        h = self.node_in(torch.cat([x, emb.unsqueeze(0).expand(b, -1, -1)], dim=-1))
        for layer in self.layers:
            h = layer(h, self.attn_mask)
        h = self.node_squeeze(h)
        return self.readout(h.reshape(b, -1))


class CountEncoder(nn.Module):
    """count 分支：x (B,N_CAT) -> h_count (B,HIDDEN)。層數/寬度跟
    v2_ddae_fsce 的 MLPAE.encoder 前四層一致，差別只是不含最後
    HIDDEN->latent_dim 的投影（該投影移到跟 GAT 分支融合之後）。
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(N_CAT, HIDDEN), nn.GELU(),
            nn.Linear(HIDDEN, HIDDEN), nn.GELU(),
            nn.Linear(HIDDEN, HIDDEN), nn.GELU(),
            nn.Linear(HIDDEN, HIDDEN), nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


class CrossAttentionFusion(nn.Module):
    """把 h_count、h_gat 疊成 2-token 序列做一次 multi-head self-attention，
    讓兩者互相參考對方資訊後更新，再各自過 LayerNorm(elementwise_affine=False)
    固定尺度，用上限鎖在 ALPHA_CAP 的 alpha 做凸組合融合成 z_pre。

    固定尺度這一步是必要的：alpha 代表「GAT 分支貢獻不超過 ALPHA_CAP」，
    這個承諾只有在兩個向量尺度可比時才有意義，否則模型可以把 GAT 分支
    內部尺度學大來繞過係數上限；用不可學習的 LayerNorm 是因為若讓它可學習，
    模型一樣能學那個縮放參數繞過去。

    dim：融合維度（等於 HIDDEN）。n_heads：cross-attention 頭數。
    """

    def __init__(self, dim=HIDDEN, n_heads=FUSION_HEADS):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.ln_cnt = nn.LayerNorm(dim, elementwise_affine=False)
        self.ln_gat = nn.LayerNorm(dim, elementwise_affine=False)
        self.alpha_raw = nn.Parameter(torch.tensor(0.0))

    def forward(self, h_cnt, h_gat):
        """h_cnt、h_gat：各自 (B,HIDDEN)。回傳 (z_pre, alpha)：z_pre (B,HIDDEN)
        是融合後、尚未投影到 latent 的向量，alpha 是純量 tensor ∈(0,ALPHA_CAP)。
        """
        seq = torch.stack([h_cnt, h_gat], dim=1)
        normed = self.norm(seq)
        a, _ = self.attn(normed, normed, normed, need_weights=False)
        seq = seq + a
        h_cnt_p, h_gat_p = seq[:, 0, :], seq[:, 1, :]
        h_cnt_n, h_gat_n = self.ln_cnt(h_cnt_p), self.ln_gat(h_gat_p)
        alpha = ALPHA_CAP * torch.sigmoid(self.alpha_raw)
        z_pre = alpha * h_gat_n + (1 - alpha) * h_cnt_n
        return z_pre, alpha


class GATAE(nn.Module):
    """count 分支 + OD-GAT 分支，經 CrossAttentionFusion 融合後投影進
    latent。decoder 結構跟 v2_ddae_fsce 的 MLPAE.decoder 完全相同。

    latent_dim：latent 維度。
    """

    def __init__(self, latent_dim=2):
        super().__init__()
        self.count_enc = CountEncoder()
        self.gat = GATBranch()
        self.fusion = CrossAttentionFusion()
        self.to_latent = nn.Linear(HIDDEN, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, HIDDEN), nn.GELU(),
            nn.Linear(HIDDEN, HIDDEN), nn.GELU(),
            nn.Linear(HIDDEN, HIDDEN), nn.GELU(),
            nn.Linear(HIDDEN, HIDDEN), nn.GELU(),
            nn.Linear(HIDDEN, N_CAT),   # 輸出是 log λ
        )

    def encode(self, x, od):
        """x：(B,N_CAT) count 向量。od：(B,N_CAT,N_CAT) OD 矩陣。
        回傳 (z, alpha)：z (B,latent_dim)，alpha 是這次融合用的純量 tensor。
        """
        h_cnt = self.count_enc(x)
        h_gat = self.gat(od)
        z_pre, alpha = self.fusion(h_cnt, h_gat)
        return self.to_latent(z_pre), alpha

    def forward(self, x, od):
        """回傳 (z, log_lam, alpha)：z (B,latent_dim)、log_lam (B,N_CAT)、
        alpha 是這次融合用的純量 tensor。
        """
        z, alpha = self.encode(x, od)
        return z, self.decoder(z), alpha


def poisson_nll(log_lam, x):
    """Poisson NLL，省略跟模型無關的 log(y!) 常數項，回傳每個 patch 一個數字。"""
    cell = torch.exp(log_lam) - x * log_lam
    return cell.mean(dim=1)


def poisson_deviance(log_lam, x):
    """Poisson deviance：2·(y·log(y/λ) - (y - λ))，越小越好、有下界，可跨 patch 比較。"""
    lam = torch.exp(log_lam)
    cell = 2 * (torch.xlogy(x, x) - x * log_lam - x + lam)
    return cell.mean(dim=1)


def build_fsce_graph(x, n_neighbors=15, metric="euclidean"):
    """x 是高維空間的特徵矩陣 (N,D)（這裡傳乾淨 count 的 log1p），在上面建一次
    UMAP 的 fuzzy simplicial set。回傳 edge_i、edge_j：邊兩端的 patch 編號 (E,)
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
    """FSCE（fuzzy set cross entropy）：z_i、z_j 是一批 pair 的 latent 座標
    (P,latent_dim)，w 是這對點在高維空間的模糊隸屬度 (P,)——正樣本填
    build_fsce_graph() 算出的 edge_w，負樣本填 0。a、b 是 UMAP 核函數參數。
    回傳每個 pair 一個數字：q（低維距離換算出的相似度）跟 w 差越多，這個數字
    越大，逼著 encoder 把 w 大的點拉近、w=0 的點推遠。
    """
    d2 = (z_i - z_j).pow(2).sum(dim=1).clamp(min=eps)
    q = (1.0 + a * d2.pow(b)).reciprocal().clamp(eps, 1 - eps)
    return -(w * q.log() + (1 - w) * (1 - q).log())
