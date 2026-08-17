"""v3_gat_literal：完全照提案的字面實作。

    POI count (N_CAT)  ──→ MLP ────────────→ h_cnt (HIDDEN) ─┐
                                                              ├─ α 混合 → z → decoder → count
    OD 矩陣 (N_CAT×N_CAT) ──→ GAT ──→ readout → h_gat (HIDDEN)─┘

規格逐條對照：
  「這個網格內使用者從 A 類別到 B 類別有多少的 OD 矩陣」
        A[s,d] = 起點在這一格的 s 類 POI、終點是 d 類 POI 的移動次數。
        終點不限距離、不限格（OD_ASSIGN="origin"）。
  「OD 矩陣可以當作 graph，用 attention 去看這條邊要給多少權重」
        節點＝N_CAT 個類別，邊的強度以 log1p(原始筆數) 加進 attention 的 logit，
        真正的權重由 attention 學。
  「乘上一個係數 alpha，跟 alpha-1 乘上原本的壓縮後 POI count，結合丟進 latent」
        alpha = sigmoid(可學習純量)，z = tanh(Linear(α·h_gat + (1-α)·h_cnt))。

刻意「沒有」的東西（都是 v3_gat_ddae 有、但規格沒提的）：
  沒有全域 OD 先驗、沒有逐列收縮   —— 邊權直接用這一格自己的原始筆數
  沒有 row-normalize               —— 所以空格 log1p(0)=0 是中性偏置，不會除以零
  沒有 in/out degree 等額外節點特徵 —— 節點特徵只有這一格該類別的 count
  沒有 OD 輔助 loss                —— loss 只有 Poisson NLL，跟 v2 完全一樣
  沒有反向邊、沒有 self-loop 權重   —— 「從 A 到 B」是有向的，只放正向

唯一一個規格沒提、但我加了的東西是兩條分支出口的 LayerNorm
（elementwise_affine=False，不帶任何可學習參數）。理由：不加的話模型可以把
h_gat 內部放大十倍來抵銷 alpha=0.1，log 裡的 alpha 就不能解讀成混合比例。
這一項已經確認過。
"""

import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(f"{os.path.dirname(__file__)}/../.."))
from config.dataset import N_CAT  # noqa: E402

HIDDEN = 64
N_HIDDEN_LAYERS = 4    # 跟 v2_ddae_tanh_fsce 對齊

GAT_DIM = 24
GAT_HEADS = 4
GAT_LAYERS = 2
GAT_FF_MULT = 2
GAT_DROPOUT = 0.1


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


class ODGraph:
    """od.npz 的 CSR 展開成 dense 的 per-patch 原始轉移計數。

    path：od.npz 的路徑。

    這一版不碰全域矩陣、不做任何正規化——A_local 就是原始筆數，直接進
    log1p 當 edge bias。(1233,10,10) float32 不到 0.5MB，整包放 device。

    屬性：A_local (N,N_CAT,N_CAT) 原始計數、n_od (N,) 每格的 OD 總量。
    """

    def __init__(self, path):
        d = np.load(path)
        cell = d["od_cell"].astype(np.int64)
        value = d["od_value"].astype(np.float32)
        offsets = d["od_offsets"].astype(np.int64)
        n = len(offsets) - 1

        dense = np.zeros((n, N_CAT * N_CAT), dtype=np.float32)
        dense[np.repeat(np.arange(n), np.diff(offsets)), cell] = value
        self.A_local = torch.from_numpy(dense.reshape(n, N_CAT, N_CAT))
        self.n_od = torch.from_numpy(d["n_od"].astype(np.float32))
        self.n = n

    def to(self, device):
        """把張量搬到 device 並回傳自己。"""
        self.A_local = self.A_local.to(device)
        self.n_od = self.n_od.to(device)
        return self


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


def _mlp_block(d_in, d_out, n_layers):
    """疊 n_layers 個 Linear(HIDDEN,HIDDEN)+GELU，前面加一層 d_in->HIDDEN，
    回傳 list of nn.Module（不含最後把 HIDDEN 投影到 d_out 的那一層）。
    """
    layers = [nn.Linear(d_in, HIDDEN), nn.GELU()]
    for _ in range(n_layers - 1):
        layers += [nn.Linear(HIDDEN, HIDDEN), nn.GELU()]
    return layers


class DenseGATLayer(nn.Module):
    """N_CAT 個節點的全連接 multi-head attention，OD 以 log1p(原始筆數) 加進 logit。

    節點只有 N_CAT 個，dense attention 的 (B,H,N_CAT,N_CAT) 張量微不足道，
    所以純 PyTorch 手寫，不需要 PyTorch Geometric。

    不做 edge masking：某條邊筆數為 0 時 log1p(0)=0 是中性偏置，attention
    退回只看節點特徵，不會出現整列 -inf 的 softmax。
    """

    def __init__(self, d=GAT_DIM, n_heads=GAT_HEADS, dropout=GAT_DROPOUT):
        super().__init__()
        self.heads, self.dk = n_heads, d // n_heads
        self.W = nn.Linear(d, d, bias=False)
        self.a_src = nn.Parameter(torch.empty(n_heads, self.dk))
        self.a_dst = nn.Parameter(torch.empty(n_heads, self.dk))
        nn.init.xavier_uniform_(self.a_src)
        nn.init.xavier_uniform_(self.a_dst)
        # OD 的權重，每個 head 一個。從 1 起步：OD 就是這一版的重點，
        # 一開始就用它，讓「模型主動把它關掉」變成有意義的觀測
        self.w_od = nn.Parameter(torch.ones(n_heads))
        self.proj = nn.Linear(d, d)
        self.ff = nn.Sequential(
            nn.Linear(d, GAT_FF_MULT * d), nn.GELU(),
            nn.Linear(GAT_FF_MULT * d, d))
        self.norm1, self.norm2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.drop = nn.Dropout(dropout)

    def forward(self, h, log_a):
        """h：(B,N_CAT,D) 節點表示。log_a：(B,N_CAT,N_CAT) = log1p(OD 原始筆數)。

        回傳 (h_out, att)：h_out (B,N_CAT,D) 已做 residual + LayerNorm，
        att (B,H,N_CAT,N_CAT) 是 attention 權重（給分析用）。
        """
        b, c, d = h.shape
        wh = self.W(h).view(b, c, self.heads, self.dk).transpose(1, 2)   # (B,H,C,dk)
        s_src = (wh * self.a_src.unsqueeze(1)).sum(-1).unsqueeze(-1)     # (B,H,C,1)
        s_dst = (wh * self.a_dst.unsqueeze(1)).sum(-1).unsqueeze(-2)     # (B,H,1,C)
        e = F.leaky_relu(s_src + s_dst, 0.2)                             # (B,H,C,C)
        e = e + self.w_od.view(1, self.heads, 1, 1) * log_a.unsqueeze(1)

        att = self.drop(torch.softmax(e, dim=-1))                        # (B,H,C,C)
        msg = (att @ wh).transpose(1, 2).reshape(b, c, d)                # (B,C,D)
        h = self.norm1(h + self.drop(self.proj(msg)))
        return self.norm2(h + self.drop(self.ff(h))), att


class GATBranch(nn.Module):
    """OD 圖分支：節點特徵只有「這一格該類別的 count」，加上類別身分 embedding。

    類別 embedding 是必要的：節點集合固定且有語意（index 0 永遠是同一類），
    沒有它的話兩個 count 剛好相同的類別完全無法區分。

    readout 用 flatten：一般 GNN 用 mean/sum pool 是為了 permutation
    invariance，但這裡節點永遠是同樣的 N_CAT 個類別、順序固定，
    invariance 在這裡不是優點，是白白丟資訊。
    """

    def __init__(self, d=GAT_DIM, n_layers=GAT_LAYERS):
        super().__init__()
        self.node_in = nn.Linear(1, d)
        self.cat_emb = nn.Embedding(N_CAT, d)
        self.layers = nn.ModuleList([DenseGATLayer(d) for _ in range(n_layers)])
        self.out = nn.Linear(N_CAT * d, HIDDEN)

    def forward(self, x_in, log_a):
        """x_in：(B,N_CAT) 加噪後的 count。log_a：(B,N_CAT,N_CAT)。

        回傳 (h_gat, att)：h_gat (B,HIDDEN)、att 是最後一層的 attention。

        x_in 必須是加噪版：餵乾淨的等於讓模型繞過 denoising，alpha 會漂到 1、
        val_dev 好得不合理，而且這個 bug 只會表現成「效果超好」，很難察覺。
        """
        h = self.node_in(torch.log1p(x_in).unsqueeze(-1)) \
            + self.cat_emb.weight.unsqueeze(0)                 # (B,C,D)
        att = None
        for layer in self.layers:
            h, att = layer(h, log_a)
        return self.out(h.reshape(h.shape[0], -1)), att


class GATLiteralAE(nn.Module):
    """count 分支 + OD 圖分支，用一個可學習純量 alpha 做凸組合後投影進 latent。

    latent_dim：latent 維度。alpha_init：sigmoid 的起點（0.5 = 兩邊等重）。
    """

    def __init__(self, latent_dim=2, alpha_init=0.5):
        super().__init__()
        self.cnt_trunk = nn.Sequential(*_mlp_block(N_CAT, HIDDEN, N_HIDDEN_LAYERS))
        self.gat = GATBranch()
        # elementwise_affine=False：不帶可學習參數，只把兩條分支的尺度釘死，
        # 這樣 alpha 才真的是混合比例而不是可以被分支內部縮放抵銷掉的數字
        self.norm_cnt = nn.LayerNorm(HIDDEN, elementwise_affine=False)
        self.norm_gat = nn.LayerNorm(HIDDEN, elementwise_affine=False)
        self.alpha_raw = nn.Parameter(torch.logit(torch.tensor(float(alpha_init))))
        self.to_z = nn.Linear(HIDDEN, latent_dim)
        self.decoder = nn.Sequential(
            *_mlp_block(latent_dim, HIDDEN, N_HIDDEN_LAYERS),
            nn.Linear(HIDDEN, N_CAT),        # 輸出是 log λ
        )

    def alpha(self):
        """回傳純量 tensor 的 alpha ∈ (0,1)。"""
        return torch.sigmoid(self.alpha_raw)

    def encode(self, x_in, a_local, knockout=False):
        """x_in：(B,N_CAT) 加噪 count。a_local：(B,N_CAT,N_CAT) 原始 OD 筆數。
        knockout：True 時把 h_gat 換成這一批的平均，切斷 GAT 分支傳遞的所有
                  per-patch 資訊——用來量這條分支到底有沒有在做事。

        回傳 (z, aux)：z (B,latent_dim)，aux 是 dict，含 h_gat / h_cnt / att。
        """
        h_gat, att = self.gat(x_in, torch.log1p(a_local))
        h_gat = self.norm_gat(h_gat)
        h_cnt = self.norm_cnt(self.cnt_trunk(x_in))
        if knockout:
            h_gat = h_gat.mean(dim=0, keepdim=True).expand_as(h_gat)
        a = self.alpha()
        z = torch.tanh(self.to_z(a * h_gat + (1 - a) * h_cnt))
        return z, {"h_gat": h_gat, "h_cnt": h_cnt, "att": att}

    def forward(self, x_in, a_local, knockout=False):
        """回傳 (z, log_lam, aux)：z (B,latent_dim)、log_lam (B,N_CAT)、aux 同 encode()。"""
        z, aux = self.encode(x_in, a_local, knockout)
        return z, self.decoder(z), aux


def poisson_nll(log_lam, x):
    """Poisson NLL，省略跟模型無關的 log(y!) 常數項，回傳每個 patch 一個數字。"""
    cell = torch.exp(log_lam) - x * log_lam
    return cell.mean(dim=1)


def poisson_deviance(log_lam, x):
    """Poisson deviance：2·(y·log(y/λ) - (y - λ))，越小越好、有下界，可跨 patch 比較。"""
    lam = torch.exp(log_lam)
    cell = 2 * (torch.xlogy(x, x) - x * log_lam - x + lam)
    return cell.mean(dim=1)
