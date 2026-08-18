"""v3_gat_literal：OD-pair（line graph）當節點的 GAT + count 分支。

    POI count (N_CAT)  ──→ MLP ────────────→ h_cnt (HIDDEN) ─┐
                                                              ├─ α 混合 → z → decoder → count
    OD 矩陣 (N_CAT×N_CAT) ──→ GAT(line graph) → readout → h_gat (HIDDEN)─┘

規格逐條對照：
  「這個網格內使用者從 A 類別到 B 類別有多少的 OD 矩陣」
        A[s,d] = 起點在這一格的 s 類 POI、終點是 d 類 POI 的移動次數。
        終點不限距離、不限格（OD_ASSIGN="origin"）。
  「OD 矩陣可以當作 graph，用 attention 去看這條邊要給多少權重」
        節點＝N_CAT×N_CAT 個 OD pair (s,d)（A_local 矩陣裡的每一格是一個節點，
        不是每個類別一個節點）。節點 (s,d) 的鄰居 = 上游流量 ∪ 下游流量 ∪ 自己：
          上游 inflow ：所有終點剛好是 s 的其他 pair (x,s)，x 跑遍 0..N_CAT-1
          下游 outflow：所有起點剛好是 d 的其他 pair (d,y)，y 跑遍 0..N_CAT-1
        這是這張「類別轉移圖」的 directed line graph：兩個 OD pair 相鄰若且唯若
        其中一個的終點是另一個的起點。attention 只在這個鄰居集合內做（固定
        boolean mask 擋掉其餘位置的 logit），不是任兩個 OD pair 都互相看。
        節點特徵是這一格自己的 log1p(原始筆數)，加上起點類別 embedding 與
        終點類別 embedding（各自獨立一份，起點/終點的語意不同）。
  「乘上一個係數 alpha，跟 alpha-1 乘上原本的壓縮後 POI count，結合丟進 latent」
        alpha = sigmoid(可學習純量)，z = Linear(α·h_gat + (1-α)·h_cnt)（latent
        前不接激活，離群值才不會被壓回來，跟 v2_ddae_fsce 系列同一個理由）。

刻意「沒有」的東西：
  沒有全域 OD 先驗、沒有逐列收縮   —— 節點特徵直接用這一格自己的原始筆數
  沒有 row-normalize               —— 所以空格 log1p(0)=0，節點特徵是中性值，不影響鄰居
  沒有 edge bias                   —— 舊版把 log1p(OD) 當 attention logit 的外部偏置
                                       （w_od），新版 OD 值已經是節點自己的特徵，
                                       不再需要這層機制，「OD 有多重要」直接由
                                       node_in 的權重學
  沒有 OD 輔助 loss                —— 沒有額外拿 OD 去監督什麼東西；訓練訊號
                                       仍然只有 Poisson NLL + FSCE（跟
                                       v2_ddae_fsce 對齊，見 train.py），
                                       GAT 分支唯一的梯度來源是這兩個 loss
                                       反傳回 encode() 的路徑
  沒有額外的可學習自環權重          —— inflow/outflow 定義本身在 s=d 時就自動包含
                                       自環，其餘節點的自環用固定 mask 補上，不
                                       另外加一個可學習純量去加權自己

行/列邊際（每個類別送出多少、接收多少）是顯式特徵，餵兩個地方：節點特徵裡
每個節點 (s,d) 除了自己那格的值，另外帶 log1p(row_sum[s]) 與 log1p(col_sum[d])；
readout 的 flatten 後面也直接接上這 2·N_CAT 個數字。

這一條是實測改的，不是預設就該有。前一版節點特徵只有自己那格的 log1p，檔頭
還寫過「in/out degree 已經隱含在 attention 的鄰居集合裡，不需要額外餵」——那個
推論是錯的：DenseGATLayer 的聚合是 softmax(e) @ wh，是**凸組合、尺度無關**，
它表示得了「從 s 出發的流量往哪些類別分配」，表示不了「從 s 出發總共有多少」。
節點 (s,d) 的鄰居集合確實就是 s 的 inflow 與 d 的 outflow，但取加權平均之後
「有多少」正好被 normalize 掉。

證據（不用訓練的 ridge 診斷，目標＝乾淨 count 的 log1p，各特徵集各自掃過
ridge alpha，5 個噪聲 seed 的驗證集 R²）：

  count 10 維單獨                     0.8005
  count + OD 全矩陣 100 維            0.8200   (+0.0195)
  count + OD 行/列邊際 20 維          0.8440   (+0.0435)  <- 最好
  count + 全矩陣 + 邊際 130 維        0.8391   (+0.0385)

OD 確實帶有 count 以外的資訊，但那個資訊集中在邊際、不在 pair 層級；在 1233
個樣本下，100 維的 pair 表示反而是在稀釋它。對應到訓練結果：把邊際餵進去以前，
gat_gain（拔掉 GAT 分支後 val_dev 變差多少）在後段只有 +0.0075，遠低於 val_dev
本身的波動（std≈0.015~0.023），也就是這條分支量不到貢獻。

readout 前多一層 GAT_DIM -> GAT_READOUT_DIM 的 bottleneck：節點數從 N_CAT（10）
變成 N_CAT²（100）後，若直接 flatten 全部節點的 GAT_DIM 維表示，光 readout 這層
就有 N_CAT²·GAT_DIM·HIDDEN ≈ 10 萬參數，訓練集只有 1233 個 patch，過擬合風險
太明顯。bottleneck 把每個節點先壓到 GAT_READOUT_DIM 維（權重在所有節點間共用，
形同 1x1 conv）再 flatten，attention 本身的表達力（GAT_DIM/GAT_HEADS）不變，
只是進 readout 之前先瘦身。

GAT_READOUT_DIM = 1：每個節點壓成一個純量，所以進 flatten 的那 100 個數字
就是「attention 重新加權過的 10x10 OD 矩陣」，節點位置語意完全保留（本來就是
為了保留它才用 flatten 而不是 pooling），只是每格從 GAT_READOUT_DIM 個數字變
成一個，順帶變得可以直接畫出來看。

參數量的分配（會影響這裡每個常數怎麼選）：cnt_trunk 與 decoder 跟
v2_ddae_fsce 逐字相同、合計 26.5k，是對照實驗的基準線不能動；能調的只有
GAT 分支。第一版 GAT_DIM=24、GAT_READOUT_DIM=4 時 GAT 分支是 33.7k，比整個
v2 模型（26.6k）還大，其中光 out 這一層 Linear(400,HIDDEN) 就佔 25.7k——在
1233 個樣本上這是最主要的記憶容量來源，實測 train_nll 每個檢查點都比 v2 低、
val_dev 卻從 epoch 450 起停在 0.74 不動。現在 GAT_DIM=16、GAT_READOUT_DIM=1
把 GAT 分支壓到 10.2k（全模型 36.9k），out 降到 6.5k。

GAT_LAYERS 維持 2 沒有跟著砍：在 GAT_DIM=16 底下第二層只要 1.7k 參數，而
「兩個 OD pair 隔一步互相看得到」正是這張 line graph 的重點，砍掉是拿掉模型
的主張去換 5% 的參數。

h_gat 的寬度被釘死在 HIDDEN：alpha 是凸組合 α·h_gat+(1-α)·h_cnt，兩邊必須
同寬，所以 out 的輸出維度不能再壓。要再省只能把 out 拆成低秩的兩層
（Linear(100,r) -> Linear(r,HIDDEN)），目前沒做，先看這一刀夠不夠。
"""

import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors
from umap.umap_ import find_ab_params, fuzzy_simplicial_set

sys.path.insert(0, os.path.abspath(f"{os.path.dirname(__file__)}/../.."))
from config.dataset import N_CAT  # noqa: E402

HIDDEN = 64
N_HIDDEN_LAYERS = 4    # 跟 v2_ddae_tanh_fsce 對齊

GAT_DIM = 16           # 4 heads x dk=4
GAT_HEADS = 4
GAT_LAYERS = 2
GAT_FF_MULT = 2
GAT_DROPOUT = 0.1
GAT_READOUT_DIM = 1    # flatten 前的 per-node bottleneck，理由見檔頭 docstring


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
    log1p 當 GAT line graph 的節點特徵。(1233,10,10) float32 不到 0.5MB。

    整包留在 CPU，跟 Patches 同一個規則：取 batch、加噪都在 CPU 做完，
    才把那一小塊 (B,N_CAT,N_CAT) 搬上 device。corrupt_od() 用的
    torch.binomial 目前沒有 MPS kernel，所以 A_local 不能預先放 device。

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


def corrupt(x, p, mode="thinning", generator=None):
    """把乾淨的 count 向量破壞成 DAE 的輸入。

    x：(B,N_CAT) 的乾淨 count 向量（float，值是非負整數）。
    p：破壞強度 ∈[0,1)。thinning 是每個 POI 被丟掉的機率，mask 是每一類
       整維被抹成 0 的機率。p=0 時直接回傳 x。
    mode："thinning" 或 "mask"。
    generator：torch.Generator，給定就用它抽亂數，讓每個 epoch 的破壞可重現。

    回傳跟 x 同 shape 的 (B,N_CAT) tensor，已經除以 1-p 做過尺度補償。
    OD 走 corrupt_od()：破壞的語意一樣（每筆觀測獨立被丟掉），但值域大太多，
    這裡的實作會爆記憶體。
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


def corrupt_od(a, p, generator=None):
    """把乾淨的 OD 計數矩陣破壞成 DAE 的輸入（binomial thinning）。

    a：(B,N_CAT,N_CAT) 的原始 OD 筆數（float，值是非負整數），必須在 CPU 上
       ——torch.binomial 目前沒有 MPS kernel。
    p：破壞強度 ∈[0,1)，每一筆移動被獨立丟掉的機率。p=0 時直接回傳 a。
    generator：torch.Generator（CPU），給定就用它抽亂數，讓每個 epoch 的
       破壞可重現。

    回傳跟 a 同 shape 的 tensor，已經除以 1-p 做過尺度補償，跟 corrupt() 同一個
    規則——節點特徵是 log1p(值)，不補償的話 GAT 分支看到的量級會被 p 系統性壓低。

    為什麼不共用 corrupt() 的 thinning 分支：那裡是開一個 (…,max_c) 的硬幣張量
    逐筆丟，OD 最大的一格有 5313 筆，(256,10,10,5313) 會直接炸掉記憶體。
    torch.binomial 一次抽完，分布上跟逐筆丟硬幣完全等價。

    這條加噪是刻意加的：原本只有 count 分支每個 epoch 被破壞成不同樣子，
    A_local 每個 epoch 都一模一樣，等於給 GAT 分支一份沒有任何 augmentation 的
    per-patch 固定指紋，可以直接記住「這格是哪一格」把乾淨的 x 背出來。
    """
    if p <= 0:
        return a
    keep = 1.0 - p
    return torch.binomial(a, torch.full_like(a, keep), generator=generator) / keep


def _mlp_block(d_in, d_out, n_layers):
    """疊 n_layers 個 Linear(HIDDEN,HIDDEN)+GELU，前面加一層 d_in->HIDDEN，
    回傳 list of nn.Module（不含最後把 HIDDEN 投影到 d_out 的那一層）。
    """
    layers = [nn.Linear(d_in, HIDDEN), nn.GELU()]
    for _ in range(n_layers - 1):
        layers += [nn.Linear(HIDDEN, HIDDEN), nn.GELU()]
    return layers


class DenseGATLayer(nn.Module):
    """固定鄰居 mask 的 multi-head attention（line graph 版）。

    節點數 C=N_CAT*N_CAT 雖然比類別版的 N_CAT 大 10 倍，(B,H,C,C) 在這個規模下
    仍然微不足道（C=100 時單一 batch 只有幾十 MB），所以維持純 PyTorch 手寫、
    dense 實作：mask 只是把不合法的位置在 softmax 前填 -inf，不省 FLOPs，但這裡
    的絕對計算量本來就小，不需要為此換成稀疏實作。
    """

    def __init__(self, d=GAT_DIM, n_heads=GAT_HEADS, dropout=GAT_DROPOUT):
        super().__init__()
        self.heads, self.dk = n_heads, d // n_heads
        self.W = nn.Linear(d, d, bias=False)
        self.a_src = nn.Parameter(torch.empty(n_heads, self.dk))
        self.a_dst = nn.Parameter(torch.empty(n_heads, self.dk))
        nn.init.xavier_uniform_(self.a_src)
        nn.init.xavier_uniform_(self.a_dst)
        self.proj = nn.Linear(d, d)
        self.ff = nn.Sequential(
            nn.Linear(d, GAT_FF_MULT * d), nn.GELU(),
            nn.Linear(GAT_FF_MULT * d, d))
        self.norm1, self.norm2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.drop = nn.Dropout(dropout)

    def forward(self, h, mask):
        """h：(B,C,D) 節點表示。mask：(C,C) bool，mask[q,k]=True 表示節點 k
        對節點 q 可見（含 inflow、outflow、self），每一層共用同一份。

        回傳 (h_out, att)：h_out (B,C,D) 已做 residual + LayerNorm，
        att (B,H,C,C) 是 attention 權重（不可見位置權重為 0，給分析用）。
        """
        b, c, d = h.shape
        wh = self.W(h).view(b, c, self.heads, self.dk).transpose(1, 2)   # (B,H,C,dk)
        s_src = (wh * self.a_src.unsqueeze(1)).sum(-1).unsqueeze(-1)     # (B,H,C,1)
        s_dst = (wh * self.a_dst.unsqueeze(1)).sum(-1).unsqueeze(-2)     # (B,H,1,C)
        e = F.leaky_relu(s_src + s_dst, 0.2)                             # (B,H,C,C)
        e = e.masked_fill(~mask, float("-inf"))

        att = self.drop(torch.softmax(e, dim=-1))                        # (B,H,C,C)
        msg = (att @ wh).transpose(1, 2).reshape(b, c, d)                # (B,C,D)
        h = self.norm1(h + self.drop(self.proj(msg)))
        return self.norm2(h + self.drop(self.ff(h))), att


class GATBranch(nn.Module):
    """OD line-graph 分支：節點＝OD pair (s,d)，鄰居＝上游 inflow ∪ 下游 outflow ∪ 自己。

    節點順序固定＝s*N_CAT+d（跟 A_local.reshape(B,-1) 的攤平順序一致），
    所以 readout 一樣用 flatten 保留位置語意，不用 permutation-invariant pooling
    （理由跟舊版類別節點一樣：節點集合固定且有語意，invariance 只是白白丟資訊）。

    起點/終點各自獨立一份 embedding：同一類別當起點（發送方）跟當終點
    （吸收方）語意不同，共用一份會讓 (s,d) 跟 (d,s) 分不清方向。

    行/列邊際走兩條路進來（理由與證據見檔頭）：一條是節點特徵，node_in 吃
    3 維（自己那格、自己起點類別的送出總量、自己終點類別的接收總量）；另一條
    是 readout 直接把 2·N_CAT 個邊際接在 flatten 後面，讓「總量」不必穿過
    softmax 那層尺度無關的聚合就能到達 h_gat。
    """

    def __init__(self, d=GAT_DIM, n_layers=GAT_LAYERS, readout_dim=GAT_READOUT_DIM):
        super().__init__()
        self.node_in = nn.Linear(3, d)
        self.cat_emb_src = nn.Embedding(N_CAT, d)
        self.cat_emb_dst = nn.Embedding(N_CAT, d)
        self.layers = nn.ModuleList([DenseGATLayer(d) for _ in range(n_layers)])
        self.readout_proj = nn.Linear(d, readout_dim)
        self.out = nn.Linear(N_CAT * N_CAT * readout_dim + 2 * N_CAT, HIDDEN)

        idx = torch.arange(N_CAT * N_CAT)
        s, dd = idx // N_CAT, idx % N_CAT
        inflow = dd.unsqueeze(0) == s.unsqueeze(1)      # 節點 k 的終點 == 節點 q 的起點
        outflow = s.unsqueeze(0) == dd.unsqueeze(1)      # 節點 k 的起點 == 節點 q 的終點
        self_loop = torch.eye(N_CAT * N_CAT, dtype=torch.bool)
        self.register_buffer("nbr_mask", inflow | outflow | self_loop, persistent=False)

    def forward(self, a_local):
        """a_local：(B,N_CAT,N_CAT) 這一格的原始 OD 筆數（未取 log）。

        回傳 (h_gat, att)：h_gat (B,HIDDEN)、att 是最後一層的 attention。
        """
        b = a_local.shape[0]
        # 邊際：row 是「這個起點類別總共送出多少」，col 是「這個終點類別總共收到多少」
        row = torch.log1p(a_local.sum(dim=2))                        # (B,N_CAT) 依 s
        col = torch.log1p(a_local.sum(dim=1))                        # (B,N_CAT) 依 d
        # 攤成節點順序 s*N_CAT+d：row 依 s 每個重複 N_CAT 次，col 依 d 整段重複 N_CAT 遍
        node_feat = torch.stack([
            torch.log1p(a_local).reshape(b, N_CAT * N_CAT),
            row.repeat_interleave(N_CAT, dim=1),
            col.repeat(1, N_CAT),
        ], dim=-1)                                                   # (B,C,3)
        h = self.node_in(node_feat) \
            + self.cat_emb_src.weight.repeat_interleave(N_CAT, dim=0).unsqueeze(0) \
            + self.cat_emb_dst.weight.repeat(N_CAT, 1).unsqueeze(0)          # (B,C,D)
        att = None
        for layer in self.layers:
            h, att = layer(h, self.nbr_mask)
        # 邊際再接一次：softmax 聚合是凸組合、把「總量」normalize 掉了，
        # 這條路讓 readout 不必穿過 attention 就拿得到總量
        flat = torch.cat([self.readout_proj(h).reshape(b, -1), row, col], dim=1)
        return self.out(flat), att


class BranchCrossAttn(nn.Module):
    """h_cnt、h_gat 視為 2 個 token，做一層 self-attention 讓兩邊互看彼此的資訊。"""

    def __init__(self, d=HIDDEN, n_heads=4, dropout=GAT_DROPOUT):
        super().__init__()
        self.mha = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d)

    def forward(self, h_cnt, h_gat):
        """h_cnt、h_gat：各自 (B,D)。回傳 (h_cnt2, h_gat2)，形狀不變。"""
        tokens = torch.stack([h_cnt, h_gat], dim=1)          # (B,2,D)
        attn_out, _ = self.mha(tokens, tokens, tokens)
        tokens = self.norm(tokens + attn_out)
        return tokens[:, 0, :], tokens[:, 1, :]


class AE(nn.Module):
    """count 分支 + OD 圖分支，互相注意力融合後各自降維，再用純量 alpha 做凸組合進 latent。

    latent_dim：latent 維度。
    alpha_init：sigmoid 的起點 / 固定值（∈(0, ALPHA_MAX)，ALPHA_MAX/2 = 兩邊等重）。
    alpha_learn：True 時 alpha 是可學習參數，False 時釘死在 alpha_init。

    alpha_learn=False 的理由：alpha 只看得到 train loss，而我們遇到的問題是
    「train 看起來沒事、val 不好」，它在定義上偵測不到。釘死之後 alpha 變成
    乾淨的超參數，可以直接掃著看混合比例的影響。

    ALPHA_MAX：GAT 分支目前實測貢獻量很小（見檔頭 docstring），把它的權重
    上限釘死在 0.2，避免它稀釋 count 分支的訊號。
    """

    ALPHA_MAX = 0.2

    def __init__(self, latent_dim=2, alpha_init=0.1, alpha_learn=True):
        super().__init__()
        self.cnt_trunk = nn.Sequential(*_mlp_block(N_CAT, HIDDEN, N_HIDDEN_LAYERS))
        self.gat = GATBranch()
        # elementwise_affine=False：不帶可學習參數，只把兩條分支的尺度釘死，
        # 這樣 alpha 才真的是混合比例而不是可以被分支內部縮放抵銷掉的數字
        self.norm_cnt = nn.LayerNorm(HIDDEN, elementwise_affine=False)
        self.norm_gat = nn.LayerNorm(HIDDEN, elementwise_affine=False)
        self.fuse = BranchCrossAttn()
        raw = torch.logit(torch.tensor(float(alpha_init) / self.ALPHA_MAX))
        self.alpha_learn = alpha_learn
        if alpha_learn:
            self.alpha_raw = nn.Parameter(raw)
        else:
            # 用 buffer 而不是 Parameter：不會進 optimizer、不吃 weight decay，
            # 但仍然留在 state_dict 裡，checkpoint 存回來還看得到當初用的比例
            self.register_buffer("alpha_raw", raw)
        self.z_cnt = nn.Linear(HIDDEN, latent_dim)
        self.z_gat = nn.Linear(HIDDEN, latent_dim)
        self.decoder = nn.Sequential(
            *_mlp_block(latent_dim, HIDDEN, N_HIDDEN_LAYERS),
            nn.Linear(HIDDEN, N_CAT),        # 輸出是 log λ
        )

    def alpha(self):
        """回傳純量 tensor 的 alpha ∈ (0, ALPHA_MAX)。"""
        return self.ALPHA_MAX * torch.sigmoid(self.alpha_raw)

    def encode(self, x_in, a_local, knockout=False):
        """x_in：(B,N_CAT) 加噪 count。a_local：(B,N_CAT,N_CAT) 原始 OD 筆數。
        knockout：True 時把 h_gat 換成這一批的平均，切斷 GAT 分支傳遞的所有
                  per-patch 資訊——用來量這條分支到底有沒有在做事。

        回傳 (z, aux)：z (B,latent_dim)，aux 是 dict，含融合後的 h_gat / h_cnt
        （即 h'_gat / h'_cnt）與 att。
        """
        h_gat, att = self.gat(a_local)
        h_gat = self.norm_gat(h_gat)
        h_cnt = self.norm_cnt(self.cnt_trunk(x_in))
        if knockout:
            h_gat = h_gat.mean(dim=0, keepdim=True).expand_as(h_gat)
        h_cnt, h_gat = self.fuse(h_cnt, h_gat)   # 互相參考彼此的資訊，變成 h'_cnt / h'_gat
        a = self.alpha()
        # 各自降維後才混合，latent 前不接激活，離群值才不會被壓回來
        z = a * self.z_gat(h_gat) + (1 - a) * self.z_cnt(h_cnt)
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

    d2 用 eps 墊底：b<1 時 d2^b 在 d2=0 處的梯度是無限大，兩個不同 patch 剛好
    被 encoder 映到同一點（或負樣本剛好抽到 i==j）並不無法排除，沒墊底會讓
    backward 炸成 NaN。
    """
    d2 = (z_i - z_j).pow(2).sum(dim=1).clamp(min=eps)
    q = (1.0 + a * d2.pow(b)).reciprocal().clamp(eps, 1 - eps)
    return -(w * q.log() + (1 - w) * (1 - q).log())
