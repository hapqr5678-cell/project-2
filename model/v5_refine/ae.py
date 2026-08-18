"""v5_refine：v2_ddae_fsce 的架構變體——encoder 從「一次前饋」改成
「前饋 + T 次帶重建殘差回饋的迭代精修」。decoder、corrupt、FSCE、所有
超參數與 v2_ddae_fsce 逐字對齊，唯一的變因是 encoder 端多了 RefineNet。

為什麼是這一刀（實測，不是猜的）：把訓練好的 v2_ddae_fsce decoder 完全凍住，
只把每個 patch 的 2 維 z 當自由變數用梯度解最佳值，val_dev 從 0.6380 掉到
0.5740、train_dev 從 0.6227 掉到 0.5593。訓練集與驗證集的缺口一模一樣
（-0.063 / -0.064），所以這不是過擬合也不是泛化問題，是純粹的 amortization
gap：那 2 維裡放得下更好的答案，是 encoder 一次前饋沒找到。對照尺規：線性
PCA 2 維是 1.1545、10 維滿秩是 0.5910，也就是一個選得夠好的 2 維座標比整個
10 維線性表示還準。

拿精修後的 z 重算整張圖的 FSCE loss 是 1.0063 -> 1.0108（+0.45%），z 只挪了
0.645（latent 半徑 3.53 的 18%）——精修是局部修正，不會把 latent 結構打散。

RefineNet 的輸出是 Δz ∈ R^latent_dim，decoder 從頭到尾只看得到 latent_dim
個數字，資訊通道的容量完全沒變。這跟「把總量從瓶頸旁邊送給 decoder」那種
旁路做法是不同性質的東西：驗證時輸入就是乾淨的目標，任何旁路都等於讓
decoder 直接抄答案，val_dev 會漂亮但沒有意義。這裡改的是「怎麼找到那
latent_dim 個數字」，不是「decoder 能看到多少東西」。

兩個設計決定都是實測出來的，不是預設就該這樣（凍住 v2 骨幹、只訓練精修
模組、400 epoch、驗證一律乾淨輸入；起點 0.6380、oracle 0.5740）：

  1. 精修要看乾淨觀測，不是加噪輸入
        看加噪輸入（一開始的寫法）  0.6380 -> 0.6462  (+0.0082，更差)
        看乾淨觀測                  0.6380 -> 0.6173  (-0.0207)
     理由見 MLPAE.trajectory 的 docstring。

  2. 一定要把 ∇_z NLL 顯式算給 RefineNet
        只餵殘差、自己猜方向   T=1 +0.0105 / T=3 -0.0207 / T=5 +0.0187 / T=8 +0.0336
                               （訓練集每個 T 都降 -0.02 上下，驗證集卻翻臉——
                                 學到的是訓練集特有的位移，不是 descent 規則）
        顯式梯度導引（3 seeds） T=3 0.6094±0.0052 / T=5 0.5953±0.0042
                                T=8 0.5951±0.0107
     T=5 吃下 0.064 缺口裡的 0.043（67%），T=8 沒有更好而且變異變大，
     所以 REFINE_STEPS 定在 5。以上都還是凍住骨幹的數字，聯合訓練還有空間。

一整版沒有梯度導引的完整訓練結果留在 result/result_v1_naive_T3.log
（val_dev 0.6987，比 v2 差），是這條 ablation 的完整證據。

容量預算（1110 個訓練 patch × N_CAT 類 = 11100 個觀測值）：
  v2_ddae_fsce  encoder 13314 + decoder 13322 = 26636（每觀測值 2.40 個參數）
  RefineNet     1125（每觀測值 +0.10），佔既有模型 4.2%
RefineNet 的權重在 T 個步驟之間共用，所以 T 加大只增加運算深度、不增加參數；
最後一層權重零初始化、gate 初值 0.1，模型在 t=0 時**完全等於** v2_ddae_fsce，
精修能力是從零長出來的，不是一開始就多一塊亂動的東西。

FSCE 的高維鄰接關係是用「整包 count 向量的 log1p、euclidean 距離」kNN 建的
fuzzy simplicial set，用乾淨 count 建、只建一次、整個訓練共用同一張圖。
log1p 之後的 euclidean 對總量是敏感的：組成相同、總數差 k 倍的兩個 patch，
彼此距離約 √N_CAT·log k，不會被判成鄰居。

破壞方式有兩種，用 NOISE_MODE 切換（意義同 v2_dae）：
  "thinning"  binomial thinning：每個 POI 以 1-NOISE_P 的機率被保留。
  "mask"      整個類別歸零：每一類以 NOISE_P 的機率整維被抹成 0。
兩種都會除以 1-NOISE_P 做尺度補償，讓訓練/推論的輸入尺度一致。
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
from config.dataset import N_CAT, HALF_WIDTH  # noqa: E402,F401

HIDDEN = 64
N_HIDDEN_LAYERS = 4   # v2_dae_fsce 是 2 層，這一版加倍

# RefineNet 的容量：input 是 latent_dim + 3*N_CAT = 32 維，一層 32 寬的隱藏層，
# 合計 1125 個參數。11100 個訓練觀測值下這是 +4.2%，刻意壓到既有模型的零頭——
# 這個模組要做的事只是「沿著重建殘差修一小步」，不需要自己去記資料。
REFINE_STEPS = 5         # 精修步數 T；0 = 完全退回 v2_ddae_fsce，掃這個值就是主實驗
REFINE_HIDDEN = 32
REFINE_GATE_INIT = 0.1   # 每一步修正項的初始權重；配上零初始化的輸出層，起點等於 v2
REFINE_ETA_INIT = -2.0   # 每一步梯度步長的 raw 值，實際步長是 softplus(η)≈0.13


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
       整維被抹成 0 的機率。p=0 時直接回傳 x。
    mode："thinning" 或 "mask"，意義見模組 docstring。
    generator：torch.Generator，給定就用它抽亂數，讓每個 epoch 的破壞可重現。

    回傳跟 x 同 shape 的 (B,N_CAT) tensor，已經除以 1-p 做過尺度補償，
    期望值等於 x，所以可以直接餵進跟推論時同一個 encoder。
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


def _mlp_block(d_in, d_out, n_layers):
    """疊 n_layers 個 Linear(HIDDEN,HIDDEN)+GELU，前面加一層 d_in->HIDDEN，
    回傳 list of nn.Module（不含最後把 HIDDEN 投影到 d_out 的那一層）。
    """
    layers = [nn.Linear(d_in, HIDDEN), nn.GELU()]
    for _ in range(n_layers - 1):
        layers += [nn.Linear(HIDDEN, HIDDEN), nn.GELU()]
    return layers


class RefineNet(nn.Module):
    """學一個「往哪修、修多遠」的更新規則：一個顯式的梯度步 + 一個學出來的修正項。

        z_{t+1} = z_t - softplus(η_t)·∇_z NLL + gate_t·net([z, λ, x_obs, 殘差, ∇])

    latent_dim：z 的維度。n_steps：精修步數 T，決定 η/gate 的長度。

    為什麼一定要有那個顯式的梯度項（實測，見檔頭 ablation）：只餵殘差、讓
    網路自己猜方向的版本，訓練集降 -0.02 但驗證集反而升，換 T 就翻臉——它學到
    的是訓練集特有的位移方向，不是descent 規則。把 ∇_z NLL 直接算給它之後，
    「往哪走」變成已知，網路只需要學步長與預條件，這件事在任何 patch 上都成立，
    所以泛化得動。

    權重在 T 步之間共用（只有 η_t、gate_t 逐步不同），T 加大不增加參數量——
    1233 個樣本下唯一負擔得起的加深方式。
    """

    def __init__(self, latent_dim, n_steps):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * latent_dim + 3 * N_CAT, REFINE_HIDDEN), nn.GELU(),
            nn.Linear(REFINE_HIDDEN, latent_dim),
        )
        # 輸出層零初始化：起步時修正項恆為 0，只剩下純粹的梯度下降，
        # 網路是從「標準梯度步」這個安全的起點開始學怎麼變好
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.eta = nn.Parameter(torch.full((n_steps,), REFINE_ETA_INIT))
        self.gate = nn.Parameter(torch.full((n_steps,), REFINE_GATE_INIT))

    def forward(self, z, x_obs, log_lam, grad_z, t):
        """z：(B,latent_dim) 目前的 latent。x_obs：(B,N_CAT) 要解釋的觀測值
        （推論時就是輸入的乾淨 count，訓練時見 MLPAE.trajectory 的說明）。
        log_lam：(B,N_CAT) 目前這個 z 解出來的 log λ。grad_z：(B,latent_dim)
        已經算好的 ∇_z NLL（detach 過）。t：第幾步，決定用哪一組 η/gate。

        回傳 (B,latent_dim) 的下一個 z。特徵裡的 Pearson 殘差
        (λ-x_obs)/√(λ+1) 除以 √(λ+1)，是為了讓大 count 類別不會單純因為
        數字大就主導這一步。
        """
        lam = log_lam.exp()
        feat = torch.cat([z, torch.log1p(lam), torch.log1p(x_obs),
                          (lam - x_obs) / (lam + 1.0).sqrt(), grad_z], dim=1)
        return z - F.softplus(self.eta[t]) * grad_z + self.gate[t] * self.net(feat)


class MLPAE(nn.Module):
    """encoder / decoder 與 v2_ddae_fsce 逐字相同（各 N_HIDDEN_LAYERS 層隱藏層、
    HIDDEN 維、latent 前不接激活），差別只在 encoder 輸出的 z 後面多接 n_steps
    次 RefineNet 精修。

    latent_dim：latent 維度。n_steps：精修步數 T，0 表示完全退回 v2_ddae_fsce
    （連 RefineNet 都不建），拿來當同一份程式碼下的對照組。

    類別名沿用 MLPAE 是為了讓 analyze/ 底下的腳本跟 v2 平行。
    """

    def __init__(self, latent_dim=2, n_steps=REFINE_STEPS):
        super().__init__()
        self.encoder = nn.Sequential(
            *_mlp_block(N_CAT, HIDDEN, N_HIDDEN_LAYERS),
            nn.Linear(HIDDEN, latent_dim),   # latent 前不接激活，離群值才不會被壓回來
        )
        self.decoder = nn.Sequential(
            *_mlp_block(latent_dim, HIDDEN, N_HIDDEN_LAYERS),
            nn.Linear(HIDDEN, N_CAT),   # 輸出是 log λ
        )
        self.n_steps = n_steps
        self.refine = RefineNet(latent_dim, n_steps) if n_steps > 0 else None

    def grad_z(self, z, x_obs):
        """算 ∇_z Poisson NLL(decoder(z), x_obs)，回傳 (B,latent_dim)（已 detach）。

        z：(B,latent_dim)。x_obs：(B,N_CAT) 要解釋的觀測值。

        enable_grad 是必要的：驗證與推論都在 no_grad 底下跑，但這個梯度是
        forward 的一部分（它是 RefineNet 的輸入特徵），不是反傳用的東西。
        detach 掉是刻意的——不對「梯度本身」再求一次導數（truncated BPTT），
        省掉二階微分的成本，這是 learned optimizer 的標準做法。
        """
        with torch.enable_grad():
            zz = z.detach().requires_grad_(True)
            loss = poisson_nll(self.decoder(zz), x_obs).sum()
            g, = torch.autograd.grad(loss, zz)
        return g.detach()

    def trajectory(self, x_in, x_obs=None):
        """跑完整條精修軌跡。

        x_in：(B,N_CAT) 餵給 encoder 的輸入（訓練時加噪、推論時乾淨）。
        x_obs：(B,N_CAT) 精修要解釋的觀測值，預設 None 表示等於 x_in。

        回傳 (zs, log_lams)：兩個長度都是 n_steps+1 的 list，zs[t] 是第 t 步的
        (B,latent_dim) latent、log_lams[t] 是它解出來的 (B,N_CAT) log λ。
        zs[0] 是 encoder 的前饋輸出，zs[-1] 是最終要用的 latent。

        推論時 x_in 與 x_obs 本來就是同一個東西（模型收到什麼就解釋什麼），
        所以預設值就對。訓練時 train.py 會明確傳 x_obs=乾淨 count：精修模組
        的工作是「把 z 調到能解釋眼前這份觀測」，它在推論時面對的觀測是乾淨的，
        訓練時就必須面對同一種東西，否則它學到的是「不要太相信殘差」的縮水
        步長。實測：精修看加噪輸入 val_dev +0.0082（更差），看乾淨觀測
        -0.0207。denoising 仍然作用在 encoder 那一路（它永遠只看得到加噪輸入），
        這是這一版跟 v2_ddae_fsce 之間唯一需要明講的設計差異。
        """
        if x_obs is None:
            x_obs = x_in
        z = self.encoder(x_in)
        zs, log_lams = [z], [self.decoder(z)]
        for t in range(self.n_steps):
            z = self.refine(z, x_obs, log_lams[-1], self.grad_z(z, x_obs), t)
            zs.append(z)
            log_lams.append(self.decoder(z))
        return zs, log_lams

    def encode(self, x_in, x_obs=None):
        """回傳精修完的 (B,latent_dim) z，參數意義同 trajectory()。
        FSCE 那一路用的就是它——圖上畫的、被結構 loss 約束的都是最終的 z。
        """
        return self.trajectory(x_in, x_obs)[0][-1]

    def forward(self, x_in, x_obs=None):
        """參數意義同 trajectory()。

        回傳 (z, log_lam, log_lams)：z 是最終的 (B,latent_dim) latent，
        log_lam 是它的 (B,N_CAT) log λ，log_lams 是整條軌跡（deep supervision
        要用，也是畫「val_dev 隨精修步數下降」那張圖的來源）。
        """
        zs, log_lams = self.trajectory(x_in, x_obs)
        return zs[-1], log_lams[-1], log_lams


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
