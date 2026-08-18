"""v0_l32_sizefactor：v0_sizefactor 開到 32 維，看組成訊號會不會回來。

跟 v0_sizefactor 唯一的差別是 LATENT_DIM，其他（正規化、loss、架構、切分）
完全相同，所以兩者的差異可以直接歸因於容量。

要解決的問題是 v0_sizefactor/analyze/latent_features.py 量出來的
（以下都是排除重疊窗之後的 R²，見該檔 docstring）：
size factor 那半邊成功了（log(POI 數) 的 kNN R² 只剩 0.079），
但密度讓出來的空間並沒有被組成填滿——

    entropy_norm  R² = -0.008      hhi     R² = -0.001
    max_lq        R² = -0.024      十類佔比的 R² 全部 < 0.06
    mean_r        R² = +0.715      ring_in R² = +0.719

2 維 latent 幾乎整個拿去描述「POI 離 patch 中心多遠」的徑向剖面。
這很可能是容量問題而不是設計問題：loss 是逐格的 Poisson NLL，
「哪一格有點」的資訊量遠大於「那個點是哪一類」，2 維不夠分的時候
模型當然先描述幾何。開到 32 維是最直接的檢驗。

判讀方式（跟 v0_sizefactor 同一張表比）：
  組成的 R² 明顯上來    -> 確實是容量問題，這條路可以繼續走
  組成的 R² 還是趴著    -> 是 loss 的問題，得改成對類別分布加權，
                          開更多維只是讓 latent 更難解釋而已
  log(POI 數) 的 R² 跟著上來 -> 多出來的維度被拿去記密度了，
                          size factor 的正規化在高維下沒有守住

已知風險：32 維之後 robust distance 與 kNN 都會受維度詛咒影響，
「鄰居」的意義變弱、距離趨於均勻，saturation.py 的 z-score 會比 2 維時保守。
散點圖一律先 PCA 投影到 2 維，但 kNN R² 與 robust distance 仍用完整 32 維算。

以下是 v0_sizefactor 原本的設計說明，除了維度以外都還適用。

---

v0_sizefactor：把「這個 patch 有幾個 POI」從 latent 裡結構性地拿掉。

要解決的問題，v0_weight_mse 的 docstring 已經量過了：latent 對 log(POI 數)
的 kNN R² 是 0.934，調 loss 權重不但搶不回來，w 越大 R² 反而越高
（0.934 -> 0.971）。原因很單純——n_poi 橫跨 10 ~ 1054 兩個數量級，
而 latent 只有 2 維，密度自然吃掉一整維。用 loss 去對抗一個尺度問題
是打不贏的。

作法照搬 scRNA-seq 的 size factor（DCA 那一套的核心其實是這個，不是 ZINB）：

    s = 這個 patch 的 POI 總數                      （= scRNA 的 library size）
    encoder 吃  log1p(x / s * S_REF)                 只剩「組成 + 空間排列」
    decoder 出  log mu_norm，再加回 log(s / S_REF)   才是真正的 log lambda
    loss 仍然對 raw count 算圓內 Poisson NLL

關鍵在最後兩行：decoder 描述的是「一個 S_REF 規模的 patch 長什麼樣」，
實際規模由 s 以 offset 的形式免費給它。模型沒有動機把容量花在記密度上，
因為密度不經過 latent 就已經到了 decoder 手上。

對專題目標的影響（重要，這是設計上的取捨不是 bug）：
「過飽和」本身就是密度現象，把密度從 latent 拿掉，latent 上的位移就只
反映組成/排列的變化。所以飽和度不再是「latent 離群」而要改成條件式的問法：
    給定這個 patch 的形狀 z，它的 n_poi 比同形狀的鄰居高多少？
這正是 analyze/saturation.py 在算的密度殘差。這樣講其實比原本乾淨——
原本那張圖上「離群」同時混著密度大和形狀怪，分不出是哪一種。

S_REF 只是把正規化後的數字拉回「典型 patch 的計數尺度」，避免 log1p
作用在 0.001 這種量級上等同線性。取全體 n_poi 的中位數（見檔案上方 S_REF，
幾何參數一改就要跟著對）。

已知風險：
  * 稀疏 patch 被放大得很兇。n_poi=10 的 patch，一個點會變成
    log1p(10/10*60) = log1p(6) = 1.95；n_poi=1000 的 patch 一個點只有
    log1p(0.06) = 0.058。也就是低密度 patch 的 latent 由極少數點決定，
    噪聲大。MIN_POI=10 目前是唯一的防線，必要時得往上調。
  * s 用的是圓內 POI 總數，跟 latents.npz 裡的 n_poi 相同（binning 只會
    把點移到邊界格，不會增減數量），所以兩者可以互相驗證。
  * 圓外 lambda 漂移的問題還在（見 v0_poisson_nll），但這一版 lambda 的
    尺度被 s 綁住了，漂移幅度應該小很多，rebuild_test 的圓周亮邊要重看。
"""

import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(f"{os.path.dirname(__file__)}/../.."))
from config.dataset import CELL, GRID, N_CAT, HALF_WIDTH  # noqa: E402,F401

IN_CH = N_CAT
# 參考規模：全體 n_poi 的中位數。改幾何參數(HALF_WIDTH/CELL)之後這個數字會變，
# 必須重新對一次——留著舊值會讓 log1p 作用在遠小於 1 的量級上而退化成線性，
# 正規化就白做了。目前值對應 HALF_WIDTH=800 / CELL=50（中位數 227）。
S_REF = 227.0



def normalize(counts):
    """raw count -> (encoder 輸入, log size factor)。

    size factor 取整個 patch 的 POI 總數（跨類別跨格子），不是每類各自算。
    每類各自算會連「哪一類多」都一起除掉，那正是我們要留在 latent 裡的東西。
    """
    s = counts.sum(dim=(1, 2, 3), keepdim=True).clamp(min=1.0)
    x_norm = torch.log1p(counts / s * S_REF)
    return x_norm, torch.log(s / S_REF)


class Patches:
    """稀疏點列表；render() 才展開成稠密矩陣。"""

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
        """回傳 (B,10,40,40) 的原始 count 矩陣；正規化交給 normalize()。"""
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
    """架構跟 v0_sizefactor 相同；差別只有 latent_dim 的預設值。"""

    def __init__(self, latent_dim=32):
        super().__init__()
        s1 = (GRID + 2*1 - 3) // 2 + 1
        s2 = (s1 + 2*1 - 3) // 2 + 1
        s3 = (s2 + 2*1 - 3) // 2 + 1
        self.enc_size = s3
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
            nn.ConvTranspose2d(32, IN_CH, 4, 2, 1),   # 20 -> 40，輸出是 log mu_norm
        )

    def forward(self, counts):
        """吃 raw count，回傳 (z, log lambda)。正規化與 offset 都包在這裡，
        呼叫端（train / analyze）跟 v0_poisson_nll 的介面完全一樣。"""
        x_norm, log_s = normalize(counts)
        z = self.encoder(x_norm)
        return z, self.decoder(z) + log_s


def poisson_nll(log_lam, x):
    """跟 v0_poisson_nll 同一個式子：每格 lambda - y*log lambda，只算圓內。"""
    cell = (torch.exp(log_lam) - x * log_lam)
    return cell.mean(dim=(1, 2, 3))


def poisson_deviance(log_lam, x):
    """Poisson deviance，完美重建為 0、永遠非負。存進 latents.npz 的 err 用這個。

    注意這一版的 deviance 天生會比 v0_poisson_nll 低：總量已經由 size factor
    免費給定，decoder 只要描述形狀。兩版的 err 不能比大小，只能各自比排序。
    """
    lam = torch.exp(log_lam)
    cell = 2 * (torch.xlogy(x, x) - x * log_lam - x + lam)
    return cell.mean(dim=(1, 2, 3))
