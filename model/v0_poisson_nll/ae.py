"""v0_poisson_nll：把重建當成「每格每類的 POI 數服從 Poisson」來算 likelihood。

跟 v0 的差別有三件事，但核心只有一個想法：count 資料不該用 MSE。

1. 輸入不再是 log1p(count)，而是原始 count
   Poisson 的 target 必須是非負整數，這一版乾脆讓 in = out 都是 raw count，
   統計解釋最乾淨：encoder 看到的就是真實計數，decoder 描述的就是它的分布。

2. decoder 的輸出不是「重建的矩陣」而是 log λ
   形狀還是 (N_CAT, 40, 40)，但每個數字是該格該類 Poisson 強度的對數。
   用 log link（而不是直接輸出 λ）是因為 λ 必須為正，且對 log λ 的梯度
   剛好是 λ - y，數值上很溫和：λ 再怎麼小梯度也只趨近 -y，不會爆。

3. loss 只算圓內的格子
   patch 是半徑 300m 的圓，40x40 的四個角永遠在圓外。MSE 下那些格子只是
   「很好猜的 0」無傷大雅，但 Poisson 會把它們的 log λ 一路推向 -inf，
   等於拿模型容量去描述一個純粹的幾何假象。所以用 MASK 把圓外整個排除，
   語義上也比較誠實：圓外不是「觀測到 0」，而是「沒有觀測」。
   圓內約佔 1257/1600 = 78.6% 的格子，loss 的分母是圓內格數 x N_CAT。

為什麼值得做：v0 的 MSE 等於假設每格的誤差是等變異的高斯，但 POI 計數
是「大部分格子 0、少數格子 3~5」的離散分布，變異數本來就隨 λ 增長。
Poisson NLL 自動給低強度區域較大的相對權重，不必像 v0_weight_mse 那樣
人工調 POS_WEIGHT——飽和/過飽和的判斷本來就該建立在「這個密度有多不尋常」
而不是「差值平方多大」上。

已知風險：
  * 輸入是未正規化的 raw count（多數格 0、少數格 5+），第一層 conv 後才有
    GroupNorm，初期梯度尺度會比 v0 大。LR 沿用 1e-3 但如果 loss 出現 nan，
    第一個要調的就是它。
  * log λ 直接 exp，沒有 clamp（跟 torch 的 PoissonNLLLoss 一樣）。
    float32 在 log λ > 88 會 overflow 成 inf。decoder 最後一層沒有 norm、
    初始化下輸出接近 0（λ≈1），正常訓練碰不到，但 loss 變 nan 就是這裡。
  * 存進 latents.npz 的 err 是 Poisson deviance 不是 NLL，跟 v0 / v0_weight_mse
    的 err 不能直接比大小（見 poisson_deviance 的說明）。
  * 圓外的格子沒有梯度，λ 會自由漂移到很大的值（實測某個 patch 圓內 λ 總和
    15.9、圓外卻是 1378）。這本身無害——encoder 的輸入圓外恆為 0，latent
    不受影響——但 conv 的感受野會讓它從邊界滲進圓內幾格，rebuild_test 的圖上
    看得到圓周附近偏亮。要根治得在遮罩之外再罰圓外的 λ，這一版沒做。
"""

import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(f"{os.path.dirname(__file__)}/../.."))
from config.dataset import CELL, GRID, N_CAT, HALF_WIDTH  # noqa: E402,F401

IN_CH = N_CAT

# 形狀 (1,1,GRID,GRID)，broadcast 到所有 batch 與 channel。


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
        """idx 是 patch 編號的 tensor，回傳 (B,10,40,40) 的原始 count 矩陣。

        跟 v0 唯一的差別：最後不套 log1p。Poisson 的 target 要是真正的計數。
        """
        b = len(idx)
        starts, ends = self.offsets[idx], self.offsets[idx + 1]
        lens = ends - starts
        # 把每個 patch 的點攤平成一條，並記住它屬於 batch 裡的第幾個
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
    """架構跟 v0 完全一樣，只是 decoder 的輸出解釋成 log λ 而不是重建值。"""

    def __init__(self, latent_dim=2):
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
            nn.ConvTranspose2d(32, IN_CH, 4, 2, 1),   # 20 -> 40，輸出是 log λ
        )

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        if out.shape[-1] != x.shape[-1] or out.shape[-2] != x.shape[-2]:
            import torch.nn.functional as F
            out = F.interpolate(out, size=(x.shape[-2], x.shape[-1]), mode='bilinear', align_corners=False)
        return z, out


def poisson_nll(log_lam, x):
    """Poisson negative log-likelihood，省略跟模型無關的 log(y!) 常數項。

    每格 = λ - y·log λ，只加總圓內的格子再除以 N_VALID，
    所以不同 patch 之間可以直接比較。回傳每個 patch 一個數字。
    這是訓練用的 loss；常數項省不省不影響梯度，也不影響 patch 之間的排序。
    """
    cell = (torch.exp(log_lam) - x * log_lam)
    return cell.mean(dim=(1, 2, 3))


def poisson_deviance(log_lam, x):
    """Poisson deviance：2·(y·log(y/λ) - (y - λ))，y=0 時第一項定義為 0。

    存進 latents.npz 當 err 的是這個而不是 NLL。理由是 deviance 減掉了
    saturated model（λ = y）的 log-likelihood，所以完美重建時剛好是 0、
    永遠非負，跟 v0 的 MSE 一樣是「越小越好、有下界」的量，
    後面 robust distance 與各張圖的語義才不用改。

    NLL 保留 log(y!) 與否都會讓密度高的 patch 天生有較大的基準值，
    用它排序離群等於又把密度混進來——deviance 沒有這個問題。

    注意數值：xlogy(0, 0) = 0，所以空格不會產生 nan。
    """
    lam = torch.exp(log_lam)
    cell = 2 * (torch.xlogy(x, x) - x * log_lam - x + lam)
    return cell.mean(dim=(1, 2, 3))
