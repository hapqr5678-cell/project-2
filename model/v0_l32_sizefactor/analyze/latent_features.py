"""用手算特徵表替 v0_l32_sizefactor 的 latent 上色，看組成訊號有沒有回來。

這支的存在理由就是 v0_sizefactor 版本量出來的那張表
（排除重疊窗之後的 R²，見下面第三點）：

    量      log_n_total   +0.079      <- size factor 生效了，密度確實出去了
    組成    entropy_norm  -0.008      hhi -0.001   max_lq -0.024
            十類佔比的 R² 全部 < 0.06
    空間    mean_r        +0.715      ring_in +0.719   ring_out +0.665

2 維 latent 幾乎整個拿去描述「POI 離 patch 中心多遠」的徑向剖面，組成完全
沒進去。假設是容量不足（逐格 Poisson NLL 之下，「哪一格有點」的資訊量遠大於
「那個點是哪一類」，維度不夠時模型當然先描述幾何），所以開到 32 維再量一次。

三種結果分別代表什麼，寫在 ae.py 的 docstring 裡，這裡只負責產生數字。
要點是「跟 v0_sizefactor 同一張表逐列比」，不是看絕對值高低。

高維帶來的三個差異：
  * 散點圖畫不出 32 維，一律先 PCA 投影到前 2 個主成分（會印出解釋變異比例；
    比例很低的話那張圖只是個示意，不要拿來下結論）。kNN R² 仍用完整 32 維算。
  * 維度詛咒會讓 kNN 的「鄰居」變得不那麼有意義，距離趨於均勻。這個方向是
    讓所有 R² 一起偏低。
  * **反方向而且嚴重得多的是重疊窗污染**。patch 中心每 50m 一個、窗寬 1600m，
    相鄰 patch 共用約 94% 的 POI，實際上是同一個地點的複本。維度越高、latent
    越能分辨細節，這些複本就越容易變成彼此的最近鄰：實測 latent 的 50 個鄰居
    裡，地理距離 <200m 的比例在 2 維版是 2.2%，32 維版是 24.7%。這會讓所有
    特徵的 R² 一起虛高，因為那等於拿一個地方跟它自己比。

    所以這裡一律印兩欄：原始 R²，以及排除地理距離 < EXCLUDE_M 的鄰居之後的
    R²。**判讀一律看第二欄**；第一欄留著是為了看污染有多嚴重。EXCLUDE_M 取
    一個窗寬（2*HALF_WIDTH），因為距離小於窗寬的兩個 patch 必定共用 POI。

其餘已知風險與 v0_sizefactor 版相同：
  * kNN R² 只測「latent 附近的點該特徵是否相近」，不保證線性或單調，也不代表
    因果。R² 高只說明資訊在 latent 裡，不說明它佔了哪幾維。
  * 單一類別不必印 lq 的 R²：lq_c = p_c / (該類全體佔比)，只是每欄乘一個常數，
    而 R² 對線性縮放不變，兩者必然相等。只有 max_lq 牽涉跨類別 argmax，
    才帶有 p_c 系列沒有的資訊。
  * features.npz 與 latents.npz 必須來自同一份 patches.npz，否則列數對不上。
    對不上時直接報錯，不要自作聰明用 lat/lon 去配。
"""

import os
import sys

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.abspath(f"{os.path.dirname(__file__)}/../../.."))
from config.dataset import (CATEGORIES, FEATURES, HALF_WIDTH,  # noqa: E402
                            PATCHES, result)

LATENTS = result("v0_l32_sizefactor", "latents.npz")
OUT = result("v0_l32_sizefactor", "latent_features.png")

KNN = 50
# 排除地理距離小於此的鄰居：一個窗寬，小於它的兩個 patch 必定共用 POI
EXCLUDE_M = 2 * HALF_WIDTH
# 先取這麼多個 latent 鄰居再篩，篩完至少要剩 MIN_NB 個才算數
POOL = 400
MIN_NB = 10
DOT = 3.0

# 六張圖：一張量、三張組成、兩張空間
COLS = [
    ("log_n_total", "log1p(POI 總數)", "量（應該低）", "viridis"),
    ("entropy_norm", "標準化 Shannon 熵", "組成：多樣性", "cividis"),
    ("hhi", "HHI = Σp²", "組成：集中度", "magma"),
    ("max_lq", "最大 LQ", "組成：最過度集中那類", "inferno"),
    ("mean_r", "平均距中心距離 (m)", "空間：擴散程度", "plasma"),
    ("vmr", "變異數/平均 (每格計數)", "空間：群聚程度", "cool"),
]

# 只印 R² 不畫圖：用來確認上面六張圖的判讀，不佔版面
EXTRA = [
    ("n_occupied", "量"),
    ("std_r", "空間"),
    ("nn_dist", "空間"),
    ("clark_evans", "空間"),
    ("ring_in", "空間"),
    ("ring_out", "空間"),
]

mpl.rcParams["font.family"] = ["Heiti TC"]
mpl.rcParams["axes.unicode_minus"] = False
mpl.rcParams["figure.dpi"] = 130


def pca2(z):
    """投影到前 2 個主成分，回傳 (座標, 兩軸各自的解釋變異比例)。"""
    zc = z - z.mean(0)
    _, s, vt = np.linalg.svd(zc, full_matrices=False)
    ratio = s ** 2 / (s ** 2).sum()
    return zc @ vt[:2].T, ratio[:2]


def neighbors(zs, xy, exclude_m):
    """latent 上的 KNN 個鄰居；exclude_m > 0 時排除地理距離太近的重疊窗。

    不足 MIN_NB 個的列補 -1，knn_r2 會把那些列整列丟掉。
    """
    idx = cKDTree(zs).query(zs, k=POOL, workers=-1)[1][:, 1:]
    if exclude_m <= 0:
        return idx[:, :KNN]
    far = np.linalg.norm(xy[idx] - xy[:, None, :], axis=2) >= exclude_m
    out = np.full((len(zs), KNN), -1, dtype=np.int64)
    for i in range(len(zs)):
        c = idx[i][far[i]][:KNN]
        if len(c) >= MIN_NB:
            out[i, :len(c)] = c
    return out


def knn_r2(nb, y):
    """用 latent 的鄰居預測 y，回傳 R²。nb 裡的 -1 是被排除掉的位置。"""
    v = np.where(nb >= 0, y[np.maximum(nb, 0)], np.nan)
    cnt = (nb >= 0).sum(1)
    pred = np.divide(np.nansum(v, axis=1), np.maximum(cnt, 1),
                     where=cnt > 0, out=np.full(len(nb), np.nan))
    ok = (cnt >= MIN_NB) & np.isfinite(y) & np.isfinite(pred)
    if ok.sum() < 2:
        return np.nan
    return 1 - np.var(y[ok] - pred[ok]) / np.var(y[ok])


def style(ax, title):
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("PC1", fontsize=8)
    ax.set_ylabel("PC2", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.15, linewidth=0.5)
    for s in ax.spines.values():
        s.set_alpha(0.3)


def scatter(ax, zp, c, title, label, cmap):
    lo, hi = np.nanpercentile(c, [1, 99])
    sc = ax.scatter(zp[:, 0], zp[:, 1], c=c, s=DOT, cmap=cmap, linewidths=0,
                    alpha=0.6, rasterized=True, vmin=lo, vmax=hi)
    cb = ax.figure.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label(label, fontsize=8)
    cb.ax.tick_params(labelsize=7)
    style(ax, title)


def main():
    z = np.load(LATENTS)["z"]
    f = np.load(FEATURES)

    if len(z) != len(f["n_total"]):
        raise SystemExit(
            f"latents 有 {len(z)} 列、features 有 {len(f['n_total'])} 列。\n"
            f"兩者來自不同的 patches.npz，請重跑 build_patches -> train -> "
            f"build_features。")

    p = np.load(PATCHES)
    xy = np.column_stack([p["center_x"], p["center_y"]]).astype(float)

    zs = (z - z.mean(0)) / z.std(0)
    nb_all = neighbors(zs, xy, 0.0)
    nb_far = neighbors(zs, xy, EXCLUDE_M)
    zp, ratio = pca2(z)

    d_all = np.linalg.norm(xy[nb_all] - xy[:, None, :], axis=2)
    print(f"latent 鄰居的地理距離：中位數 {np.median(d_all):.0f}m，"
          f"其中 {(d_all < EXCLUDE_M).mean():.1%} 落在一個窗寬內（＝重疊窗，"
          f"拿自己跟自己比）")
    kept = (nb_far >= 0).sum(1)
    print(f"排除後每個 patch 平均剩 {kept.mean():.1f} 個鄰居，"
          f"{(kept < MIN_NB).sum()} 個 patch 不足 {MIN_NB} 個已整列剔除\n")

    print(f"latent 能還原多少（kNN R², k={KNN}，{len(z)} 個 patch，"
          f"{z.shape[1]} 維）")
    print(f"{'':<18}{'':<14}{'原始':>9}{'排除重疊窗':>12}  <- 看這欄\n")

    def row(label, key, y):
        print(f"  {label:<18}{key:<14}"
              f"{knn_r2(nb_all, y):>9.3f}{knn_r2(nb_far, y):>12.3f}")

    for key, _, group, _ in COLS:
        row(group, key, f[key].astype(float))
    print()
    for key, group in EXTRA:
        row(group, key, f[key].astype(float))

    print("\n各類別佔比的 R²（lq 的 R² 與 p 相同，見 docstring）")
    print(f"  {'類別':<38}{'原始':>7}{'排除後':>9}{'LQ 中位數':>11}{'LQ p99':>9}")
    for name in CATEGORIES:
        k = name.split()[0].lower()
        y, lq = f["p_" + k].astype(float), f["lq_" + k].astype(float)
        print(f"  {name:<36}"
              f"{knn_r2(nb_all, y):>7.3f}{knn_r2(nb_far, y):>9.3f}"
              f"{np.median(lq):>11.2f}{np.percentile(lq, 99):>9.2f}")

    print(f"\nPCA 前兩軸解釋 {ratio[0]:.1%} + {ratio[1]:.1%} = "
          f"{ratio.sum():.1%} 的變異（散點圖只看得到這麼多）")

    fig, axes = plt.subplots(2, 3, figsize=(17, 10))
    for ax, (key, label, group, cmap) in zip(axes.ravel(), COLS):
        scatter(ax, zp, f[key].astype(float), f"色={group}", label, cmap)

    fig.suptitle(
        f"v0_l32_sizefactor latent space 依手算特徵上色\n"
        f"（{len(z)} 個 patch，latent_dim={z.shape[1]}，散點為 PCA 前 2 維，"
        f"解釋 {ratio.sum():.0%} 變異；R² 用完整 {z.shape[1]} 維算）",
        fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT, bbox_inches="tight")
    print(f"\n已存 {OUT}")


main()
