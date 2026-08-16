"""用手算特徵表替 v0_sizefactor 的 latent 上色，驗證它是不是真的只剩形狀。

v0_sizefactor 的整個設計前提是：把 size factor 當 offset 餵給 decoder 之後，
latent 就不需要記密度，只剩「組成 + 空間排列」。這句話一直沒有被正面檢查過——
saturation.py 直接假設它成立，在 latent 上找「形狀鄰居」再比 n_poi。
如果 latent 其實還是編了密度，那整套飽和度 z-score 就變成拿密度去比密度，
數字再漂亮也沒有意義。這支腳本就是那個檢查。

判讀方式（kNN R² = 從 latent 的 50 個鄰居能還原多少該特徵）：
  log_n_total   應該「低」。v0 是 0.934，這一版要明顯掉下來才算成功。
  entropy_norm / hhi / max_lq  應該「高」。這些是組成，latent 該編的東西。
  mean_r / vmr  應該「中高」。這些是空間排列，同樣是 latent 該編的東西。

max_lq 是這版新加的：entropy 與 hhi 只說得出「組成單一」，說不出單一在哪一類。
再把每一類的佔比 R² 印出來，就能看出 latent 的軸到底跟著哪個類別走——
如果只有某一兩類的 R² 特別高，那 latent 其實是一張「那類店的密度圖」。

  注意單一類別不必印 lq 的 R²：lq_c = p_c / (該類全體佔比)，只是每欄乘一個
  常數，而 R² 對線性縮放不變，兩者必然相等。LQ 的價值在絕對值可解讀
  （LQ=3 就是三倍過度集中），不在 R²。只有 max_lq 因為牽涉跨類別 argmax，
  才帶有 p_c 系列沒有的資訊。

已知風險：
  * kNN R² 只測「latent 附近的點該特徵是否相近」，不保證是線性或單調關係，
    也不代表因果。R² 高只說明資訊在 latent 裡，不說明它佔了哪一維。
  * patch 中心是 CENTER_STEP=50m 的規則格點、窗寬 1600m，相鄰 patch 共用
    約 94% 的 POI，實際上是同一個地點的複本。這些複本很容易成為彼此在
    latent 上的最近鄰，等於拿一個地方跟它自己比，會讓所有 R² 一起虛高。
    所以一律印兩欄：原始 R²，以及排除地理距離 < EXCLUDE_M 的鄰居之後的 R²。
    **判讀一律看第二欄**；第一欄留著是為了看污染有多嚴重。EXCLUDE_M 取一個
    窗寬（2*HALF_WIDTH），因為距離小於窗寬的兩個 patch 必定共用 POI。
    這一版（2 維）的污染其實不嚴重——鄰居落在窗寬內的只有 4.0%，
    32 維版是 31.5%，維度越高複本越容易變成最近鄰。
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

LATENTS = result("v0_sizefactor", "latents.npz")
OUT = result("v0_sizefactor", "latent_features.png")

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
    ax.set_xlabel("z1", fontsize=8)
    ax.set_ylabel("z2", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.15, linewidth=0.5)
    for s in ax.spines.values():
        s.set_alpha(0.3)


def scatter(ax, z, c, title, label, cmap):
    lo, hi = np.nanpercentile(c, [1, 99])
    sc = ax.scatter(z[:, 0], z[:, 1], c=c, s=DOT, cmap=cmap, linewidths=0,
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

    fig, axes = plt.subplots(2, 3, figsize=(17, 10))
    for ax, (key, label, group, cmap) in zip(axes.ravel(), COLS):
        scatter(ax, z, f[key].astype(float), f"色={group}", label, cmap)

    fig.suptitle(
        "v0_sizefactor latent space 依手算特徵上色\n"
        f"（{len(z)} 個 patch；設計上「量」應該掉下來、「組成/空間」應該留著）",
        fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT, bbox_inches="tight")
    print(f"\n已存 {OUT}")


main()
