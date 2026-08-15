"""手算每個 patch 的特徵表，完全不經過模型。

只吃 config 指定的 patch 快取（原始 POI 的相對座標與類別），輸出一張
每個 patch 一列的表，之後用來檢查 latent space 到底編碼了什麼。

四個維度，刻意分開量、組成、集中度、空間，因為 latent 離群同時混著這幾種
東西的時候，「過飽和」這個結論就沒有內容：

  量      n_total / log1p / n_occupied
  組成    n_c、p_c、entropy、entropy_norm、max_p
  集中度  hhi、lq_c、max_lq、max_lq_cat
  空間    mean_r、std_r、nn_dist、clark_evans、vmr、內中外三環佔比

LQ（location quotient）是這一版新增的重點：entropy 與 max_p 只能說「這裡
組成很單一」，說不出單一在哪一類。LQ = 該類在本 patch 的佔比 / 該類在全體
的佔比，>1 就是相對全東京過度集中，能直接指名是餐飲洗版還是零售洗版。

  LQ 的基準用來源 CSV 的全體 POI（每個 POI 算一次），不是把所有 patch 的
  計數 pool 起來——後者會因為 patch 高度重疊而讓密集區的 POI 被重複計算
  幾十次，基準會被市中心的組成拉走。

幾何跟 build_patches.py 對齊：那邊用 p=inf 取的是邊長 2*HALF_WIDTH 的
「方形」窗不是圓，所以三環也改成同心方環（Chebyshev 距離），邊界取
HALF_WIDTH 的 1/3 與 2/3，換 config 參數不用改這裡。mean_r / std_r 仍用
歐氏距離，它們是「散得多開」的尺度而不是分環，不需要跟窗形一致。

已知風險：
  * VMR 與 n_occupied 對 CELL 極度敏感（MAUP）。換了 CELL 之後這兩欄
    不能跟舊表比較，entropy / HHI / LQ 則不受影響（它們只吃點計數）。
  * Clark-Evans R 沒有做邊界修正，靠窗邊的 POI 的最近鄰可能落在窗外卻
    看不到，nn_dist 被高估、R 偏大。而且實測東京 POI 從 5m 到 300m 的
    g(r) 全程遠大於 1（沒有任何排斥尺度，一棟雑居ビル在 2D 上就是同一點），
    所以 R 幾乎必然 < 1，它只有 patch 之間相對比較的意義，不能拿來論證
    「這裡已經擠到極限」。
  * LQ 在稀有類別上方差極大：n_total=40 的 patch 多一家夜生活店，LQ 就
    跳一大截。所以 max_lq 只在 n_c >= LQ_MIN_N 的類別裡挑，全部不合格時
    退回用計數最多的類別。看單一類別的 lq_ 欄位時務必同時看 n_ 欄位。
  * entropy / HHI / LQ 完全不看空間排列：組成相同但一邊聚成一團、一邊
    均勻散開的兩個 patch 會拿到一樣的值。這正是還要留 mean_r / VMR 的原因。
"""

import os
import sys

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.abspath(f"{os.path.dirname(__file__)}/../.."))
from config.dataset import (CAT_COL, CATEGORIES, CELL, CSV,  # noqa: E402
                            FEATURES, FEATURES_CSV, GRID, HALF_WIDTH,
                            N_CAT, PATCHES)

OUT_NPZ = FEATURES
OUT_CSV = FEATURES_CSV

# 同心方環的邊界，取 HALF_WIDTH 的幾分之幾（最後一環就是窗邊界，不用寫）
RING_FRAC = [1 / 3, 2 / 3]

# 算 max_lq 時，類別至少要有這麼多個 POI 才納入，避免單一店造成的假訊號
LQ_MIN_N = 5


def nn_dist(dx, dy, offsets):
    """每個 patch 內，POI 到最近鄰的平均距離(公尺)。重疊點算 0。"""
    out = np.zeros(len(offsets) - 1)
    for i in range(len(out)):
        s, e = offsets[i], offsets[i + 1]
        pts = np.column_stack([dx[s:e], dy[s:e]])
        if len(pts) < 2:
            out[i] = np.nan
            continue
        d, _ = cKDTree(pts).query(pts, k=2)
        out[i] = d[:, 1].mean()
    return out


def global_p():
    """全體 POI 的各類別佔比，當 LQ 的分母。"""
    src = pd.read_csv(CSV)
    idx = {name: i for i, name in enumerate(CATEGORIES)}
    c = src[CAT_COL].map(idx)
    c = c[c.notna()].to_numpy(dtype=np.int64)
    return np.bincount(c, minlength=N_CAT) / len(c), len(c), len(src)


def main():
    d = np.load(PATCHES)
    dx, dy, cat, offsets = d["dx"], d["dy"], d["cat"].astype(np.int64), d["offsets"]
    n_total = np.diff(offsets).astype(np.int64)
    n = len(n_total)
    owner = np.repeat(np.arange(n), n_total)
    n_cell = GRID * GRID

    # 量 -----------------------------------------------------------------
    # 佔用格子數與 VMR：用跟模型輸入一樣的 binning，但不分 channel
    ix = np.clip(np.floor(dx / CELL + GRID / 2).astype(np.int64), 0, GRID - 1)
    iy = np.clip(np.floor(dy / CELL + GRID / 2).astype(np.int64), 0, GRID - 1)
    key = owner * n_cell + iy * GRID + ix
    uniq, cnt = np.unique(key, return_counts=True)
    which = uniq // n_cell
    n_occupied = np.bincount(which, minlength=n)

    # VMR：把整個 patch 當 n_cell 個 quadrat，空格也算進去（所以是母體變異數）
    # Poisson(完全隨機) 時期望 1，>1 群聚、<1 規則
    sum_sq = np.bincount(which, weights=cnt.astype(float) ** 2, minlength=n)
    mean_c = n_total / n_cell
    var_c = sum_sq / n_cell - mean_c ** 2
    vmr = var_c / np.maximum(mean_c, 1e-12)

    # 組成 ---------------------------------------------------------------
    n_c = np.bincount(owner * N_CAT + cat,
                      minlength=n * N_CAT).reshape(n, N_CAT)
    p_c = n_c / n_total[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        logp = np.where(p_c > 0, np.log2(p_c), 0.0)
    entropy = -(p_c * logp).sum(1)
    entropy_norm = entropy / np.log2(N_CAT)
    max_p = p_c.max(1)
    hhi = (p_c ** 2).sum(1)          # 0~1，沒有乘 10000

    # 集中度：LQ -----------------------------------------------------------
    p_g, n_src_used, n_src = global_p()
    lq = np.where(p_g > 0, p_c / np.maximum(p_g, 1e-12), np.nan)

    elig = n_c >= LQ_MIN_N
    max_lq_cat = np.where(elig, lq, -np.inf).argmax(1)
    none = ~elig.any(1)
    max_lq_cat[none] = n_c[none].argmax(1)
    max_lq = lq[np.arange(n), max_lq_cat]

    # 空間 ---------------------------------------------------------------
    r = np.hypot(dx, dy).astype(np.float64)
    mean_r = np.bincount(owner, weights=r, minlength=n) / n_total
    mean_r2 = np.bincount(owner, weights=r ** 2, minlength=n) / n_total
    std_r = np.sqrt(np.maximum(mean_r2 - mean_r ** 2, 0.0))

    cheb = np.maximum(np.abs(dx), np.abs(dy)).astype(np.float64)
    ring = np.digitize(cheb, [f * HALF_WIDTH for f in RING_FRAC])
    ring_frac = np.stack([
        np.bincount(owner, weights=(ring == k), minlength=n) / n_total
        for k in range(len(RING_FRAC) + 1)
    ], axis=1)

    nn = nn_dist(dx, dy, offsets)

    # Clark-Evans R = 觀察到的平均最近鄰距離 / 完全隨機下的期望 1/(2*sqrt(密度))
    area = (2.0 * HALF_WIDTH) ** 2
    lam = n_total / area
    clark_evans = nn / (1.0 / (2.0 * np.sqrt(lam)))

    # 存表 ---------------------------------------------------------------
    tbl = {
        "lat": d["center_lat"], "lon": d["center_lon"],
        "n_total": n_total, "log_n_total": np.log1p(n_total),
        "n_occupied": n_occupied,
        "entropy": entropy, "entropy_norm": entropy_norm,
        "max_p": max_p, "hhi": hhi,
        "max_lq": max_lq, "max_lq_cat": max_lq_cat,
        "mean_r": mean_r, "std_r": std_r, "nn_dist": nn,
        "clark_evans": clark_evans, "vmr": vmr,
        "ring_in": ring_frac[:, 0], "ring_mid": ring_frac[:, 1],
        "ring_out": ring_frac[:, 2],
    }
    for i, name in enumerate(CATEGORIES):
        k = name.split()[0].lower()
        tbl[f"n_{k}"] = n_c[:, i]
        tbl[f"p_{k}"] = p_c[:, i]
        tbl[f"lq_{k}"] = lq[:, i]

    np.savez_compressed(OUT_NPZ, **tbl)
    df = pd.DataFrame(tbl)
    df["max_lq_name"] = [CATEGORIES[i] for i in max_lq_cat]
    df.to_csv(OUT_CSV, index=False, float_format="%.5f")

    print(f"{n} 個 patch，{df.shape[1]} 欄")
    print(f"LQ 基準用 {CSV} 的 {n_src_used}/{n_src} 筆（類別對不上的已丟）")
    print(f"已存 {OUT_NPZ} 與 {OUT_CSV}\n")

    cols = ["n_total", "n_occupied", "entropy_norm", "max_p", "hhi", "max_lq",
            "mean_r", "std_r", "nn_dist", "clark_evans", "vmr",
            "ring_in", "ring_mid", "ring_out"]
    print(f"{'欄位':<14}{'p10':>9}{'中位數':>10}{'p90':>9}")
    for c in cols:
        v = df[c].to_numpy(dtype=float)
        p10, med, p90 = np.nanpercentile(v, [10, 50, 90])
        print(f"{c:<14}{p10:>9.2f}{med:>10.2f}{p90:>9.2f}")

    # 三環佔比要跟「面積佔比」比才有意義：環是同心方環，面積不相等
    edge = [f * HALF_WIDTH for f in RING_FRAC] + [HALF_WIDTH]
    a = np.diff([0.0] + [(e / HALF_WIDTH) ** 2 for e in edge])
    print(f"（三環若均勻分布應為 {a[0]:.2f} / {a[1]:.2f} / {a[2]:.2f}，"
          f"內環偏高代表 POI 集中在 patch 中心）")

    print(f"\n{'類別':<38}{'全體佔比':>9}{'patch平均':>10}{'LQ中位數':>10}"
          f"{'LQ p99':>9}{'最集中類':>9}")
    for i, name in enumerate(CATEGORIES):
        share = (max_lq_cat == i).mean()
        print(f"  {name:<36}{p_g[i]:>9.3f}{p_c[:, i].mean():>10.3f}"
              f"{np.median(lq[:, i]):>10.2f}{np.percentile(lq[:, i], 99):>9.2f}"
              f"{share:>9.1%}")


main()
