"""手算每個 patch 的特徵表，完全不經過模型。

只吃 config 指定的 patch 快取（原始 POI 的相對座標與類別），輸出一張
每個 patch 一列的表，之後用來檢查 latent space 到底編碼了什麼。

三個維度：
  量    n_total / log1p / n_occupied
  組成  n_c、p_c、entropy、max_p
  空間  mean_r、std_r、nn_dist、內中外三環佔比
"""

import os
import sys

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.abspath(f"{os.path.dirname(__file__)}/../.."))
from config.dataset import (CATEGORIES, CELL, FEATURES,  # noqa: E402
                            FEATURES_CSV, GRID, N_CAT, PATCHES)

OUT_NPZ = FEATURES
OUT_CSV = FEATURES_CSV

# 三環邊界(公尺)：內圈 <100，中圈 100~200，外圈 200~300
RINGS = [100.0, 200.0, 300.0]


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


def main():
    d = np.load(PATCHES)
    dx, dy, cat, offsets = d["dx"], d["dy"], d["cat"].astype(np.int64), d["offsets"]
    n_total = np.diff(offsets).astype(np.int64)
    n = len(n_total)
    owner = np.repeat(np.arange(n), n_total)

    # 量 -----------------------------------------------------------------
    # 佔用格子數：用跟模型輸入一樣的 binning，但不分 channel，看空間覆蓋
    ix = np.clip(np.floor(dx / CELL + GRID / 2).astype(np.int64), 0, GRID - 1)
    iy = np.clip(np.floor(dy / CELL + GRID / 2).astype(np.int64), 0, GRID - 1)
    cell_id = np.unique(owner * GRID * GRID + iy * GRID + ix)
    n_occupied = np.bincount(cell_id // (GRID * GRID), minlength=n)

    # 組成 ---------------------------------------------------------------
    n_c = np.bincount(owner * N_CAT + cat,
                      minlength=n * N_CAT).reshape(n, N_CAT)
    p_c = n_c / n_total[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        logp = np.where(p_c > 0, np.log2(p_c), 0.0)
    entropy = -(p_c * logp).sum(1)
    max_p = p_c.max(1)

    # 空間 ---------------------------------------------------------------
    r = np.hypot(dx, dy).astype(np.float64)
    mean_r = np.bincount(owner, weights=r, minlength=n) / n_total
    mean_r2 = np.bincount(owner, weights=r ** 2, minlength=n) / n_total
    std_r = np.sqrt(np.maximum(mean_r2 - mean_r ** 2, 0.0))

    ring = np.digitize(r, RINGS)          # 0=內 1=中 2=外
    ring_frac = np.stack([
        np.bincount(owner, weights=(ring == k), minlength=n) / n_total
        for k in range(3)
    ], axis=1)

    nn = nn_dist(dx, dy, offsets)

    # 存表 ---------------------------------------------------------------
    tbl = {
        "lat": d["center_lat"], "lon": d["center_lon"],
        "n_total": n_total, "log_n_total": np.log1p(n_total),
        "n_occupied": n_occupied,
        "entropy": entropy, "max_p": max_p,
        "mean_r": mean_r, "std_r": std_r, "nn_dist": nn,
        "ring_in": ring_frac[:, 0], "ring_mid": ring_frac[:, 1],
        "ring_out": ring_frac[:, 2],
    }
    for i, name in enumerate(CATEGORIES):
        key = name.split()[0].lower()
        tbl[f"n_{key}"] = n_c[:, i]
        tbl[f"p_{key}"] = p_c[:, i]

    np.savez_compressed(OUT_NPZ, **tbl)
    df = pd.DataFrame(tbl)
    df.to_csv(OUT_CSV, index=False, float_format="%.5f")

    print(f"{n} 個 patch，{df.shape[1]} 欄")
    print(f"已存 {OUT_NPZ} 與 {OUT_CSV}\n")

    cols = ["n_total", "n_occupied", "entropy", "max_p",
            "mean_r", "std_r", "nn_dist", "ring_in", "ring_mid", "ring_out"]
    print(f"{'欄位':<12}{'p10':>9}{'中位數':>10}{'p90':>9}")
    for c in cols:
        v = df[c].to_numpy(dtype=float)
        p10, med, p90 = np.nanpercentile(v, [10, 50, 90])
        print(f"{c:<12}{p10:>9.2f}{med:>10.2f}{p90:>9.2f}")

    print("\n各類別平均佔比")
    for i, name in enumerate(CATEGORIES):
        print(f"  {name:<38}{p_c[:, i].mean():.3f}")


main()
