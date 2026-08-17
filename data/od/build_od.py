"""從 tky_raw.txt 的打卡序列抽出類別間的 OD（origin-destination）矩陣，存成 od.npz。

一筆 OD = 同一個使用者「連續兩筆打卡」，且時間間隔 < OD_MAX_GAP_H、
空間位移 < OD_MAX_DIST。起點打卡的類別是 origin、終點的是 destination。

只有 fsq 資料集能用：tky_clean.csv 的 poi_id 就是 tky_raw.txt 的 venueID，
POI count 與 OD 出自同一批 venue、同一套類別表，不需要任何類別對照。
不在 tky_clean.csv 裡的打卡（Train / Taxi / Boat 這類移動載具，被
data/preprocess/clean.py 的 EXCLUDE_NAMES 剔掉）會先從序列中移除再取相鄰
配對——這樣「餐廳 → 電車 → 商店」會正確變成一筆「餐廳 → 商店」，
而不是斷成兩筆都被丟掉。

輸出兩層：
  A_global   全東京一張 N_CAT x N_CAT，密（樣本多），當每個 patch 的先驗。
  per-patch  每個 patch 一張 N_CAT x N_CAT，稀疏（樣本少），存成 CSR。
模型端用逐列的經驗貝氏收縮把兩者混起來，見 model/v3_gat_ddae/ae.py 的 shrink()。

同樣會存 config_hash，train.py 前呼叫 config.dataset.ensure_od() 自動檢查、
對不上就重跑這支腳本，慣例跟 data/patch/build_patches.py 一致。
"""

import os
import sys

import numpy as np
import pandas as pd
from pyproj import Transformer

sys.path.insert(0, os.path.abspath(f"{os.path.dirname(__file__)}/../.."))
from config.dataset import (CATEGORIES, CENTER_STEP, CRS, CSV,  # noqa: E402
                            DATASET, N_CAT, OD, OD_ASSIGN, OD_MAX_DIST,
                            OD_MAX_GAP_H, OD_POOL_DECAY, OD_POOL_RING, OD_RAW,
                            PATCHES, od_fingerprint)

OUT = OD


def _transitions():
    """讀 tky_raw.txt，回傳過濾後的移動配對。

    回傳 (xy_o, xy_d, cat_o, cat_d)：
      xy_o / xy_d  (T,2) float64，起訖點在 CRS 下的座標（公尺）
      cat_o/cat_d  (T,)  int64，起訖點的類別 index，值域 0..N_CAT-1
    """
    poi = pd.read_csv(CSV)
    tf = Transformer.from_crs("EPSG:4326", CRS, always_xy=True)
    px, py = tf.transform(poi["lon"].values, poi["lat"].values)

    cat_index = {name: i for i, name in enumerate(CATEGORIES)}
    poi_cat = poi["category"].map(cat_index)
    ok = poi_cat.notna()
    keys = poi["poi_id"].values[ok.values]
    v_cat = dict(zip(keys, poi_cat.values[ok.values].astype(np.int64)))
    v_x = dict(zip(keys, np.asarray(px)[ok.values]))
    v_y = dict(zip(keys, np.asarray(py)[ok.values]))

    raw = pd.read_csv(OD_RAW, sep="\t", names=["uid", "vid", "cid", "cname",
                                               "lat", "lon", "tz", "t"])
    n_raw = len(raw)
    # 先把不在 POI 清單裡的打卡整列丟掉，再取相鄰配對，鏈才不會被移動載具切斷
    raw = raw[raw["vid"].isin(v_cat)].copy()
    raw["ts"] = pd.to_datetime(raw["t"], format="%a %b %d %H:%M:%S %z %Y", utc=True)
    raw = raw.sort_values(["uid", "ts"]).reset_index(drop=True)
    print(f"打卡 {n_raw} -> 對得到 POI 的 {len(raw)}"
          f"（{len(v_cat)} 個 venue，{n_raw - len(raw)} 筆是移動載具或查無類別）")

    xy = np.column_stack([raw["vid"].map(v_x).values,
                          raw["vid"].map(v_y).values]).astype(np.float64)
    cat = raw["vid"].map(v_cat).values.astype(np.int64)

    nxt = raw.shift(-1)
    gap = (nxt["ts"] - raw["ts"]).dt.total_seconds().values / 3600.0
    same_user = (raw["uid"].values == nxt["uid"].values)

    xy_d = np.roll(xy, -1, axis=0)
    dist = np.abs(xy - xy_d).max(axis=1)          # chebyshev，跟方形窗一致
    keep = same_user & (gap < OD_MAX_GAP_H) & (dist < OD_MAX_DIST)
    print(f"相鄰配對 {int(same_user.sum())} -> "
          f"gap<{OD_MAX_GAP_H}h 且位移<{OD_MAX_DIST}m 的 {int(keep.sum())}")
    return xy[keep], xy_d[keep], cat[keep], np.roll(cat, -1)[keep]


def _patch_cells():
    """從 patches.npz 的中心座標反推每個 patch 佔的格子。

    回傳 dict[(cx, cy)] -> patch 編號。build_patches.py 的中心是
    (cell + 0.5) * CENTER_STEP，這裡把它反算回整數格座標。
    2*HALF_WIDTH == CENTER_STEP，所以方形窗剛好無縫鋪滿、一個點只屬於一格。
    """
    with np.load(PATCHES) as d:
        cx = np.rint(d["center_x"] / CENTER_STEP - 0.5).astype(np.int64)
        cy = np.rint(d["center_y"] / CENTER_STEP - 0.5).astype(np.int64)
    return {(int(a), int(b)): i for i, (a, b) in enumerate(zip(cx, cy))}, len(cx)


def _assign(xy_o, xy_d):
    """決定每一筆 transition 要記到哪一格。

    xy_o / xy_d：(T,2) 起訖點座標。
    回傳 (cells, mask)：cells (T,2) int64 是目標格座標，mask (T,) bool 標記
    這一筆是否有效（OD_ASSIGN="both" 時兩端不同格就無效）。
    """
    c_o = np.floor(xy_o / CENTER_STEP).astype(np.int64)
    if OD_ASSIGN == "origin":
        return c_o, np.ones(len(c_o), dtype=bool)
    if OD_ASSIGN == "both":
        c_d = np.floor(xy_d / CENTER_STEP).astype(np.int64)
        return c_o, (c_o == c_d).all(axis=1)
    if OD_ASSIGN == "midpoint":
        mid = np.floor((xy_o + xy_d) / 2 / CENTER_STEP).astype(np.int64)
        return mid, np.ones(len(mid), dtype=bool)
    raise ValueError(f"未知的 OD_ASSIGN：{OD_ASSIGN}")


def build():
    if DATASET != "fsq":
        raise RuntimeError(
            f"OD 只有 fsq 資料集能建（目前 DATASET={DATASET}）："
            "count 與 OD 必須同源，否則類別空間對不上。")

    xy_o, xy_d, cat_o, cat_d = _transitions()

    a_global = np.zeros((N_CAT, N_CAT), dtype=np.float64)
    np.add.at(a_global, (cat_o, cat_d), 1.0)

    cellmap, n_patch = _patch_cells()
    cells, valid = _assign(xy_o, xy_d)
    flat = (cat_o * N_CAT + cat_d).astype(np.int64)

    # (patch, cell) -> 權重。POOL_RING>0 時一筆 OD 會散到周圍好幾格
    acc = {}
    n_land = 0
    ring = max(0, int(OD_POOL_RING))
    for dx in range(-ring, ring + 1):
        for dy in range(-ring, ring + 1):
            w = OD_POOL_DECAY ** max(abs(dx), abs(dy))
            pid = np.fromiter(
                (cellmap.get((int(cx) + dx, int(cy) + dy), -1) for cx, cy in cells),
                dtype=np.int64, count=len(cells))
            m = valid & (pid >= 0)
            if dx == 0 and dy == 0:
                n_land = int(m.sum())
            key = pid[m] * (N_CAT * N_CAT) + flat[m]
            uniq, cnt = np.unique(key, return_counts=True)
            for k, c in zip(uniq, cnt):
                acc[int(k)] = acc.get(int(k), 0.0) + w * float(c)

    print(f"落進有效 patch 的 transition {n_land}"
          f"（assign={OD_ASSIGN}, pool_ring={ring}）")

    keys = np.fromiter(sorted(acc), dtype=np.int64, count=len(acc))
    vals = np.fromiter((acc[int(k)] for k in keys), dtype=np.float64, count=len(acc))
    owner = keys // (N_CAT * N_CAT)
    od_cell = (keys % (N_CAT * N_CAT)).astype(np.int16)

    per_patch = np.bincount(owner, minlength=n_patch)
    od_offsets = np.concatenate([[0], np.cumsum(per_patch)]).astype(np.int64)
    n_od = np.bincount(owner, weights=vals, minlength=n_patch).astype(np.float32)

    np.savez_compressed(
        OUT,
        od_cell=od_cell,
        od_value=vals.astype(np.float32),
        od_offsets=od_offsets,
        n_od=n_od,
        A_global=a_global,
        config_hash=od_fingerprint(),
    )

    has = n_od > 0
    print(f"\npatch {n_patch}，非零 cell {len(keys)}，有 OD 的 patch {int(has.sum())}")
    print(f"每 patch OD 量 中位數 {np.median(n_od):.1f}  "
          f"p75 {np.percentile(n_od, 75):.1f}  p90 {np.percentile(n_od, 90):.1f}  "
          f"p99 {np.percentile(n_od, 99):.1f}  max {n_od.max():.1f}")
    print(f"OD 量 >= {N_CAT * N_CAT}（矩陣平均每格 1 筆）的 patch："
          f"{int((n_od >= N_CAT * N_CAT).sum())}")
    print(f"全域矩陣：總量 {a_global.sum():.0f}，"
          f"非零 cell {int((a_global > 0).sum())}/{a_global.size}")

    dead = a_global.sum(axis=1) == 0
    if dead.any():
        print(f"全域矩陣整列為零的類別（收縮時退回均勻分布）："
              f"{[CATEGORIES[i] for i in np.where(dead)[0]]}")
    print(f"\n已存 {OUT}")


if __name__ == "__main__":
    build()
