"""把來源 POI 切成以規則格點為中心的鄰域 patch，存成稀疏點列表。

來源資料集、類別表與幾何參數都在 config/dataset.py。
不直接存 40x40 矩陣，因為 patch 要在訓練時做隨機旋轉再 binning，
存相對座標(公尺)才能正確旋轉。
"""

import os
import sys

import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.abspath(f"{os.path.dirname(__file__)}/../.."))
from config.dataset import (CAT_COL, CATEGORIES, CENTER_STEP, CRS,  # noqa: E402
                            CSV, DATASET, MIN_POI, PATCHES, HALF_WIDTH)

OUT = PATCHES


def main():
    print(f"資料集 {DATASET}：{CSV}")
    df = pd.read_csv(CSV)
    transformer = Transformer.from_crs("EPSG:4326", CRS, always_xy=True)
    x, y = transformer.transform(df["lon"].values, df["lat"].values)
    coords = np.column_stack([x, y]).astype(np.float64)

    cat_index = {name: i for i, name in enumerate(CATEGORIES)}
    cats = df[CAT_COL].map(cat_index).values.astype(np.int8)

    # 中心點：POI 佔用過的格點中心，避開完全沒有 POI 的大片空地
    cells = np.unique(np.floor(coords / CENTER_STEP).astype(np.int64), axis=0)
    centers = (cells + 0.5) * CENTER_STEP

    tree = cKDTree(coords)
    neighbors = tree.query_ball_point(centers, HALF_WIDTH, p=np.inf)

    keep = np.array([len(n) >= MIN_POI for n in neighbors])
    centers = centers[keep]
    neighbors = [n for n, k in zip(neighbors, keep) if k]

    counts = np.array([len(n) for n in neighbors], dtype=np.int32)
    offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    idx = np.concatenate([np.asarray(n, dtype=np.int64) for n in neighbors])

    # 相對於中心的位移，單位公尺
    rel = coords[idx] - np.repeat(centers, counts, axis=0)

    inv = Transformer.from_crs(CRS, "EPSG:4326", always_xy=True)
    clon, clat = inv.transform(centers[:, 0], centers[:, 1])

    np.savez_compressed(
        OUT,
        dx=rel[:, 0].astype(np.float32),
        dy=rel[:, 1].astype(np.float32),
        cat=cats[idx],
        offsets=offsets,
        center_x=centers[:, 0].astype(np.float32),
        center_y=centers[:, 1].astype(np.float32),
        center_lat=np.asarray(clat, dtype=np.float32),
        center_lon=np.asarray(clon, dtype=np.float32),
        n_poi=counts,
    )

    print(f"patch {len(centers)}，總點數 {len(idx)}")
    print(f"每 patch POI 數 中位數 {np.median(counts):.0f}  "
          f"p10 {np.percentile(counts, 10):.0f}  p90 {np.percentile(counts, 90):.0f}  "
          f"max {counts.max()}")


main()
