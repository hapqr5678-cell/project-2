"""把 tky_clean.csv 切成以規則格點為中心的鄰域 patch，存成稀疏點列表。

不直接存 40x40 矩陣，因為 patch 要在訓練時做隨機旋轉再 binning，
存相對座標(公尺)才能正確旋轉。
"""

import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.spatial import cKDTree

CSV = "dataOverture/tokyo_places.csv"
OUT = "modelOverture/patches.npz"

CRS = "EPSG:6677"   # 日本平面直角座標系第9系，涵蓋東京都，單位公尺
RADIUS = 300        # patch 半徑(公尺)
CENTER_STEP = 50    # patch 中心的格點間距(公尺)
MIN_POI = 10        # 圓內少於這個數量的中心直接丟掉

# 10 個最大類，順序即 channel 順序
CATEGORIES = [
    'food_and_drink',  'services_and_business',
    'lifestyle_services',  'education',
    'health_care', 'shopping',
    'arts_and_entertainment',  'cultural_and_historic',
    'travel_and_transportation',  'sports_and_recreation',
    'lodging',  'community_and_government',
    'geographic_entities'
]


def main():
    df = pd.read_csv(CSV)
    transformer = Transformer.from_crs("EPSG:4326", CRS, always_xy=True)
    x, y = transformer.transform(df["lon"].values, df["lat"].values)
    coords = np.column_stack([x, y]).astype(np.float64)

    cat_index = {name: i for i, name in enumerate(CATEGORIES)}
    cats = df["category"].map(cat_index).values.astype(np.int8)

    # 中心點：POI 佔用過的格點中心，避開完全沒有 POI 的大片空地
    cells = np.unique(np.floor(coords / CENTER_STEP).astype(np.int64), axis=0)
    centers = (cells + 0.5) * CENTER_STEP

    tree = cKDTree(coords)
    neighbors = tree.query_ball_point(centers, RADIUS)

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
