from scipy.spatial import cKDTree
from pyproj import Transformer
import numpy as np
import pandas as pd

df = pd.read_csv('data/tky_clean.csv')  # lat, lon 為 WGS84 經緯度

# EPSG:6677 = 日本平面直角座標系第9系,涵蓋東京都,單位為公尺
transformer = Transformer.from_crs('EPSG:4326', 'EPSG:6677', always_xy=True)
x, y = transformer.transform(df['lon'].values, df['lat'].values)
coords = np.column_stack([x, y])  # 投影座標系(公尺),不是經緯度

tree = cKDTree(coords)
dist, _ = tree.query(coords, k=2)  # k=2 因為最近的是自己
nn_dist = dist[:, 1]  # 單位:公尺
print(np.percentile(nn_dist, [10, 25, 50, 75, 90]))