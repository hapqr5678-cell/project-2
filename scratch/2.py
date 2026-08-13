from pyproj import Transformer
import numpy as np
import pandas as pd

df = pd.read_csv('data/tky_clean.csv')  # lat, lon 為 WGS84 經緯度

# EPSG:6677 = 日本平面直角座標系第9系,涵蓋東京都,單位為公尺
transformer = Transformer.from_crs('EPSG:4326', 'EPSG:6677', always_xy=True)
x, y = transformer.transform(df['lon'].values, df['lat'].values)
coords = np.column_stack([x, y])  # 投影座標系(公尺)

cell_size = 15  # 你要測試的候選值
grid_idx = np.floor(coords / cell_size).astype(int)
unique, counts = np.unique(grid_idx, axis=0, return_counts=True)

collision_rate = (counts > 1).sum() / len(unique)
print(f"碰撞率: {collision_rate:.2%}")
print(f"最多同格POI數: {counts.max()}")

print("\n同格數量 2~max 的分布(佔全部格子比率):")
for n in range(2, counts.max() + 1):
    rate = (counts == n).sum() / len(unique)
    print(f"  {n} 個POI同格: {rate:.5%}")

