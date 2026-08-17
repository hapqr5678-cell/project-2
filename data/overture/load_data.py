"""從 Overture Maps 的公開 S3 抓東京的 places，存成 overture_raw.parquet。

為什麼用 duckdb 直接讀 S3：Overture 的 parquet 依經緯度排序並帶 bbox 欄位，
下 bbox 條件時 parquet 的 row-group statistics 會直接跳過不相干的檔案，
所以不用先下載整個日本（數 GB）就能只取東京這塊。

為什麼存 parquet 不存 csv：raw 要保留巢狀欄位（alternate 類別、taxonomy 階層），
csv 會把 list 壓成字串，之後 clean 還要自己解析。

半徑 8.2km 是照「約 25 萬筆」回推的：以東京車站為中心，
±5km≈13.5萬、±8km≈24.4萬、±10km≈29.0萬（2026-07-22.0 release 實測）。

已知風險：RELEASE 是寫死的。Overture 的 bucket 只留最近幾個 release，
舊的會被刪掉，跑不動時先去 list bucket 看現在有哪些版本。
"""

import os
import time

import duckdb

RELEASE = "2026-07-22.0"
SRC = f"s3://overturemaps-us-west-2/release/{RELEASE}/theme=places/type=place/*.parquet"
OUT = f"{os.path.dirname(os.path.abspath(__file__))}/overture_raw.parquet"
XMIN = None
# bounding box，留空（None）則預設：東京車站為中心，取正方形範圍
# XMIN, YMIN, XMAX, YMAX = 138.9 , 35.34, 140.0,36.0

def main():
    if XMIN is not None:
        xmin, ymin, xmax, ymax = XMIN, YMIN, XMAX, YMAX
    else:
        CENTER_LON, CENTER_LAT = 139.7671, 35.6812
        HALF_KM = 20
        KM_PER_LON, KM_PER_LAT = 90.3, 111.0
        dlon = HALF_KM / KM_PER_LON
        dlat = HALF_KM / KM_PER_LAT
        xmin, xmax = CENTER_LON - dlon, CENTER_LON + dlon
        ymin, ymax = CENTER_LAT - dlat, CENTER_LAT + dlat

    print(f"release {RELEASE}")
    print(f"bbox lon [{xmin:.4f}, {xmax:.4f}]  lat [{ymin:.4f}, {ymax:.4f}]")

    con = duckdb.connect()
    con.execute("INSTALL spatial; LOAD spatial;")
    con.execute("INSTALL httpfs; LOAD httpfs; SET s3_region='us-west-2';")

    # places 的 geometry 都是 point，所以 bbox.xmin/ymin 就是它的經緯度，
    # 拿來當過濾條件即可（真正的座標還是從 geometry 取，精度比較好）。
    query = f"""
    COPY (
        SELECT
            id,
            names.primary                       AS name,
            categories.primary                  AS category,
            categories.alternate                AS category_alt,
            taxonomy.hierarchy                  AS taxonomy,
            taxonomy.hierarchy[1]               AS top_category,
            basic_category,
            confidence,
            operating_status,
            ST_Y(geometry)                      AS lat,
            ST_X(geometry)                      AS lon,
        FROM read_parquet('{SRC}')
        WHERE bbox.xmin BETWEEN {xmin} AND {xmax}
          AND bbox.ymin BETWEEN {ymin} AND {ymax}
    ) TO '{OUT}' (FORMAT parquet, COMPRESSION zstd)
    """
    t = time.time()
    con.execute(query)
    n = con.execute(f"SELECT count(*) FROM '{OUT}'").fetchone()[0]
    mb = os.path.getsize(OUT) / 1e6
    print(f"{n} 筆 -> {OUT}（{mb:.1f} MB，{time.time() - t:.0f}s）")

    print("\ntop-level 類別分佈：")
    rows = con.execute(f"""
        SELECT coalesce(top_category, '(null)') AS c, count(*) AS n
        FROM '{OUT}' GROUP BY 1 ORDER BY n DESC
    """).fetchall()
    for c, k in rows:
        print(f"  {k:7d}  {c}")

    print("\n輸出分布圖...")
    import pandas as pd
    import matplotlib.pyplot as plt
    df = pd.read_parquet(OUT)
    plt.figure(figsize=(10, 8))
    plt.scatter(df['lon'], df['lat'], s=0.1, alpha=0.3, c='blue')
    plt.title(f"Overture Points (Total: {n})")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.grid(True, linestyle='--', alpha=0.6)
    
    img_out = OUT.replace(".parquet", "_map.png")
    plt.savefig(img_out, dpi=150, bbox_inches='tight')
    print(f"已儲存分布圖至 {img_out}")

main()
