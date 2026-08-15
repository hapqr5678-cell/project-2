"""把 overture_raw.parquet 清成唯一地點清單 overture_clean.csv。

輸出欄位刻意跟 data/tky_clean.csv 對齊（都有 category / lat / lon），
config/dataset.py 的 CAT_COL 直接指到 category，build_patches 不用改。

清理規則與理由：
1. 沒有 categories.primary 的直接丟掉（約 6000 筆）——沒類別就沒有 channel 可放。
2. taxonomy.hierarchy 有少數葉節點是空的（doctor、iron_work 共 935 筆），
   用 MANUAL_TOP 手動補上層，不然這些醫療點會被整批丟掉。
3. 剔除線狀/面狀的地理實體（橋、河、山、湖…）。這跟 fsq 版 clean.py 剔除
   Bridge / River / Road 是同一個理由：它們不是「一個點」，
   放進 40x40 矩陣時位置沒有意義。溫泉、植物園、marina 這種有實體範圍
   但仍算場所的則保留。
4. 去重：Overture 本身已經做過 conflation，id 全唯一，同名同類別又在 ~11m
   內的只剩個位數，這步基本上是保險，留 confidence 高的那筆。

沒有做的事：不依 confidence 過濾。confidence 分佈很散（0.1 以下也有兩千多筆），
砍掉會直接改變密度分佈，而密度正是這個專題要看的東西，
所以把 confidence 原樣留在輸出裡，要篩之後再說。
"""

import csv
import os
import sys

import duckdb

sys.path.insert(0, os.path.abspath(f"{os.path.dirname(__file__)}/../.."))
from config.dataset import SOURCES  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = f"{HERE}/overture_raw.parquet"
OUT = f"{HERE}/overture_clean.csv"

# taxonomy.hierarchy 是空的葉節點 -> 手動指定 top-level
MANUAL_TOP = {
    "doctor": "health_care",
    "iron_work": "services_and_business",
}

# 線狀/面狀，不是單一地點 -> 整筆剔除
EXCLUDE_LEAF = {
    "bridge", "river", "canal", "lake", "island", "forest", "mountain",
    "waterfall", "beach", "cave", "nature_reserve", "quay", "pier",
    "structure_and_geography",
}

# 只保留 config 列出的 top-level（順序即之後的 channel 順序）
KEEP_TOP = set(SOURCES["overture"]["categories"])


def main():
    con = duckdb.connect()
    total = con.execute(f"SELECT count(*) FROM '{RAW}'").fetchone()[0]

    manual = ", ".join(f"'{k}': '{v}'" for k, v in MANUAL_TOP.items())
    exclude = ", ".join(f"'{c}'" for c in sorted(EXCLUDE_LEAF))
    keep = ", ".join(f"'{c}'" for c in sorted(KEEP_TOP))

    rows = con.execute(f"""
        WITH tagged AS (
            SELECT
                id, name, category AS category_leaf, lat, lon, confidence,
                coalesce(top_category, MAP {{{manual}}}[category]) AS category
            FROM '{RAW}'
            WHERE category IS NOT NULL
              AND category NOT IN ({exclude})
        ),
        ranked AS (
            SELECT *, row_number() OVER (
                PARTITION BY name, category_leaf, round(lat, 4), round(lon, 4)
                ORDER BY confidence DESC
            ) AS rk
            FROM tagged
            WHERE category IN ({keep})
        )
        SELECT id, name, category_leaf, category, lat, lon, confidence
        FROM ranked WHERE rk = 1
    """).fetchall()

    with open(OUT, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["poi_id", "name", "category_leaf", "category",
                         "lat", "lon", "confidence"])
        writer.writerows(rows)

    print(f"原始 {total} -> 輸出 {len(rows)}（剔除 {total - len(rows)}）")
    print(f"{OUT}")

    print("\n類別分佈：")
    counts = {}
    for r in rows:
        counts[r[3]] = counts.get(r[3], 0) + 1
    for cat in SOURCES["overture"]["categories"]:
        print(f"  {counts.get(cat, 0):7d}  {cat}")


main()
