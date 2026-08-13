"""把 tky_raw.txt 的打卡紀錄壓成唯一地點清單 tky_clean.csv。"""

import collections
import csv

RAW = "data/tky_raw.txt"
CATEGORY_CSV = "data/preprocess/FSD_category_api.csv"
OUT = "data/tky_clean.csv"

# 移動載具與線狀/面狀，非固定地點 -> 整個 POI 剔除
EXCLUDE_NAMES = {"Train", "Boat or Ferry", "Moving Target", "Taxi", "Plane",
                 "Bus Line", "Road", "Bridge", "River", "Hiking Trail",
                 "Neighborhood", "City"}
EXCLUDE_IDS = {"4e51a0c0bd41d3446defbb2e"}  # CSV 查無的 Ferry


def load_categories():
    """category_id -> (顯示名稱, 最大類)"""
    table = {}
    with open(CATEGORY_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            label = row["Category Label"]
            table[row["Category ID"]] = (row["Category Name"], label.split(" > ")[0])
    return table


def scan_raw():
    """每個 poi_id 保留最晚一筆打卡的 category 與座標，另計打卡次數。"""
    latest = {}
    counts = collections.Counter()
    with open(RAW, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 8:
                continue
            _, poi_id, category_id, _, lat, lon, _, _ = parts
            latest[poi_id] = (category_id, lat, lon)
            counts[poi_id] += 1
    return latest, counts


def main():
    table = load_categories()
    latest, counts = scan_raw()

    rows = []
    excluded = 0
    for poi_id, (category_id, lat, lon) in latest.items():
        if category_id in EXCLUDE_IDS or category_id not in table:
            excluded += 1
            continue
        category_name, category = table[category_id]
        if category_name in EXCLUDE_NAMES:
            excluded += 1
            continue
        rows.append([poi_id, category_id, category_name, category, lat, lon,
                     counts[poi_id]])

    with open(OUT, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["poi_id", "category_id", "category_name", "category",
                         "lat", "lon", "checkin_count"])
        writer.writerows(rows)

    print(f"唯一 POI {len(latest)} -> 輸出 {len(rows)}（排除 {excluded}）")
    print("\n最大類分佈：")
    for category, n in collections.Counter(r[3] for r in rows).most_common():
        print(f"  {n:6d}  {category}")


main()
