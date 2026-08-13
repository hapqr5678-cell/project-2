"""資料集來源與共用參數的唯一設定點。

換資料集只要改下面的 DATASET，其他腳本
（build_patches / build_features / v0 / v0_l16 / v1）都從這裡拿：
  來源 CSV、類別表（channel 順序）、中文標籤與配色
  幾何參數（半徑、cell、格數、中心間距）
  輸出路徑（patches、特徵表、各版本的 result 目錄）

注意：輸出路徑不分資料集，換 DATASET 重跑會蓋掉前一份的 patches 與 result。

路徑都是絕對路徑（以本檔位置推算 repo root），所以腳本從哪個目錄跑都一樣。
用法：
    import os, sys
    sys.path.insert(0, os.path.abspath(f"{os.path.dirname(__file__)}/../.."))
    from config.dataset import CATEGORIES, PATCHES, result
"""

import os

# 要用哪個資料集：SOURCES 的 key
DATASET = "fsq"

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


RADIUS = 100.0
CELL = 15.0
GRID = int(RADIUS * 2 / CELL)
CENTER_STEP = 50       # patch 中心的格點間距(公尺)
MIN_POI = 10           # 圓內少於這個數量的中心直接丟掉
CRS = "EPSG:6677"      # 日本平面直角座標系第9系，涵蓋東京都，單位公尺

SOURCES = {
    # Foursquare 打卡紀錄清乾淨後的唯一地點清單，類別是 FSQ 的最大類
    "fsq": {
        "csv": "data/tky_clean.csv",
        "cat_col": "category",
        "categories": [
            "Dining and Drinking",
            "Retail",
            "Nightlife Spot",
            "Community and Government",
            "Travel and Transportation",
            "Business and Professional Services",
            "Landmarks and Outdoors",
            "Arts and Entertainment",
            "Health and Medicine",
            "Sports and Recreation",
        ],
        "zh": [
            "餐飲", "零售", "夜生活", "社區/政府", "交通",
            "商業服務", "地標/戶外", "藝文娛樂", "醫療", "運動休閒",
        ],
    },
    # Overture Maps 的 places，類別是它的 top-level category
    "overture": {
        "csv": "dataOverture/overture_clean.csv",
        "cat_col": "category",
        "categories": [
            "food_and_drink",
            "services_and_business",
            "shopping",
            "lifestyle_services",
            "health_care",
            "arts_and_entertainment",
            "education",
            "sports_and_recreation",
            "travel_and_transportation",
            "community_and_government",
            "cultural_and_historic",
            "lodging",
            "geographic_entities",
        ],
        "zh": [
            "餐飲", "商業服務", "購物", "生活服務", "醫療",
            "藝文娛樂", "教育", "運動休閒", "交通", "社區/政府",
            "文化古蹟", "住宿", "地理實體",
        ],
    },
}

# 類別配色，取前 N_CAT 個
PALETTE = [
    "#e6194b", "#3cb44b", "#911eb4", "#4363d8", "#f58231",
    "#46f0f0", "#008080", "#f032e6", "#9a6324", "#808000",
    "#000075", "#bcf60c", "#fabebe",
]

_src = SOURCES[DATASET]
CSV = os.path.join(ROOT, _src["csv"])
CAT_COL = _src["cat_col"]
CATEGORIES = _src["categories"]     # 順序即 channel 順序
CAT_ZH = _src["zh"]
N_CAT = len(CATEGORIES)
CAT_COLORS = PALETTE[:N_CAT]


def result(version, name):
    """某個模型版本的輸出：model/<version>/result/<name>。"""
    d = os.path.join(ROOT, "model", version, "result")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, name)


PATCHES = os.path.join(ROOT, "model", "patch", "patches_100.npz")
# 手算特徵表，跟模型無關；沿用既有位置
FEATURES = result("v0", "features.npz")
FEATURES_CSV = result("v0", "features.csv")
