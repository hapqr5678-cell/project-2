import hashlib
import json
import os

import numpy as np

# 要用哪個資料集：SOURCES 的 key

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


DATASET = "fsq"
HALF_WIDTH = 50.0
CELL = 15
GRID = int(HALF_WIDTH * 2 / CELL)
CENTER_STEP = 100       # patch 中心的格點間距(公尺)
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
        "csv": "data/overture/overture_clean.csv",
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


PATCHES = os.path.join(ROOT, "data", "patch", "patches.npz")
FEATURES = os.path.join(ROOT, "data", "patch", "features.npz")
FEATURES_CSV = os.path.join(ROOT, "data", "patch", "features.csv")

# 會影響 patches.npz 內容的參數：改了這些，舊的 patches 就作廢
_PATCH_PARAMS = {
    "dataset": DATASET,
    "csv": _src["csv"],
    "cat_col": CAT_COL,
    "categories": CATEGORIES,
    "half_width": HALF_WIDTH,
    "cell": CELL,
    "center_step": CENTER_STEP,
    "min_poi": MIN_POI,
    "crs": CRS,
}


def patch_fingerprint():
    """目前這組參數（含來源 CSV 的 mtime）的指紋。

    CSV 的 mtime 也算進去：參數沒變但資料本身換過，一樣要重建。
    """
    params = dict(_PATCH_PARAMS)
    if os.path.exists(CSV):
        params["csv_mtime"] = os.path.getmtime(CSV)
    blob = json.dumps(params, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()


def patches_fresh():
    """patches.npz 是否存在，且是用目前這組參數建的。"""
    if not os.path.exists(PATCHES):
        return False
    with np.load(PATCHES) as d:
        if "config_hash" not in d:
            return False
        return d["config_hash"].item() == patch_fingerprint()


def ensure_patches():
    """train 前先確保 patches.npz 對得上目前的 config，對不上就重跑 build_patches。"""
    if patches_fresh():
        return
    print("[config] patches.npz 跟目前設定不一致，重新產生...")
    from data.patch.build_patches import build
    build()
