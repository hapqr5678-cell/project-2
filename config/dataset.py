import hashlib
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASET = "fsq"

if DATASET == "fsq":
    HALF_WIDTH = 50.0
    CELL = 15
    CENTER_STEP = 100       # patch 中心的格點間距(公尺)
    MIN_POI = 10           # 圓內少於這個數量的中心直接丟掉
elif DATASET == "overture":
    HALF_WIDTH = 200.0
    CELL = 15
    CENTER_STEP = 400       # patch 中心的格點間距(公尺)
    MIN_POI = 15           # 圓內少於這個數量的中心直接丟掉


GRID = int(HALF_WIDTH * 2 / CELL)
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

# OD（類別間人流）：從 tky_raw.txt 的打卡序列抽「同一使用者連續兩筆打卡」，
# 得到每個 patch 內部的 N_CAT x N_CAT 轉移矩陣。只有 fsq 資料集能用——
# POI count 與 OD 同源（同一批 venue、同一套類別），不需要任何類別對照。
OD = os.path.join(ROOT, "data", "od", "od.npz")
OD_RAW = os.path.join(ROOT, "data", "tky_raw.txt")

OD_MAX_GAP_H = 6.0      # 相鄰兩筆打卡間隔超過這麼久，不算同一趟移動
OD_MAX_DIST = float("inf")   # 位移上限；inf = 不限，終點多遠都算
# 一筆移動要記進哪一格：
#   "origin"    起點所在的格。語意是「在這格的 A 類店的人，下一步去了哪種店」，
#               終點可以在任何地方。資料量最多（195,921 筆，1233/1233 格都有）。
#   "midpoint"  起訖中點所在的格。
#   "both"      兩端都在同一格才算（最嚴格，只剩 19,840 筆、中位數 3）。
OD_ASSIGN = "origin"
OD_POOL_RING = 0        # >0 時一筆 OD 也記進周圍 ring 圈的格，權重 OD_POOL_DECAY**ring
OD_POOL_DECAY = 0.5

_OD_PARAMS = {
    "max_gap_h": OD_MAX_GAP_H,
    "max_dist": OD_MAX_DIST,
    "assign": OD_ASSIGN,
    "pool_ring": OD_POOL_RING,
    "pool_decay": OD_POOL_DECAY,
}

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


def od_fingerprint():
    """目前這組 OD 參數的指紋，回傳 sha256 hex 字串。

    把 patch_fingerprint() 也算進去：patch 的幾何一變，格子的切法就變，
    舊的 od.npz 必定作廢，所以不需要第二個 hash 欄位。
    tky_raw.txt 的 mtime 也算進去，理由同 patch_fingerprint()。
    """
    params = dict(_OD_PARAMS)
    params["patches"] = patch_fingerprint()
    if os.path.exists(OD_RAW):
        params["raw_mtime"] = os.path.getmtime(OD_RAW)
    blob = json.dumps(params, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()


def od_fresh():
    """od.npz 是否存在，且是用目前這組參數建的。回傳 bool。"""
    if not os.path.exists(OD):
        return False
    with np.load(OD) as d:
        if "config_hash" not in d:
            return False
        return d["config_hash"].item() == od_fingerprint()


def ensure_od():
    """train 前先確保 od.npz 對得上目前的 config，對不上就重跑 build_od。無回傳值。

    會先 ensure_patches()：od 的指紋含 patch 指紋，且 build_od 需要
    patches.npz 的 center_x/center_y 才能把格子對回 patch 編號。
    """
    ensure_patches()
    if od_fresh():
        return
    print("[config] od.npz 跟目前設定不一致，重新產生...")
    from data.od.build_od import build
    build()
