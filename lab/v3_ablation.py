"""掃 v3_gat_ddae 的 ablation，每組跑完把 result.log 最後一行的指標收成一張表。

用法：
    python lab/v3_ablation.py            # 跑第一批（判定要不要繼續）
    python lab/v3_ablation.py second     # 跑第二批（第一批通過才跑）
    python lab/v3_ablation.py table      # 不訓練，只把已有的 result.log 收成表

每組是一個獨立的 VERSION（各自的 result 目錄），透過環境變數 V3_OVERRIDE
把設定灌進 model/v3_gat_ddae/train.py，不複製檔案。

=== 第一批要回答的五個問題 ===

  base      v3 完整版，其他組的參照點
  nograph   不用圖，只把 log1p(n_od) 與 has_od 兩個純量接在 count 後面。
            ★ 最重要的一組。Foursquare 的覆蓋不是隨機缺失，這兩個純量本身
            就是「熱鬧程度」的代理，而熱鬧程度跟 count 高度相關。如果這組
            就拿走 base 大部分的增益，那 GAT 是裝飾。
  noedge    w_g / w_lf / w_lb / w_self 全部凍結為 0，OD 完全不進 attention。
            增益是不是只來自「多了一條分支、多了參數」？
  globalod  kappa 開到很大，A 恆等於全域矩陣，local OD 完全失效。
            全域結構的 inductive bias 值多少？
  shuffled  全域矩陣的元素隨機置換（seed 固定），結構被打散但邊際統計還在。
            ★ OD 的「結構」有沒有用，還是隨便一張矩陣都行？

判準：若 globalod 打不贏 shuffled 和 nograph，就停手——這代表 OD 的圖結構
沒有貢獻，該把資源轉去做「OD 只當監督訊號、不當輸入」的設計。
"""

import json
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN = os.path.join(ROOT, "model", "v3_gat_ddae", "train.py")

FIRST = {
    "v3_base": {},
    "v3_nograph": {"USE_GRAPH": False},
    "v3_noedge": {"FREEZE_EDGE_BIAS": True},
    "v3_globalod": {"KAPPA": 1e9},
    "v3_shuffled": {"SHUFFLE_GLOBAL": True},
}

SECOND = {
    "v3_alpha0": {"ALPHA_MODE": "fixed", "ALPHA_INIT": 0.0},
    "v3_alpha1": {"ALPHA_MODE": "fixed", "ALPHA_INIT": 1.0},
    "v3_alpha_learned": {"ALPHA_MODE": "learned"},
    "v3_kappa2": {"KAPPA": 2.0},
    "v3_kappa32": {"KAPPA": 32.0},
    "v3_readout_mean": {"READOUT": "mean"},
    "v3_od0": {"LAMBDA_OD": 0.0},
    "v3_od_const": {"LAMBDA_OD": 0.3, "OD_SCHEDULE": "const"},
    "v3_od3": {"LAMBDA_OD": 3.0},
}

COLUMNS = [("val_dev", "val_dev↓"), ("expl_dev", "expl_dev↑"),
           ("knock_dev", "knock↑"), ("expl_od", "expl_od↑"),
           ("alpha", "alpha"), ("w_lf", "w_lf"), ("w_g", "w_g")]


def run_one(version, override):
    """跑一組設定。version 是 VERSION 字串，override 是要覆寫的常數 dict。

    無回傳值；訓練的輸出直接透到終端機，指標留在 model/<version>/result/result.log。
    """
    env = dict(os.environ)
    env["V3_OVERRIDE"] = json.dumps({"VERSION": version, **override})
    print(f"\n{'=' * 70}\n{version}  {override}\n{'=' * 70}", flush=True)
    subprocess.run([sys.executable, TRAIN], env=env, cwd=ROOT, check=True)


def last_metrics(version):
    """讀 model/<version>/result/result.log 最後一行 epoch 記錄的指標。

    回傳 dict[str, float]；檔案不存在或沒有任何 epoch 行時回傳空 dict。
    """
    path = os.path.join(ROOT, "model", version, "result", "result.log")
    if not os.path.exists(path):
        return {}
    line = ""
    with open(path, encoding="utf-8") as f:
        for row in f:
            if row.startswith("epoch "):
                line = row
    if not line:
        return {}
    out = {}
    for key, _ in COLUMNS:
        m = re.search(rf"\b{key}\s+(-?[\d.]+|nan)", line)
        if m:
            out[key] = float(m.group(1))
    return out


def table(versions):
    """把多個版本的指標印成對齊的表格。versions 是 VERSION 字串的 list。"""
    head = f"{'version':20s}" + "".join(f"{lab:>11s}" for _, lab in COLUMNS)
    print(f"\n{head}\n{'-' * len(head)}")
    for v in versions:
        m = last_metrics(v)
        if not m:
            print(f"{v:20s}{'(沒有 result.log)':>11s}")
            continue
        print(f"{v:20s}" + "".join(
            f"{m.get(k, float('nan')):>11.4f}" for k, _ in COLUMNS))
    print("\nknock 是切斷 GAT 分支後的 val_dev：它比 val_dev 高越多，"
          "代表這條分支真的在做事。\n兩者相等就是 GAT 分支已死，"
          "此時 alpha 顯示多少都不算證據。")


def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else "first"
    groups = {"first": FIRST, "second": SECOND, "table": {**FIRST, **SECOND}}
    if arg not in groups:
        raise SystemExit(f"用法：{sys.argv[0]} [first|second|table]")
    names = list(groups[arg])
    if arg != "table":
        for v, o in groups[arg].items():
            run_one(v, o)
    table(names)


if __name__ == "__main__":
    main()
