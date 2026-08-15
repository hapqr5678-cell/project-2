"""找出 patches.npz 中 POI 最密集的 patches。

支援：
1. 依全體 POI 總數排序，或依特定類別（例如 `--cat 夜生活`）排序。
2. 空間去重（--dedup）：自動過濾半徑內重疊的相鄰 patch，找出城市中不同的獨立核心熱點。
3. 輸出 patch 編號（可直接複製給 `rebuild_test.py --n <id>` 或 `patch_lab.py` 使用）。

用法範例：
  p3 data/patch/find_dense_patches.py
  p3 data/patch/find_dense_patches.py --top 10 --dedup
  p3 data/patch/find_dense_patches.py --cat 夜生活 --top 15
  p3 data/patch/find_dense_patches.py --csv top_dense.csv
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(f"{os.path.dirname(__file__)}/../.."))
from config.dataset import CAT_ZH, CATEGORIES, PATCHES, HALF_WIDTH  # noqa: E402


def calculate_category_counts(d, n_patches, n_cats):
    """計算每個 patch 在各類別的 POI 數量矩陣 (N, N_CAT)。"""
    offsets = d["offsets"]
    cats = d["cat"]
    
    # 每個點屬於第幾個 patch
    lens = offsets[1:] - offsets[:-1]
    patch_ids = np.repeat(np.arange(n_patches), lens)
    
    # flat index = patch_id * N_CAT + cat
    flat = patch_ids * n_cats + cats.astype(np.int64)
    counts = np.bincount(flat, minlength=n_patches * n_cats)
    return counts.reshape(n_patches, n_cats)


def main():
    parser = argparse.ArgumentParser(description="找出 patches.npz 中最密集的 patches")
    parser.add_argument("--top", type=int, default=20, help="顯示前幾名（預設 20）")
    parser.add_argument("--cat", type=str, default=None,
                        help=f"指定只看特定類別，可填中文或英文名稱：{', '.join(CAT_ZH)}")
    parser.add_argument("--dedup", nargs="?", const=HALF_WIDTH, type=float, default=None,
                        help="空間去重半徑（公尺，預設為 HALF_WIDTH）。開啟後會避開重疊的鄰近 patch，找出獨立熱點")
    parser.add_argument("--csv", type=str, default=None, help="將結果輸出為 CSV 檔案")
    args = parser.parse_args()

    assert os.path.exists(PATCHES), f"找不到 patches 檔案：{PATCHES}"
    d = np.load(PATCHES)
    n_patches = len(d["n_poi"])
    n_cats = len(CAT_ZH)

    target_cat_idx = None
    if args.cat is not None:
        if args.cat in CAT_ZH:
            target_cat_idx = CAT_ZH.index(args.cat)
        elif args.cat in CATEGORIES:
            target_cat_idx = CATEGORIES.index(args.cat)
        else:
            try:
                target_cat_idx = int(args.cat)
                assert 0 <= target_cat_idx < n_cats
            except (ValueError, AssertionError):
                print(f"錯誤：找不到類別 '{args.cat}'，可選類別為：{CAT_ZH}")
                return

    # 計算排序分數
    cat_counts = calculate_category_counts(d, n_patches, n_cats)
    if target_cat_idx is not None:
        sort_metric = cat_counts[:, target_cat_idx]
        metric_name = f"類別[{CAT_ZH[target_cat_idx]}] POI 數"
    else:
        sort_metric = d["n_poi"]
        metric_name = "總 POI 數"

    sorted_indices = np.argsort(-sort_metric)

    # 空間去重 (Non-Maximum Suppression)
    if args.dedup is not None:
        dedup_dist = args.dedup
        cx = d["center_x"]
        cy = d["center_y"]
        selected_indices = []
        selected_coords = []

        for idx in sorted_indices:
            if sort_metric[idx] <= 0:
                break
            pt = np.array([cx[idx], cy[idx]])
            if len(selected_coords) > 0:
                dists = np.linalg.norm(np.array(selected_coords) - pt, axis=1)
                if np.min(dists) < dedup_dist:
                    continue
            selected_indices.append(idx)
            selected_coords.append(pt)
            if len(selected_indices) >= args.top:
                break
        final_indices = np.array(selected_indices)
    else:
        final_indices = sorted_indices[:args.top]

    # 印出結果表格
    print("\n" + "=" * 90)
    title = f"最密集的 Patches 排行榜（共 {n_patches} 個 patches，評估指標：{metric_name}）"
    if args.dedup is not None:
        title += f" [空間去重半徑: {args.dedup:.0f}m]"
    print(f" {title}")
    print("=" * 90)
    header = f"{'名次':>4} | {'Patch ID':>8} | {metric_name:>14} | {'總 POI':>7} | {'緯度 (Lat)':>10} | {'經度 (Lon)':>10} | 主要類別分布"
    print(header)
    print("-" * 90)

    rows_for_csv = []
    for rank, idx in enumerate(final_indices, 1):
        total_p = d["n_poi"][idx]
        metric_val = sort_metric[idx]
        lat = d["center_lat"][idx]
        lon = d["center_lon"][idx]
        
        # 統計該 patch 前 3 大類別
        p_cats = cat_counts[idx]
        top_cats = sorted(zip(CAT_ZH, p_cats), key=lambda x: -x[1])[:3]
        top_str = ", ".join(f"{name}:{cnt}" for name, cnt in top_cats if cnt > 0)

        print(f"{rank:4d} | {idx:8d} | {metric_val:14d} | {total_p:7d} | {lat:10.5f} | {lon:10.5f} | {top_str}")

        if args.csv:
            rows_for_csv.append({
                "rank": rank,
                "patch_id": int(idx),
                "metric_name": metric_name,
                "metric_value": int(metric_val),
                "total_poi": int(total_p),
                "lat": float(lat),
                "lon": float(lon),
                "top_categories": top_str,
                "gmaps_link": f"https://www.google.com/maps?q={lat},{lon}",
            })

    print("=" * 90)
    print("提示：可直接複製 Patch ID 搭配 rebuild_test 或 patch_lab 查看：")
    if len(final_indices) > 0:
        sample_id = final_indices[0]
        print(f"  例如: p3 model/v0_poisson_nll/analyze/rebuild_test.py --n {sample_id}\n")

    if args.csv and rows_for_csv:
        import pandas as pd
        df_out = pd.DataFrame(rows_for_csv)
        df_out.to_csv(args.csv, index=False, encoding="utf-8-sig")
        print(f"已輸出結果至 CSV: {args.csv}\n")


if __name__ == "__main__":
    main()
