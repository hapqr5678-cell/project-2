import re
import os
import matplotlib.pyplot as plt

# 可以直接填入檔案路徑，或者使用 tuple: ("檔案路徑", "自訂標籤")
# 若只有路徑，預設會使用檔案名稱作為標籤
log1 = ("model/v2_ddae_fsce/result/result_base.log", 'v2_ddae_fsce')
log2 = ""
log3 = ""
log4 = ("model/v2_ddae_fsce_moe/result/result.log", 'v2_ddae_fsce_moe')
log5 = ("model/v2_ddae_fsce_moe_way1/result/result.log", 'v2_ddae_fsce_moe_way1')
log6 = ("model/v3_ddae_fsce_moe_way2/result/result.log", 'v3_ddae_fsce_moe_way2')

# 將所有要比較的 log 變數放入列表中，並自動過濾掉空字串
LOG_FILES = [log for log in [log1, log2, log3, log4, log5, log6] if log]

MIN_EPOCH = 100

def parse_metrics(log_path):
    """
    傳入值: log_path (str) - result.log 檔案路徑
    return值: (epochs, train_nlls, val_devs) - 三個等長 list，
              分別為 epoch 數、對應的 train_nll、val_dev 數值
    """
    epochs = []
    train_nlls = []
    val_devs = []
    pattern = re.compile(
        r"epoch\s+(\d+)\s*\|.*?train_nll\s+([-\d.]+).*?val_dev\s+([-\d.]+)"
    )
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = pattern.search(line)
            if m:
                epochs.append(int(m.group(1)))
                train_nlls.append(float(m.group(2)))
                val_devs.append(float(m.group(3)))
    return epochs, train_nlls, val_devs


def main():
    if not LOG_FILES:
        print("沒有設定任何 log 檔案")
        return

    fig, (ax_train, ax_val) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

    for item in LOG_FILES:
        # 支援 ("路徑", "標籤") 或 "路徑" 兩種格式
        if isinstance(item, tuple) and len(item) == 2:
            log_path, name = item
        else:
            log_path = item
            name = os.path.basename(os.path.dirname(os.path.dirname(log_path))) + "/" + os.path.basename(log_path) # 用部分路徑當預設標籤，比較清楚
            # 或者單純用 log_path 也行，這裡改用原來的 log_path 保持一致
            name = log_path

        if not os.path.exists(log_path):
            print(f"找不到檔案，略過: {log_path}")
            continue

        epochs, train, val = parse_metrics(log_path)
        
        # 過濾掉小於 MIN_EPOCH 的數據
        filtered = [(e, t, v) for e, t, v in zip(epochs, train, val) if e >= MIN_EPOCH]
        if not filtered:
            continue
            
        epochs, train, val = zip(*filtered)

        ax_train.plot(epochs, train, label=name)
        ax_val.plot(epochs, val, label=name)

    ax_train.set_ylabel("train_nll")
    ax_train.set_title("train_nll vs val_dev")
    ax_train.legend()
    ax_train.grid(True, alpha=0.3)

    ax_val.set_xlabel("epoch")
    ax_val.set_ylabel("val_dev")
    ax_val.legend()
    ax_val.grid(True, alpha=0.3)

    out_path = os.path.join(os.path.dirname(__file__), "log_compare.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"saved to {out_path}")


if __name__ == "__main__":
    main()
