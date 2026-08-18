import re
import os
import matplotlib.pyplot as plt

log1 = "model/v2_ddae_fsce/result/result64.log"
log2 = "model/v2_ddae_fsce/result/result.log"

MIN_EPOCH = 50

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
    with open(log_path, "r") as f:
        for line in f:
            m = pattern.search(line)
            if m:
                epochs.append(int(m.group(1)))
                train_nlls.append(float(m.group(2)))
                val_devs.append(float(m.group(3)))
    return epochs, train_nlls, val_devs




def main():
    epochs1, train1, val1 = parse_metrics(log1)
    epochs2, train2, val2 = parse_metrics(log2)

    epochs1, train1, val1 = zip(
        *[(e, t, v) for e, t, v in zip(epochs1, train1, val1) if e >= MIN_EPOCH]
    )
    epochs2, train2, val2 = zip(
        *[(e, t, v) for e, t, v in zip(epochs2, train2, val2) if e >= MIN_EPOCH]
    )

    name1 = log1
    name2 = log2

    fig, (ax_train, ax_val) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

    ax_train.plot(epochs1, train1, label=name1)
    ax_train.plot(epochs2, train2, label=name2)
    ax_train.set_ylabel("train_nll")
    ax_train.set_title("train_nll vs val_dev (divergence = overfitting)")
    ax_train.legend()
    ax_train.grid(True, alpha=0.3)

    ax_val.plot(epochs1, val1, label=name1)
    ax_val.plot(epochs2, val2, label=name2)
    ax_val.set_xlabel("epoch")
    ax_val.set_ylabel("val_dev")
    ax_val.legend()
    ax_val.grid(True, alpha=0.3)

    out_path = os.path.join(os.path.dirname(__file__), "log_compare.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"saved to {out_path}")


if __name__ == "__main__":
    main()
