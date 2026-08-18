import re
import os
import sys
import matplotlib.pyplot as plt

file = 'model/v3_gat/result/result.log'
MIN_EPOCH = 10

log = sys.argv[1] if len(sys.argv) > 1 else file

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
    epochs, train_nlls, val_devs = parse_metrics(log)
    epochs, train_nlls, val_devs = zip(
        *[(e, t, v) for e, t, v in zip(epochs, train_nlls, val_devs) if e >= MIN_EPOCH]
    )

    fig, ax_nll = plt.subplots(figsize=(8, 5))
    ax_dev = ax_nll.twinx()

    l1, = ax_nll.plot(epochs, train_nlls, color="tab:blue", label="train_nll")
    l2, = ax_dev.plot(epochs, val_devs, color="tab:orange", label="val_dev")

    ax_nll.set_xlabel("epoch")
    ax_nll.set_ylabel("train_nll", color="tab:blue")
    ax_dev.set_ylabel("val_dev", color="tab:orange")
    ax_nll.tick_params(axis="y", labelcolor="tab:blue")
    ax_dev.tick_params(axis="y", labelcolor="tab:orange")

    version = os.path.basename(os.path.dirname(os.path.dirname(log)))
    ax_nll.set_title(f"{version}: train_nll vs val_dev (divergence = overfitting)")
    ax_nll.legend(handles=[l1, l2], loc="best")
    ax_nll.grid(True, alpha=0.3)

    name = "train_vs_val.png" if len(sys.argv) <= 1 else f"train_vs_val_{version}.png"
    out_path = os.path.join(os.path.dirname(__file__), name)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"saved to {out_path}")


if __name__ == "__main__":
    main()
