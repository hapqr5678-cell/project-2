"""測量一：v5_refine 自己的 amortization gap 還剩多少。

做法跟 ae.py 檔頭當初量 v2_ddae_fsce 的那一次完全一樣——把訓練好的 decoder
整個凍住，把每個 patch 的 latent 當自由變數用梯度解到底（多起點，避開局部
極小），得到「這個 decoder 在這個 latent 維度下所能達到的最好重建」，也就是
oracle。然後跟模型實際輸出的 deviance 相減。

這個數字決定下一步該往哪加架構：
  refined - oracle 還很大  → 瓶頸仍在「怎麼找到那 latent_dim 個數字」，
                             該做的是更好的精修（預條件、更多步、多起點）
  refined - oracle ≈ 0     → 精修已經到頂，瓶頸換成 decoder 的 2 維流形本身
                             穿不過那些點，該動的是 decoder 的表達形式

注意 oracle 是「拿要被評分的那份 x 去解 z」，所以它對 z 是樂觀有偏的，不是
可達成的泛化上限——它衡量的是「同一個 decoder + 同一份輸入下，前饋+精修
離最佳解還有多遠」。要知道這個 oracle 值本身的地板在哪，跑 noise_floor.py。

用法：python model/v5_refine/analyze/oracle.py [--restarts 4] [--steps 1500]
輸出印在 stdout，同時寫進 result/oracle.log。
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.abspath(f"{os.path.dirname(__file__)}"))
sys.path.insert(0, os.path.abspath(f"{os.path.dirname(__file__)}/.."))
sys.path.insert(0, os.path.abspath(f"{os.path.dirname(__file__)}/../../.."))
from ae import MLPAE, Patches, poisson_deviance, poisson_nll  # noqa: E402
from oracle_fit import fit_z  # noqa: E402
from config.dataset import PATCHES, result  # noqa: E402

VERSION = "v5_refine"
LATENT_DIM = 2
# 下面三個必須跟 train.py 一致，否則切出來的 val 不是訓練時那一組
SEED = 0
VAL_FRAC = 0.1
BATCH = 256

device = ("mps" if torch.backends.mps.is_available()
          else "cuda" if torch.cuda.is_available() else "cpu")


def forward_all(model, data, idx):
    """對 idx 這批 patch 跑完整條精修（乾淨輸入、eval 模式），分批做完再接起來。

    model：MLPAE。data：Patches。idx：patch 編號的 LongTensor。

    回傳 (x, z, dev_steps)：
        x (B,N_CAT) 乾淨 count（在 device 上），
        z (B,LATENT_DIM) 精修完的最終 latent，
        dev_steps (T+1,B) 軌跡上每一步的 Poisson deviance。
    """
    xs, zs, devs = [], [], []
    with torch.no_grad():
        for i in range(0, len(idx), BATCH):
            x = data.agg(idx[i:i + BATCH]).to(device)
            z, _, log_lams = model(x)
            xs.append(x)
            zs.append(z)
            devs.append(torch.stack([poisson_deviance(ll, x) for ll in log_lams]))
    return torch.cat(xs), torch.cat(zs), torch.cat(devs, dim=1)


def report(name, model, data, idx, restarts, steps, log):
    """量一個 split 的前饋 / 精修 / oracle 三個 deviance 並印出來。

    name：這個 split 的顯示名稱。model / data / idx 同 forward_all。
    restarts / steps：轉給 fit_z。log：印訊息的函式。

    回傳 dict，欄位 feedforward / refined / oracle 是三個平均 deviance。
    """
    x, z, dev_steps = forward_all(model, data, idx)
    z_star, _, gnorm = fit_z(model.decoder, x, z, poisson_nll,
                             restarts=restarts, steps=steps, seed=SEED)
    with torch.no_grad():
        dev_star = poisson_deviance(model.decoder(z_star), x).mean().item()
        shift = (z_star - z).norm(dim=1).mean().item()
        radius = z.norm(dim=1).mean().item()

    ff = dev_steps[0].mean().item()
    refined = dev_steps[-1].mean().item()
    log(f"\n[{name}]（{len(idx)} 個 patch）")
    log("  精修軌跡 " + "→".join(f"{v:.4f}" for v in dev_steps.mean(dim=1).tolist()))
    log(f"  encoder 前饋      {ff:.4f}")
    log(f"  精修 T 步後       {refined:.4f}   （精修吃掉 {ff - refined:.4f}）")
    log(f"  oracle（自由解 z）{dev_star:.4f}   （剩餘 gap {refined - dev_star:+.4f}，"
        f"佔前饋→oracle 全距的 {(refined - dev_star) / (ff - dev_star) * 100:.1f}%）")
    log(f"  oracle 把 z 挪了 {shift:.4f}（latent 平均半徑 {radius:.4f} 的 "
        f"{shift / radius * 100:.1f}%）；收斂檢查 |∇_z NLL| = {gnorm:.2e}")
    return {"feedforward": ff, "refined": refined, "oracle": dev_star}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--restarts", type=int, default=4, help="除了模型的 z 之外的隨機起點數")
    ap.add_argument("--steps", type=int, default=1500, help="每個起點的 Adam 步數")
    args = ap.parse_args()

    f = open(result(VERSION, "oracle.log"), "w")

    def log(msg):
        print(msg)
        f.write(msg + "\n")
        f.flush()

    data = Patches(PATCHES)
    model = MLPAE(LATENT_DIM)
    model.load_state_dict(torch.load(result(VERSION, "ae.pt"), map_location="cpu"))
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)   # decoder 是被凍住的，只有 z 是自由變數

    g = torch.Generator().manual_seed(SEED)
    perm = torch.randperm(data.n, generator=g)
    n_val = int(data.n * VAL_FRAC)
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    log(f"{VERSION} oracle 測量：{data.n} 個 patch，device={device}，"
        f"精修 T={model.n_steps}，起點 1+{args.restarts} 個 × {args.steps} 步")

    v = report("val", model, data, val_idx, args.restarts, args.steps, log)
    t = report("train", model, data, train_idx, args.restarts, args.steps, log)

    log("\n[判讀]")
    gap_v = v["refined"] - v["oracle"]
    gap_t = t["refined"] - t["oracle"]
    log(f"  val 剩餘 gap {gap_v:+.4f}／train 剩餘 gap {gap_t:+.4f}")
    if abs(gap_v - gap_t) < 0.015:
        log("  兩邊的 gap 差不多 → 這是純粹的 amortization gap，不是過擬合。")
    else:
        log("  兩邊的 gap 明顯不同 → 除了 amortization 之外還混了泛化的成分，"
            "看 val/train 哪邊大再判斷。")
    log("  注意：gap 不等於 0 不代表精修還有得賺。oracle 是拿要被評分的那份 x "
        "去解 z，10 個觀測值配 2 個自由參數，它必然會吸掉一部分 Poisson 噪聲——"
        "就算模型完全正確，這個 gap 也不會是 0。")
    log("  要判斷這個 gap 是「精修沒做好」還是「解 z 吸噪聲的必然代價」，"
        "必須跟 noise_floor.py（測量二）在合成資料上量到的同一個 gap 相減。"
        "兩者一樣大 = 精修已經到頂，這條路沒有東西可以拿。")


main()
