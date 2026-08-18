"""測量二：Poisson 取樣噪聲的地板在哪，也就是 deviance 再怎麼努力也降不下去
的那一條線。

沒有這條線，0.57 這個數字是「還有一倍空間可挖」還是「已經貼著地板」根本
無法判斷，任何架構的改動都在猜。

做法是 parametric bootstrap：拿模型自己解出來的 λ̂ 當成「真值」，重抽
x* ~ Poisson(λ̂)。這批合成資料在建構上完全落在 decoder 的 2 維流形上、
而且 λ 完全正確，所以在它上面量到的 deviance 就是各個評分方式的地板。

量三條線（都在同一批合成資料上）：
  A  dev(λ̂, x*)          λ 完全正確、而且 z 直接給你——純粹的 Poisson 取樣噪聲
  B  dev(model(x*), x*)   整個模型（encoder+精修）跑在合成資料上——對應 val_dev
  C  dev(decoder(z*), x*) 凍住 decoder 自由解 z——對應 oracle.py 的 oracle

B、C 兩條會低於 A：z 是拿同一份 x* 解出來的，會吸掉一部分噪聲（10 個觀測值
配 2 個自由參數）。這正是為什麼要分開量——真實資料上的 refined / oracle 也
是同一種樂觀偏誤，只有跟同樣有偏的地板比才對得起來。

用法：python model/v5_refine/analyze/noise_floor.py [--reps 10] [--split val]
輸出印在 stdout，同時寫進 result/noise_floor.log。
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
SEED = 0          # 跟 train.py 一致，才切得出訓練時那一組 val
VAL_FRAC = 0.1
BATCH = 256

device = ("mps" if torch.backends.mps.is_available()
          else "cuda" if torch.cuda.is_available() else "cpu")


def model_forward(model, x):
    """乾淨輸入跑完整條精修，分批做完再接起來。

    model：MLPAE。x：(B,N_CAT) count（在 device 上）。

    回傳 (z, log_lam)：最終的 (B,LATENT_DIM) latent 與它的 (B,N_CAT) log λ。
    """
    zs, lls = [], []
    with torch.no_grad():
        for i in range(0, len(x), BATCH):
            z, log_lam, _ = model(x[i:i + BATCH])
            zs.append(z)
            lls.append(log_lam)
    return torch.cat(zs), torch.cat(lls)


def mean_std(v):
    """v 是 list of float，回傳 (平均, 標準差) 字串用的兩個 float。"""
    t = torch.tensor(v)
    return t.mean().item(), (t.std().item() if len(v) > 1 else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=10, help="bootstrap 重抽次數")
    ap.add_argument("--split", default="val", choices=["val", "train", "all"],
                    help="在哪一組 patch 上量")
    ap.add_argument("--restarts", type=int, default=4, help="oracle 的隨機起點數")
    ap.add_argument("--steps", type=int, default=1500, help="oracle 的 Adam 步數")
    args = ap.parse_args()

    f = open(result(VERSION, "noise_floor.log"), "w")

    def log(msg):
        print(msg)
        f.write(msg + "\n")
        f.flush()

    data = Patches(PATCHES)
    model = MLPAE(LATENT_DIM)
    model.load_state_dict(torch.load(result(VERSION, "ae.pt"), map_location="cpu"))
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    g = torch.Generator().manual_seed(SEED)
    perm = torch.randperm(data.n, generator=g)
    n_val = int(data.n * VAL_FRAC)
    idx = {"val": perm[:n_val], "train": perm[n_val:],
           "all": torch.arange(data.n)}[args.split]

    log(f"{VERSION} noise floor 測量：split={args.split}（{len(idx)} 個 patch），"
        f"device={device}，精修 T={model.n_steps}，bootstrap {args.reps} 次")

    # ---- 真實資料上的兩個對照數字 ----
    x_real = data.agg(idx).to(device)
    z_real, log_lam_hat = model_forward(model, x_real)
    real_per = poisson_deviance(log_lam_hat, x_real)          # (B,) 每個 patch 一個
    real_refined = real_per.mean().item()
    z_star, _, gnorm = fit_z(model.decoder, x_real, z_real, poisson_nll,
                             restarts=args.restarts, steps=args.steps, seed=SEED)
    with torch.no_grad():
        real_oracle = poisson_deviance(model.decoder(z_star), x_real).mean().item()
    log(f"真實資料：refined {real_refined:.4f}／oracle {real_oracle:.4f}"
        f"（|∇_z NLL| = {gnorm:.2e}）")

    # ---- 合成資料上的三條地板 ----
    lam_hat = log_lam_hat.exp().cpu()
    A, B, C = [], [], []
    b_per = []          # 每個 rep 的逐 patch deviance，配對比較要用
    gs = torch.Generator().manual_seed(SEED)
    for r in range(args.reps):
        xb = torch.poisson(lam_hat, generator=gs).to(device)
        a = poisson_deviance(log_lam_hat, xb).mean().item()
        zb, ll_b = model_forward(model, xb)
        bp = poisson_deviance(ll_b, xb)
        b_per.append(bp)
        b = bp.mean().item()
        zb_star, _, _ = fit_z(model.decoder, xb, zb, poisson_nll,
                              restarts=args.restarts, steps=args.steps, seed=SEED + r)
        with torch.no_grad():
            c = poisson_deviance(model.decoder(zb_star), xb).mean().item()
        A.append(a)
        B.append(b)
        C.append(c)
        log(f"  rep {r + 1:2d}/{args.reps}  A(λ已知) {a:.4f}  "
            f"B(整個模型) {b:.4f}  C(oracle) {c:.4f}")

    am, asd = mean_std(A)
    bm, bsd = mean_std(B)
    cm, csd = mean_std(C)

    log("\n[地板]（合成資料，λ 在建構上完全正確）")
    log(f"  A  λ 已知、z 已知      {am:.4f} ± {asd:.4f}")
    log(f"  B  整個模型跑一遍      {bm:.4f} ± {bsd:.4f}   ← 對應 val_dev")
    log(f"  C  凍 decoder 自由解 z {cm:.4f} ± {csd:.4f}   ← 對應 oracle")

    # 配對比較：真實與合成的差是逐 patch 對齊的（同一個 patch、同一個 λ̂），
    # 所以誤差棒要用「patch 之間的變異」算，不是用 rep 之間的變異。前者才是
    # 「這 123 個 patch 只是所有 patch 的一個抽樣」帶來的不確定性。
    diff = (real_per - torch.stack(b_per).mean(dim=0))
    dm = diff.mean().item()
    dse = (diff.std() / len(diff) ** 0.5).item()
    mc_se = bsd / args.reps ** 0.5

    log("\n[誤差棒]")
    log(f"  合成地板 B 的 Monte-Carlo 標準誤（{args.reps} 次 bootstrap）±{mc_se:.4f}")
    log(f"  逐 patch 配對差 real - floor = {dm:+.4f} ± {dse:.4f}（{len(diff)} 個 patch 的標準誤）")
    log(f"  → {abs(dm) / dse:.1f} 倍標準誤；小於 2 就是連「這個差存在」都證不了")

    log("\n[可挖空間]")
    log(f"  refined  {real_refined:.4f} - 地板 {bm:.4f} = {real_refined - bm:+.4f}")
    log(f"  oracle   {real_oracle:.4f} - 地板 {cm:.4f} = {real_oracle - cm:+.4f}")
    log("  第一列是「換更好的架構最多還能拿回多少」的總預算；第二列是其中"
        "「decoder 的 2 維流形本身沒穿過資料」的那一份。")
    log("  兩列相減就是找 z 那一路還欠的量，應該跟 oracle.py 量到的剩餘 gap 對得上。")
    if real_refined - bm < 0.05:
        log("  總預算已經很薄：0.57 基本上是貼著 Poisson 噪聲地板，"
            "再堆架構的邊際報酬很低，該換題目而不是換模型。")


main()
