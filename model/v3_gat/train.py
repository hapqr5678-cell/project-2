"""訓練 v3_gat_literal：完全照提案字面的 count + OD 圖雙分支模型。

跟 v2_ddae_tanh_fsce 逐字保持一致的部分（不然比較就不乾淨）：SEED、
VAL_FRAC 與切分方式、BATCH、LR、EPOCHS、WEIGHT_DECAY、破壞方式、
Poisson NLL、explained deviance、latents.npz 的欄位。唯一的差別就是
encoder 多了一條 OD 圖分支。loss 也只有 Poisson NLL，沒有任何附加項。

result.log 多印三個純觀測的欄位（不影響訓練，只是量測）：

  alpha      sigmoid(可學習純量)。因為兩條分支出口都過了 LayerNorm，
             這個數字可以直接讀成「OD 分支佔多少比重」。
  w_od       GAT 各層 edge bias 的權重絕對值平均。OD 進 attention 的強度，
             w_od -> 0 就是 attention 不再理會 OD 矩陣。
  knock_dev  驗證時把 h_gat 換成該批平均後重算的 val_dev。
             knock_dev - val_dev ≈ 0 代表不管 alpha 顯示多少，GAT 分支
             對重建都沒有貢獻——alpha 的數值不是證據，Δloss 才是。
"""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(f"{os.path.dirname(__file__)}/../.."))
from ae import (GATLiteralAE, ODGraph, Patches, corrupt,  # noqa: E402
                poisson_deviance, poisson_nll)
from config.dataset import OD, PATCHES, ensure_od, result  # noqa: E402
from config.train_log import open_log  # noqa: E402

VERSION = "v3_gat"
OUT = result(VERSION, "latents.npz")
CKPT = result(VERSION, "ae.pt")

LATENT_DIM = 2
EPOCHS = 1200
BATCH = 256
LR = 1e-3
WEIGHT_DECAY = 1e-4      # 跟 v2_ddae_tanh_fsce 一致
VAL_FRAC = 0.1
SEED = 0

ALPHA_INIT = 0.5         # sigmoid 的起點，0.5 = 兩條分支等重

NOISE_P = 0.3
NOISE_MODE = "thinning"

device = ("mps" if torch.backends.mps.is_available()
          else "cuda" if torch.cuda.is_available() else "cpu")


def evaluate(model, data, od, idx, log_lam_null):
    """驗證集上的一輪評估，全部用乾淨輸入、eval 模式。

    model：GATLiteralAE。data：Patches。od：ODGraph。
    idx：要評估的 patch 編號 LongTensor。log_lam_null：(1,N_CAT) 空模型的 log λ。

    回傳 (dev, expl, knock_dev)：Poisson deviance、explained deviance、
    切斷 GAT 分支後的 deviance。
    """
    dev, dev_null, dev_ko = [], [], []
    with torch.no_grad():
        for i in range(0, len(idx), BATCH):
            batch = idx[i:i + BATCH]
            x = data.agg(batch).to(device)
            a_local = od.A_local[batch.to(device)]
            _, log_lam, _ = model(x, a_local)
            dev.append(poisson_deviance(log_lam, x))
            dev_null.append(poisson_deviance(log_lam_null.expand(len(batch), -1), x))
            _, log_lam_ko, _ = model(x, a_local, knockout=True)
            dev_ko.append(poisson_deviance(log_lam_ko, x))
    d = torch.cat(dev).mean().item()
    return d, 1 - d / torch.cat(dev_null).mean().item(), torch.cat(dev_ko).mean().item()


def run(data, od, train_idx, val_idx, log):
    """訓練並回傳 (z, err)：z 是 (N,LATENT_DIM) 的 latent、err 是 (N,) 的
    Poisson deviance，兩者都用乾淨輸入、eval 模式算出來。

    data 是 Patches；od 是已搬上 device 的 ODGraph；
    train_idx / val_idx 是 patch 編號的 LongTensor；
    log 是 config.train_log.open_log() 回傳的函式。
    訓練中按 Ctrl-C 會提前跳出，用當下的模型存 checkpoint 跟 latent。
    """
    torch.manual_seed(SEED)
    model = GATLiteralAE(LATENT_DIM, ALPHA_INIT).to(device)

    # edge bias 的 w_od 與 alpha 不套 weight decay：套了的話它們會被 L2
    # 單調往 0 拉，「模型主動關掉 OD」跟「被 weight decay 壓下去」就分不出來
    free = [p for layer in model.gat.layers for p in [layer.w_od]] + [model.alpha_raw]
    free_ids = {id(p) for p in free}
    opt = torch.optim.Adam(
        [{"params": [p for p in model.parameters() if id(p) not in free_ids],
          "weight_decay": WEIGHT_DECAY},
         {"params": free, "weight_decay": 0.0}], lr=LR)

    log(f"參數量 {sum(p.numel() for p in model.parameters())}")

    # 空模型（λ=訓練集全域平均）的 log_lam，當 explained deviance 的分母
    log_lam_null = data.agg(train_idx).mean(dim=0, keepdim=True) \
        .clamp_min(1e-8).log().to(device)

    for epoch in range(EPOCHS):
        try:
            model.train()
            g = torch.Generator().manual_seed(SEED + epoch)
            perm = train_idx[torch.randperm(len(train_idx), generator=g)]
            total = 0.0
            for i in range(0, len(perm), BATCH):
                batch = perm[i:i + BATCH]
                x = data.agg(batch)                                  # 乾淨目標（CPU）
                x_in = corrupt(x, NOISE_P, NOISE_MODE, generator=g)  # 加噪輸入
                x, x_in = x.to(device), x_in.to(device)
                a_local = od.A_local[batch.to(device)]                # OD 不加噪

                _, log_lam, _ = model(x_in, a_local)
                loss = poisson_nll(log_lam, x).mean()   # 目標永遠是乾淨的 x
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += loss.item() * len(batch)

            if (epoch + 1) % 5 == 0 or epoch == 0:
                model.eval()
                dev, expl, knock = evaluate(model, data, od, val_idx, log_lam_null)
                w_od = np.mean([float(l.w_od.detach().abs().mean())
                                for l in model.gat.layers])
                log(f"epoch {epoch + 1:4d} | train_nll {total / len(perm):.5f} | "
                    f"val_dev {dev:.5f} | expl_dev {expl:.5f} | "
                    f"knock_dev {knock:.5f} | alpha {model.alpha().item():.4f} | "
                    f"w_od {w_od:.4f}")
        except KeyboardInterrupt:
            log(f"\n[中斷] epoch {epoch + 1} 收到 Ctrl-C，用目前模型存 checkpoint 跟 latent")
            break

    torch.save(model.state_dict(), CKPT)

    model.eval()
    zs, errs = [], []
    with torch.no_grad():
        for i in range(0, data.n, BATCH):
            idx = torch.arange(i, min(i + BATCH, data.n))
            x = data.agg(idx).to(device)   # 推論不加噪
            z, log_lam, _ = model(x, od.A_local[idx.to(device)])
            zs.append(z.cpu())
            errs.append(poisson_deviance(log_lam, x).cpu())
    return torch.cat(zs).numpy(), torch.cat(errs).numpy()


def main():
    log = open_log(VERSION, {
        "LATENT_DIM": LATENT_DIM, "EPOCHS": EPOCHS, "BATCH": BATCH, "LR": LR,
        "WEIGHT_DECAY": WEIGHT_DECAY, "VAL_FRAC": VAL_FRAC, "SEED": SEED,
        "ALPHA_INIT": ALPHA_INIT,
        "NOISE_P": NOISE_P, "NOISE_MODE": NOISE_MODE,
    })

    ensure_od()          # 內含 ensure_patches()
    data = Patches(PATCHES)
    od = ODGraph(OD).to(device)
    assert od.n == data.n, f"od.npz 有 {od.n} 個 patch，patches.npz 有 {data.n} 個"

    log(f"{data.n} 個 patch，device={device}，噪聲 {NOISE_MODE} p={NOISE_P}")
    log(f"OD：{(od.n_od > 0).float().mean().item() * 100:.1f}% 的 patch 有觀測，"
        f"總量中位數 {od.n_od.median().item():.0f}，"
        f"最大 {od.n_od.max().item():.0f}")

    g = torch.Generator().manual_seed(SEED)
    perm = torch.randperm(data.n, generator=g)
    n_val = int(data.n * VAL_FRAC)
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    z, err = run(data, od, train_idx, val_idx, log)
    np.savez(OUT, n_poi=data.n_poi, lat=data.lat, lon=data.lon, z=z, err=err)
    log(f"已存 {OUT}")


main()
