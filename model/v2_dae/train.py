"""訓練 v2_dae 的 denoising MLP AutoEncoder，輸出每個 patch 的 latent 與 Poisson deviance。

跟 v2_ae 的 train.py 相比只多一件事：訓練迴圈裡，餵給模型的是
corrupt() 破壞過的 count 向量，Poisson NLL 仍然拿乾淨的原始 count 當目標，
也就是「從殘缺的觀測重建完整的分布」。驗證與最後輸出 latent 的推論階段
一律不加噪——噪聲是正則化手段，不是資料本身的性質。

破壞是每個 step 重新抽的（generator 綁 SEED+epoch，可重現），同一個 patch
在不同 epoch 會看到不同的殘缺版本，模型才學得到「哪些類別彼此可以互相推」。

其餘超參數（latent 2 維、EPOCHS、lr 1e-3、切分 seed）全部跟 v2_ae 對齊，
這樣兩邊的 latent 才是在同一組資料/loss 下比較，差別只有加噪這件事。
"""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(f"{os.path.dirname(__file__)}/../.."))
from ae import MLPAE, Patches, corrupt, poisson_deviance, poisson_nll  # noqa: E402
from config.dataset import ensure_patches, PATCHES, result  # noqa: E402
from config.train_log import open_log  # noqa: E402

VERSION = "v2_dae"
OUT = result(VERSION, "latents.npz")
CKPT = result(VERSION, "ae.pt")

LATENT_DIM = 2
EPOCHS = 1000
BATCH = 256
LR = 1e-3
VAL_FRAC = 0.1
SEED = 0

NOISE_P = 0.3            # 破壞強度：thinning 是每個 POI 被丟掉的機率
NOISE_MODE = "thinning"  # "thinning"（逐 POI 丟）或 "mask"（整類歸零）

device = ("mps" if torch.backends.mps.is_available()
          else "cuda" if torch.cuda.is_available() else "cpu")


def run(data, train_idx, val_idx, log):
    """訓練並回傳全體 patch 的 (z, err)：z 是 (N,LATENT_DIM) 的 latent 座標，
    err 是 (N,) 的 Poisson deviance，兩者都用乾淨輸入、eval 模式算出來。
    data 是 Patches，train_idx / val_idx 是 patch 編號的 LongTensor。
    log 是 config.train_log.open_log() 回傳的函式，訓練過程的訊息都灌進去。
    訓練中按 Ctrl-C 會提前跳出迴圈，用當下的模型狀態存 checkpoint 跟 latent，
    不會整個丟掉重來。
    """
    torch.manual_seed(SEED)
    model = MLPAE(LATENT_DIM).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    # 空模型（λ=訓練集全域平均）的 log_lam，當 explained deviance 的分母，
    # 用訓練集估、驗證集算 deviance，不然分母本身就偷看了驗證集
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
                _, log_lam = model(x_in)
                loss = poisson_nll(log_lam, x).mean()   # 目標永遠是乾淨的 x
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += loss.item() * len(batch)

            if (epoch + 1) % 20 == 0 or epoch == 0:
                model.eval()
                with torch.no_grad():
                    vl, vd, vdn = [], [], []
                    for i in range(0, len(val_idx), BATCH):
                        batch = val_idx[i:i + BATCH]
                        x = data.agg(batch).to(device)   # 驗證不加噪
                        _, log_lam = model(x)
                        vl.append(poisson_nll(log_lam, x))
                        vd.append(poisson_deviance(log_lam, x))
                        vdn.append(poisson_deviance(log_lam_null.expand(len(batch), -1), x))
                    val = torch.cat(vl).mean().item()
                    dev = torch.cat(vd).mean().item()
                    dev_null = torch.cat(vdn).mean().item()
                    expl = 1 - dev / dev_null
                log(f"epoch {epoch + 1:4d} | train_nll {total / len(perm):.5f} | "
                    f"val_nll {val:.5f} | val_dev {dev:.5f} | expl_dev {expl:.5f}")
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
            z, log_lam = model(x)
            zs.append(z.cpu())
            errs.append(poisson_deviance(log_lam, x).cpu())
    return torch.cat(zs).numpy(), torch.cat(errs).numpy()


def main():
    log = open_log(VERSION, {
        "LATENT_DIM": LATENT_DIM, "EPOCHS": EPOCHS, "BATCH": BATCH, "LR": LR,
        "VAL_FRAC": VAL_FRAC, "SEED": SEED,
        "NOISE_P": NOISE_P, "NOISE_MODE": NOISE_MODE,
    })

    ensure_patches()
    data = Patches(PATCHES)
    log(f"{data.n} 個 patch，device={device}，"
        f"噪聲 {NOISE_MODE} p={NOISE_P}")

    g = torch.Generator().manual_seed(SEED)
    perm = torch.randperm(data.n, generator=g)
    n_val = int(data.n * VAL_FRAC)
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    z, err = run(data, train_idx, val_idx, log)
    np.savez(OUT, n_poi=data.n_poi, lat=data.lat, lon=data.lon, z=z, err=err)
    log(f"已存 {OUT}")


main()
