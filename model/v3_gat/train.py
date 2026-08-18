"""訓練 v3_gat_literal：count + OD 圖雙分支模型，loss 是 Poisson NLL + FSCE。

跟 v2_ddae_fsce 逐字保持一致的部分（不然比較就不乾淨）：SEED、VAL_FRAC
與切分方式、BATCH、LR、EPOCHS、WEIGHT_DECAY、破壞方式、Poisson NLL、
FSCE（N_NEIGHBORS、GRAPH_METRIC、EDGE_BATCH、LAMBDA_FSCE、WARMUP_EPOCHS
全部同值）、explained deviance、latents.npz 的欄位。唯一的差別就是 encoder
多了一條 OD 圖分支——這樣 val_dev 才能直接拿來回答「在 v2_ddae_fsce 這個
已經有 FSCE 的配方上，多加 GAT 分支有沒有幫助」，而不是把「有沒有 FSCE」
跟「有沒有 GAT」兩個變因混在一起比。

FSCE 的 pair（正樣本、負樣本）一樣要過 AE.encode()，所以每個 pair 除了
count 向量還要各自查一份 OD（od.A_local[idx]），跟 v2_ddae_fsce 唯一的
差別只有這裡多一個引數。

OD 跟 count 一樣要加噪（NOISE_OD_P，binomial thinning）。上一版只破壞
count、A_local 每個 epoch 原封不動，結果是 DAE 的正則化只保護了兩條分支
中的一條：GAT 分支拿到的是一份沒有任何 augmentation 的 per-patch 固定
指紋，可以靠記住「這格是哪一格」把乾淨的 x 背出來——train_nll 每個檢查點
都比 v2_ddae_fsce 低，val_dev 卻從 epoch 450 左右起就停在 0.74 不動。
加噪的地方跟 count 完全對齊：recon 那一路、FSCE 的 pair 都加，evaluate()
與最後輸出 latent 的推論都用乾淨的 A_local。

result.log 多印兩個純觀測的欄位（不影響訓練，只是量測）：

  alpha      兩條分支的混合比例。因為兩條分支出口都過了 LayerNorm，這個數字
             可以直接讀成「OD 分支佔多少比重」。ALPHA_LEARN=False 時它是固定
             的超參數，每行都印同一個值（留著是為了 log 格式穩定、也方便掃）。
  gat_gain   knock_dev - val_dev：把 h_gat 換成該批平均後重算的 val_dev，
             減去真正的 val_dev。≈0 代表不管 alpha 顯示多少，GAT 分支對
             重建都沒有貢獻；>0 代表 GAT 分支有幫忙（拔掉之後誤差變大）；
             alpha 的數值不是證據，這個差值才是。
"""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(f"{os.path.dirname(__file__)}/../.."))
from ae import (AE, GAT_DIM, GAT_HEADS, GAT_LAYERS,  # noqa: E402
                GAT_READOUT_DIM, ODGraph, Patches, build_fsce_graph,
                corrupt, corrupt_od, fsce_loss, poisson_deviance, poisson_nll)
from config.dataset import OD, PATCHES, ensure_od, result  # noqa: E402
from config.train_log import open_log  # noqa: E402

VERSION = "v3_gat"
OUT = result(VERSION, "latents.npz")
CKPT = result(VERSION, "ae.pt")

LATENT_DIM = 2
EPOCHS = 1000
BATCH = 256
LR = 1e-3
WEIGHT_DECAY = 1e-6      # 跟 v2_ddae_fsce 一致
VAL_FRAC = 0.1
SEED = 0

ALPHA_INIT = 0.1         # 兩條分支的混合比例，上限 ae.AE.ALPHA_MAX=0.2，0.1 = 等重
ALPHA_LEARN = False      # False = 釘死在 ALPHA_INIT，不當參數學（理由見 ae.AE）
GAT_LR_MULT = 0.2        # GAT 分支的 lr 倍率，讓它學得比 count 分支慢

N_NEIGHBORS = 15             # 建高維 fuzzy graph 的 kNN 數，跟 v2_ddae_fsce 一致
GRAPH_METRIC = "euclidean"   # 在 log1p 上算，組成與總量都敏感
EDGE_BATCH = 256              # 每個 step 抽的正樣本邊數，負樣本抽等量
LAMBDA_FSCE = 0.01            # FSCE loss 的權重，warm-up 結束後的最終值
WARMUP_EPOCHS = 200           # lambda 從 0 線性升到 LAMBDA_FSCE 所花的 epoch 數

NOISE_P = 0.3
NOISE_MODE = "thinning"
NOISE_OD_P = 0.3         # OD 的 binomial thinning 強度，跟 count 同值

device = ("mps" if torch.backends.mps.is_available()
          else "cuda" if torch.cuda.is_available() else "cpu")


def evaluate(model, data, od, idx, log_lam_null):
    """驗證集上的一輪評估，全部用乾淨輸入、eval 模式。

    model：AE。data：Patches。od：ODGraph。
    idx：要評估的 patch 編號 LongTensor。log_lam_null：(1,N_CAT) 空模型的 log λ。

    count 跟 OD 都用乾淨的（不加噪），跟 v2_ddae_fsce 的驗證流程一致。

    回傳 (dev, expl, knock_dev)：Poisson deviance、explained deviance、
    切斷 GAT 分支後的 deviance。
    """
    dev, dev_null, dev_ko = [], [], []
    with torch.no_grad():
        for i in range(0, len(idx), BATCH):
            batch = idx[i:i + BATCH]
            x = data.agg(batch).to(device)
            a_local = od.A_local[batch].to(device)
            _, log_lam, _ = model(x, a_local)
            dev.append(poisson_deviance(log_lam, x))
            dev_null.append(poisson_deviance(log_lam_null.expand(len(batch), -1), x))
            _, log_lam_ko, _ = model(x, a_local, knockout=True)
            dev_ko.append(poisson_deviance(log_lam_ko, x))
    d = torch.cat(dev).mean().item()
    return d, 1 - d / torch.cat(dev_null).mean().item(), torch.cat(dev_ko).mean().item()


def run(data, od, train_idx, val_idx, edge_i, edge_j, edge_w, a, b, log):
    """訓練並回傳 (z, err)：z 是 (N,LATENT_DIM) 的 latent、err 是 (N,) 的
    Poisson deviance，兩者都用乾淨輸入、eval 模式算出來。

    data 是 Patches；od 是 ODGraph（整包留在 CPU，取 batch、加噪後才上 device）；
    train_idx / val_idx 是 patch 編號的 LongTensor；
    edge_i / edge_j / edge_w / a / b 是 build_fsce_graph() 的回傳值；
    log 是 config.train_log.open_log() 回傳的函式。
    訓練中按 Ctrl-C 會提前跳出，用當下最好的模型存 checkpoint 跟 latent。

    選模是「val_dev 最低的那一次評估」，不是最後一個 epoch：val_dev 在後段的
    epoch 間波動有 std≈0.015~0.023，跟我們在追的效果同一個量級，拿最後一個
    epoch 當代表等於在讀雜訊。代價是 val 同時被拿來選模，val_dev 因此是
    樂觀偏誤的估計——只要拿它跟別的版本比時雙方用同一套規則就還是公平的，
    但別把它當成乾淨的 held-out 數字。best/final 兩個都會印，要哪個都拿得到。
    """
    torch.manual_seed(SEED)
    model = AE(LATENT_DIM, ALPHA_INIT, ALPHA_LEARN).to(device)
    n_edges = len(edge_i)

    # GAT 分支自成一組、lr 乘 GAT_LR_MULT：讓它學得比 count 分支慢。
    # 注意一定要調 lr，不能用 hook 把梯度乘上係數——Adam 的更新量是
    # m/sqrt(v)，梯度整體縮放會在分子分母上抵銷掉，等於什麼都沒做。
    gat_p = list(model.gat.parameters())
    gat_ids = {id(p) for p in gat_p}
    # alpha 若可學習則不套 weight decay：套了的話它會被 L2 單調往 0 拉，
    # sigmoid 後 alpha 就會被推向 0.5，「模型自己選的比例」跟「被 weight decay
    # 壓下去」就分不出來
    free = [model.alpha_raw] if ALPHA_LEARN else []
    free_ids = {id(p) for p in free}
    groups = [
        {"params": [p for p in model.parameters()
                    if id(p) not in free_ids and id(p) not in gat_ids],
         "weight_decay": WEIGHT_DECAY, "lr": LR},
        {"params": gat_p, "weight_decay": WEIGHT_DECAY, "lr": LR * GAT_LR_MULT},
    ]
    if free:
        groups.append({"params": free, "weight_decay": 0.0, "lr": LR})
    opt = torch.optim.Adam(groups, lr=LR)

    log(f"參數量 {sum(p.numel() for p in model.parameters())}"
        f"（GAT 分支 {sum(p.numel() for p in gat_p)}，lr x{GAT_LR_MULT}）")
    log(f"alpha {'可學習' if ALPHA_LEARN else '固定'} = {model.alpha().item():.4f}")

    # 空模型（λ=訓練集全域平均）的 log_lam，當 explained deviance 的分母
    log_lam_null = data.agg(train_idx).mean(dim=0, keepdim=True) \
        .clamp_min(1e-8).log().to(device)

    best = {"dev": float("inf"), "epoch": 0, "expl": 0.0, "gain": 0.0, "alpha": 0.0}
    best_state = None
    last = None

    for epoch in range(EPOCHS):
        try:
            model.train()
            lam_t = LAMBDA_FSCE * min(1.0, (epoch + 1) / WARMUP_EPOCHS)
            g = torch.Generator().manual_seed(SEED + epoch)
            perm = train_idx[torch.randperm(len(train_idx), generator=g)]
            total, total_fsce = 0.0, 0.0
            for i in range(0, len(perm), BATCH):
                batch = perm[i:i + BATCH]
                x = data.agg(batch)                                  # 乾淨目標（CPU）
                x_in = corrupt(x, NOISE_P, NOISE_MODE, generator=g)  # 加噪輸入
                x, x_in = x.to(device), x_in.to(device)
                a_local = corrupt_od(od.A_local[batch], NOISE_OD_P,
                                     generator=g).to(device)          # OD 也加噪

                _, log_lam, _ = model(x_in, a_local)
                recon = poisson_nll(log_lam, x).mean()   # 目標永遠是乾淨的 x

                eg = torch.randint(0, n_edges, (EDGE_BATCH,), generator=g)
                pi, pj, pw = edge_i[eg], edge_j[eg], edge_w[eg]
                ni = torch.randint(0, data.n, (EDGE_BATCH,), generator=g)
                nj = torch.randint(0, data.n, (EDGE_BATCH,), generator=g)
                idx_i, idx_j = torch.cat([pi, ni]), torch.cat([pj, nj])
                # pair 的 count 跟 OD 都加噪，encoder 訓練時看到的輸入分布才跟
                # recon 那一路一致
                xi = corrupt(data.agg(idx_i), NOISE_P, NOISE_MODE, generator=g).to(device)
                xj = corrupt(data.agg(idx_j), NOISE_P, NOISE_MODE, generator=g).to(device)
                a_i = corrupt_od(od.A_local[idx_i], NOISE_OD_P, generator=g).to(device)
                a_j = corrupt_od(od.A_local[idx_j], NOISE_OD_P, generator=g).to(device)
                w = torch.cat([pw, torch.zeros(EDGE_BATCH)]).to(device)
                zi, _ = model.encode(xi, a_i)
                zj, _ = model.encode(xj, a_j)
                fsce = fsce_loss(zi, zj, w, a, b).mean()

                loss = recon + lam_t * fsce
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += recon.item() * len(batch)
                total_fsce += fsce.item() * len(batch)

            if (epoch + 1) % 5 == 0 or epoch == 0:
                model.eval()
                dev, expl, knock = evaluate(model, data, od, val_idx, log_lam_null)
                alpha = model.alpha().item()
                last = {"dev": dev, "epoch": epoch + 1, "expl": expl,
                        "gain": knock - dev, "alpha": alpha}
                star = ""
                if dev < best["dev"]:
                    best = dict(last)
                    # 搬到 CPU 再 clone，否則存的是 device 上會被下一步覆寫的張量
                    best_state = {k: v.detach().cpu().clone()
                                  for k, v in model.state_dict().items()}
                    star = " *"      # 標記放行尾：行首格式不能動，scratch/log 的
                                     # 兩支繪圖腳本是用 r"epoch\s+(\d+)\s*\|" 在抓
                log(f"epoch {epoch + 1:4d} | train_nll {total / len(perm):.5f} | "
                    f"train_fsce {total_fsce / len(perm):.5f} | val_dev {dev:.5f} | "
                    f"expl_dev {expl:.5f} | gat_gain {knock - dev:+.5f} | "
                    f"alpha {alpha:.4f}{star}")
        except KeyboardInterrupt:
            log(f"\n[中斷] epoch {epoch + 1} 收到 Ctrl-C，用目前最好的模型存 "
                f"checkpoint 跟 latent")
            break

    if best_state is None:
        raise RuntimeError("一次評估都沒跑完就中斷了，沒有可用的模型")

    log(f"\n[best ] epoch {best['epoch']:4d} | val_dev {best['dev']:.5f} | "
        f"expl_dev {best['expl']:.5f} | gat_gain {best['gain']:+.5f} | "
        f"alpha {best['alpha']:.4f}")
    log(f"[final] epoch {last['epoch']:4d} | val_dev {last['dev']:.5f} | "
        f"expl_dev {last['expl']:.5f} | gat_gain {last['gain']:+.5f} | "
        f"alpha {last['alpha']:.4f}")
    log(f"latent 與 checkpoint 都用 best（epoch {best['epoch']}）那份權重")

    model.load_state_dict(best_state)
    torch.save(best_state, CKPT)

    model.eval()
    zs, errs = [], []
    with torch.no_grad():
        for i in range(0, data.n, BATCH):
            idx = torch.arange(i, min(i + BATCH, data.n))
            x = data.agg(idx).to(device)   # 推論不加噪，count 跟 OD 都一樣
            z, log_lam, _ = model(x, od.A_local[idx].to(device))
            zs.append(z.cpu())
            errs.append(poisson_deviance(log_lam, x).cpu())
    return torch.cat(zs).numpy(), torch.cat(errs).numpy()


def main():
    log = open_log(VERSION, {
        "LATENT_DIM": LATENT_DIM, "EPOCHS": EPOCHS, "BATCH": BATCH, "LR": LR,
        "WEIGHT_DECAY": WEIGHT_DECAY, "VAL_FRAC": VAL_FRAC, "SEED": SEED,
        "ALPHA_INIT": ALPHA_INIT, "ALPHA_LEARN": ALPHA_LEARN,
        "GAT_LR_MULT": GAT_LR_MULT,
        # GAT 分支的形狀寫進 header：這幾個常數正在被拿來做減量的對照，
        # 不記下來的話幾份 result.log 分不出是哪個設定跑的
        "GAT_DIM": GAT_DIM, "GAT_HEADS": GAT_HEADS, "GAT_LAYERS": GAT_LAYERS,
        "GAT_READOUT_DIM": GAT_READOUT_DIM,
        "N_NEIGHBORS": N_NEIGHBORS, "GRAPH_METRIC": GRAPH_METRIC,
        "EDGE_BATCH": EDGE_BATCH, "LAMBDA_FSCE": LAMBDA_FSCE,
        "WARMUP_EPOCHS": WARMUP_EPOCHS,
        "NOISE_P": NOISE_P, "NOISE_MODE": NOISE_MODE,
        "NOISE_OD_P": NOISE_OD_P,
    })

    ensure_od()          # 內含 ensure_patches()
    data = Patches(PATCHES)
    od = ODGraph(OD)
    assert od.n == data.n, f"od.npz 有 {od.n} 個 patch，patches.npz 有 {data.n} 個"

    log(f"{data.n} 個 patch，device={device}，噪聲 {NOISE_MODE} p={NOISE_P}")
    log(f"OD：{(od.n_od > 0).float().mean().item() * 100:.1f}% 的 patch 有觀測，"
        f"總量中位數 {od.n_od.median().item():.0f}，"
        f"最大 {od.n_od.max().item():.0f}")

    # fuzzy graph 用乾淨 count 建：鄰接關係是資料的性質，不是噪聲的性質
    x_all = np.log1p(data.agg(torch.arange(data.n)).numpy())
    edge_i, edge_j, edge_w, a, b = build_fsce_graph(
        x_all, n_neighbors=N_NEIGHBORS, metric=GRAPH_METRIC)
    log(f"FSCE graph：{len(edge_i)} 條邊，a={a:.4f} b={b:.4f}")

    g = torch.Generator().manual_seed(SEED)
    perm = torch.randperm(data.n, generator=g)
    n_val = int(data.n * VAL_FRAC)
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    z, err = run(data, od, train_idx, val_idx, edge_i, edge_j, edge_w, a, b, log)
    np.savez(OUT, n_poi=data.n_poi, lat=data.lat, lon=data.lon, z=z, err=err)
    log(f"已存 {OUT}")


main()
