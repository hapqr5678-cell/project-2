
"""訓練 v5_refine：v2_ddae_fsce 的 encoder 從一次前饋改成「前饋 + T 次帶
重建殘差回饋的迭代精修」，其餘一切逐字對齊。

單一變因：decoder、corrupt、FSCE、LATENT_DIM、EPOCHS、BATCH、LR、
WEIGHT_DECAY、VAL_FRAC、SEED、N_NEIGHBORS、GRAPH_METRIC、EDGE_BATCH、
LAMBDA_FSCE、WARMUP_EPOCHS、NOISE_P/NOISE_MODE 全部跟 v2_ddae_fsce 相同，
差別只有 REFINE_STEPS（見 ae.py）多出來的那個模組。REFINE_STEPS=0 時
連 RefineNet 都不會建，是同一份程式碼下的嚴格對照組。

為什麼加這個模組、為什麼它不是「繞過瓶頸」，見 ae.py 檔頭的實測數字。

encoder 吃加噪輸入（denoising 的正則化留在這一路），精修模組解釋的則是
乾淨觀測——它在推論時面對的觀測就是乾淨的，訓練時必須面對同一種東西。
這是本版與 v2_ddae_fsce 唯一需要明講的設計差異，兩條路的實測對照見
ae.py 檔頭。

deep supervision：loss 對軌跡上每一步的 log λ 都算一次 NLL 再平均，而不是
只罰最後一步。理由有二——(1) 中間步驟拿得到梯度，T 大時才不會只有最後
一步在學；(2) 「val_dev 隨精修步數單調下降」這件事本身是這一版要證明的
東西，deep supervision 讓每一步都是可讀的重建結果，不是隨便的中間狀態。

FSCE 掛在最終的 z（trajectory 的最後一步）：圖上要畫的、要被結構 loss
約束的都是它。pair 那一路因此也要跑完整條精修，訓練成本大約是 v2 的
(1+T) 倍。

驗證與最後輸出 latent 的推論階段一律不加噪。破壞是每個 step 重新抽的
（generator 綁 SEED+epoch，可重現）。
"""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(f"{os.path.dirname(__file__)}/../.."))
from ae import (MLPAE, REFINE_ETA_INIT, REFINE_GATE_INIT,  # noqa: E402
                REFINE_HIDDEN, REFINE_STEPS, Patches, build_fsce_graph,
                corrupt, fsce_loss, poisson_deviance, poisson_nll)
from config.dataset import ensure_patches, PATCHES, result  # noqa: E402
from config.train_log import open_log  # noqa: E402

VERSION = "v5_refine"
OUT = result(VERSION, "latents.npz")
CKPT = result(VERSION, "ae.pt")

LATENT_DIM = 2
EPOCHS = 1500
BATCH = 256
LR = 1e-3
WEIGHT_DECAY = 1e-6
VAL_FRAC = 0.1
SEED = 0

N_NEIGHBORS = 15         # 建高維 fuzzy graph 的 kNN 數，跟 data/patch/umap_grid.py 一致
GRAPH_METRIC = "euclidean"  # 在 log1p 上算，組成與總量都敏感——跟 v2_ddae_fsce 對齊
EDGE_BATCH = 256          # 每個 step 抽的正樣本邊數，負樣本抽等量
LAMBDA_FSCE = 0.01        # FSCE loss 的權重，warm-up 結束後的最終值
WARMUP_EPOCHS = 200        # lambda 從 0 線性升到 LAMBDA_FSCE 所花的 epoch 數

NOISE_P = 0.3            # 破壞強度：thinning 是每個 POI 被丟掉的機率
NOISE_MODE = "thinning"  # "thinning"（逐 POI 丟）或 "mask"（整類歸零）

# 精修模組在訓練時要解釋哪一份觀測。"clean" 是本版的設計（推論時它面對的
# 觀測就是乾淨的，訓練時必須面對同一種東西）；"noisy" 是對照組，讓它跟
# encoder 一樣只看得到加噪輸入。encoder 那一路不受這個開關影響，永遠吃加噪。
REFINE_OBS = "clean"

device = ("mps" if torch.backends.mps.is_available()
          else "cuda" if torch.cuda.is_available() else "cpu")


def run(data, train_idx, val_idx, edge_i, edge_j, edge_w, a, b, log):
    """訓練並回傳全體 patch 的 (z, err)：z 是 (N,LATENT_DIM) 的最終 latent
    （精修完的那一個），err 是 (N,) 的 Poisson deviance，兩者都用乾淨輸入、
    eval 模式算出來，欄位語意跟 v2_ddae_fsce 完全一致。
    data 是 Patches；train_idx / val_idx 是 patch 編號的 LongTensor；
    edge_i / edge_j / edge_w / a / b 是 build_fsce_graph() 的回傳值。
    log 是 config.train_log.open_log() 回傳的函式，訓練過程的訊息都灌進去。
    訓練中按 Ctrl-C 會提前跳出迴圈，用當下的模型狀態存 checkpoint 跟 latent。
    """
    torch.manual_seed(SEED)
    model = MLPAE(LATENT_DIM, REFINE_STEPS).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    n_edges = len(edge_i)

    n_enc = sum(p.numel() for p in model.encoder.parameters())
    n_dec = sum(p.numel() for p in model.decoder.parameters())
    n_ref = sum(p.numel() for p in model.refine.parameters()) if model.refine else 0
    n_obs = len(train_idx) * data.agg(train_idx[:1]).shape[1]
    log(f"參數量 encoder {n_enc} + decoder {n_dec} + refine {n_ref} = "
        f"{n_enc + n_dec + n_ref}；訓練觀測值 {n_obs}（每觀測值 "
        f"{(n_enc + n_dec + n_ref) / n_obs:.2f} 個參數，refine 佔 "
        f"{n_ref / (n_enc + n_dec) * 100:.1f}%）")

    # 空模型（λ=訓練集全域平均）的 log_lam，當 explained deviance 的分母，
    # 用訓練集估、驗證集算 deviance，不然分母本身就偷看了驗證集
    log_lam_null = data.agg(train_idx).mean(dim=0, keepdim=True) \
        .clamp_min(1e-8).log().to(device)

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
                # encoder 吃加噪、精修解釋乾淨觀測——推論時它面對的就是乾淨的，
                # 訓練時餵加噪會讓它學成縮水的步長（見 ae.py 檔頭的 ablation）
                obs = x if REFINE_OBS == "clean" else x_in
                _, _, log_lams = model(x_in, obs)
                # deep supervision：軌跡上每一步都罰，目標永遠是乾淨的 x
                recon = torch.stack([poisson_nll(ll, x).mean()
                                     for ll in log_lams]).mean()

                eg = torch.randint(0, n_edges, (EDGE_BATCH,), generator=g)
                pi, pj, pw = edge_i[eg], edge_j[eg], edge_w[eg]
                ni = torch.randint(0, data.n, (EDGE_BATCH,), generator=g)
                nj = torch.randint(0, data.n, (EDGE_BATCH,), generator=g)
                # pair 也加噪，encoder 訓練時看到的輸入分布才跟 recon 那一路一致；
                # 精修那一路同樣拿乾淨觀測，規則跟上面完全一致
                ci = data.agg(torch.cat([pi, ni]))
                cj = data.agg(torch.cat([pj, nj]))
                xi = corrupt(ci, NOISE_P, NOISE_MODE, generator=g).to(device)
                xj = corrupt(cj, NOISE_P, NOISE_MODE, generator=g).to(device)
                ci, cj = ci.to(device), cj.to(device)
                w = torch.cat([pw, torch.zeros(EDGE_BATCH)]).to(device)
                # encode() 會跑完整條精修，FSCE 約束的是最終的 z
                oi, oj = (ci, cj) if REFINE_OBS == "clean" else (xi, xj)
                zi, zj = model.encode(xi, oi), model.encode(xj, oj)
                fsce = fsce_loss(zi, zj, w, a, b).mean()

                loss = recon + lam_t * fsce
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += recon.item() * len(batch)
                total_fsce += fsce.item() * len(batch)

            if (epoch + 1) % 5 == 0 or epoch == 0:
                model.eval()
                with torch.no_grad():
                    steps, vdn = [], []
                    for i in range(0, len(val_idx), BATCH):
                        batch = val_idx[i:i + BATCH]
                        x = data.agg(batch).to(device)   # 驗證不加噪
                        _, _, log_lams = model(x)
                        steps.append(torch.stack(
                            [poisson_deviance(ll, x) for ll in log_lams]))
                        vdn.append(poisson_deviance(
                            log_lam_null.expand(len(batch), -1), x))
                    per_step = torch.cat(steps, dim=1).mean(dim=1)   # (T+1,)
                    dev = per_step[-1].item()
                    dev_null = torch.cat(vdn).mean().item()
                    expl = 1 - dev / dev_null
                    # 訓練集也量一次：amortization gap 是否也在訓練集上被關掉，
                    # 是判斷這個模組有沒有變成記憶體的關鍵（見 ae.py 檔頭）
                    td = []
                    for i in range(0, len(train_idx), BATCH):
                        batch = train_idx[i:i + BATCH]
                        x = data.agg(batch).to(device)
                        _, log_lam, _ = model(x)
                        td.append(poisson_deviance(log_lam, x))
                    tdev = torch.cat(td).mean().item()
                log(f"epoch {epoch + 1:4d} | train_nll {total / len(perm):.5f} | "
                    f"train_fsce {total_fsce / len(perm):.5f} | val_dev {dev:.5f} | "
                    f"expl_dev {expl:.5f} | train_dev {tdev:.5f} | "
                    f"gap {dev - tdev:+.5f} | "
                    f"step " + "→".join(f"{v:.4f}" for v in per_step.tolist()))
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
            z, log_lam, _ = model(x)
            zs.append(z.cpu())
            errs.append(poisson_deviance(log_lam, x).cpu())
    return torch.cat(zs).numpy(), torch.cat(errs).numpy()


def main():
    log = open_log(VERSION, {
        "LATENT_DIM": LATENT_DIM, "EPOCHS": EPOCHS, "BATCH": BATCH, "LR": LR,
        "WEIGHT_DECAY": WEIGHT_DECAY, "VAL_FRAC": VAL_FRAC, "SEED": SEED,
        "N_NEIGHBORS": N_NEIGHBORS, "GRAPH_METRIC": GRAPH_METRIC,
        "EDGE_BATCH": EDGE_BATCH, "LAMBDA_FSCE": LAMBDA_FSCE,
        "WARMUP_EPOCHS": WARMUP_EPOCHS,
        "NOISE_P": NOISE_P, "NOISE_MODE": NOISE_MODE,
        "REFINE_OBS": REFINE_OBS, "REFINE_STEPS": REFINE_STEPS, "REFINE_HIDDEN": REFINE_HIDDEN,
        "REFINE_GATE_INIT": REFINE_GATE_INIT,
        "REFINE_ETA_INIT": REFINE_ETA_INIT,
    })

    ensure_patches()
    data = Patches(PATCHES)
    log(f"{data.n} 個 patch，device={device}，"
        f"噪聲 {NOISE_MODE} p={NOISE_P}，精修 T={REFINE_STEPS}")

    # fuzzy graph 用乾淨 count 建：鄰接關係是資料的性質，不是噪聲的性質
    x_all = np.log1p(data.agg(torch.arange(data.n)).numpy())
    edge_i, edge_j, edge_w, a, b = build_fsce_graph(
        x_all, n_neighbors=N_NEIGHBORS, metric=GRAPH_METRIC)
    log(f"FSCE graph：{len(edge_i)} 條邊，a={a:.4f} b={b:.4f}")

    g = torch.Generator().manual_seed(SEED)
    perm = torch.randperm(data.n, generator=g)
    n_val = int(data.n * VAL_FRAC)
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    z, err = run(data, train_idx, val_idx, edge_i, edge_j, edge_w, a, b, log)
    np.savez(OUT, n_poi=data.n_poi, lat=data.lat, lon=data.lon, z=z, err=err)
    log(f"已存 {OUT}")


main()
