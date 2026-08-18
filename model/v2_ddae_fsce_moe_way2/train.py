
"""訓練 v2_ddae_fsce_moe：v2_ddae_fsce 的公平 ablation 版本。

唯一實驗變因：加入 residual decoder MoE（2 experts, scale=0.1）。
其餘所有超參數、模型主幹、loss、訓練流程與 v2_ddae_fsce 完全相同。
"""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(f"{os.path.dirname(__file__)}/../.."))
from ae import (MoEMLPAE, Patches, build_fsce_graph, corrupt,  # noqa: E402
                fsce_loss, poisson_deviance, poisson_nll, moe_balance_loss,
                N_EXPERTS, ROUTER_TEMPERATURE, MOE_SCALE)
from config.dataset import ensure_patches, PATCHES, result  # noqa: E402
from config.train_log import open_log  # noqa: E402

VERSION = "v3_ddae_fsce_moe_way2"
OUT = result(VERSION, "latents.npz")
CKPT = result(VERSION, "ae.pt")

LATENT_DIM = 2
EPOCHS = 1000
BATCH = 256
LR_BASE = 1e-3
LR_MOE = 5e-3
WEIGHT_DECAY = 1e-6
VAL_FRAC = 0.1
SEED = 0

N_NEIGHBORS = 15         # 建高維 fuzzy graph 的 kNN 數，跟 data/patch/umap_grid.py 一致
GRAPH_METRIC = "euclidean"  # 在 log1p 上算，組成與總量都敏感——跟 Poisson NLL 的要求一致
EDGE_BATCH = 256          # 每個 step 抽的正樣本邊數，負樣本抽等量
LAMBDA_FSCE = 0.01       # FSCE loss 的權重，warm-up 結束後的最終值
WARMUP_EPOCHS = 200        # lambda 從 0 線性升到 LAMBDA_FSCE 所花的 epoch 數
LAMBDA_BALANCE = 0.05    # MoE balance loss 的權重

NOISE_P = 0.3            # 破壞強度：thinning 是每個 POI 被丟掉的機率
NOISE_MODE = "thinning"  # "thinning"（逐 POI 丟）或 "mask"（整類歸零）

device = ("mps" if torch.backends.mps.is_available()
          else "cuda" if torch.cuda.is_available() else "cpu")


def run(data, train_idx, val_idx, edge_i, edge_j, edge_w, a, b, log):
    """訓練並回傳全體 patch 的 (z, err, gates, expert_id)：z 是 (N,LATENT_DIM)
    的 latent 座標，err 是 (N,) 的 Poisson deviance，gates 是 (N, N_EXPERTS)
    的 router 權重，expert_id 是 (N,) 的最大 gate 編號。
    全部都用乾淨輸入、eval 模式算出來。
    data 是 Patches；train_idx / val_idx 是 patch 編號的 LongTensor；
    edge_i / edge_j / edge_w / a / b 是 build_fsce_graph() 的回傳值。
    log 是 config.train_log.open_log() 回傳的函式，訓練過程的訊息都灌進去。
    訓練中按 Ctrl-C 會提前跳出迴圈，用當下的模型狀態存 checkpoint 跟 latent，
    不會整個丟掉重來。
    """
    torch.manual_seed(SEED)
    model = MoEMLPAE(LATENT_DIM).to(device)
    opt = torch.optim.Adam([
        {"params": model.encoder.parameters(), "lr": LR_BASE},
        {"params": model.decoder.parameters(), "lr": LR_BASE},
        {"params": model.router.parameters(), "lr": LR_MOE},
        {"params": model.residual_experts.parameters(), "lr": LR_MOE},
    ], weight_decay=WEIGHT_DECAY)
    n_edges = len(edge_i)

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
            total, total_fsce, total_balance = 0.0, 0.0, 0.0
            gate_sum = torch.zeros(N_EXPERTS, device=device)
            gate_count = 0
            for i in range(0, len(perm), BATCH):
                batch = perm[i:i + BATCH]
                x = data.agg(batch)                                  # 乾淨目標（CPU）
                x_in = corrupt(x, NOISE_P, NOISE_MODE, generator=g)  # 加噪輸入
                x, x_in = x.to(device), x_in.to(device)
                _, log_lam, gates = model.forward_with_gates(x_in)
                recon = poisson_nll(log_lam, x).mean()   # 目標永遠是乾淨的 x

                eg = torch.randint(0, n_edges, (EDGE_BATCH,), generator=g)
                pi, pj, pw = edge_i[eg], edge_j[eg], edge_w[eg]
                ni = torch.randint(0, data.n, (EDGE_BATCH,), generator=g)
                nj = torch.randint(0, data.n, (EDGE_BATCH,), generator=g)
                # pair 也加噪，encoder 訓練時看到的輸入分布才跟 recon 那一路一致
                xi = corrupt(data.agg(torch.cat([pi, ni])), NOISE_P,
                             NOISE_MODE, generator=g).to(device)
                xj = corrupt(data.agg(torch.cat([pj, nj])), NOISE_P,
                             NOISE_MODE, generator=g).to(device)
                w = torch.cat([pw, torch.zeros(EDGE_BATCH)]).to(device)
                zi, zj = model.encode(xi), model.encode(xj)
                fsce = fsce_loss(zi, zj, w, a, b).mean()
                balance = moe_balance_loss(gates)

                loss = recon + lam_t * fsce + LAMBDA_BALANCE * balance
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += recon.item() * len(batch)
                total_fsce += fsce.item() * len(batch)
                total_balance += balance.item() * len(batch)
                gate_sum += gates.detach().sum(dim=0)
                gate_count += len(batch)

            if (epoch + 1) % 5 == 0 or epoch == 0:
                model.eval()
                with torch.no_grad():
                    vd, vdn = [], []
                    for i in range(0, len(val_idx), BATCH):
                        batch = val_idx[i:i + BATCH]
                        x = data.agg(batch).to(device)   # 驗證不加噪
                        _, log_lam = model(x)
                        vd.append(poisson_deviance(log_lam, x))
                        vdn.append(poisson_deviance(log_lam_null.expand(len(batch), -1), x))
                    dev = torch.cat(vd).mean().item()
                    dev_null = torch.cat(vdn).mean().item()
                    expl = 1 - dev / dev_null
                usage = ",".join(
                    f"{v:.3f}" for v in (gate_sum / gate_count).tolist()
                )
                log(f"epoch {epoch + 1:4d} | train_nll {total / len(perm):.5f} | "
                    f"train_fsce {total_fsce / len(perm):.5f} | "
                    f"moe_balance {total_balance / len(perm):.5f} | "
                    f"gate_usage [{usage}] | val_dev {dev:.5f} | "
                    f"expl_dev {expl:.5f}")
        except KeyboardInterrupt:
            log(f"\n[中斷] epoch {epoch + 1} 收到 Ctrl-C，用目前模型存 checkpoint 跟 latent")
            break

    torch.save(model.state_dict(), CKPT)

    model.eval()
    zs, errs, all_gates = [], [], []
    with torch.no_grad():
        for i in range(0, data.n, BATCH):
            idx = torch.arange(i, min(i + BATCH, data.n))
            x = data.agg(idx).to(device)   # 推論不加噪
            z, log_lam, gates = model.forward_with_gates(x)
            zs.append(z.cpu())
            errs.append(poisson_deviance(log_lam, x).cpu())
            all_gates.append(gates.cpu())
    z = torch.cat(zs).numpy()
    err = torch.cat(errs).numpy()
    gates = torch.cat(all_gates).numpy()
    expert_id = gates.argmax(axis=1)
    return z, err, gates, expert_id


def main():
    log = open_log(VERSION, {
        "LATENT_DIM": LATENT_DIM, "EPOCHS": EPOCHS, "BATCH": BATCH,
        "LR_BASE": LR_BASE, "LR_MOE": LR_MOE,
        "WEIGHT_DECAY": WEIGHT_DECAY, "VAL_FRAC": VAL_FRAC, "SEED": SEED,
        "N_NEIGHBORS": N_NEIGHBORS, "GRAPH_METRIC": GRAPH_METRIC,
        "EDGE_BATCH": EDGE_BATCH, "LAMBDA_FSCE": LAMBDA_FSCE,
        "WARMUP_EPOCHS": WARMUP_EPOCHS, "LAMBDA_BALANCE": LAMBDA_BALANCE,
        "NOISE_P": NOISE_P, "NOISE_MODE": NOISE_MODE,
        "N_EXPERTS": N_EXPERTS, "ROUTER_TEMPERATURE": ROUTER_TEMPERATURE,
        "MOE_SCALE": MOE_SCALE,
    })

    ensure_patches()
    data = Patches(PATCHES)
    log(f"{data.n} 個 patch，device={device}，"
        f"噪聲 {NOISE_MODE} p={NOISE_P}")

    # fuzzy graph 用乾淨 count 建：鄰接關係是資料的性質，不是噪聲的性質
    x_all = np.log1p(data.agg(torch.arange(data.n)).numpy())
    edge_i, edge_j, edge_w, a, b = build_fsce_graph(
        x_all, n_neighbors=N_NEIGHBORS, metric=GRAPH_METRIC)
    log(f"FSCE graph：{len(edge_i)} 條邊，a={a:.4f} b={b:.4f}")

    g = torch.Generator().manual_seed(SEED)
    perm = torch.randperm(data.n, generator=g)
    n_val = int(data.n * VAL_FRAC)
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    z, err, gates, expert_id = run(data, train_idx, val_idx,
                                   edge_i, edge_j, edge_w, a, b, log)
    np.savez(OUT, n_poi=data.n_poi, lat=data.lat, lon=data.lon,
             z=z, err=err, gates=gates, expert_id=expert_id)
    log(f"已存 {OUT}")


main()
