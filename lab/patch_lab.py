import os
import sys

import numpy as np
import torch
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, RadioButtons
from scipy.spatial import cKDTree

ROOT = os.path.abspath(f"{os.path.dirname(__file__)}/..")
sys.path.insert(0, ROOT)
from config.dataset import (CAT_COLORS, CAT_ZH, CELL, GRID, N_CAT,  # noqa: E402
                            PATCHES, HALF_WIDTH, result)

MODEL_VERSION = "v2_ddae_fsce_euc"
MODELS = {
    "v0": dict(latent_dim=2, log1p=True),
    "v0_poisson_nll": dict(latent_dim=2, log1p=False),
    "v0_l16": dict(latent_dim=16, log1p=True),
    "v0_l32_poisson_nll": dict(latent_dim=32, log1p=False),
    "v0_l32_nb": dict(latent_dim=32, log1p=False),
    # v1 的輸入/輸出介面不同（counts×mask、回傳 log_lam），要接再另外處理
    "v2_ae": dict(latent_dim=2, v2=True),
    "v2_l16_ae": dict(latent_dim=16, v2=True),
    "v2_vae": dict(latent_dim=2, v2=True, vae=True),
    "v2_perceiver": dict(latent_dim=2, v2=True, perceiver=True),
    "v2_ddae": dict(latent_dim=2, v2=True),
    "v2_ddae_fsce_euc": dict(latent_dim=2, v2=True),
    "v2_ddae_tanh_fsce": dict(latent_dim=2, v2=True),
}
cfg = MODELS[MODEL_VERSION]
PATCH_IDX = None     # None = 從 robust 距離最小（最典型）的 patch 起手；或填 patch 編號
BATCH_N = 50         # 「+N 隨機」一次灑幾顆
OUTLIER_PCT = 99.5   # 全體 robust 距離的離群門檻百分位
ZOOM_PCT = (1, 99)   # latent 圖初始視野
SEED = 0
DOT = 3.0

sys.path.insert(0, os.path.join(ROOT, "model", MODEL_VERSION))
if cfg.get("v2"):
    if cfg.get("perceiver"):
        from ae import PerceiverAE as ModelClass  # type: ignore # noqa: E402
    elif cfg.get("vae"):
        from ae import VAE as ModelClass  # type: ignore # noqa: E402
    else:
        from ae import MLPAE as ModelClass  # type: ignore # noqa: E402
else:
    from ae import ConvAE as ModelClass  # type: ignore # noqa: E402

mpl.rcParams["font.family"] = ["Heiti TC"]
mpl.rcParams["axes.unicode_minus"] = False


def robust_distance(z, ref):
    """用全體 latent 的 中位數 / MAD 標準化後，算 z 到 latent 中心的距離。"""
    med = np.median(ref, axis=0)
    mad = np.median(np.abs(ref - med), axis=0) * 1.4826
    return np.linalg.norm((z - med) / mad, axis=-1)


def render(dx, dy, cat):
    """把一組點列表 binning 成 (1,N_CAT,40,40) 的輸入矩陣。"""
    ix = np.clip(np.floor(dx / CELL + GRID / 2), 0, GRID - 1).astype(np.int64)
    iy = np.clip(np.floor(dy / CELL + GRID / 2), 0, GRID - 1).astype(np.int64)
    flat = (cat.astype(np.int64) * GRID + iy) * GRID + ix
    counts = np.bincount(flat, minlength=N_CAT * GRID * GRID)
    counts = counts.reshape(1, N_CAT, GRID, GRID).astype(np.float32)
    if cfg.get("log1p", True):
        counts = np.log1p(counts)
    return torch.from_numpy(counts)


def sample_disk(rng, k):
    """在半徑 HALF_WIDTH 的圓內均勻灑 k 個點。"""
    r = HALF_WIDTH * np.sqrt(rng.random(k))
    t = rng.random(k) * 2 * np.pi
    return r * np.cos(t), r * np.sin(t)


def patch_points(p, i):
    """回傳第 i 個 patch 的 (dx, dy, cat)。"""
    s, e = p["offsets"][i], p["offsets"][i + 1]
    return p["dx"][s:e], p["dy"][s:e], p["cat"][s:e]


def fit_to2d(z_all, latent_dim):
    """latent 超過 2 維時，用全體的 PCA 前 2 主成分作圖（距離仍用完整維度）。"""
    if latent_dim == 2:
        return lambda z: z
    mean = z_all.mean(axis=0)
    _, _, vt = np.linalg.svd(z_all - mean, full_matrices=False)
    return lambda z: (z - mean) @ vt[:2].T


# ---- 載入資料與模型 ----

cfg = MODELS[MODEL_VERSION]
p = np.load(PATCHES)
d = np.load(result(MODEL_VERSION, "latents.npz"))
z_all, n_poi = d["z"], d["n_poi"]
assert len(z_all) == len(p["n_poi"]), (
    f"latents({len(z_all)}) 與 patches({len(p['n_poi'])}) 的 patch 數不一致，"
    f"請先重跑 model/{MODEL_VERSION}/train.py")

model = ModelClass(cfg["latent_dim"])
model.load_state_dict(torch.load(result(MODEL_VERSION, "ae.pt"),
                                 map_location="cpu"))
model.eval()

dist_all = robust_distance(z_all, z_all)
thr = np.percentile(dist_all, OUTLIER_PCT)
to2d = fit_to2d(z_all, cfg["latent_dim"])
z2_all = to2d(z_all)

# 點擊找最近 patch：兩軸各自除以視野跨度，讓「最近」跟眼睛看到的一致
lo = np.percentile(z2_all, ZOOM_PCT[0], axis=0)
hi = np.percentile(z2_all, ZOOM_PCT[1], axis=0)
scale = np.where(hi - lo > 0, hi - lo, 1.0)
tree = cKDTree(z2_all / scale)

# ---- UI 佈局 ----

fig = plt.figure(figsize=(14, 7))
ax_lat = fig.add_axes([0.06, 0.08, 0.40, 0.80])
ax_map = fig.add_axes([0.53, 0.08, 0.29, 0.80])
ax_radio = fig.add_axes([0.86, 0.32, 0.12, 0.54])
ax_batch = fig.add_axes([0.86, 0.23, 0.12, 0.05])
ax_undo = fig.add_axes([0.86, 0.16, 0.12, 0.05])
ax_reset = fig.add_axes([0.86, 0.09, 0.12, 0.05])
status = fig.text(0.06, 0.94, "", fontsize=10)

axis2 = "z" if cfg["latent_dim"] == 2 else "PC"
ax_lat.scatter(z2_all[:, 0], z2_all[:, 1], s=DOT, c="#b8bcc4",
               linewidths=0, alpha=0.4, rasterized=True)
pad = (hi - lo) * 0.05
ax_lat.set_xlim(lo[0] - pad[0], hi[0] + pad[0])
ax_lat.set_ylim(lo[1] - pad[1], hi[1] + pad[1])
ax_lat.set_title(f"latent space（{MODEL_VERSION}，點一下換 patch）", fontsize=10)
ax_lat.set_xlabel(f"{axis2}1", fontsize=8)
ax_lat.set_ylabel(f"{axis2}2", fontsize=8)
ax_lat.tick_params(labelsize=7)
ax_lat.grid(alpha=0.15, linewidth=0.5)

traj_line, = ax_lat.plot([], [], c="#c0392b", lw=1.5, zorder=3)
star = ax_lat.scatter([], [], s=130, marker="*", c="#1a1a1a", zorder=4)
cur = ax_lat.scatter([], [], s=60, c="#c0392b", edgecolors="k",
                     linewidths=0.6, zorder=5)

state = dict(idx=-1, base=None, cat=0,
             added=[],    # list of (dx_arr, dy_arr, cat_id)，一格 = 一個 undo 批次
             traj=[],     # 完整維度 latent；len(traj) == len(added) + 1
             artists=[])  # 與 added 平行的 ax_map scatter handle
rng = np.random.default_rng(SEED)


def encode_current():
    dx0, dy0, cat0 = state["base"]
    dx = np.concatenate([dx0] + [b[0] for b in state["added"]])
    dy = np.concatenate([dy0] + [b[1] for b in state["added"]])
    cat = np.concatenate([cat0] + [np.full(len(b[0]), b[2], dtype=cat0.dtype)
                                   for b in state["added"]])
    with torch.no_grad():
        if cfg.get("v2"):
            if cfg.get("perceiver"):
                T = len(cat)
                tok = torch.from_numpy(cat).unsqueeze(0).long()
                pad_mask = torch.zeros(1, T, dtype=torch.bool)
                if T == 0:
                    tok = torch.zeros(1, 1, dtype=torch.long)
                    pad_mask = torch.ones(1, 1, dtype=torch.bool)
                z, _ = model(tok, pad_mask)
            else:
                counts = np.bincount(cat, minlength=N_CAT)
                x = torch.from_numpy(counts).unsqueeze(0).float()
                out = model(x)
                z = out[0] if not cfg.get("vae") else out[2]
        else:
            z, _ = model(render(dx, dy, cat))
    return z[0].numpy()


def update():
    t2 = to2d(np.array(state["traj"]))
    traj_line.set_data(t2[:, 0], t2[:, 1])
    star.set_offsets(t2[:1])
    cur.set_offsets(t2[-1:])
    dist = robust_distance(state["traj"][-1], z_all)
    out = dist > thr
    n_added = sum(len(b[0]) for b in state["added"])
    status.set_text(f"patch {state['idx']}｜原始 POI {n_poi[state['idx']]}｜"
                    f"已加 {n_added} 顆｜robust 距離 {dist:.2f} / "
                    f"門檻 {thr:.2f}｜" + ("★ 離群" if out else "正常"))
    status.set_color("#c0392b" if out else "#1a1a1a")
    fig.canvas.draw_idle()


def draw_base_map():
    ax_map.clear()
    dx0, dy0, cat0 = state["base"]
    for k in range(N_CAT):
        m = cat0 == k
        if m.any():
            ax_map.scatter(dx0[m], dy0[m], s=16, c=CAT_COLORS[k],
                           linewidths=0, alpha=0.85)
    ax_map.set_xlim(-HALF_WIDTH * 1.1, HALF_WIDTH * 1.1)
    ax_map.set_ylim(-HALF_WIDTH * 1.1, HALF_WIDTH * 1.1)
    ax_map.set_aspect("equal")
    ax_map.set_title(f"patch {state['idx']} 的 POI（點一下加一顆）", fontsize=10)
    ax_map.set_xlabel("東西向位移 (m)", fontsize=8)
    ax_map.set_ylabel("南北向位移 (m)", fontsize=8)
    ax_map.tick_params(labelsize=7)
    ax_map.grid(alpha=0.15, linewidth=0.5)


def set_patch(i):
    state["idx"] = int(i)
    state["base"] = patch_points(p, state["idx"])
    state["added"], state["artists"] = [], []
    draw_base_map()
    state["traj"] = [encode_current()]
    update()


def add_batch(dx, dy):
    cat_id = state["cat"]
    state["added"].append((dx, dy, cat_id))
    state["traj"].append(encode_current())
    art = ax_map.scatter(dx, dy, s=34, marker="x", c=CAT_COLORS[cat_id],
                         linewidths=1.4, zorder=3)
    state["artists"].append(art)
    update()


def on_click(event):
    if getattr(fig.canvas.toolbar, "mode", "") != "":
        return  # pan/zoom 模式中，拖曳不觸發
    if event.inaxes is ax_lat:
        _, i = tree.query(np.array([event.xdata, event.ydata]) / scale)
        set_patch(i)
    elif event.inaxes is ax_map:
        add_batch(np.array([event.xdata], dtype=np.float32),
                  np.array([event.ydata], dtype=np.float32))


def on_radio(label):
    state["cat"] = CAT_ZH.index(label)


def on_batch(event):
    bx, by = sample_disk(rng, BATCH_N)
    add_batch(bx.astype(np.float32), by.astype(np.float32))


def on_undo(event):
    if not state["added"]:
        return
    state["added"].pop()
    state["traj"].pop()
    state["artists"].pop().remove()
    update()


def on_reset(event):
    for a in state["artists"]:
        a.remove()
    state["added"], state["artists"] = [], []
    state["traj"] = state["traj"][:1]
    update()


ax_radio.set_title("要加的類別", fontsize=9)
radio = RadioButtons(ax_radio, CAT_ZH)
for i, t in enumerate(radio.labels):
    t.set_color(CAT_COLORS[i])
    t.set_fontsize(9)
btn_batch = Button(ax_batch, f"+{BATCH_N} 隨機")
btn_undo = Button(ax_undo, "undo")
btn_reset = Button(ax_reset, "reset")
radio.on_clicked(on_radio)
btn_batch.on_clicked(on_batch)
btn_undo.on_clicked(on_undo)
btn_reset.on_clicked(on_reset)
fig.canvas.mpl_connect("button_press_event", on_click)

set_patch(int(np.argmin(dist_all)) if PATCH_IDX is None else PATCH_IDX)

z0 = state["traj"][0]
if not np.allclose(z0, z_all[state["idx"]], atol=1e-2):
    print(f"警告：現算的 latent {z0} 與 latents.npz 的 {z_all[state['idx']]} "
          f"不符，checkpoint 與 patches 可能不一致")
print(f"patch {state['idx']}：POI {n_poi[state['idx']]}，"
      f"robust 距離 {dist_all[state['idx']]:.2f}，離群門檻 {thr:.2f}")

plt.show()
