"""v2_ae / v2_perceiver / v2_vae 及各自的 tanh 對照組（v2_tanh_ae /
v2_tanh_perceiver / v2_tanh_vae）共六個版本的 deviance-based pseudo R²。

R² = 1 - deviance_model / deviance_null

  deviance_model：用各自模型重建出的 λ 算的 Poisson deviance（跟
                  poisson_deviance() 定義一致）。
  deviance_null ：「空模型」——不管是哪個 patch，每個類別都只用
                  全體 1233 個 patch 在該類別的平均值當 λ（也就是完全
                  不看這個 patch 的組成，只知道「這個城市平均而言每類
                  大概幾個」）。三個模型共用同一個 null，因為 null 只
                  跟資料有關、跟架構無關。

R² 的意義跟迴歸的 R² 一樣：0 = 模型不比「只報全體平均」更準，
1 = 完美重建；如果是負的，代表模型比空模型還爛。
"""

import importlib.util
import os
import sys

import numpy as np
import torch
import matplotlib as mpl
import matplotlib.pyplot as plt

ROOT = os.path.abspath(f"{os.path.dirname(__file__)}/../../..")
sys.path.insert(0, ROOT)
from config.dataset import PATCHES, result  # noqa: E402

OUT = result("v2_ae", "r2_deviance.png")

mpl.rcParams["font.family"] = ["Heiti TC"]
mpl.rcParams["axes.unicode_minus"] = False
mpl.rcParams["figure.dpi"] = 130


def _load_module(version):
    path = os.path.join(ROOT, "model", version, "ae.py")
    spec = importlib.util.spec_from_file_location(f"{version}_ae", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def deviance(x, lam):
    """逐 cell 的 Poisson deviance，(N_patch, N_CAT)。"""
    log_lam = torch.log(lam.clamp_min(1e-8))
    return 2 * (torch.xlogy(x, x) - x * log_lam - x + lam)


def main():
    ae_mod = _load_module("v2_ae")
    data = ae_mod.Patches(PATCHES)
    idx = torch.arange(data.n)
    x = data.agg(idx)

    lam_null = x.mean(dim=0, keepdim=True).expand_as(x)  # 每個類別的全體平均，跟 patch 無關
    dev_null_total = deviance(x, lam_null).sum().item()

    versions = {}

    m = ae_mod.MLPAE(2)
    m.load_state_dict(torch.load(result("v2_ae", "ae.pt"), map_location="cpu"))
    m.eval()
    with torch.no_grad():
        _, log_lam = m(x)
    versions["v2_ae"] = torch.exp(log_lam)

    perc_mod = _load_module("v2_perceiver")
    tok, pad_mask = perc_mod.Patches(PATCHES).tokens(idx)
    m = perc_mod.PerceiverAE(2)
    m.load_state_dict(torch.load(result("v2_perceiver", "ae.pt"), map_location="cpu"))
    m.eval()
    with torch.no_grad():
        _, log_lam = m(tok, pad_mask)
    versions["v2_perceiver"] = torch.exp(log_lam)

    vae_mod = _load_module("v2_vae")
    m = vae_mod.VAE(2)
    m.load_state_dict(torch.load(result("v2_vae", "ae.pt"), map_location="cpu"))
    m.eval()
    with torch.no_grad():
        _, _, _, log_lam = m(x)
    versions["v2_vae"] = torch.exp(log_lam)

    tanh_mod = _load_module("v2_tanh_ae")
    m = tanh_mod.MLPAE(2)
    m.load_state_dict(torch.load(result("v2_tanh_ae", "ae.pt"), map_location="cpu"))
    m.eval()
    with torch.no_grad():
        _, log_lam = m(x)
    versions["v2_tanh_ae"] = torch.exp(log_lam)

    tanh_perc_mod = _load_module("v2_tanh_perceiver")
    tok2, pad_mask2 = tanh_perc_mod.Patches(PATCHES).tokens(idx)
    m = tanh_perc_mod.PerceiverAE(2)
    m.load_state_dict(torch.load(result("v2_tanh_perceiver", "ae.pt"), map_location="cpu"))
    m.eval()
    with torch.no_grad():
        _, log_lam = m(tok2, pad_mask2)
    versions["v2_tanh_perceiver"] = torch.exp(log_lam)

    tanh_vae_mod = _load_module("v2_tanh_vae")
    m = tanh_vae_mod.VAE(2)
    m.load_state_dict(torch.load(result("v2_tanh_vae", "ae.pt"), map_location="cpu"))
    m.eval()
    with torch.no_grad():
        _, _, _, log_lam = m(x)
    versions["v2_tanh_vae"] = torch.exp(log_lam)

    print(f"deviance_null（空模型，全體平均）= {dev_null_total:.2f}\n")

    r2s = {}
    for name, lam in versions.items():
        dev_model_total = deviance(x, lam).sum().item()
        r2 = 1 - dev_model_total / dev_null_total
        r2s[name] = r2
        print(f"{name:<14}deviance = {dev_model_total:10.2f}   R² = {r2:+.4f}")

    colors = ["#3a6ea5" if not n.startswith("v2_tanh") else "#c0392b" for n in r2s]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    names = list(r2s.keys())
    vals = [r2s[n] for n in names]
    bars = ax.bar(names, vals, color=colors, alpha=0.8, width=0.6)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}",
                ha="center", va="bottom", fontsize=10)
    ax.axhline(0, color="#888", linewidth=0.8)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylabel("R² = 1 - deviance_model / deviance_null", fontsize=9)
    ax.tick_params(labelsize=9)
    ax.grid(alpha=0.15, linewidth=0.5, axis="y")
    for s in ax.spines.values():
        s.set_alpha(0.3)

    fig.suptitle(f"v2 系列 deviance-based R²（{data.n} 個 patch）", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT, bbox_inches="tight")
    print(f"\n已存 {OUT}")


main()
