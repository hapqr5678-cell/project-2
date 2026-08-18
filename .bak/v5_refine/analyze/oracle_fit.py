"""共用工具：凍住 decoder，對每個 patch 的 latent 做自由最佳化（多起點）。

oracle.py（測量一）與 noise_floor.py（測量二）都要用同一套解法，抽出來放這裡
才能保證兩邊量到的「oracle」是同一個東西、可以直接相減。
"""

import torch


def fit_z(decoder, x, z0, nll, restarts=4, steps=1500, lr=0.05, seed=0):
    """凍住 decoder，對每個 patch 自由解 z* = argmin_z NLL(decoder(z), x)。

    decoder：nn.Module，吃 (B,latent_dim) 回傳 (B,N_CAT) 的 log λ。呼叫端要
        先 eval() 並把參數 requires_grad_(False)，本函式不會替你關。
    x：(B,N_CAT) 要解釋的 count 向量。
    z0：(B,latent_dim) 第一個起點，通常是模型自己輸出的 z。
    nll：函式 (log_lam, x) -> (B,)，直接傳 ae.poisson_nll。
    restarts：除了 z0 之外再撒幾個隨機起點（範圍取 z0 每一維的 min~max），
        每個 patch 各自取最好的那一個。NLL 在 z 上是非凸的，不多起點量到的
        會是「某個局部極小」而不是 oracle。
    steps：Adam 的步數。lr：初始學習率，cosine 衰減到 lr/100。
    seed：隨機起點的種子。

    回傳 (z_best, nll_best, grad_norm)：
        z_best (B,latent_dim)：每個 patch 最好的解。
        nll_best (B,)：對應的 NLL。
        grad_norm：純量，最後一步所有起點的 |∇_z NLL| 平均。這是收斂證據——
        它沒有小到接近 0，量到的 oracle 就是偏高的（還沒走到底）。
    """
    device = z0.device
    B, D = z0.shape
    R = restarts + 1

    g = torch.Generator().manual_seed(seed)
    lo, hi = z0.min(dim=0).values.cpu(), z0.max(dim=0).values.cpu()
    extra = torch.rand(restarts, B, D, generator=g) * (hi - lo) + lo
    z = torch.cat([z0.detach().cpu().unsqueeze(0), extra], dim=0).to(device)
    z = z.requires_grad_(True)

    xr = x.repeat(R, 1)
    opt = torch.optim.Adam([z], lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, eta_min=lr / 100)
    for _ in range(steps):
        loss = nll(decoder(z.view(R * B, D)), xr).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
    grad_norm = z.grad.norm(dim=-1).mean().item()

    with torch.no_grad():
        per = nll(decoder(z.view(R * B, D)), xr).view(R, B)
        best = per.argmin(dim=0)
        ar = torch.arange(B, device=device)
        z_best = z.view(R, B, D)[best, ar].detach()
        nll_best = per[best, ar]
    return z_best, nll_best, grad_norm
