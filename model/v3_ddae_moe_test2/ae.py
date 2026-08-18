"""Strict-2D count autoencoder with competitive composition experts."""

import os
import sys

import numpy as np
import torch
import torch.nn as nn
from sklearn.neighbors import NearestNeighbors
from umap.umap_ import find_ab_params, fuzzy_simplicial_set

sys.path.insert(0, os.path.abspath(f"{os.path.dirname(__file__)}/../.."))
from config.dataset import HALF_WIDTH, N_CAT  # noqa: E402,F401


HIDDEN = 64
N_EXPERTS = 2
ROUTER_TEMPERATURE = 0.7
EXPERT_INIT_NOISE = 1e-3

MODE_BASE = 0
MODE_HARD = 1
MODE_SOFT = 2
MODE_NAMES = {
    MODE_BASE: "base",
    MODE_HARD: "hard",
    MODE_SOFT: "soft",
}


class Patches:
    """Sparse POI lists aggregated into one count vector per patch."""

    def __init__(self, path):
        data = np.load(path)
        self.cat = torch.from_numpy(data["cat"].astype(np.int64))
        self.offsets = torch.from_numpy(data["offsets"])
        self.n_poi = data["n_poi"]
        self.lat = data["center_lat"]
        self.lon = data["center_lon"]
        self.n = len(self.n_poi)

    def agg(self, idx):
        batch_size = len(idx)
        starts, ends = self.offsets[idx], self.offsets[idx + 1]
        lengths = ends - starts
        positions = (
            torch.repeat_interleave(starts, lengths)
            + torch.arange(int(lengths.sum()))
            - torch.repeat_interleave(torch.cumsum(lengths, 0) - lengths, lengths)
        )
        owners = torch.repeat_interleave(torch.arange(batch_size), lengths)
        categories = self.cat[positions]
        flat = owners * N_CAT + categories
        counts = torch.bincount(flat, minlength=batch_size * N_CAT)
        return counts.view(batch_size, N_CAT).float()


def corrupt(x, probability, mode="thinning", generator=None):
    """Corrupt counts and rescale them so their expectation stays unchanged."""
    if probability <= 0:
        return x
    keep = 1.0 - probability
    if mode == "thinning":
        max_count = int(x.max().item())
        if max_count == 0:
            return x
        coins = (
            torch.rand(
                x.shape + (max_count,), generator=generator, device=x.device
            )
            < keep
        )
        alive = torch.arange(max_count, device=x.device) < x.unsqueeze(-1)
        noisy = (coins & alive).sum(dim=-1).float()
    elif mode == "mask":
        mask = (
            torch.rand(x.shape, generator=generator, device=x.device) < keep
        )
        noisy = x * mask.float()
    else:
        raise ValueError(f"unknown corruption mode: {mode}")
    return noisy / keep


def _hidden_stack(input_dim, output_dim, depth=4):
    layers = []
    current_dim = input_dim
    for _ in range(depth):
        layers.extend(
            [
                nn.Linear(current_dim, HIDDEN),
                nn.LayerNorm(HIDDEN),
                nn.GELU(),
            ]
        )
        current_dim = HIDDEN
    layers.append(nn.Linear(current_dim, output_dim))
    return nn.Sequential(*layers)


def arithmetic_mixture_log_rate(expert_log_lam, gates):
    """Compute ``log(sum_e gate_e * lambda_e)`` stably."""
    log_gates = gates.clamp_min(torch.finfo(gates.dtype).tiny).log()
    return torch.logsumexp(log_gates.unsqueeze(-1) + expert_log_lam, dim=1)


def hard_mixture_log_rate(expert_log_lam, gates):
    """Select exactly one complete expert for each patch."""
    selected = gates.argmax(dim=1)
    row = torch.arange(len(selected), device=selected.device)
    return expert_log_lam[row, selected]


class MLPAE(nn.Module):
    """Strict 2D baseline plus competitive residual composition experts.

    The shared total head predicts one expected total count. The base and every
    expert predict a category composition on the probability simplex. Experts
    therefore cannot improve by cancelling total-count scale or by emitting an
    incomplete rate vector.
    """

    def __init__(
        self,
        latent_dim=2,
        n_experts=N_EXPERTS,
        router_temperature=ROUTER_TEMPERATURE,
        expert_init_noise=EXPERT_INIT_NOISE,
    ):
        super().__init__()
        if latent_dim != 2:
            raise ValueError("v3_ddae_moe_test2 requires latent_dim=2")
        if n_experts < 2:
            raise ValueError("n_experts must be at least 2")
        if router_temperature <= 0:
            raise ValueError("router_temperature must be positive")

        self.latent_dim = latent_dim
        self.n_experts = n_experts
        self.router_temperature = router_temperature
        self.encoder = _hidden_stack(N_CAT, latent_dim, depth=4)
        self.total_decoder = _hidden_stack(latent_dim, 1, depth=2)
        self.composition_decoder = _hidden_stack(latent_dim, N_CAT, depth=4)

        self.residual_experts = nn.ModuleList(
            [_hidden_stack(latent_dim, N_CAT, depth=2) for _ in range(n_experts)]
        )
        for expert in self.residual_experts:
            nn.init.normal_(expert[-1].weight, mean=0.0, std=expert_init_noise)
            nn.init.zeros_(expert[-1].bias)

        self.router = nn.Sequential(
            nn.Linear(latent_dim, HIDDEN),
            nn.GELU(),
            nn.Linear(HIDDEN, n_experts),
        )
        nn.init.zeros_(self.router[-1].weight)
        nn.init.zeros_(self.router[-1].bias)

        # The restored state records which validation candidate won.
        self.register_buffer("inference_mode", torch.tensor(MODE_BASE))

    def set_inference_mode(self, mode):
        if mode not in MODE_NAMES:
            raise ValueError(f"unknown inference mode: {mode}")
        self.inference_mode.fill_(mode)

    def encode(self, x):
        """Return the model's only latent representation, shape ``(B, 2)``."""
        return self.encoder(x)

    def decode_base(self, z):
        """Decode only the single-model baseline path."""
        log_total = self.total_decoder(z)
        base_logits = self.composition_decoder(z)
        base_log_lam = log_total + torch.log_softmax(base_logits, dim=-1)
        return log_total, base_logits, base_log_lam

    def decode_all(self, z):
        log_total, base_logits, base_log_lam = self.decode_base(z)

        residuals = torch.stack(
            [expert(z) for expert in self.residual_experts], dim=1
        )
        expert_logits = base_logits.unsqueeze(1) + residuals
        expert_log_lam = log_total.unsqueeze(1) + torch.log_softmax(
            expert_logits, dim=-1
        )
        router_logits = self.router(z)
        gates = torch.softmax(
            router_logits / self.router_temperature, dim=-1
        )
        hard_log_lam = hard_mixture_log_rate(expert_log_lam, gates)
        soft_log_lam = arithmetic_mixture_log_rate(expert_log_lam, gates)
        return {
            "base_log_lam": base_log_lam,
            "hard_log_lam": hard_log_lam,
            "soft_log_lam": soft_log_lam,
            "expert_log_lam": expert_log_lam,
            "residuals": residuals,
            "router_logits": router_logits,
            "gates": gates,
            "log_total": log_total.squeeze(-1),
        }

    def forward_base(self, x):
        z = self.encode(x)
        return z, self.decode_base(z)[2]

    def forward_with_experts(self, x):
        z = self.encode(x)
        return z, self.decode_all(z)

    def forward(self, x):
        z = self.encode(x)
        outputs = self.decode_all(z)
        mode = int(self.inference_mode.item())
        key = {
            MODE_BASE: "base_log_lam",
            MODE_HARD: "hard_log_lam",
            MODE_SOFT: "soft_log_lam",
        }[mode]
        return z, outputs[key]


def poisson_nll(log_lam, x):
    """Poisson NLL without the target-only ``log(x!)`` term, per patch."""
    return (torch.exp(log_lam) - x * log_lam).mean(dim=-1)


def poisson_deviance(log_lam, x):
    """Mean category-wise Poisson deviance, per patch."""
    lam = torch.exp(log_lam)
    cell = 2 * (torch.xlogy(x, x) - x * log_lam - x + lam)
    return cell.mean(dim=-1)


def expert_nlls(expert_log_lam, x):
    expanded_x = x.unsqueeze(1).expand_as(expert_log_lam)
    return poisson_nll(expert_log_lam, expanded_x)


def expert_deviances(expert_log_lam, x):
    expanded_x = x.unsqueeze(1).expand_as(expert_log_lam)
    return poisson_deviance(expert_log_lam, expanded_x)


def build_fsce_graph(x, n_neighbors=15, metric="euclidean"):
    """Build the high-dimensional fuzzy graph used by FSCE."""
    knn = NearestNeighbors(n_neighbors=n_neighbors, metric=metric).fit(x)
    knn_dists, knn_idx = knn.kneighbors(x)
    graph, _, _ = fuzzy_simplicial_set(
        x,
        n_neighbors=n_neighbors,
        random_state=0,
        metric=metric,
        knn_indices=knn_idx,
        knn_dists=knn_dists,
    )
    graph = graph.tocoo()
    edge_i = torch.from_numpy(graph.row).long()
    edge_j = torch.from_numpy(graph.col).long()
    edge_weight = torch.from_numpy(graph.data).float()
    a, b = find_ab_params(spread=1.0, min_dist=0.1)
    return edge_i, edge_j, edge_weight, a, b


def fsce_loss(z_i, z_j, weight, a, b, eps=1e-4):
    """Fuzzy-set cross entropy for positive and sampled negative pairs."""
    distance_sq = (z_i - z_j).pow(2).sum(dim=1).clamp(min=eps)
    similarity = (1.0 + a * distance_sq.pow(b)).reciprocal()
    similarity = similarity.clamp(eps, 1 - eps)
    return -(
        weight * similarity.log()
        + (1 - weight) * (1 - similarity).log()
    )
