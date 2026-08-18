"""DDAE + FSCE with a zero-initialized residual decoder MoE.

The baseline encoder and decoder keep exactly the same parameter names as
``v2_ddae_fsce`` so its checkpoint can be loaded directly.  Two decoder
experts predict corrections to the baseline log-rate.  Their final layers are
initialized to zero, therefore the initial model is exactly the baseline:

    log_lambda = base_log_lambda + moe_scale * sum(gate_e * residual_e)
"""

import os
import sys

import numpy as np
import torch
import torch.nn as nn
from sklearn.neighbors import NearestNeighbors
from umap.umap_ import find_ab_params, fuzzy_simplicial_set

sys.path.insert(0, os.path.abspath(f"{os.path.dirname(__file__)}/../.."))
from config.dataset import N_CAT  # noqa: E402


HIDDEN = 64
N_EXPERTS = 2
ROUTER_TEMPERATURE = 0.7
MOE_SCALE = 0.1


class Patches:
    """Sparse POI list; aggregate patches into category-count vectors."""

    def __init__(self, path):
        data = np.load(path)
        self.cat = torch.from_numpy(data["cat"].astype(np.int64))
        self.offsets = torch.from_numpy(data["offsets"])
        self.n_poi = data["n_poi"]
        self.lat = data["center_lat"]
        self.lon = data["center_lon"]
        self.n = len(self.n_poi)

    def agg(self, idx):
        idx = torch.as_tensor(idx, dtype=torch.long)
        batch_size = len(idx)
        starts, ends = self.offsets[idx], self.offsets[idx + 1]
        lengths = ends - starts
        positions = (
            torch.repeat_interleave(starts, lengths)
            + torch.arange(int(lengths.sum()))
            - torch.repeat_interleave(
                torch.cumsum(lengths, 0) - lengths, lengths
            )
        )
        owners = torch.repeat_interleave(torch.arange(batch_size), lengths)
        categories = self.cat[positions]
        flat = owners * N_CAT + categories
        counts = torch.bincount(flat, minlength=batch_size * N_CAT)
        return counts.view(batch_size, N_CAT).float()


def corrupt(x, p, mode="thinning", generator=None):
    """Corrupt count inputs and preserve their expected scale."""
    if p <= 0:
        return x
    keep = 1.0 - p
    if mode == "thinning":
        max_count = int(x.max().item())
        if max_count == 0:
            return x
        coin = torch.rand(
            x.shape + (max_count,), generator=generator, device=x.device
        ) < keep
        alive = torch.arange(max_count, device=x.device) < x.unsqueeze(-1)
        noisy = (coin & alive).sum(dim=-1).float()
    elif mode == "mask":
        mask = torch.rand(
            x.shape, generator=generator, device=x.device
        ) < keep
        noisy = x * mask.float()
    else:
        raise ValueError(f"unknown corruption mode: {mode}")
    return noisy / keep


class ResidualDecoderMoEAE(nn.Module):
    """Baseline DDAE with gated residual experts on decoder log-rates."""

    def __init__(
        self,
        latent_dim=2,
        n_experts=N_EXPERTS,
        router_temperature=ROUTER_TEMPERATURE,
        moe_scale=MOE_SCALE,
    ):
        super().__init__()
        if n_experts < 2:
            raise ValueError("n_experts must be at least 2")
        if router_temperature <= 0:
            raise ValueError("router_temperature must be positive")
        if moe_scale < 0:
            raise ValueError("moe_scale must be non-negative")

        self.n_experts = n_experts
        self.router_temperature = router_temperature
        self.moe_scale = moe_scale

        # Keep these module names and layer indices identical to v2_ddae_fsce.
        self.encoder = nn.Sequential(
            nn.Linear(N_CAT, HIDDEN),
            nn.GELU(),
            nn.Linear(HIDDEN, HIDDEN),
            nn.GELU(),
            nn.Linear(HIDDEN, HIDDEN),
            nn.GELU(),
            nn.Linear(HIDDEN, HIDDEN),
            nn.GELU(),
            nn.Linear(HIDDEN, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, HIDDEN),
            nn.GELU(),
            nn.Linear(HIDDEN, HIDDEN),
            nn.GELU(),
            nn.Linear(HIDDEN, HIDDEN),
            nn.GELU(),
            nn.Linear(HIDDEN, HIDDEN),
            nn.GELU(),
            nn.Linear(HIDDEN, N_CAT),
        )

        self.router = nn.Sequential(
            nn.Linear(latent_dim, HIDDEN),
            nn.GELU(),
            nn.Linear(HIDDEN, n_experts),
        )
        self.residual_experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(latent_dim, HIDDEN),
                nn.GELU(),
                nn.Linear(HIDDEN, N_CAT),
            )
            for _ in range(n_experts)
        ])
        for expert in self.residual_experts:
            nn.init.zeros_(expert[-1].weight)
            nn.init.zeros_(expert[-1].bias)

    def encode(self, x):
        return self.encoder(x)

    def route(self, z):
        return torch.softmax(
            self.router(z) / self.router_temperature, dim=-1
        )

    def decode_with_gates(self, z):
        base_log_lam = self.decoder(z)
        gates = self.route(z)
        residuals = torch.stack(
            [expert(z) for expert in self.residual_experts], dim=1
        )
        residual = (gates.unsqueeze(-1) * residuals).sum(dim=1)
        return base_log_lam + self.moe_scale * residual, gates

    def decode(self, z):
        log_lam, _ = self.decode_with_gates(z)
        return log_lam

    def forward_with_gates(self, x):
        z = self.encode(x)
        log_lam, gates = self.decode_with_gates(z)
        return z, log_lam, gates

    def forward(self, x):
        z, log_lam, _ = self.forward_with_gates(x)
        return z, log_lam

    def set_base_trainable(self, trainable):
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(trainable)
        for parameter in self.decoder.parameters():
            parameter.requires_grad_(trainable)


def load_baseline_checkpoint(model, path, map_location="cpu"):
    """Load v2_ddae_fsce weights and verify only MoE keys are missing."""
    try:
        checkpoint = torch.load(
            path, map_location=map_location, weights_only=True
        )
    except TypeError:
        checkpoint = torch.load(path, map_location=map_location)
    state = checkpoint.get("model_state", checkpoint)
    incompatible = model.load_state_dict(state, strict=False)
    allowed_prefixes = ("router.", "residual_experts.")
    invalid_missing = [
        key for key in incompatible.missing_keys
        if not key.startswith(allowed_prefixes)
    ]
    if invalid_missing or incompatible.unexpected_keys:
        raise RuntimeError(
            "baseline checkpoint does not match model: "
            f"missing={invalid_missing}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    return incompatible.missing_keys


def moe_balance_loss(gates):
    """Optional global-usage regularizer; disabled in the first experiment."""
    mean_usage = gates.mean(dim=0)
    target = torch.full_like(mean_usage, 1.0 / gates.shape[1])
    return (mean_usage - target).pow(2).mean()


def poisson_nll(log_lam, x):
    cell = torch.exp(log_lam) - x * log_lam
    return cell.mean(dim=1)


def poisson_deviance(log_lam, x):
    lam = torch.exp(log_lam)
    cell = 2 * (torch.xlogy(x, x) - x * log_lam - x + lam)
    return cell.mean(dim=1)


def build_fsce_graph(x, n_neighbors=15, metric="euclidean"):
    knn = NearestNeighbors(n_neighbors=n_neighbors, metric=metric).fit(x)
    knn_distances, knn_indices = knn.kneighbors(x)
    graph, _, _ = fuzzy_simplicial_set(
        x,
        n_neighbors=n_neighbors,
        random_state=0,
        metric=metric,
        knn_indices=knn_indices,
        knn_dists=knn_distances,
    )
    graph = graph.tocoo()
    edge_i = torch.from_numpy(graph.row).long()
    edge_j = torch.from_numpy(graph.col).long()
    edge_weight = torch.from_numpy(graph.data).float()
    a, b = find_ab_params(spread=1.0, min_dist=0.1)
    return edge_i, edge_j, edge_weight, a, b


def fsce_loss(z_i, z_j, weight, a, b, eps=1e-4):
    distance_squared = (z_i - z_j).pow(2).sum(dim=1).clamp(min=eps)
    similarity = (1.0 + a * distance_squared.pow(b)).reciprocal()
    similarity = similarity.clamp(eps, 1.0 - eps)
    return -(
        weight * similarity.log()
        + (1.0 - weight) * (1.0 - similarity).log()
    )
