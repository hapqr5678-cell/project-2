"""DDAE + FSCE + OD-GAT with a class-balanced soft-composition objective.

The OD/count architecture and leakage controls are inherited from
``v2_ddae_gat``.  This experiment adds a *linear* composition head on the
fused two-dimensional latent.  A linear head is intentional: it makes the
latent itself carry category-composition information instead of delegating
all separation to another deep decoder.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


BASE_AE = Path(__file__).resolve().parents[1] / "v2_ddae_gat" / "ae.py"
_SPEC = importlib.util.spec_from_file_location("_v2_ddae_gat_ae", BASE_AE)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"cannot load base model from {BASE_AE}")
_BASE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _BASE
_SPEC.loader.exec_module(_BASE)

# Re-export the unchanged data, graph, reconstruction, and FSCE components so
# train.py remains explicit while this experiment stays focused on its change.
Patches = _BASE.Patches
ODGraph = _BASE.ODGraph
corrupt = _BASE.corrupt
load_od_graph = _BASE.load_od_graph
build_fsce_graph = _BASE.build_fsce_graph
fsce_loss = _BASE.fsce_loss
poisson_nll = _BASE.poisson_nll
poisson_deviance = _BASE.poisson_deviance


class CountODCompositionAE(_BASE.CountODFusedAE):
    """Fused count/OD autoencoder with a linear latent-composition head."""

    def __init__(
        self,
        latent_dim,
        node_feature_dim,
        edge_feature_dim,
        n_categories,
        alpha_od=0.3,
    ):
        super().__init__(
            latent_dim=latent_dim,
            node_feature_dim=node_feature_dim,
            edge_feature_dim=edge_feature_dim,
            alpha_od=alpha_od,
        )
        self.composition_head = nn.Linear(latent_dim, n_categories)
        nn.init.xavier_uniform_(self.composition_head.weight)
        nn.init.zeros_(self.composition_head.bias)

    def composition_logits(self, z):
        return self.composition_head(z)

    def composition_probabilities(self, z):
        return self.composition_logits(z).softmax(dim=1)


def count_composition(x):
    """Convert non-negative counts to a per-patch category distribution."""
    return x / x.sum(dim=1, keepdim=True).clamp_min(1.0)


def make_class_weights(
    training_counts,
    power=0.5,
    minimum=0.25,
    maximum=4.0,
):
    """Inverse-frequency weights estimated from training counts only.

    Square-root inverse frequency is deliberately milder than full inverse
    frequency because the rarest category has far fewer observations than
    Dining.  Clipping prevents a rare category from destabilizing training.
    """
    frequency = training_counts.sum(dim=0)
    frequency = frequency / frequency.sum().clamp_min(1.0)
    if (frequency <= 0).any():
        missing = torch.nonzero(frequency <= 0, as_tuple=False).flatten().tolist()
        raise ValueError(f"training split has zero count for categories {missing}")
    weights = frequency.pow(-power)
    weights = weights / weights.mean()
    weights = weights.clamp(min=minimum, max=maximum)
    # Per-patch loss normalization below makes a second global normalization
    # unnecessary; omitting it also preserves the requested clipping bounds.
    return weights


def class_balanced_soft_ce(logits, target, class_weights):
    """Soft CE with rare-category *patch* weighting.

    Weighting each category inside ``target`` would change the optimal
    predicted composition from ``target`` to a reweighted distribution.  We
    instead weight the complete patch loss according to its composition.  A
    perfect prediction is therefore still the original clean proportion.
    """
    per_patch = -(target * F.log_softmax(logits, dim=1)).sum(dim=1)
    patch_weight = (target * class_weights.unsqueeze(0)).sum(dim=1)
    patch_weight = patch_weight / patch_weight.mean().detach().clamp_min(1e-12)
    return per_patch * patch_weight


def composition_js(probability, target, eps=1e-8):
    """Per-patch Jensen-Shannon divergence in natural-log units."""
    probability = probability.clamp_min(eps)
    target = target.clamp_min(eps)
    midpoint = 0.5 * (probability + target)
    return 0.5 * (
        (target * (target.log() - midpoint.log())).sum(dim=1)
        + (probability * (probability.log() - midpoint.log())).sum(dim=1)
    )
