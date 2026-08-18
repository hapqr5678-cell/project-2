"""Balanced soft-composition pair geometry for the DDAE + OD-GAT model.

The unchanged count encoder, Poisson decoder, OD graph, and GAT are shared
with ``v2_ddae_gat``.  This experiment replaces the count-kNN FSCE graph and
the failed composition classifier with an implicit, training-only fuzzy graph:

* sample categories uniformly;
* sample patches in proportion to their membership in that category;
* compare same-category and cross-category pairs;
* assign every pair a continuous, prior-corrected Hellinger membership.

Mixed patches can participate in several categories and therefore remain
bridges instead of being forced into one hard class.
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F


BASE_AE = Path(__file__).resolve().parents[1] / "v2_ddae_gat" / "ae.py"
_SPEC = importlib.util.spec_from_file_location("_v2_ddae_gat_balanced_base", BASE_AE)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"cannot load base model from {BASE_AE}")
_BASE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _BASE
_SPEC.loader.exec_module(_BASE)

Patches = _BASE.Patches
ODGraph = _BASE.ODGraph
CountODFusedAE = _BASE.CountODFusedAE
corrupt = _BASE.corrupt
load_od_graph = _BASE.load_od_graph
poisson_nll = _BASE.poisson_nll
poisson_deviance = _BASE.poisson_deviance
fsce_loss = _BASE.fsce_loss


@dataclass
class BalancedCompositionPairs:
    """Training-only distributions used to sample an implicit fuzzy graph."""

    composition: torch.Tensor
    sampling_weights: torch.Tensor
    hellinger_embedding: torch.Tensor
    prior: torch.Tensor
    membership_power: float
    a: float
    b: float

    @property
    def n_categories(self):
        return self.composition.shape[1]

    @property
    def n_patches(self):
        return self.composition.shape[0]


def count_composition(counts):
    return counts / counts.sum(dim=1, keepdim=True).clamp_min(1.0)


def build_balanced_composition_pairs(
    training_counts,
    sampling_power=0.75,
    membership_power=2.0,
):
    """Build a leakage-safe sampler from training counts only.

    ``sqrt(p / prior)`` removes the global category prior before cosine
    similarity.  Without this correction, Dining is shared by nearly every
    patch and dominates ordinary Hellinger/Bhattacharyya similarity.
    """
    if not 0.0 < sampling_power <= 1.0:
        raise ValueError("sampling_power must be in (0, 1]")
    if membership_power <= 0.0:
        raise ValueError("membership_power must be positive")
    composition = count_composition(training_counts.float()).cpu()
    prior = composition.mean(dim=0)
    if (prior <= 0).any():
        missing = torch.nonzero(prior <= 0, as_tuple=False).flatten().tolist()
        raise ValueError(f"training split has no membership for categories {missing}")

    # Rows are categories, columns are local training-patch indices.
    sampling_weights = composition.pow(sampling_power).T.contiguous()
    if (sampling_weights.sum(dim=1) <= 0).any():
        raise ValueError("a category has no patch available for pair sampling")

    embedding = torch.sqrt(composition / prior.unsqueeze(0))
    embedding = F.normalize(embedding, p=2, dim=1, eps=1e-12)
    a, b = _BASE.find_ab_params(spread=1.0, min_dist=0.1)
    return BalancedCompositionPairs(
        composition=composition,
        sampling_weights=sampling_weights,
        hellinger_embedding=embedding,
        prior=prior,
        membership_power=float(membership_power),
        a=float(a),
        b=float(b),
    )


def _balanced_categories(n_pairs, n_categories, generator):
    repeats = (n_pairs + n_categories - 1) // n_categories
    categories = torch.arange(n_categories).repeat(repeats)[:n_pairs]
    order = torch.randperm(n_pairs, generator=generator)
    return categories[order]


def _sample_patch(sampler, categories, generator):
    rows = sampler.sampling_weights[categories]
    return torch.multinomial(
        rows,
        num_samples=1,
        replacement=True,
        generator=generator,
    ).squeeze(1)


def _resample_identical(sampler, left, right, categories, generator):
    """Avoid zero-gradient self-pairs when the category has enough support."""
    for _ in range(3):
        identical = left == right
        if not identical.any():
            break
        right[identical] = _sample_patch(
            sampler,
            categories[identical],
            generator,
        )
    return right


def sample_balanced_composition_pairs(sampler, n_pairs, generator):
    """Return equal numbers of same- and cross-category soft pairs.

    ``n_pairs`` is the number in each group; the returned arrays contain
    ``2 * n_pairs`` pairs.  Membership always comes from the full composition,
    so even the cross-category group is not incorrectly forced to zero when
    two mixed patches are genuinely similar.
    """
    categories = _balanced_categories(
        n_pairs,
        sampler.n_categories,
        generator,
    )
    different_offset = torch.randint(
        1,
        sampler.n_categories,
        (n_pairs,),
        generator=generator,
    )
    other_categories = (categories + different_offset) % sampler.n_categories

    same_left = _sample_patch(sampler, categories, generator)
    same_right = _sample_patch(sampler, categories, generator)
    same_right = _resample_identical(
        sampler, same_left, same_right, categories, generator
    )

    cross_left = _sample_patch(sampler, categories, generator)
    cross_right = _sample_patch(sampler, other_categories, generator)
    cross_right = _resample_identical(
        sampler, cross_left, cross_right, other_categories, generator
    )

    left = torch.cat([same_left, cross_left])
    right = torch.cat([same_right, cross_right])
    membership = (
        sampler.hellinger_embedding[left]
        * sampler.hellinger_embedding[right]
    ).sum(dim=1)
    membership = membership.clamp(0.0, 1.0).pow(sampler.membership_power)
    return left, right, membership
