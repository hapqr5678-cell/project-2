"""DDAE + FSCE + POI OD-GAT latent fusion.

The count branch is the four-hidden-layer model used by
``v2_ddae_fsce_taryn``.  The OD branch applies a directed, edge-aware GAT to
POI nodes and then attention-pools POIs into patch embeddings.  Both branches
produce ``latent_dim`` features and are fused as

    z = (1 - alpha_od) * z_count + alpha_od * z_od

For patches without OD nodes, ``z_count`` is used unchanged.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors
from umap.umap_ import find_ab_params, fuzzy_simplicial_set


sys.path.insert(0, os.path.abspath(f"{os.path.dirname(__file__)}/../.."))
from config.dataset import CATEGORIES, N_CAT  # noqa: E402


HIDDEN = 64
N_HIDDEN_LAYERS = 4


class Patches:
    """Sparse patch POIs; ``agg`` returns one category-count vector per patch."""

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
            - torch.repeat_interleave(torch.cumsum(lengths, 0) - lengths, lengths)
        )
        owners = torch.repeat_interleave(torch.arange(batch_size), lengths)
        categories = self.cat[positions]
        flat = owners * N_CAT + categories
        counts = torch.bincount(flat, minlength=batch_size * N_CAT)
        return counts.view(batch_size, N_CAT).float()


def corrupt(x, p, mode="thinning", generator=None):
    """Corrupt count inputs while preserving their expected scale."""
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
        mask = torch.rand(x.shape, generator=generator, device=x.device) < keep
        noisy = x * mask.float()
    else:
        raise ValueError(f"unknown corruption mode: {mode}")
    return noisy / keep


def _mlp_block(input_dim, n_layers):
    layers = [nn.Linear(input_dim, HIDDEN), nn.GELU()]
    for _ in range(n_layers - 1):
        layers += [nn.Linear(HIDDEN, HIDDEN), nn.GELU()]
    return layers


class CountEncoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.network = nn.Sequential(
            *_mlp_block(N_CAT, N_HIDDEN_LAYERS),
            nn.Linear(HIDDEN, latent_dim),
        )

    def forward(self, x):
        return self.network(x)


class PoissonDecoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.network = nn.Sequential(
            *_mlp_block(latent_dim, N_HIDDEN_LAYERS),
            nn.Linear(HIDDEN, N_CAT),
        )

    def forward(self, z):
        return self.network(z)


@dataclass
class ODGraph:
    """Tensor representation of full and training-only POI message graphs."""

    node_features: torch.Tensor
    node_patch: torch.Tensor
    train_edge_index: torch.Tensor
    train_edge_features: torch.Tensor
    full_edge_index: torch.Tensor
    full_edge_features: torch.Tensor
    patch_has_od: torch.Tensor
    n_patches: int
    n_raw_edges: int
    n_train_raw_edges: int

    @property
    def node_feature_dim(self):
        return self.node_features.shape[1]

    @property
    def edge_feature_dim(self):
        return self.full_edge_features.shape[1]

    def to(self, device):
        return ODGraph(
            node_features=self.node_features.to(device),
            node_patch=self.node_patch.to(device),
            train_edge_index=self.train_edge_index.to(device),
            train_edge_features=self.train_edge_features.to(device),
            full_edge_index=self.full_edge_index.to(device),
            full_edge_features=self.full_edge_features.to(device),
            patch_has_od=self.patch_has_od.to(device),
            n_patches=self.n_patches,
            n_raw_edges=self.n_raw_edges,
            n_train_raw_edges=self.n_train_raw_edges,
        )


def _message_graph(origin, destination, edge_features, node_mask):
    """Expose both incoming and outgoing context while preserving direction."""
    # ``pandas.Series.map(...).to_numpy()`` may return a read-only view.
    # ``torch.tensor`` deliberately copies it and avoids undefined write behaviour.
    origin = torch.tensor(origin, dtype=torch.long)
    destination = torch.tensor(destination, dtype=torch.long)
    edge_features = torch.tensor(edge_features, dtype=torch.float32)
    node_ids = torch.nonzero(
        torch.as_tensor(node_mask, dtype=torch.bool), as_tuple=False
    ).flatten()

    forward_direction = torch.ones((len(origin), 1), dtype=torch.float32)
    reverse_direction = -forward_direction
    self_features = torch.zeros(
        (len(node_ids), edge_features.shape[1] + 1), dtype=torch.float32
    )
    forward_features = torch.cat([edge_features, forward_direction], dim=1)
    reverse_features = torch.cat([edge_features, reverse_direction], dim=1)

    edge_index = torch.stack(
        [
            torch.cat([origin, destination, node_ids]),
            torch.cat([destination, origin, node_ids]),
        ]
    )
    message_features = torch.cat(
        [forward_features, reverse_features, self_features], dim=0
    )
    return edge_index, message_features


def load_od_graph(od_csv, poi_csv, n_patches, train_idx):
    """Load and standardize OD data using training patches only.

    The training message graph contains only edges whose two endpoint patches
    are training patches.  Full-graph tensors are reserved for validation and
    final inference, so validation topology never updates model parameters.
    """
    required_edge_columns = {
        "origin_poi_id",
        "destination_poi_id",
        "origin_patch_id",
        "destination_patch_id",
        "log1p_trip_count",
        "log1p_unique_user_count",
        "distance_m",
        "median_travel_minutes",
    }
    edges = pd.read_csv(od_csv)
    missing = required_edge_columns - set(edges.columns)
    if missing:
        raise ValueError(f"OD CSV missing columns: {sorted(missing)}")
    if edges[list(required_edge_columns)].isna().any().any():
        raise ValueError("OD CSV contains missing values in required columns")
    if edges.duplicated(["origin_poi_id", "destination_poi_id"]).any():
        raise ValueError("OD CSV contains duplicate directed POI pairs")

    patch_columns = edges[["origin_patch_id", "destination_patch_id"]]
    if (patch_columns.to_numpy() < 0).any() or (
        patch_columns.to_numpy() >= n_patches
    ).any():
        raise ValueError("OD CSV patch IDs do not match current patches.npz")

    node_ids = np.union1d(
        edges["origin_poi_id"].astype(str).unique(),
        edges["destination_poi_id"].astype(str).unique(),
    )
    node_lookup = {poi_id: idx for idx, poi_id in enumerate(node_ids)}
    origin = edges["origin_poi_id"].astype(str).map(node_lookup).to_numpy()
    destination = edges["destination_poi_id"].astype(str).map(node_lookup).to_numpy()

    endpoint_mapping = pd.concat(
        [
            edges[["origin_poi_id", "origin_patch_id"]].rename(
                columns={"origin_poi_id": "poi_id", "origin_patch_id": "patch_id"}
            ),
            edges[["destination_poi_id", "destination_patch_id"]].rename(
                columns={
                    "destination_poi_id": "poi_id",
                    "destination_patch_id": "patch_id",
                }
            ),
        ],
        ignore_index=True,
    )
    if (endpoint_mapping.groupby("poi_id")["patch_id"].nunique() > 1).any():
        raise ValueError("one POI is assigned to multiple patches")
    patch_by_poi = (
        endpoint_mapping.drop_duplicates("poi_id").set_index("poi_id")["patch_id"]
    )
    node_patch = patch_by_poi.loc[node_ids].to_numpy(dtype=np.int64)

    poi = pd.read_csv(poi_csv).set_index("poi_id")
    required_poi_columns = {"category", "checkin_count"}
    if not required_poi_columns.issubset(poi.columns):
        raise ValueError(
            f"POI CSV missing columns: {sorted(required_poi_columns - set(poi.columns))}"
        )
    node_table = poi.loc[node_ids]
    category_lookup = {name: idx for idx, name in enumerate(CATEGORIES)}
    category_index = node_table["category"].map(category_lookup)
    if category_index.isna().any():
        unknown = sorted(node_table.loc[category_index.isna(), "category"].unique())
        raise ValueError(f"OD nodes contain unknown categories: {unknown}")

    train_patch_mask = np.zeros(n_patches, dtype=bool)
    train_patch_mask[np.asarray(train_idx, dtype=np.int64)] = True
    train_node_mask = train_patch_mask[node_patch]
    popularity = np.log1p(node_table["checkin_count"].to_numpy(dtype=np.float32))
    popularity_mean = popularity[train_node_mask].mean()
    popularity_std = max(popularity[train_node_mask].std(), 1e-6)
    popularity = (popularity - popularity_mean) / popularity_std
    category_one_hot = np.eye(N_CAT, dtype=np.float32)[
        category_index.to_numpy(dtype=np.int64)
    ]
    node_features = np.column_stack([category_one_hot, popularity]).astype(np.float32)

    raw_edge_features = np.column_stack(
        [
            edges["log1p_trip_count"].to_numpy(dtype=np.float32),
            edges["log1p_unique_user_count"].to_numpy(dtype=np.float32),
            np.log1p(edges["distance_m"].to_numpy(dtype=np.float32)),
            np.log1p(edges["median_travel_minutes"].to_numpy(dtype=np.float32)),
        ]
    )
    edge_patch_origin = edges["origin_patch_id"].to_numpy(dtype=np.int64)
    edge_patch_destination = edges["destination_patch_id"].to_numpy(dtype=np.int64)
    train_edge_mask = (
        train_patch_mask[edge_patch_origin]
        & train_patch_mask[edge_patch_destination]
    )
    if not train_edge_mask.any():
        raise ValueError("training-only OD graph has no edges")
    edge_mean = raw_edge_features[train_edge_mask].mean(axis=0)
    edge_std = np.maximum(raw_edge_features[train_edge_mask].std(axis=0), 1e-6)
    edge_features = (raw_edge_features - edge_mean) / edge_std

    train_edge_index, train_edge_features = _message_graph(
        origin[train_edge_mask],
        destination[train_edge_mask],
        edge_features[train_edge_mask],
        train_node_mask,
    )
    full_edge_index, full_edge_features = _message_graph(
        origin,
        destination,
        edge_features,
        np.ones(len(node_ids), dtype=bool),
    )
    patch_has_od = np.bincount(node_patch, minlength=n_patches) > 0

    return ODGraph(
        node_features=torch.from_numpy(node_features),
        node_patch=torch.tensor(node_patch, dtype=torch.long),
        train_edge_index=train_edge_index,
        train_edge_features=train_edge_features,
        full_edge_index=full_edge_index,
        full_edge_features=full_edge_features,
        patch_has_od=torch.from_numpy(patch_has_od),
        n_patches=n_patches,
        n_raw_edges=len(edges),
        n_train_raw_edges=int(train_edge_mask.sum()),
    )


def _segment_softmax(scores, segment, n_segments):
    """Softmax over incoming edges (or pooled nodes) in each segment."""
    if scores.ndim == 1:
        scores = scores[:, None]
        squeeze = True
    else:
        squeeze = False
    index = segment[:, None].expand(-1, scores.shape[1])
    maxima = torch.full(
        (n_segments, scores.shape[1]),
        -torch.inf,
        dtype=scores.dtype,
        device=scores.device,
    )
    maxima.scatter_reduce_(0, index, scores, reduce="amax", include_self=True)
    exponent = torch.exp(scores - maxima[segment])
    denominator = torch.zeros_like(maxima)
    denominator.index_add_(0, segment, exponent)
    result = exponent / denominator[segment].clamp_min(1e-12)
    return result[:, 0] if squeeze else result


class EdgeGATLayer(nn.Module):
    """Multi-head GAT layer whose attention and messages use OD edge features."""

    def __init__(self, input_dim, edge_dim, head_dim=16, heads=4, dropout=0.1):
        super().__init__()
        self.head_dim = head_dim
        self.heads = heads
        self.output_dim = head_dim * heads
        self.dropout = dropout
        self.node_projection = nn.Linear(input_dim, self.output_dim, bias=False)
        self.edge_projection = nn.Linear(edge_dim, self.output_dim, bias=False)
        self.residual = nn.Linear(input_dim, self.output_dim, bias=False)
        self.attention_source = nn.Parameter(torch.empty(heads, head_dim))
        self.attention_destination = nn.Parameter(torch.empty(heads, head_dim))
        self.attention_edge = nn.Parameter(torch.empty(heads, head_dim))
        self.bias = nn.Parameter(torch.zeros(self.output_dim))
        self.normalization = nn.LayerNorm(self.output_dim)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.node_projection.weight)
        nn.init.xavier_uniform_(self.edge_projection.weight)
        nn.init.xavier_uniform_(self.residual.weight)
        nn.init.xavier_uniform_(self.attention_source)
        nn.init.xavier_uniform_(self.attention_destination)
        nn.init.xavier_uniform_(self.attention_edge)
        nn.init.zeros_(self.bias)

    def forward(self, node_features, edge_index, edge_features):
        source, destination = edge_index
        n_nodes = len(node_features)
        node_hidden = self.node_projection(node_features).view(
            n_nodes, self.heads, self.head_dim
        )
        edge_hidden = self.edge_projection(edge_features).view(
            len(edge_features), self.heads, self.head_dim
        )
        scores = (
            (node_hidden[source] * self.attention_source).sum(dim=-1)
            + (node_hidden[destination] * self.attention_destination).sum(dim=-1)
            + (edge_hidden * self.attention_edge).sum(dim=-1)
        )
        scores = F.leaky_relu(scores, negative_slope=0.2)
        attention = _segment_softmax(scores, destination, n_nodes)
        attention = F.dropout(attention, p=self.dropout, training=self.training)

        messages = (node_hidden[source] + edge_hidden) * attention.unsqueeze(-1)
        output = torch.zeros(
            (n_nodes, self.heads, self.head_dim),
            dtype=node_features.dtype,
            device=node_features.device,
        )
        output.index_add_(0, destination, messages)
        output = output.reshape(n_nodes, self.output_dim) + self.residual(node_features)
        output = self.normalization(output + self.bias)
        return F.gelu(output)


class ODGATEncoder(nn.Module):
    """Encode POI OD interactions and attention-pool them into patch latents."""

    def __init__(
        self,
        node_feature_dim,
        edge_feature_dim,
        latent_dim,
        head_dim=16,
        heads=4,
        dropout=0.1,
    ):
        super().__init__()
        self.layer1 = EdgeGATLayer(
            node_feature_dim, edge_feature_dim, head_dim, heads, dropout
        )
        self.layer2 = EdgeGATLayer(
            self.layer1.output_dim, edge_feature_dim, head_dim, heads, dropout
        )
        self.pool_score = nn.Linear(self.layer2.output_dim, 1)
        self.to_latent = nn.Sequential(
            nn.Linear(self.layer2.output_dim, HIDDEN),
            nn.GELU(),
            nn.Linear(HIDDEN, latent_dim),
        )

    def forward(
        self,
        node_features,
        edge_index,
        edge_features,
        node_patch,
        n_patches,
    ):
        hidden = self.layer1(node_features, edge_index, edge_features)
        hidden = self.layer2(hidden, edge_index, edge_features)
        scores = self.pool_score(hidden).squeeze(-1)
        weights = _segment_softmax(scores, node_patch, n_patches)
        pooled = torch.zeros(
            (n_patches, hidden.shape[1]),
            dtype=hidden.dtype,
            device=hidden.device,
        )
        pooled.index_add_(0, node_patch, hidden * weights[:, None])
        return self.to_latent(pooled)


class CountODFusedAE(nn.Module):
    def __init__(
        self,
        latent_dim,
        node_feature_dim,
        edge_feature_dim,
        alpha_od=0.3,
    ):
        super().__init__()
        if not 0.0 <= alpha_od <= 1.0:
            raise ValueError("alpha_od must be in [0, 1]")
        self.alpha_od = float(alpha_od)
        self.count_encoder = CountEncoder(latent_dim)
        self.od_encoder = ODGATEncoder(
            node_feature_dim=node_feature_dim,
            edge_feature_dim=edge_feature_dim,
            latent_dim=latent_dim,
        )
        self.decoder = PoissonDecoder(latent_dim)

    def encode_count(self, x):
        return self.count_encoder(x)

    def encode_od(self, graph, training_graph=True):
        edge_index = (
            graph.train_edge_index if training_graph else graph.full_edge_index
        )
        edge_features = (
            graph.train_edge_features
            if training_graph
            else graph.full_edge_features
        )
        return self.od_encoder(
            graph.node_features,
            edge_index,
            edge_features,
            graph.node_patch,
            graph.n_patches,
        )

    def fuse(self, z_count, z_od, has_od):
        effective_alpha = has_od.to(z_count.dtype).unsqueeze(-1) * self.alpha_od
        return (1.0 - effective_alpha) * z_count + effective_alpha * z_od

    def decode(self, z):
        return self.decoder(z)


def poisson_nll(log_lam, x):
    cell = torch.exp(log_lam) - x * log_lam
    return cell.mean(dim=1)


def poisson_deviance(log_lam, x):
    lam = torch.exp(log_lam)
    cell = 2 * (torch.xlogy(x, x) - x * log_lam - x + lam)
    return cell.mean(dim=1)


def build_fsce_graph(x, n_neighbors=15, metric="cosine"):
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
    edge_w = torch.from_numpy(graph.data).float()
    a, b = find_ab_params(spread=1.0, min_dist=0.1)
    return edge_i, edge_j, edge_w, a, b


def fsce_loss(z_i, z_j, weight, a, b, eps=1e-4):
    distance_squared = (z_i - z_j).pow(2).sum(dim=1).clamp(min=eps)
    similarity = (1.0 + a * distance_squared.pow(b)).reciprocal()
    similarity = similarity.clamp(eps, 1.0 - eps)
    return -(
        weight * similarity.log()
        + (1.0 - weight) * (1.0 - similarity).log()
    )
