"""Train a DAE with Poisson reconstruction and fuzzy latent regularization.

The reconstruction branch is identical to v2_dae: binomial-thinned counts are
encoded and decoded toward the clean count vector. Separately, an exact fuzzy
kNN graph is built from log1p(clean count) over the training split only. Clean
graph endpoints are encoded and optimized with UMAP-style fuzzy cross entropy.
"""

import os
import sys

import numpy as np
import torch
from torch.distributions import Binomial

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(f"{os.path.dirname(__file__)}/../.."))
from ae import (  # noqa: E402
    MLPAE,
    Patches,
    build_fuzzy_graph,
    fuzzy_set_cross_entropy,
    poisson_deviance,
    poisson_nll,
    sample_edge_batch,
)
from config.dataset import ensure_patches, PATCHES, result  # noqa: E402

VERSION = "v2_dae_fuzzy"
OUT = result(VERSION, "latents.npz")
CKPT = result(VERSION, "dae_fuzzy.pt")

LATENT_DIM = 2
EPOCHS = 1000
BATCH = 256
LR = 1e-3
VAL_FRAC = 0.1
SEED = 0
KEEP_PROB = 0.8

N_NEIGHBORS = 15
NEGATIVE_SAMPLE_RATE = 5
FUZZY_BATCH = BATCH
ALPHA = 0.1
UMAP_A = 1.576943460
UMAP_B = 0.895060879

device = (
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)


def binomial_thinning(x, keep_prob):
    """Use binomial thinning to create a noisy DAE input."""
    try:
        distribution = Binomial(
            total_count=x,
            probs=torch.tensor(keep_prob, device=x.device, dtype=torch.float32),
        )
        return distribution.sample()
    except Exception:
        # PyTorch may not implement Binomial on MPS; sample on CPU in that case.
        distribution = Binomial(
            total_count=x.cpu(),
            probs=torch.tensor(keep_prob, device="cpu", dtype=torch.float32),
        )
        return distribution.sample().to(x.device)


def fuzzy_loss_for_batch(model, train_clean, graph, generator):
    """Sample global graph edges and compute CE on clean-input latents."""
    pos_head, pos_tail, neg_head, neg_tail = sample_edge_batch(
        graph,
        n_positive=FUZZY_BATCH,
        negative_sample_rate=NEGATIVE_SAMPLE_RATE,
        generator=generator,
    )

    node_ids = torch.cat((pos_head, pos_tail, neg_head, neg_tail))
    unique_nodes, inverse = torch.unique(node_ids, return_inverse=True)
    z_unique = model.encoder(train_clean[unique_nodes].to(device))
    z = z_unique[inverse.to(device)]

    n_pos = len(pos_head)
    n_neg = len(neg_head)
    z_pos_head = z[:n_pos]
    z_pos_tail = z[n_pos:2 * n_pos]
    z_neg_head = z[2 * n_pos:2 * n_pos + n_neg]
    z_neg_tail = z[2 * n_pos + n_neg:]
    return fuzzy_set_cross_entropy(
        z_pos_head,
        z_pos_tail,
        z_neg_head,
        z_neg_tail,
        a=UMAP_A,
        b=UMAP_B,
    )


def run(data, train_idx, val_idx):
    torch.manual_seed(SEED)
    model = MLPAE(LATENT_DIM).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    # Local graph node indices refer to rows of train_clean, never val_idx.
    train_clean = data.agg(train_idx)
    graph_features = torch.log1p(train_clean)
    graph = build_fuzzy_graph(graph_features, n_neighbors=N_NEIGHBORS)
    print(
        f"fuzzy graph: {graph.n_nodes} training nodes, {graph.n_edges} edges, "
        f"k={N_NEIGHBORS}, transform=log1p"
    )
    edge_generator = torch.Generator().manual_seed(SEED)

    for epoch in range(EPOCHS):
        model.train()
        shuffle_generator = torch.Generator().manual_seed(SEED + epoch)
        perm = train_idx[
            torch.randperm(len(train_idx), generator=shuffle_generator)
        ]

        total_sum = 0.0
        recon_sum = 0.0
        fuzzy_sum = 0.0
        attraction_sum = 0.0
        repulsion_sum = 0.0
        n_steps = 0

        for i in range(0, len(perm), BATCH):
            opt.zero_grad()
            batch = perm[i:i + BATCH]
            x_clean = data.agg(batch).to(device)
            x_noisy = binomial_thinning(x_clean, KEEP_PROB)

            _, log_lam = model(x_noisy)
            recon_loss = poisson_nll(log_lam, x_clean).mean()
            fuzzy_loss, attraction, repulsion = fuzzy_loss_for_batch(
                model, train_clean, graph, edge_generator
            )
            total_loss = recon_loss + ALPHA * fuzzy_loss

            loss_values = torch.stack((
                total_loss.detach(),
                recon_loss.detach(),
                fuzzy_loss.detach(),
                attraction.detach(),
                repulsion.detach(),
            ))
            if not torch.isfinite(loss_values).all():
                raise FloatingPointError(
                    f"non-finite loss at epoch={epoch + 1}, batch_start={i}: "
                    f"total={total_loss.item()}, recon={recon_loss.item()}, "
                    f"fuzzy={fuzzy_loss.item()}, pull={attraction.item()}, "
                    f"push={repulsion.item()}"
                )

            total_loss.backward()
            for name, parameter in model.named_parameters():
                if (
                    parameter.grad is not None
                    and not torch.isfinite(parameter.grad).all()
                ):
                    raise FloatingPointError(
                        f"non-finite gradient at epoch={epoch + 1}, "
                        f"batch_start={i}, parameter={name}"
                    )
            opt.step()

            batch_size = len(batch)
            total_sum += total_loss.item() * batch_size
            recon_sum += recon_loss.item() * batch_size
            fuzzy_sum += fuzzy_loss.item()
            attraction_sum += attraction.item()
            repulsion_sum += repulsion.item()
            n_steps += 1

        if (epoch + 1) % 20 == 0 or epoch == 0:
            model.eval()
            with torch.no_grad():
                val_nll, val_deviance = [], []
                for i in range(0, len(val_idx), BATCH):
                    batch = val_idx[i:i + BATCH]
                    x_clean = data.agg(batch).to(device)
                    _, log_lam = model(x_clean)
                    val_nll.append(poisson_nll(log_lam, x_clean))
                    val_deviance.append(poisson_deviance(log_lam, x_clean))
                val = torch.cat(val_nll).mean().item()
                deviance = torch.cat(val_deviance).mean().item()

            print(
                f"  epoch {epoch + 1:4d}  "
                f"train total {total_sum / len(perm):.5f}  "
                f"recon {recon_sum / len(perm):.5f}  "
                f"fuzzy {fuzzy_sum / n_steps:.5f} "
                f"(pull {attraction_sum / n_steps:.5f}, "
                f"push {repulsion_sum / n_steps:.5f})  "
                f"clean val NLL {val:.5f}  "
                f"clean val deviance {deviance:.5f}"
            )

    torch.save(model.state_dict(), CKPT)

    model.eval()
    latents, errors = [], []
    with torch.no_grad():
        for i in range(0, data.n, BATCH):
            idx = torch.arange(i, min(i + BATCH, data.n))
            x_clean = data.agg(idx).to(device)
            z, log_lam = model(x_clean)
            latents.append(z.cpu())
            errors.append(poisson_deviance(log_lam, x_clean).cpu())
    return torch.cat(latents).numpy(), torch.cat(errors).numpy()


def main():
    ensure_patches()
    data = Patches(PATCHES)
    print(f"{data.n} patches, device={device}")

    generator = torch.Generator().manual_seed(SEED)
    perm = torch.randperm(data.n, generator=generator)
    n_val = int(data.n * VAL_FRAC)
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    z, err = run(data, train_idx, val_idx)
    np.savez(
        OUT,
        n_poi=data.n_poi,
        lat=data.lat,
        lon=data.lon,
        z=z,
        err=err,
    )
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
