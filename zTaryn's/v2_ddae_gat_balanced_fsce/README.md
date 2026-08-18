# v2_ddae_gat_balanced_fsce

This experiment replaces the failed linear composition head with a loss that
acts directly on the geometry of the fused two-dimensional latent.

It keeps:

- denoising count encoder;
- Poisson decoder and clean count target;
- directed edge-aware OD-GAT;
- `ALPHA_OD=0.3`;
- `WEIGHT_DECAY=0`.

The unchanged model/graph components are imported from `../v2_ddae_gat/ae.py`.

## Why this differs from the composition-head version

The composition head could predict a smooth Dining proportion along one axis
without forming category neighborhoods.  It also remained dominated by the
large number of Dining patches.

This version has no composition classifier.  Instead, each optimizer step
builds part of an implicit fuzzy graph from training patches only:

1. sample each POI category equally often;
2. sample a patch with probability proportional to its soft membership in the
   selected category;
3. form equal numbers of same-category and cross-category pairs;
4. calculate a continuous similarity from the two complete compositions;
5. use fuzzy cross-entropy to match that similarity with latent distance.

Mixed patches can be sampled under several categories and keep a continuous
pair membership.  They are not forced into one hard class.

## Prior-corrected Hellinger membership

Ordinary composition similarity is still dominated by Dining because Dining
appears in almost every patch.  Training-only composition prior `prior[c]` is
removed before pair similarity:

```text
h_i[c] = sqrt(p_i[c] / prior[c])
h_i    = L2_normalize(h_i)
membership(i, j) = dot(h_i, h_j) ** MEMBERSHIP_POWER
```

Default `MEMBERSHIP_POWER=2.0` increases contrast while keeping membership in
`[0, 1]`.  In the current data, balanced sampling produces approximately:

```text
same-category median membership  >  cross-category median membership
```

The exact medians are printed before training.

## Loss

```text
total_loss = reconstruction_loss + scheduled_BETA_PAIR * fuzzy_pair_loss
```

Defaults:

- `BETA_PAIR=0.1`
- pair warm-up: 100 epochs
- category sampling power: `0.75`
- `MEMBERSHIP_POWER=2.0`

This replaces the original log-count kNN FSCE graph; it is not added on top of
that graph.  Reconstruction continues to preserve total POI count information.

## Leakage controls

- Split patches before estimating the composition prior or sampler.
- Pair sampling contains training patches only.
- Composition prior is estimated from training patches only.
- GAT optimizer steps use only edges whose endpoints are both training
  patches.
- Validation counts and full OD topology are used only in evaluation mode.

## Train

```powershell
python model/v2_ddae_gat_balanced_fsce/train.py
```

Short smoke test:

```powershell
python model/v2_ddae_gat_balanced_fsce/train.py --epochs 1 --no-save
```

Recommended coefficient ablations can be preserved separately:

```powershell
python model/v2_ddae_gat_balanced_fsce/train.py --beta-pair 0.05 --output-tag beta005
python model/v2_ddae_gat_balanced_fsce/train.py --beta-pair 0.10 --output-tag beta010
```

The default run writes `result/ae.pt` and `result/latents.npz`.  A tagged run
writes filenames such as `ae_beta005.pt` and `latents_beta005.npz`.

Training reports:

- clean validation NLL and Poisson deviance;
- held-out latent kNN macro-F1;
- confident-patch macro-F1;
- non-Dining neighbor Dining fraction;
- PC1 variance ratio, to detect one-dimensional latent collapse.

## Compare results

After the default full run:

```powershell
python model/v2_ddae_gat_balanced_fsce/analyze/compare_latents.py
```

It compares the original GAT, failed composition-head model, and the new
balanced-pair model with the same saved split.  Outputs:

- `result/balanced_fsce_comparison.png`
- `result/balanced_fsce_metrics.json`
