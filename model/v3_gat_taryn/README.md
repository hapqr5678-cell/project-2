# v2_ddae_fsce_gat

This version adds a POI-level OD-GAT branch to the `weight_decay=0`
`v2_ddae_fsce_taryn` count model.

## Architecture

```text
noisy patch POI counts -> 4-layer count encoder -> z_count ---+
                                                            |
POI nodes + directed OD edges -> 2-layer edge-aware GAT      |
                           -> patch attention pooling -> z_od +
                                                            |
z = (1 - ALPHA_OD) * z_count + ALPHA_OD * z_od <------------+
                           |
                    Poisson decoder
                           |
                  clean patch POI counts
```

The fused latent is also optimized by the training-only FSCE loss.

## OD features

POI node features:

- FSQ top-level category one-hot;
- standardized `log1p(checkin_count)`.

OD edge features:

- `log1p_trip_count`;
- `log1p_unique_user_count`;
- `log1p(distance_m)`;
- `log1p(median_travel_minutes)`;
- direction (`+1` for the observed direction, `-1` for its reverse message,
  `0` for a self-loop).

Feature normalization statistics are calculated from training patches only.

## Leakage controls

- Split patches before building the FSCE and OD training views.
- FSCE graph: training patches only.
- OD training graph: edges whose origin and destination patches are both train.
- Validation counts and OD topology never update parameters.
- Full OD graph is used only for eval-mode validation and final inference.

## Train

```powershell
python model/v2_ddae_fsce_gat/train.py
```

Useful checks:

```powershell
python model/v2_ddae_fsce_gat/train.py --epochs 1 --no-save
python model/v2_ddae_fsce_gat/train.py --alpha-od 0.1
```

The default is `ALPHA_OD=0.3` and `WEIGHT_DECAY=0`.

`latents.npz` stores the fused `z`, branch outputs `z_count` and `z_od`, clean
Poisson deviance `err`, and explicit `is_train` / `is_val` masks.
