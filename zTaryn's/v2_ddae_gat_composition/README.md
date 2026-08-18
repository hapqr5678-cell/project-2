# v2_ddae_gat_composition

This experiment tests whether the Dining overlap is reduced when the fused
two-dimensional latent is explicitly supervised with each patch's complete
POI category composition.

It keeps the DDAE + FSCE + OD-GAT architecture and `WEIGHT_DECAY=0`.  The
unchanged graph/model components are imported from `../v2_ddae_gat/ae.py`;
all experiment-specific code and outputs live in this new folder.

## Architecture

```text
noisy category counts -> count encoder -> z_count --------+
                                                         |
POI nodes + directed OD graph -> GAT -> z_od ------------+
                                                         |
                z = (1 - ALPHA_OD) z_count + ALPHA_OD z_od
                              |
                +-------------+----------------+
                |                              |
        Poisson decoder              linear composition head
                |                              |
       clean category counts            10-category softmax
                                               |
                                  clean category proportions
```

The composition target is not a hard dominant label:

```text
target[c] = clean_count[c] / sum(clean_count)
```

The linear head is intentional.  It forces category-composition information
to be readable from the two-dimensional latent instead of allowing another
deep network to hide the separation.

## Loss

```text
total_loss = reconstruction_loss
           + scheduled_FSCE_weight * FSCE_loss
           + scheduled_BETA_COMPOSITION * balanced_soft_composition_loss
```

Default `BETA_COMPOSITION=0.5`, warmed up over 50 epochs.  Composition class
weights use square-root inverse count frequency, are clipped for stability,
and are estimated from the training split only.  They weight each complete
patch loss according to its composition; they do not alter the proportions
inside the soft target, so the loss remains calibrated to the clean mixture.

`MIXED_MARGIN=0.15` affects evaluation only.  A patch is reported as Mixed
when its largest and second-largest category proportions differ by less than
0.15.  The model is still trained against its full soft composition.

## Leakage controls

- Patch split happens before either graph or class weight is constructed.
- FSCE graph uses training patches only.
- GAT optimizer steps use only OD edges whose two endpoints are training
  patches.
- Composition weights use training counts only.
- Clean validation counts are used only for metrics.
- The full OD graph is used only in evaluation mode.

## Train

```powershell
python model/v2_ddae_gat_composition/train.py
```

Short smoke test without saving:

```powershell
python model/v2_ddae_gat_composition/train.py --epochs 1 --no-save
```

Optional ablation:

```powershell
python model/v2_ddae_gat_composition/train.py --beta-composition 0.25
```

During training it reports clean validation NLL/deviance, composition JS,
hard-label macro-F1, confident-only macro-F1, Mixed-aware macro-F1, and the
fraction of non-Dining patches predicted as Dining.

## Analyze

After a full training run:

```powershell
python model/v2_ddae_gat_composition/analyze/latent_composition_analysis.py
```

The analysis compares the new latent with the original GAT using the same
saved train/validation mask.  It writes:

- `result/latent_composition_analysis.png`
- `result/composition_metrics.json`

`latents.npz` also keeps `composition_target`, `composition_pred`, per-patch
Jensen-Shannon error, target/predicted margins, Mixed masks, class weights,
and the explicit split masks.
