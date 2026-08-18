# v3_ddae_moe_test2

This version keeps a strict two-dimensional latent and uses a two-stage,
competitive mixture of residual composition experts.

```text
counts -> encoder -> z (exactly 2D)
                       |-> shared total decoder -> log(total rate)
                       |-> base composition decoder
                       |-> residual composition expert 0
                       |-> residual composition expert 1
                       +-> router
```

Every candidate rate has the form:

```text
lambda_e = shared_total * softmax(base_logits + residual_e)
```

Consequently, experts cannot cancel one another by predicting different total
counts, and every expert always emits a complete category distribution.

Training has two stages:

1. Train the factorized single-decoder baseline and restore its lowest
   validation-deviance checkpoint.
2. Cluster its 2D latent, freeze the baseline for 75 expert-specialization
   epochs, then use hard-EM assignments and low-learning-rate joint tuning.

The saved model is automatically selected from the baseline, hard-routed MoE,
and soft-routed MoE according to validation deviance. If MoE is worse, the
baseline checkpoint is saved instead.

Full run:

```powershell
python model/v3_ddae_moe_test2/train.py
```

Smoke test:

```powershell
python model/v3_ddae_moe_test2/train.py --base-epochs 2 --moe-epochs 2 --no-save
```

The normal MoE run should use at least 76 MoE epochs so it reaches the hard-EM
phase. After training:

```powershell
python model/v3_ddae_moe_test2/analyze/diagnostics.py
```

`latents.npz` stores the strict `(N, 2)` latent, all reconstruction modes,
per-expert errors and rates, gates, split masks, selected mode, and best metrics.
