# v3_ddae_moe_test

This experiment targets lower Poisson deviance without destabilizing the
two-dimensional FSCE latent space.

## Design

```text
count -> baseline encoder -> z -> baseline decoder -> base log(lambda)
                              \
                               -> router + 2 residual experts

log(lambda) = base log(lambda) + 0.1 * gated residual
```

- Warm-start from `model/v2_ddae_fsce/result/ae.pt`.
- Zero-initialized expert output layers make epoch 0 exactly the baseline.
- Freeze the baseline for 50 epochs, then fine-tune everything.
- Use `LR=3e-4` for MoE warm-up and `LR=1e-4` with cosine decay afterward.
- Use train-only FSCE edges and reduce `LAMBDA_FSCE` to `0.005`.
- Restore and save the lowest-validation-deviance checkpoint, including epoch 0.

## Run

```powershell
python model/v3_ddae_moe_test/train.py
```

Quick pipeline check:

```powershell
python model/v3_ddae_moe_test/train.py --epochs 2 --no-save
```

After training, compare against the baseline:

```powershell
python model/v3_ddae_moe_test/analyze/compare.py
```

`latents.npz` includes `gates`, `expert_id`, train/validation masks,
`best_epoch`, and `best_val_deviance`.
