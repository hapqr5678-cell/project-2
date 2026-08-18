# v3_ddae_mot

This version enforces a strict two-dimensional bottleneck:

```text
noisy counts -> 4-layer encoder -> z (2D) -> shared decoder -> base log(lambda)
                                     |       residual expert 1 -> correction
                                     |       residual expert 2 -> correction
                                     +-----> router ------------> gate weights
```

- `z` is the model's only latent representation.
- FSCE, the shared decoder, the router, and every expert see only this 2D `z`.
- There is no reconstruction-only latent or hidden-state skip connection.
- Expert output layers start at zero for stable residual learning.
- The FSCE graph is built from training patches only.
- Base parameters use `LR=1e-3`; router/experts use `LR=3e-4`.
- Training restores the checkpoint with the lowest validation deviance.

Run:

```powershell
python model/v3_ddae_mot/train.py
```

Quick check:

```powershell
python model/v3_ddae_mot/train.py --epochs 2 --no-save
```

`latents.npz` stores the 2D `z`, `gates`, `expert_id`, train/validation masks,
and the best epoch/deviance.
