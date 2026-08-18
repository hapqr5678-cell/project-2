# AE diagnostic benchmark

This directory contains a read-only diagnostic of the existing strict-2D count
autoencoders. It does not retrain or overwrite any model.

Run from the repository root:

```powershell
& "C:\Users\楊仟愉\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" research/ae_diagnostic/diagnose.py
```

The script requires only NumPy. Generated artifacts are placed in `output/`:

- `report.md`: human-readable conclusions and next experiment gate
- `diagnostics.json`: all machine-readable measurements
- `model_comparison.csv`: paired bootstrap comparison on one saved mask
- `pca_rate_distortion.csv` and `rate_distortion.svg`: dimensionality test
- `training_log_stability.csv`: best/final/tail statistics parsed from logs

The PCA dimensions above two are diagnostic controls only. They do not change
the requirement that a production latent representation remain two-dimensional.
