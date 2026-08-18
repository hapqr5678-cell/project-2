# AE quantitative diagnostic report

Generated from 1233 patches; validation n=123 (saved masks from v3_ddae_moe_test2).

## Decision summary

1. Treat the current 0.61387 result as a transductive/single-run reference, not an inductive target, until it is reproduced with train-only graph construction and multiple seeds.
2. The data intrinsic dimension is about 4.89 at k=15, while the required latent is 2-D.
3. Linear log-count PCA improves from 1.0162 at 2-D to 0.7994 at 4-D. This is evidence for a bottleneck before the decoder.
4. In old test2, exact observed-total rescaling changes deviance from 0.6900 to 0.6601; total and composition errors should be modeled and reported separately.
5. The saved MoE expert oracle is 0.9594, versus routed 0.6900; these experts do not contain a hidden better solution for the router to discover.

## Existing model comparison (same saved validation mask)

Lower deviance is better. CI is a paired patch bootstrap against the currently saved v2 checkpoint.

| model | val deviance | delta vs v2 | 95% CI | P(better) | trust | continuity | kNN overlap |
|---|---:|---:|---:|---:|---:|---:|---:|
| v3_ddae_moe_test | 0.63979 | -0.02184 | [-0.0745, 0.0318] | 0.794 | 0.798 | 0.848 | 0.167 |
| v2_ddae_fsce | 0.66163 | +0.00000 | reference | - | 0.840 | 0.843 | 0.220 |
| v3_ddae_mot | 0.67431 | +0.01268 | [-0.0392, 0.0621] | 0.306 | 0.818 | 0.851 | 0.198 |
| v3_ddae_moe_test2 | 0.69002 | +0.02838 | [-0.0426, 0.1257] | 0.269 | 0.803 | 0.852 | 0.169 |
| v3_gat_taryn | 0.75060 | +0.08897 | [0.0311, 0.1461] | 0.001 | 0.752 | 0.825 | 0.118 |
| v2_ae | 0.78565 | +0.12402 | [0.0466, 0.2134] | 0.001 | 0.772 | 0.848 | 0.118 |
| v2_dae | 0.80870 | +0.14707 | [0.0773, 0.2256] | 0.000 | 0.787 | 0.848 | 0.144 |
| v2_dae_fuzzy | 0.81002 | +0.14839 | [0.0747, 0.2417] | 0.000 | 0.795 | 0.853 | 0.163 |

## Rate-distortion diagnostic

Other dimensions are diagnostics only; the production representation can remain 2-D.

| dimension | explained variance | validation Poisson deviance |
|---:|---:|---:|
| 1 | 0.165 | 1.23043 |
| 2 | 0.312 | 1.01621 |
| 3 | 0.425 | 0.98182 |
| 4 | 0.526 | 0.79945 |
| 5 | 0.625 | 0.76583 |
| 6 | 0.717 | 0.54709 |
| 7 | 0.802 | 0.39377 |
| 8 | 0.880 | 0.24018 |
| 9 | 0.949 | 0.13329 |
| 10 | 1.000 | 0.00000 |

## Count diagnostics

- Category marginal variance/mean range: 1.013–4.590.
- Total-count variance/mean: 5.978.
- Conditional Pearson dispersion range under old test2: 0.367–6.309.
- Deviance share: total 4.3%, composition 95.7%.
- Validation gate usage: [0.34 0.66]; entropy 0.554.

## All-data graph contamination surface

- 20.1% of directed kNN edges touch validation.
- 11.2% of neighbors attached to a training source are validation points.
- With the old all-data negative sampler, a random pair touches validation with probability 19.0%.
- These are valid transductive edges only if the intended task is embedding this one fixed complete dataset. They are contamination for inductive validation.

## Next experiment gate

- First reproduce the train-only-graph 2-D baseline with 5 seeds and one untouched test split. Report mean, SD, and paired CI; do not select models by the minimum of the same validation curve.
- Add a dimension sweep to the same trainer. If the 2-D to 4-D gap persists across seeds, decoder MoE is not the next priority; test a multi-chart 2-D representation.
- Measure encoder gradient cosine between reconstruction and FSCE before adding PCGrad. Only use gradient surgery if negative-gradient frequency is substantial.
- Compare Poisson and negative-binomial/factorized total-composition decoders only after the protocol above is fixed.

## Important limitations

- Saved checkpoints come from different training protocols and some from different devices. This report can diagnose failure modes but cannot certify a winning architecture.
- Exact-total rescaling uses the observed target total and is therefore an oracle diagnostic, not a fair deployable model unless total count is explicitly allowed as an input side channel.
- Marginal variance/mean does not by itself prove conditional negative-binomial dispersion.
