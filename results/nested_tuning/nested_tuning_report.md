# Nested tuning sensitivity analysis

## Design

The original outer splits were retained unchanged. Hyperparameters were selected
within each outer-training set only. The inner resampling structure matched the
outer regime: shuffled row-level K-fold CV for RRCV, GroupKFold by `exp_id` for
EGCV, and GroupKFold by `aut_id` for SGCV. Selection minimized RMSE on the
original `qe` scale. Imputation, scaling, target clipping, and model fitting were
refit inside every inner-training fold.

## Original versus nested-tuned performance

| Regime | Model | Original RMSE (mean +/- SD) | Tuned RMSE (mean +/- SD) | Original R2 (mean +/- SD) | Tuned R2 (mean +/- SD) |
|---|---|---:|---:|---:|---:|
| RRCV | EN | 2.146 +/- 0.293 | 1.591 +/- 0.251 | 0.013 +/- 0.033 | 0.459 +/- 0.052 |
| RRCV | LGBM | 0.762 +/- 0.224 | 0.655 +/- 0.143 | 0.871 +/- 0.070 | 0.905 +/- 0.031 |
| RRCV | MLP | 0.618 +/- 0.118 | 0.935 +/- 0.260 | 0.912 +/- 0.039 | 0.803 +/- 0.093 |
| EGCV | EN | 2.028 +/- 0.879 | 1.871 +/- 1.007 | 0.016 +/- 0.097 | 0.177 +/- 0.347 |
| EGCV | LGBM | 0.969 +/- 0.387 | 1.123 +/- 0.585 | 0.725 +/- 0.203 | 0.658 +/- 0.189 |
| EGCV | MLP | 1.097 +/- 0.537 | 2.164 +/- 2.149 | 0.703 +/- 0.120 | -0.252 +/- 1.831 |
| SGCV | EN | 1.674 +/- 1.671 | 1.712 +/- 1.622 | -0.905 +/- 1.444 | -2.381 +/- 3.847 |
| SGCV | LGBM | 1.906 +/- 1.460 | 1.799 +/- 1.548 | -9.755 +/- 14.157 | -8.390 +/- 17.004 |
| SGCV | MLP | 19.481 +/- 22.380 | 3.618 +/- 3.831 | -21744.538 +/- 48263.885 | -54.893 +/- 109.536 |

## RRCV-to-SGCV optimism gap

| Model | Original RMSE gap (SGCV - RRCV) | Tuned RMSE gap | Original SGCV/RRCV ratio | Tuned ratio | Change in gap after tuning (95% bootstrap CI) |
|---|---:|---:|---:|---:|---:|
| EN | -0.473 | 0.121 | 0.780 | 1.076 | 0.594 (0.496, 0.687) |
| LGBM | 1.143 | 1.144 | 2.500 | 2.747 | 0.001 (-0.238, 0.201) |
| MLP | 18.862 | 2.683 | 31.506 | 3.870 | -16.208 (-34.591, 1.389) |

The bootstrap comparison used paired tuned-minus-original RMSE changes within
each regime and independently resampled the 15 RRCV and five SGCV outer-fold
changes. A confidence interval excluding zero was treated as evidence that
tuning changed the gap.

## Interpretation

Regime-specific nested tuning did not remove the generalization gap. For LGBM,
the absolute RMSE gap was effectively unchanged, because tuning improved RRCV
and SGCV by similar amounts. For MLP, tuning removed the two extreme legacy
collapse values and substantially reduced the point estimate of the gap, but
the tuned SGCV RMSE remained 3.87 times the tuned RRCV RMSE. With only five
study folds and high fold-to-fold variability, the estimated reduction in the
MLP gap was not statistically distinguishable from zero. EN improved strongly
under RRCV but not SGCV, causing its gap to increase rather than disappear.

Thus, fixed hyperparameters contributed to the severity of the legacy MLP
collapse, but they do not explain the broader RRCV-to-SGCV optimism pattern.
The especially stable LGBM result provides the clearest sensitivity evidence:
the SGCV penalty persisted almost exactly after fully nested,
regime-matched tuning.

## Important limitation

The external `ml_benchmark` package that constructed the original fixed model
specifications was not available in the current environment. This analysis
retained the locked 25-feature dataset, original outer splits, fold-wise
preprocessing, and target transformation, but used explicit prespecified
search grids implemented in the standalone nested-tuning runner. It should be
described as a nested-tuning sensitivity analysis, not as an exact
hyperparameter-by-hyperparameter reconstruction of the unavailable legacy
model builder.
