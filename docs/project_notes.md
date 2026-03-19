# Project Notes

## Design Decisions

- The fresh study reuses only cleaned inputs and limited metadata from the legacy repository.
- The target is modeled as `log1p(qe)` to handle heavy right skew while reporting metrics back on the original `qe` scale.
- The primary scientific question is reliability under hierarchical validation, not best-score benchmarking.
- The study keeps the reliability-focused framing but reuses the stronger legacy training pipeline for feature engineering and preprocessing; this includes legacy engineered features plus synthetic null features for RQ2.
- All imputation and scaling are fit on training folds only.
- Mechanism-group summaries are treated as predictive interpretation aids, not causal claims.
