# Project Notes

## Design Decisions

- The fresh study reuses only cleaned inputs and limited metadata from the legacy repository.
- The target is modeled as `log1p(qe)` to handle heavy right skew while reporting metrics back on the original `qe` scale.
- The primary scientific question is reliability under hierarchical validation, not best-score benchmarking.
- The study keeps the reliability-focused framing but reuses the stronger legacy training pipeline for feature engineering and preprocessing; this includes legacy engineered features plus synthetic null features for RQ2.
- All imputation and scaling are fit on training folds only.
- The optional stress-test regime is `leave_one_study_out`, which is more severe than five-fold grouped study CV.
- `leave_one_study_out` is retained as an auxiliary stress test only; the main monotonic performance comparison for H1-H3 should rely primarily on `random_cv`, `group_exp`, and `group_aut`.
- Mechanism-group summaries are treated as predictive interpretation aids, not causal claims.
