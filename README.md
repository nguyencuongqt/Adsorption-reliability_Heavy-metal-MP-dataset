# Adsorption Reliability Study: Grouped Validation and Null-Anchored Interpretation

This folder is the final active project in this repository. It focuses on reliability of prediction and reliability of interpretation under grouped literature-derived environmental data, while reusing the stronger legacy training pipeline for feature engineering and train-fold preprocessing.

## Research Questions

- `RQ1.` How does validation design under grouped literature-derived data affect estimated predictive performance and out-of-domain generalization?
- `RQ2.` How sensitive are mechanistic conclusions to the choice of feature-importance method, and can synthetic null features identify more trustworthy interpretation methods?

## Core Hypotheses

- `H1.` Random validation overestimates predictive performance relative to experiment-level and study-level grouped validation.
- `H2.` Predictive performance declines systematically from row-level interpolation to experiment-level and study-level transfer.
- `H3.` Different model classes respond differently to the hierarchy of generalization difficulty.
- `H4.` Different feature-importance methods yield materially different feature and mechanism-group rankings.
- `H5.` Some importance methods assign non-trivial importance to synthetic null features, indicating susceptibility to spurious interpretation.
- `H6.` Methods that suppress null features and maintain stable rankings across resampling and grouped validation provide more trustworthy evidence of predictive relevance, not causality.

## Generalization Regimes

- `random_cv`: row-level interpolation within the observed literature domain
- `group_exp`: transfer to unseen experiments while some study context remains represented
- `group_aut`: transfer to unseen studies
- `leave_one_study_out`: optional stronger stress test for study-level domain shift

Grouped validation matters because this dataset is hierarchical: rows are nested inside experiments and studies, so naive random splitting can leak study fingerprints and inflate apparent performance.

## Interpretation Philosophy

Feature importance is method-sensitive and model-sensitive. Importance can indicate predictive relevance under a particular model and validation regime, but it is not causal evidence. This study therefore stress-tests interpretation with multiple methods and synthetic null features.

## Fresh Inputs Used

- [05_modeling_dataset_final.csv](G:\My Drive\Multi_metal MP prediction\adsorption_reliability_study\data\inputs\05_modeling_dataset_final.csv)
- [ml_feature_inventory.csv](G:\My Drive\Multi_metal MP prediction\adsorption_reliability_study\data\inputs\ml_feature_inventory.csv)
- [author_year_registry.csv](G:\My Drive\Multi_metal MP prediction\adsorption_reliability_study\data\inputs\author_year_registry.csv)

## Reproducible Runs

- `python scripts/01_run_rq1.py`
- `python scripts/02_run_rq2.py`
- `python scripts/03_run_full_study.py`

## Environment Setup

Install public Python dependencies:

```powershell
pip install -r requirements.txt
```

This project also reuses legacy code from `ml_benchmark`, which is not bundled inside this repository. Before running the study scripts, make sure the legacy `ml_benchmark` source is available in your broader workspace and importable from Python.

## Output Layout

- `results/tables/`: machine-readable result tables
- `results/figures/`: manuscript-support figures
- `results/logs/`: run manifests and hashes
- `docs/`: project notes and discussion notes

## Git Workflow

Basic daily workflow in this repository:

```powershell
git pull
git status
git add .
git commit -m "Short description of the change"
git push
```

Create a new branch for a focused change:

```powershell
git checkout -b feature/short-name
```

Switch back to the main branch:

```powershell
git checkout master
git pull
```

Useful checks:

- `git status`: see changed files
- `git log --oneline -5`: see recent commits
- `git remote -v`: confirm the connected GitHub repository
