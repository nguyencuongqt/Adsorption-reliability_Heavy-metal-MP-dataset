# Reliability of Heavy-Metal Adsorption Models for Microplastics

This repository contains the analysis dataset, source code, configuration, and
machine-readable results supporting a study of predictive and interpretive
reliability in literature-derived adsorption data.

The dataset contains 1,009 observations from 149 experiments nested within
23 studies.

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

Grouped validation matters because this dataset is hierarchical: rows are nested inside experiments and studies, so naive random splitting can leak study fingerprints and inflate apparent performance.

## Interpretation Philosophy

Feature importance is method-sensitive and model-sensitive. Importance can indicate predictive relevance under a particular model and validation regime, but it is not causal evidence. This study therefore stress-tests interpretation with multiple methods and synthetic null features.

## Repository contents

- `data/inputs/05_modeling_dataset_final.csv`: locked modeling dataset
- `data/inputs/ml_feature_inventory.csv`: feature inventory
- `data/inputs/author_year_registry.csv`: study registry
- `configs/study_config.json`: analysis configuration
- `src/adsorption_reliability_study/`: analysis modules
- `scripts/`: command-line runners
- `results/tables/`: primary machine-readable results
- `results/nested_tuning/`: nested-tuning sensitivity results

## Environment setup

Create the Conda environment:

```powershell
conda env create -f environment.yml
conda activate adsorption-reliability-study
```

Alternatively, install the dependencies with pip:

```powershell
python -m pip install -r requirements.txt
```

## Reproducible sensitivity analysis

The nested-tuning analysis is self-contained within this repository:

```powershell
python scripts/13_run_nested_tuning.py
```

It retains the locked dataset and outer validation splits, while selecting
hyperparameters only within each outer-training set. Outputs are written to
`results/nested_tuning/`.

## Archived primary analysis

The primary RQ1 and RQ2 tables and figures are retained as analysis artifacts.
Their original fixed-model pipeline used an external legacy model builder that
is not distributed in this repository. Therefore, the commands
`scripts/01_run_rq1.py` through `scripts/03_run_full_study.py` document the
original workflow but are not presented as a fully standalone reconstruction.
The nested-tuning runner above is the independently executable sensitivity
analysis.

## Validation design

- `random_cv`: repeated row-level cross-validation
- `group_exp`: grouped cross-validation by experiment
- `group_aut`: grouped cross-validation by study

All reported imputation, scaling, target transformation, and model fitting are
performed within training folds. Feature importance is interpreted as
predictive relevance under a specified model and validation design, not as
causal evidence.

## Core results

- Row-level validation produced more optimistic performance than study-level
  transfer for the nonlinear models.
- Study-level transfer was weak and variable across model classes.
- Importance rankings varied across models, methods, and validation regimes.
- Synthetic null features were used to audit spurious attribution.

See `results/tables/` for the primary summaries and
`results/nested_tuning/nested_tuning_report.md` for the tuning sensitivity
analysis.

## Scope

Submission documents, final figures, working notes, copyrighted literature
files, temporary renders, and duplicate high-resolution exports are
intentionally excluded. Final figures were refined separately from the
code-generated diagnostic plots and are not presented as reproducible outputs
of this repository.
