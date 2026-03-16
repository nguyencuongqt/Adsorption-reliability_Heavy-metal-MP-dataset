# Legacy Audit

## Reused Inputs

- Final modeling dataset: [05_modeling_dataset_final.csv](G:\My Drive\Multi_metal MP prediction\data\processed\05_modeling_dataset_final.csv)
- Feature inventory: [ml_feature_inventory.csv](G:\My Drive\Multi_metal MP prediction\results\ml_feature_inventory.csv)
- Author metadata: [author_year_registry.csv](G:\My Drive\Multi_metal MP prediction\data\metadata\author_year_registry.csv)

## Potentially Reusable Utility Logic

- [pipeline_common.py](G:\My Drive\Multi_metal MP prediction\scripts\pipeline_common.py): useful for understanding prior fold logic and data normalization conventions
- [data.py](G:\My Drive\Multi_metal MP prediction\src\ml_benchmark\data.py): useful as legacy reference for feature naming conventions only
- [models.py](G:\My Drive\Multi_metal MP prediction\src\ml_benchmark\models.py): useful as legacy reference for baseline model defaults only

## Reused in the current reliability-study version

- Legacy feature engineering from [data.py](G:\My Drive\Multi_metal MP prediction\src\ml_benchmark\data.py)
- Legacy preprocessing and model construction logic from [models.py](G:\My Drive\Multi_metal MP prediction\src\ml_benchmark\models.py)
- New grouped-validation, null-feature auditing, and manuscript outputs remain specific to `adsorption_reliability_study`

## Legacy Artifacts Not Treated as Current Results

- Everything in [results/ml_benchmark](G:\My Drive\Multi_metal MP prediction\results\ml_benchmark)
- Old `model_eval_*` summaries in [results](G:\My Drive\Multi_metal MP prediction\results)
- Old figures in [submission_package](G:\My Drive\Multi_metal MP prediction\submission_package)
- Old notebooks in [notebooks](G:\My Drive\Multi_metal MP prediction\notebooks)
- Old manuscript package and text

## Why the Fresh Study Layer Exists

The new study is about reliability of prediction and interpretation under grouped literature-derived data. That requires a clean separation from older benchmark-focused outputs so that no claim in the new study depends on stale artifacts.
