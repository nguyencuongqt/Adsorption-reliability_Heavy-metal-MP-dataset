#!/usr/bin/env python3
"""External validation using the exact locked 25-feature release specification.

This script fits each pre-specified released model once on all 1,009 locked
development observations, then scores an independent external table. It uses
the release repository's feature engineering, imputation, log-target transform,
clipping, model hyperparameters, and seed convention.

Example:
  python scripts/19_run_external_validation_release.py ^
    --release-root "G:\\My Drive\\adsorption_reliability_study\\release_repository" ^
    --external "G:\\My Drive\\adsorption_reliability_study\\external data 2026\\extracted_external_holdout\\external_holdout_2026_model_ready.xlsx" ^
    --sheet Model_Input_25Features ^
    --outdir results\\external_validation_2026

The fitted models are deployment refits of the locked specification, not new
model selection. No external row is used for fitting or tuning.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--release-root", type=Path, required=True, help="Root containing configs.json, data/, and src/.")
    parser.add_argument("--external", type=Path, required=True)
    parser.add_argument("--sheet", default="Model_Input_25Features")
    parser.add_argument("--models", nargs="+", default=["elastic_net", "lightgbm", "mlp_regressor"])
    parser.add_argument("--study-column", default="aut_id")
    parser.add_argument("--target", default="qe")
    parser.add_argument("--outdir", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def metrics(frame: pd.DataFrame, target: str, prediction: str, study: str) -> dict[str, float | int | None]:
    y = pd.to_numeric(frame[target], errors="coerce").to_numpy(float)
    p = pd.to_numeric(frame[prediction], errors="coerce").to_numpy(float)
    keep = np.isfinite(y) & np.isfinite(p)
    y, p = y[keep], p[keep]
    if not len(y):
        return {"n_rows": 0, "n_studies": 0, "mae": None, "rmse": None, "r2": None, "bias": None, "pearson_r": None}
    return {
        "n_rows": int(len(y)),
        "n_studies": int(frame.loc[keep, study].nunique(dropna=True)),
        "mae": float(mean_absolute_error(y, p)),
        "rmse": float(np.sqrt(mean_squared_error(y, p))),
        "r2": float(r2_score(y, p)) if len(y) > 1 and np.unique(y).size > 1 else None,
        "bias": float(np.mean(p - y)),
        "pearson_r": float(np.corrcoef(y, p)[0, 1]) if len(y) > 1 and np.std(y) > 0 and np.std(p) > 0 else None,
    }


def main() -> int:
    args = parse_args()
    if not args.release_root.exists():
        raise FileNotFoundError(args.release_root)
    if str(args.release_root / "src") not in sys.path:
        sys.path.insert(0, str(args.release_root / "src"))
    from adsorption_reliability import analysis  # imported only after selecting the release root

    args.outdir.mkdir(parents=True, exist_ok=True)
    config = analysis.load_config()
    development = analysis.load_dataset(config)
    external = pd.read_excel(args.external, sheet_name=args.sheet)
    expected = list(analysis.RAW_FEATURES)
    missing = sorted(set(expected + [args.target, args.study_column]) - set(external.columns))
    if missing:
        raise ValueError(f"External sheet misses required columns: {missing}")
    entirely_missing = external[expected].columns[external[expected].isna().all()].tolist()
    # This is valid only for continuous variables with their explicit missingness
    # indicators. The released preprocessor imputes using development-data
    # medians; no external value is fabricated here.
    indicator_for = {"ph": "ph_missing", "temp": "temp_missing", "rpm": "rpm_missing", "sa": "sa_missing"}
    unsupported = [
        feature for feature in entirely_missing
        if feature not in indicator_for or not (external[indicator_for[feature]] == 1).all()
    ]
    if unsupported:
        raise ValueError(
            "Features entirely missing without an all-one documented missingness indicator: " + ", ".join(unsupported)
        )

    x_train = analysis.engineer_features(development)
    x_external = analysis.engineer_features(external)
    y_train = development[config["target_column"]].to_numpy(float)
    prediction_table = external.copy()
    pooled_rows, study_rows = [], []
    artifacts = {}
    for offset, model_name in enumerate(args.models):
        if model_name not in config["models"]:
            raise ValueError(f"Unknown released model '{model_name}'.")
        fitted = analysis.fit_model(
            model_name, x_train, y_train, config,
            random_state=int(config["random_state"]) + offset,
        )
        column = f"prediction_{model_name}"
        prediction_table[column] = fitted.predict(x_external)
        prediction_table[f"residual_{model_name}"] = prediction_table[column] - prediction_table[args.target]
        prediction_table[f"absolute_error_{model_name}"] = prediction_table[f"residual_{model_name}"].abs()
        overall = metrics(prediction_table, args.target, column, args.study_column)
        overall.update({"model": model_name, "aggregation": "pooled_rows"})
        pooled_rows.append(overall)
        for study_id, group in prediction_table.groupby(args.study_column, dropna=False):
            row = metrics(group, args.target, column, args.study_column)
            row.update({"model": model_name, args.study_column: study_id})
            study_rows.append(row)
        artifacts[model_name] = str(args.outdir / f"deployment_refit_{model_name}.joblib")
        joblib.dump(fitted, artifacts[model_name])

    pooled = pd.DataFrame(pooled_rows)
    study_metrics = pd.DataFrame(study_rows)
    # Equal-study summaries keep the two source studies from being treated as 35 independent studies.
    study_average = study_metrics.groupby("model", as_index=False).agg(
        n_studies=(args.study_column, "nunique"),
        macro_study_mae=("mae", "mean"),
        macro_study_rmse=("rmse", "mean"),
        macro_study_r2=("r2", "mean"),
    )
    prediction_table.to_csv(args.outdir / "external_predictions_all_models.csv", index=False)
    pooled.to_csv(args.outdir / "external_metrics_pooled.csv", index=False)
    study_metrics.to_csv(args.outdir / "external_metrics_by_study.csv", index=False)
    study_average.to_csv(args.outdir / "external_metrics_equal_study_average.csv", index=False)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_mode": "deployment_refit_to_locked_specification",
        "release_root": str(args.release_root.resolve()),
        "development_dataset": str((args.release_root / config["dataset_path"]).resolve()),
        "development_dataset_sha256": sha256(args.release_root / config["dataset_path"]),
        "external_dataset": str(args.external.resolve()),
        "external_sheet": args.sheet,
        "n_development_rows": int(len(development)),
        "n_external_rows": int(len(external)),
        "n_external_studies": int(external[args.study_column].nunique()),
        "raw_feature_contract": expected,
        "entirely_missing_external_features_imputed_by_locked_preprocessor": entirely_missing,
        "models": args.models,
        "random_states": {model: int(config["random_state"]) + offset for offset, model in enumerate(args.models)},
        "interpretation_note": "Pooled row metrics are descriptive. Study-level and equal-study summaries are required because the external rows are clustered within source studies.",
        "model_artifacts": artifacts,
    }
    (args.outdir / "external_validation_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(pooled.to_string(index=False))
    print("\nEqual-study average metrics:\n", study_average.to_string(index=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)
