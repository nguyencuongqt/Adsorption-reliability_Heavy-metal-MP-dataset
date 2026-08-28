#!/usr/bin/env python3
"""External validation of a locked adsorption-prediction model.

The preferred invocation scores with the exact fitted pipeline that was locked
after model development. It never silently refits that model.

Example (recommended):
    python scripts/run_external_validation.py ^
      --pipeline PATH_TO_LOCKED_PIPELINE.joblib ^
      --external "G:\\My Drive\\adsorption_reliability_study\\external data 2026\\extracted_external_holdout\\external_holdout_2026_core.xlsx" ^
      --sheet Model_Input --target qe_mg_g ^
      --outdir results\\external_validation_2026

Fallback (only if the exact locked pipeline is unavailable):
    python scripts/run_external_validation.py ^
      --train PATH_TO_LOCKED_TRAINING_DATA.csv ^
      --external PATH_TO_EXTERNAL.xlsx --sheet Model_Input ^
      --target qe_mg_g --allow-refit --outdir results\\external_validation_2026

Fallback output is explicitly marked REFITTED_FALLBACK__NOT_LOCKED_MODEL.
It must not be reported as external validation of the originally locked model.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


METAL_ALIASES = {
    "cadmium": "Cd", "cd": "Cd", "cd(ii)": "Cd",
    "chromium": "Cr", "cr": "Cr", "cr(iii)": "Cr", "cr(vi)": "Cr",
    "mercury": "Hg", "hg": "Hg", "hg(ii)": "Hg",
}


@dataclass
class Metrics:
    n_rows: int
    n_studies: int
    mae: float | None
    rmse: float | None
    r2: float | None
    bias_pred_minus_observed: float | None
    mean_observed: float | None
    mean_predicted: float | None
    pearson_r: float | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score an independent external hold-out against a locked model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    model_source = parser.add_mutually_exclusive_group(required=True)
    model_source.add_argument("--pipeline", type=Path, help="Fitted, locked sklearn Pipeline/joblib artifact.")
    model_source.add_argument("--train", type=Path, help="Training table; requires --allow-refit and is not locked-model validation.")
    parser.add_argument("--external", type=Path, required=True, help="External table: .xlsx, .csv, or .parquet.")
    parser.add_argument("--sheet", default="Model_Input", help="Worksheet name when input is Excel.")
    parser.add_argument("--target", default="qe_mg_g", help="Observed response variable.")
    parser.add_argument("--study-column", default="study_id", help="External source-study identifier.")
    parser.add_argument("--id-column", default="record_id", help="Optional external row identifier.")
    parser.add_argument(
        "--feature-columns", nargs="+",
        help="Exact raw input columns. Needed if the pipeline does not expose feature_names_in_, and required to control a fallback refit.",
    )
    parser.add_argument("--drop-columns", nargs="*", default=[], help="Columns excluded only when fallback features are inferred.")
    parser.add_argument("--allow-refit", action="store_true", help="Explicitly permit fallback model refitting.")
    parser.add_argument("--random-state", type=int, default=2025, help="Fallback refit seed.")
    parser.add_argument("--n-estimators", type=int, default=1000, help="Fallback random-forest trees.")
    parser.add_argument("--outdir", type=Path, required=True, help="Directory for reproducible outputs.")
    return parser.parse_args()


def read_table(path: Path, sheet: str | None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input does not exist: {path}")
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path, sheet_name=sheet)
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported input type '{suffix}'. Use XLSX, CSV, or Parquet.")


def normalize_common_labels(data: pd.DataFrame) -> pd.DataFrame:
    """Make common metal labels compatible without changing values that are unknown."""
    result = data.copy()
    for column in ("metal", "metal_species"):
        if column in result.columns:
            result[column] = result[column].map(
                lambda value: METAL_ALIASES.get(str(value).strip().lower().replace(" ", ""), value)
                if pd.notna(value) else value
            )
    return result


def recover_pipeline_features(pipeline: Any, override: list[str] | None) -> list[str]:
    if override:
        return list(override)
    for candidate in (pipeline, getattr(pipeline, "named_steps", {}).get("preprocess")):
        names = getattr(candidate, "feature_names_in_", None) if candidate is not None else None
        if names is not None:
            return list(names)
    raise ValueError(
        "The pipeline does not expose feature_names_in_. Supply its exact development feature list with --feature-columns."
    )


def fallback_pipeline(training: pd.DataFrame, target: str, features: list[str], args: argparse.Namespace) -> Pipeline:
    if not args.allow_refit:
        raise RuntimeError("Fallback refitting is disabled; supply --pipeline or add --allow-refit.")
    missing = sorted(set(features + [target]) - set(training.columns))
    if missing:
        raise ValueError(f"Training data is missing required columns: {missing}")
    y = pd.to_numeric(training[target], errors="coerce")
    valid = y.notna()
    x, y = training.loc[valid, features].copy(), y.loc[valid]
    numeric = x.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    categorical = [column for column in features if column not in numeric]
    preprocess = ColumnTransformer(
        [("numeric", Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True))]), numeric),
         ("categorical", Pipeline([
             ("impute", SimpleImputer(strategy="most_frequent")),
             ("one_hot", OneHotEncoder(handle_unknown="ignore")),
         ]), categorical)],
        remainder="drop",
    )
    model = RandomForestRegressor(
        n_estimators=args.n_estimators, random_state=args.random_state, n_jobs=-1,
    )
    fitted = Pipeline([("preprocess", preprocess), ("model", model)])
    return fitted.fit(x, y)


def summarize_metrics(observed: pd.Series, predicted: pd.Series, studies: pd.Series) -> Metrics:
    y_true = pd.to_numeric(observed, errors="coerce").to_numpy(dtype=float)
    y_pred = pd.to_numeric(predicted, errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[valid], y_pred[valid]
    if y_true.size == 0:
        return Metrics(0, 0, None, None, None, None, None, None, None)
    r2 = float(r2_score(y_true, y_pred)) if y_true.size > 1 and np.unique(y_true).size > 1 else None
    correlation = (
        float(np.corrcoef(y_true, y_pred)[0, 1])
        if y_true.size > 1 and np.std(y_true) > 0 and np.std(y_pred) > 0 else None
    )
    return Metrics(
        n_rows=int(y_true.size),
        n_studies=int(pd.Series(studies).nunique(dropna=True)),
        mae=float(mean_absolute_error(y_true, y_pred)),
        rmse=float(np.sqrt(mean_squared_error(y_true, y_pred))),
        r2=r2,
        bias_pred_minus_observed=float(np.mean(y_pred - y_true)),
        mean_observed=float(np.mean(y_true)),
        mean_predicted=float(np.mean(y_pred)),
        pearson_r=correlation,
    )


def main() -> int:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    external = normalize_common_labels(read_table(args.external, args.sheet))
    for required in (args.target, args.study_column):
        if required not in external.columns:
            raise ValueError(f"External data is missing required column '{required}'.")

    if args.pipeline is not None:
        pipeline = joblib.load(args.pipeline)
        features = recover_pipeline_features(pipeline, args.feature_columns)
        validation_mode = "LOCKED_PIPELINE"
        model_source = str(args.pipeline.resolve())
    else:
        if not args.allow_refit:
            raise RuntimeError("--train is allowed only with --allow-refit.")
        training = normalize_common_labels(read_table(args.train, sheet=None))
        blocked = {args.target, args.study_column, args.id_column, *args.drop_columns}
        features = args.feature_columns or [c for c in training.columns if c in external.columns and c not in blocked]
        if not features:
            raise ValueError("No shared input features. Provide --feature-columns explicitly.")
        pipeline = fallback_pipeline(training, args.target, features, args)
        validation_mode = "REFITTED_FALLBACK__NOT_LOCKED_MODEL"
        model_source = str(args.train.resolve())
        joblib.dump(pipeline, args.outdir / "refitted_pipeline__not_locked_model.joblib")

    missing_external = [feature for feature in features if feature not in external.columns]
    if missing_external:
        raise ValueError("External data is missing locked-model features: " + ", ".join(missing_external))

    compatibility = pd.DataFrame({
        "feature": features,
        "missing_in_external": [int(external[feature].isna().sum()) for feature in features],
        "external_dtype": [str(external[feature].dtype) for feature in features],
    })
    compatibility.to_csv(args.outdir / "feature_compatibility.csv", index=False)

    scored = external.copy()
    scored["prediction_qe_mg_g"] = pipeline.predict(external[features])
    scored["residual_pred_minus_observed_mg_g"] = (
        scored["prediction_qe_mg_g"] - pd.to_numeric(scored[args.target], errors="coerce")
    )
    scored["absolute_error_mg_g"] = scored["residual_pred_minus_observed_mg_g"].abs()
    scored["validation_mode"] = validation_mode
    scored.to_csv(args.outdir / "external_predictions.csv", index=False)

    pooled = summarize_metrics(scored[args.target], scored["prediction_qe_mg_g"], scored[args.study_column])
    study_rows = []
    for study_id, group in scored.groupby(args.study_column, dropna=False):
        result = asdict(summarize_metrics(group[args.target], group["prediction_qe_mg_g"], group[args.study_column]))
        result[args.study_column] = study_id
        study_rows.append(result)
    pd.DataFrame(study_rows).to_csv(args.outdir / "metrics_by_study.csv", index=False)

    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "validation_mode": validation_mode,
        "model_source": model_source,
        "external_source": str(args.external.resolve()),
        "target": args.target,
        "study_column": args.study_column,
        "n_features": len(features),
        "features": features,
        "pooled_metrics": asdict(pooled),
        "interpretation_note": (
            "Rows are clustered within source studies. Pooled row-level metrics are descriptive; "
            "interpret study-level metrics and do not count rows as independent studies."
        ),
    }
    (args.outdir / "validation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
