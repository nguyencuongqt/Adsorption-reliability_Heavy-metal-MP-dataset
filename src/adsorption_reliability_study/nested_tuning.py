from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from scipy.stats import wilcoxon
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, GroupKFold, KFold, ParameterGrid
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning

from .config import STUDY_ROOT, load_config
from .data import dataset_hash, get_real_feature_columns, load_dataset
from .validation import build_splits, regression_metrics, summarize_metrics

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but LGBMRegressor was fitted with feature names",
)


class ClippedLogTargetRegressor(BaseEstimator, RegressorMixin):
    """Leakage-safe target transform used by the fixed legacy workflow."""

    def __init__(
        self,
        regressor: BaseEstimator,
        clip_quantile: float = 0.995,
        max_log_prediction: float = 4.0,
    ):
        self.regressor = regressor
        self.clip_quantile = clip_quantile
        self.max_log_prediction = max_log_prediction

    def fit(self, X, y):
        y_array = np.asarray(y, dtype=float).reshape(-1)
        upper = float(np.quantile(y_array, self.clip_quantile)) if len(y_array) else 0.0
        self.target_clip_upper_ = max(upper, 0.0)
        y_transformed = np.log1p(np.clip(y_array, 0.0, self.target_clip_upper_))
        self.regressor_ = clone(self.regressor)
        self.regressor_.fit(X, y_transformed)
        return self

    def predict(self, X):
        pred = np.asarray(self.regressor_.predict(X), dtype=float).reshape(-1)
        pred = np.nan_to_num(pred, nan=0.0, posinf=self.max_log_prediction, neginf=0.0)
        pred = np.clip(pred, 0.0, self.max_log_prediction)
        return np.clip(np.expm1(pred), 0.0, None)


def _rmse_scorer(estimator, X, y) -> float:
    return -float(np.sqrt(mean_squared_error(y, estimator.predict(X))))


def _build_estimator(model_name: str, feature_names: list[str], random_state: int):
    scale = model_name in {"elastic_net", "mlp_regressor"}
    numeric_steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        numeric_steps.append(("scaler", StandardScaler()))
    preprocessor = ColumnTransformer(
        [("num", Pipeline(numeric_steps), feature_names)],
        remainder="drop",
        verbose_feature_names_out=True,
    )

    if model_name == "elastic_net":
        model = ElasticNet(max_iter=20000, random_state=random_state)
        grid = {
            "regressor__model__alpha": [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0],
            # Keep strictly inside the Elastic Net/Lasso family. The exact
            # l1_ratio=0 boundary is a Ridge problem and is poorly conditioned
            # for sklearn's coordinate-descent ElasticNet solver.
            "regressor__model__l1_ratio": [0.1, 0.25, 0.5, 0.75, 1.0],
        }
    elif model_name == "lightgbm":
        model = LGBMRegressor(
            objective="regression",
            verbosity=-1,
            n_jobs=1,
            random_state=random_state,
        )
        grid = [
            {
                "regressor__model__n_estimators": [200],
                "regressor__model__learning_rate": [0.03, 0.08],
                "regressor__model__num_leaves": [15, 31],
                "regressor__model__min_child_samples": [10, 30],
                "regressor__model__subsample": [0.8],
                "regressor__model__colsample_bytree": [0.8],
                "regressor__model__reg_lambda": [0.0, 1.0],
            }
        ]
    elif model_name == "mlp_regressor":
        model = MLPRegressor(
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=30,
            max_iter=1500,
            random_state=random_state,
        )
        # A compact, prespecified grid keeps the full 25-outer-fold nested run tractable.
        grid = [
            {
                "regressor__model__hidden_layer_sizes": [(32,), (64,), (64, 32)],
                "regressor__model__activation": ["relu"],
                "regressor__model__solver": ["adam"],
                "regressor__model__alpha": [0.0001, 0.01],
                "regressor__model__learning_rate_init": [0.001, 0.01],
            }
        ]
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])
    return ClippedLogTargetRegressor(pipeline), grid


def _inner_cv(regime: str, groups: np.ndarray | None, random_state: int):
    if regime == "random_cv":
        return KFold(n_splits=5, shuffle=True, random_state=random_state), None
    if groups is None:
        raise ValueError(f"Groups are required for {regime}")
    n_groups = len(np.unique(groups))
    return GroupKFold(n_splits=min(5, n_groups)), groups


def _bootstrap_gap_change(
    random_delta: np.ndarray,
    study_delta: np.ndarray,
    seed: int,
    n_boot: int = 20000,
) -> tuple[float, float, float]:
    """Bootstrap change in (SGCV - RRCV) RMSE after tuning."""
    rng = np.random.default_rng(seed)
    random_draws = rng.choice(random_delta, size=(n_boot, len(random_delta)), replace=True).mean(axis=1)
    study_draws = rng.choice(study_delta, size=(n_boot, len(study_delta)), replace=True).mean(axis=1)
    values = study_draws - random_draws
    return float(values.mean()), float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def _paired_change_table(tuned: pd.DataFrame, original: pd.DataFrame, random_state: int) -> pd.DataFrame:
    paired = tuned.merge(
        original[["regime", "split_id", "model", "rmse", "r2"]],
        on=["regime", "split_id", "model"],
        suffixes=("_tuned", "_original"),
        validate="one_to_one",
    )
    paired["rmse_change"] = paired["rmse_tuned"] - paired["rmse_original"]
    paired["r2_change"] = paired["r2_tuned"] - paired["r2_original"]

    rows: list[dict[str, Any]] = []
    for model_name, model_frame in paired.groupby("model"):
        random_delta = model_frame.loc[model_frame["regime"] == "random_cv", "rmse_change"].to_numpy()
        study_delta = model_frame.loc[model_frame["regime"] == "group_aut", "rmse_change"].to_numpy()
        gap_change, ci_low, ci_high = _bootstrap_gap_change(
            random_delta,
            study_delta,
            seed=random_state + len(rows),
        )
        for regime, regime_frame in model_frame.groupby("regime"):
            changes = regime_frame["rmse_change"].to_numpy()
            try:
                p_value = float(wilcoxon(changes).pvalue)
            except ValueError:
                p_value = np.nan
            rows.append(
                {
                    "model": model_name,
                    "regime": regime,
                    "n_outer_folds": len(regime_frame),
                    "mean_rmse_change_tuned_minus_original": float(changes.mean()),
                    "median_rmse_change_tuned_minus_original": float(np.median(changes)),
                    "paired_wilcoxon_p": p_value,
                    "gap_change_rmse": gap_change if regime == "group_aut" else np.nan,
                    "gap_change_rmse_ci95_low": ci_low if regime == "group_aut" else np.nan,
                    "gap_change_rmse_ci95_high": ci_high if regime == "group_aut" else np.nan,
                }
            )
    return pd.DataFrame(rows)


def run_nested_tuning() -> dict[str, str]:
    config = load_config()
    output_dir = STUDY_ROOT / "results" / "nested_tuning"
    output_dir.mkdir(parents=True, exist_ok=True)
    fold_path = output_dir / "nested_tuned_fold_metrics.csv"
    params_path = output_dir / "nested_tuned_best_params.csv"

    df = load_dataset(config)
    features = get_real_feature_columns(df, config)
    X = df[features].apply(pd.to_numeric, errors="coerce")
    y = df[config.target_column].to_numpy(dtype=float)
    splits = build_splits(df, config)

    completed: set[tuple[str, str, str]] = set()
    metric_rows: list[dict[str, Any]] = []
    param_rows: list[dict[str, Any]] = []
    if fold_path.exists():
        existing = pd.read_csv(fold_path)
        metric_rows = existing.to_dict(orient="records")
        completed = set(zip(existing["regime"], existing["split_id"], existing["model"]))
    if params_path.exists():
        param_rows = pd.read_csv(params_path).to_dict(orient="records")

    for split_number, split in enumerate(splits, start=1):
        X_train, X_test = X.iloc[split.train_index], X.iloc[split.test_index]
        y_train, y_test = y[split.train_index], y[split.test_index]
        if split.regime == "group_exp":
            inner_groups = df.iloc[split.train_index]["exp_id"].to_numpy()
        elif split.regime == "group_aut":
            inner_groups = df.iloc[split.train_index]["aut_id"].to_numpy()
        else:
            inner_groups = None

        for model_offset, model_name in enumerate(config.rq1_models):
            key = (split.regime, split.split_id, model_name)
            if key in completed:
                continue
            seed = config.random_state + 10000 + split_number * 10 + model_offset
            estimator, grid = _build_estimator(model_name, features, seed)
            cv, fit_groups = _inner_cv(split.regime, inner_groups, seed)
            search = GridSearchCV(
                estimator,
                grid,
                scoring=_rmse_scorer,
                cv=cv,
                n_jobs=-1,
                refit=True,
                return_train_score=False,
                error_score="raise",
            )
            search.fit(X_train, y_train, groups=fit_groups)
            y_pred = search.best_estimator_.predict(X_test)
            metrics = regression_metrics(y_test, y_pred)
            metric_rows.append(
                {
                    "regime": split.regime,
                    "split_id": split.split_id,
                    "model": model_name,
                    **metrics,
                    "n_train": len(split.train_index),
                    "n_test": len(split.test_index),
                    "inner_best_rmse": -float(search.best_score_),
                    "n_candidates": len(list(ParameterGrid(grid))),
                }
            )
            param_rows.append(
                {
                    "regime": split.regime,
                    "split_id": split.split_id,
                    "model": model_name,
                    "best_params_json": json.dumps(search.best_params_, sort_keys=True),
                }
            )
            pd.DataFrame(metric_rows).to_csv(fold_path, index=False)
            pd.DataFrame(param_rows).to_csv(params_path, index=False)
            print(
                f"{split.split_id} {model_name}: "
                f"outer RMSE={metrics['rmse']:.4f}, R2={metrics['r2']:.4f}, "
                f"inner RMSE={-search.best_score_:.4f}",
                flush=True,
            )

    metrics_df = pd.DataFrame(metric_rows)
    summary = summarize_metrics(metrics_df)
    summary["rmse_median"] = metrics_df.groupby(["regime", "model"])["rmse"].median().reindex(
        pd.MultiIndex.from_frame(summary[["regime", "model"]])
    ).to_numpy()

    original = pd.read_csv(STUDY_ROOT / "results" / "tables" / "rq1_fold_metrics.csv")
    original_summary = pd.read_csv(
        STUDY_ROOT / "results" / "tables" / "table_performance_across_validation_regimes.csv"
    )
    comparison = summary.merge(
        original_summary,
        on=["regime", "model"],
        suffixes=("_tuned", "_original"),
        validate="one_to_one",
    )
    for metric in ["rmse_mean", "r2_mean"]:
        comparison[f"{metric}_change"] = comparison[f"{metric}_tuned"] - comparison[f"{metric}_original"]

    gap_rows: list[dict[str, Any]] = []
    for label, frame in [("original", original_summary), ("nested_tuned", summary)]:
        pivot_rmse = frame.pivot(index="model", columns="regime", values="rmse_mean")
        pivot_r2 = frame.pivot(index="model", columns="regime", values="r2_mean")
        for model_name in pivot_rmse.index:
            gap_rows.append(
                {
                    "analysis": label,
                    "model": model_name,
                    "rmse_gap_sgcv_minus_rrcv": pivot_rmse.loc[model_name, "group_aut"]
                    - pivot_rmse.loc[model_name, "random_cv"],
                    "rmse_ratio_sgcv_over_rrcv": pivot_rmse.loc[model_name, "group_aut"]
                    / pivot_rmse.loc[model_name, "random_cv"],
                    "r2_drop_rrcv_minus_sgcv": pivot_r2.loc[model_name, "random_cv"]
                    - pivot_r2.loc[model_name, "group_aut"],
                }
            )

    paired_changes = _paired_change_table(metrics_df, original, config.random_state)
    summary.to_csv(output_dir / "nested_tuned_summary.csv", index=False)
    comparison.to_csv(output_dir / "nested_tuned_vs_original.csv", index=False)
    pd.DataFrame(gap_rows).to_csv(output_dir / "optimism_gap_comparison.csv", index=False)
    paired_changes.to_csv(output_dir / "paired_change_inference.csv", index=False)
    manifest = {
        "dataset_sha256": dataset_hash(config.dataset_path),
        "outer_splits": "Identical to the original RQ1 analysis",
        "inner_cv": {
            "random_cv": "5-fold shuffled KFold within outer-training rows",
            "group_exp": "5-fold GroupKFold by experiment within outer-training data",
            "group_aut": "5-fold GroupKFold by study within outer-training data",
        },
        "selection_metric": "RMSE on the original qe scale",
        "n_outer_results": len(metrics_df),
        "target_transform": "Training-fold 0.995 quantile clipping, log1p fit, inverse transform, prediction clipping",
    }
    (output_dir / "nested_tuning_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    return {
        "summary": str(output_dir / "nested_tuned_summary.csv"),
        "comparison": str(output_dir / "nested_tuned_vs_original.csv"),
        "gaps": str(output_dir / "optimism_gap_comparison.csv"),
        "inference": str(output_dir / "paired_change_inference.csv"),
    }
