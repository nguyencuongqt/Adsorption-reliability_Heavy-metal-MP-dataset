from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import shap
from sklearn.inspection import permutation_importance

from .data import feature_group, null_feature_type


def _normalize_importance(frame: pd.DataFrame) -> pd.DataFrame:
    working = frame.copy()
    total = working["importance"].sum()
    working["importance_share"] = 0.0 if total <= 0 else working["importance"] / total
    working["mechanism_group"] = working["feature"].map(feature_group)
    working["null_type"] = working["feature"].map(null_feature_type)
    working["feature_type"] = np.where(working["null_type"] == "real", "real", "null")
    return working


def elastic_net_coef_importance(fitted_model) -> pd.DataFrame:
    values = np.abs(np.asarray(fitted_model.model_step.coef_, dtype=float).reshape(-1))
    transformed_names = list(fitted_model.estimator.named_steps["preprocessor"].get_feature_names_out())
    return _normalize_importance(pd.DataFrame({"feature": transformed_names, "importance": values}))


def permutation_importance_frame(
    fitted_model,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    random_state: int,
    n_repeats: int = 5,
) -> pd.DataFrame:
    X_for_perm = X_val[fitted_model.feature_names].copy()
    result = permutation_importance(
        fitted_model.estimator,
        X_for_perm,
        y_val,
        n_repeats=n_repeats,
        random_state=random_state,
        scoring="neg_root_mean_squared_error",
    )
    values = np.clip(result.importances_mean, a_min=0.0, a_max=None)
    return _normalize_importance(pd.DataFrame({"feature": fitted_model.feature_names, "importance": values}))


def lightgbm_builtin_importance(fitted_model) -> pd.DataFrame:
    values = np.asarray(fitted_model.model_step.feature_importances_, dtype=float)
    transformed_names = list(fitted_model.estimator.named_steps["preprocessor"].get_feature_names_out())
    return _normalize_importance(pd.DataFrame({"feature": transformed_names, "importance": values}))


def lightgbm_shap_importance(fitted_model, X_val: pd.DataFrame, max_rows: int = 100) -> pd.DataFrame:
    transformed = fitted_model.transformed_validation_frame(X_val)
    if len(transformed) > max_rows:
        transformed = transformed.sample(n=max_rows, random_state=0)
    explainer = shap.TreeExplainer(fitted_model.model_step)
    shap_values = explainer.shap_values(transformed)
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    values = np.abs(np.asarray(shap_values, dtype=float)).mean(axis=0)
    return _normalize_importance(pd.DataFrame({"feature": fitted_model.feature_names, "importance": values}))


def summarize_method_reliability(importances: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    split_level = (
        importances.groupby(["method", "regime", "split_id"], as_index=False)
        .apply(_split_reliability_record, include_groups=False)
        .reset_index(drop=True)
    )

    stability_rows = []
    for (method, regime), subset in importances.groupby(["method", "regime"]):
        stability_rows.append(
            {
                "method": method,
                "regime": regime,
                "rank_stability": _mean_pairwise_spearman(subset, feature_type="real"),
            }
        )
    stability_df = pd.DataFrame(stability_rows)

    robustness_rows = []
    for method, subset in importances.groupby("method"):
        robustness_rows.append({"method": method, "regime_robustness": _regime_robustness(subset)})
    robustness_df = pd.DataFrame(robustness_rows)

    summary = (
        split_level.groupby(["method", "regime"], as_index=False)
        .agg(
            mean_null_share=("null_share", "mean"),
            mean_real_null_gap=("real_null_gap", "mean"),
            mean_top10_null_fraction=("top10_null_fraction", "mean"),
        )
        .merge(stability_df, on=["method", "regime"], how="left")
        .merge(robustness_df, on="method", how="left")
    )
    summary["heuristic_reliability_score"] = (
        (1.0 - summary["mean_null_share"]).clip(lower=0.0)
        + summary["mean_real_null_gap"].clip(lower=0.0) * 10.0
        + summary["rank_stability"].fillna(0.0).clip(lower=0.0)
        + summary["regime_robustness"].fillna(0.0).clip(lower=0.0)
        - summary["mean_top10_null_fraction"].clip(lower=0.0)
    )
    summary = summary.sort_values(["heuristic_reliability_score", "mean_null_share"], ascending=[False, True]).reset_index(drop=True)
    return summary, split_level, stability_df


def aggregate_mechanism_importance(importances: pd.DataFrame) -> pd.DataFrame:
    return (
        importances.groupby(["method", "regime", "mechanism_group"], as_index=False)
        .agg(mean_importance_share=("importance_share", "mean"))
        .sort_values(["method", "regime", "mean_importance_share"], ascending=[True, True, False])
        .reset_index(drop=True)
    )


def aggregate_feature_importance(importances: pd.DataFrame) -> pd.DataFrame:
    return (
        importances.groupby(["method", "regime", "feature", "feature_type", "null_type", "mechanism_group"], as_index=False)
        .agg(mean_importance=("importance", "mean"), mean_importance_share=("importance_share", "mean"))
        .sort_values(["method", "regime", "mean_importance_share"], ascending=[True, True, False])
        .reset_index(drop=True)
    )


def _split_reliability_record(frame: pd.DataFrame) -> pd.Series:
    ordered = frame.sort_values("importance", ascending=False).reset_index(drop=True)
    null_mask = ordered["feature_type"] == "null"
    null_share = float(ordered.loc[null_mask, "importance_share"].sum())
    real_values = ordered.loc[~null_mask, "importance_share"]
    null_values = ordered.loc[null_mask, "importance_share"]
    real_median = float(real_values.median()) if not real_values.empty else 0.0
    null_median = float(null_values.median()) if not null_values.empty else 0.0
    top10 = ordered.head(10)
    top10_null_fraction = float((top10["feature_type"] == "null").mean())
    return pd.Series(
        {
            "null_share": null_share,
            "real_null_gap": real_median - null_median,
            "top10_null_fraction": top10_null_fraction,
        }
    )


def _mean_pairwise_spearman(frame: pd.DataFrame, feature_type: str = "real") -> float:
    pivot = (
        frame.loc[frame["feature_type"] == feature_type]
        .pivot_table(index="feature", columns="split_id", values="importance_share", fill_value=0.0)
    )
    if pivot.shape[1] < 2:
        return np.nan
    correlations = []
    for left, right in itertools.combinations(pivot.columns, 2):
        corr = pivot[left].rank(ascending=False).corr(pivot[right].rank(ascending=False), method="spearman")
        if pd.notna(corr):
            correlations.append(float(corr))
    return float(np.mean(correlations)) if correlations else np.nan


def _regime_robustness(frame: pd.DataFrame) -> float:
    aggregated = (
        frame.loc[frame["feature_type"] == "real"]
        .groupby(["regime", "feature"], as_index=False)["importance_share"]
        .mean()
        .pivot(index="feature", columns="regime", values="importance_share")
        .fillna(0.0)
    )
    if aggregated.shape[1] < 2:
        return np.nan
    correlations = []
    for left, right in itertools.combinations(aggregated.columns, 2):
        corr = aggregated[left].rank(ascending=False).corr(aggregated[right].rank(ascending=False), method="spearman")
        if pd.notna(corr):
            correlations.append(float(corr))
    return float(np.mean(correlations)) if correlations else np.nan
