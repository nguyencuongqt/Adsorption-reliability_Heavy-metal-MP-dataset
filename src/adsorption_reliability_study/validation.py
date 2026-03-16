from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut, RepeatedKFold

from .config import StudyConfig


@dataclass(frozen=True)
class SplitSpec:
    regime: str
    split_id: str
    train_index: np.ndarray
    test_index: np.ndarray


def build_splits(df: pd.DataFrame, config: StudyConfig) -> list[SplitSpec]:
    indices = np.arange(len(df))
    splits: list[SplitSpec] = []

    random_cv = RepeatedKFold(
        n_splits=int(config.random_cv["n_splits"]),
        n_repeats=int(config.random_cv["n_repeats"]),
        random_state=config.random_state,
    )
    for split_number, (train_index, test_index) in enumerate(random_cv.split(indices), start=1):
        splits.append(SplitSpec("random_cv", f"random_cv_{split_number:02d}", train_index, test_index))

    for regime, group_column in [("group_exp", "exp_id"), ("group_aut", "aut_id")]:
        cv = GroupKFold(n_splits=int(config.group_cv["n_splits"]))
        groups = df[group_column].to_numpy()
        for split_number, (train_index, test_index) in enumerate(cv.split(indices, groups=groups), start=1):
            splits.append(SplitSpec(regime, f"{regime}_{split_number:02d}", train_index, test_index))

    if config.include_optional_regime and config.optional_regime == "leave_one_study_out":
        logo = LeaveOneGroupOut()
        groups = df["aut_id"].to_numpy()
        for split_number, (train_index, test_index) in enumerate(logo.split(indices, groups=groups), start=1):
            splits.append(SplitSpec("leave_one_study_out", f"leave_one_study_out_{split_number:02d}", train_index, test_index))

    return splits


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }


def summarize_metrics(metrics_df: pd.DataFrame) -> pd.DataFrame:
    return (
        metrics_df.groupby(["regime", "model"], as_index=False)
        .agg(
            n_splits=("split_id", "nunique"),
            r2_mean=("r2", "mean"),
            r2_sd=("r2", "std"),
            rmse_mean=("rmse", "mean"),
            rmse_sd=("rmse", "std"),
            mae_mean=("mae", "mean"),
            mae_sd=("mae", "std"),
        )
        .sort_values(["regime", "rmse_mean", "mae_mean"])
        .reset_index(drop=True)
    )


def build_generalization_gap_table(summary_df: pd.DataFrame) -> pd.DataFrame:
    baseline = summary_df.loc[summary_df["regime"] == "random_cv", ["model", "r2_mean", "rmse_mean", "mae_mean"]].rename(
        columns={
            "r2_mean": "baseline_r2",
            "rmse_mean": "baseline_rmse",
            "mae_mean": "baseline_mae",
        }
    )
    comparison = summary_df.merge(baseline, on="model", how="left")
    comparison = comparison.loc[comparison["regime"] != "random_cv"].copy()
    comparison["delta_r2_vs_random"] = comparison["r2_mean"] - comparison["baseline_r2"]
    comparison["delta_rmse_vs_random"] = comparison["rmse_mean"] - comparison["baseline_rmse"]
    comparison["delta_mae_vs_random"] = comparison["mae_mean"] - comparison["baseline_mae"]
    return comparison[
        [
            "regime",
            "model",
            "delta_r2_vs_random",
            "delta_rmse_vs_random",
            "delta_mae_vs_random",
        ]
    ].sort_values(["regime", "delta_rmse_vs_random"])
