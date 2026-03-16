from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REGIME_ORDER = ["random_cv", "group_exp", "group_aut", "leave_one_study_out"]


def save_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def plot_validation_hierarchy(summary_df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    order = [regime for regime in REGIME_ORDER if regime in summary_df["regime"].unique()]
    x = np.arange(len(order))
    for model, subset in summary_df.groupby("model"):
        subset = subset.set_index("regime").reindex(order)
        ax.plot(x, subset["rmse_mean"], marker="o", label=model)
    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=15)
    ax.set_ylabel("RMSE on original qe scale")
    ax.set_title("Validation hierarchy and performance drop")
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def plot_mechanism_heatmap(mechanism_df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plot_df = (
        mechanism_df.groupby(["method", "mechanism_group"], as_index=False)["mean_importance_share"]
        .mean()
        .pivot(index="method", columns="mechanism_group", values="mean_importance_share")
        .fillna(0.0)
    )
    fig, ax = plt.subplots(figsize=(10, 4.5))
    image = ax.imshow(plot_df.to_numpy(), aspect="auto", cmap="viridis")
    ax.set_xticks(np.arange(plot_df.shape[1]))
    ax.set_xticklabels(plot_df.columns, rotation=30, ha="right")
    ax.set_yticks(np.arange(plot_df.shape[0]))
    ax.set_yticklabels(plot_df.index)
    ax.set_title("Mechanism-group importance across methods")
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("Mean importance share")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def plot_null_benchmark(reliability_df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    method_scores = reliability_df.groupby("method", as_index=False)["mean_null_share"].mean().sort_values("mean_null_share")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(method_scores["method"], method_scores["mean_null_share"], color="#5b7db1")
    ax.set_ylabel("Mean null importance share")
    ax.set_title("Null-feature benchmarking across importance methods")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def plot_stability_heatmap(stability_df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plot_df = stability_df.pivot(index="method", columns="regime", values="rank_stability").fillna(0.0)
    plot_df = plot_df.reindex(columns=[column for column in REGIME_ORDER if column in plot_df.columns])
    fig, ax = plt.subplots(figsize=(8, 4.5))
    image = ax.imshow(plot_df.to_numpy(), aspect="auto", cmap="magma", vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(plot_df.shape[1]))
    ax.set_xticklabels(plot_df.columns, rotation=25, ha="right")
    ax.set_yticks(np.arange(plot_df.shape[0]))
    ax.set_yticklabels(plot_df.index)
    ax.set_title("Importance stability across methods and regimes")
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("Mean pairwise Spearman correlation")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
