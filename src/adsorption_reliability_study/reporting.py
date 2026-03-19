from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REGIME_ORDER = ["random_cv", "group_exp", "group_aut"]
MAIN_REGIME_ORDER = ["random_cv", "group_exp", "group_aut"]
REGIME_LABELS = {
    "random_cv": "Random CV",
    "group_exp": "Grouped\nexperiment",
    "group_aut": "Grouped\nstudy",
}
MODEL_LABELS = {
    "elastic_net": "EN",
    "lightgbm": "LGBM",
    "mlp_regressor": "MLP",
}
MODEL_COLORS = {
    "elastic_net": "#4C78A8",
    "lightgbm": "#F58518",
    "mlp_regressor": "#54A24B",
}
MODEL_MARKERS = {
    "elastic_net": "o",
    "lightgbm": "s",
    "mlp_regressor": "^",
}
METHOD_LABELS = {
    "elastic_net_coef": "EN coefficient",
    "elastic_net_permutation": "EN permutation",
    "lightgbm_permutation": "LGBM permutation",
    "lightgbm_builtin_gain": "LGBM gain",
    "lightgbm_shap": "LGBM SHAP",
}


def save_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _style_axis(ax: plt.Axes) -> None:
    ax.grid(axis="y", alpha=0.18, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_validation_hierarchy(summary_df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plot_df = summary_df.loc[summary_df["regime"].isin(MAIN_REGIME_ORDER), ["regime", "model", "rmse_mean", "rmse_sd"]].copy()
    plot_df["regime"] = pd.Categorical(plot_df["regime"], categories=MAIN_REGIME_ORDER, ordered=True)
    plot_df = plot_df.sort_values(["model", "regime"]).reset_index(drop=True)

    baseline_df = (
        plot_df.loc[plot_df["regime"] == "random_cv", ["model", "rmse_mean"]]
        .rename(columns={"rmse_mean": "random_cv_rmse"})
    )
    ratio_df = plot_df.merge(baseline_df, on="model", how="left")
    ratio_df["rmse_ratio_vs_random"] = ratio_df["rmse_mean"] / ratio_df["random_cv_rmse"]
    ratio_df = ratio_df.loc[ratio_df["regime"] != "random_cv"].copy()

    order = [regime for regime in MAIN_REGIME_ORDER if regime in plot_df["regime"].astype(str).unique()]
    ratio_order = [regime for regime in ["group_exp", "group_aut"] if regime in ratio_df["regime"].astype(str).unique()]
    x_main = np.arange(len(order))
    x_ratio = np.arange(len(ratio_order))

    upper_bounds = plot_df["rmse_mean"] + plot_df["rmse_sd"]
    finite_upper = np.sort(upper_bounds.to_numpy(dtype=float))
    if len(finite_upper) > 1 and finite_upper[-1] > finite_upper[-2] * 3.0:
        display_top = max(finite_upper[-2] * 1.10, 3.0)
    else:
        display_top = max(finite_upper[-1] * 1.08, 3.0)
    clipped_level = display_top * 0.94

    fig = plt.figure(figsize=(10.8, 4.8))
    grid = fig.add_gridspec(1, 2, width_ratios=[1.55, 1.0], wspace=0.28)
    ax_main = fig.add_subplot(grid[0, 0])
    ax_ratio = fig.add_subplot(grid[0, 1])

    outlier_notes: list[str] = []
    for model in plot_df["model"].drop_duplicates():
        subset = plot_df.loc[plot_df["model"] == model].set_index("regime").reindex(order)
        means = subset["rmse_mean"].to_numpy(dtype=float)
        sds = subset["rmse_sd"].to_numpy(dtype=float)
        uppers = means + sds
        color = MODEL_COLORS.get(model, "#4C78A8")
        marker = MODEL_MARKERS.get(model, "o")

        ax_main.plot(
            x_main,
            np.minimum(means, clipped_level),
            color=color,
            marker=marker,
            linewidth=2.3,
            markersize=7,
            label=MODEL_LABELS.get(model, model),
        )

        for idx, (x_value, mean, sd, upper) in enumerate(zip(x_main, means, sds, uppers, strict=False)):
            if not np.isfinite(mean):
                continue
            if upper <= display_top:
                lower = max(mean - sd, 0.0)
                ax_main.vlines(x_value, lower, upper, color=color, linewidth=1.5, alpha=0.8)
                ax_main.hlines([lower, upper], x_value - 0.05, x_value + 0.05, color=color, linewidth=1.5, alpha=0.8)
            else:
                regime = order[idx]
                ax_main.scatter([x_value], [clipped_level], color=color, s=55, marker=marker, zorder=3)
                ax_main.annotate(
                    f"{mean:.2f} +/- {sd:.2f}",
                    xy=(x_value, clipped_level),
                    xytext=(x_value - 0.26, display_top * 0.985),
                    fontsize=8.5,
                    color=color,
                    ha="left",
                    va="bottom",
                    arrowprops={"arrowstyle": "-|>", "color": color, "lw": 1.1},
                )
                outlier_notes.append(f"{MODEL_LABELS.get(model, model)} {regime}")

    ax_main.set_xticks(x_main, [REGIME_LABELS.get(regime, regime) for regime in order])
    ax_main.set_ylim(0.0, display_top)
    ax_main.set_ylabel("Mean RMSE on original $q_e$ scale")
    ax_main.text(-0.14, 1.03, "A", transform=ax_main.transAxes, fontsize=12, fontweight="bold")
    _style_axis(ax_main)
    ax_main.legend(frameon=False, ncol=3, loc="upper left")

    for model in ratio_df["model"].drop_duplicates():
        subset = ratio_df.loc[ratio_df["model"] == model].set_index("regime").reindex(ratio_order)
        ratios = subset["rmse_ratio_vs_random"].to_numpy(dtype=float)
        color = MODEL_COLORS.get(model, "#4C78A8")
        marker = MODEL_MARKERS.get(model, "o")
        ax_ratio.plot(
            x_ratio,
            ratios,
            color=color,
            marker=marker,
            linewidth=2.3,
            markersize=7,
        )
        for x_value, ratio in zip(x_ratio, ratios, strict=False):
            if np.isfinite(ratio):
                ax_ratio.text(
                    x_value,
                    ratio * (1.05 if ratio < 5 else 1.03),
                    f"{ratio:.1f}x",
                    color=color,
                    fontsize=8,
                    ha="center",
                    va="bottom",
                )

    ax_ratio.axhline(1.0, color="#7A7A7A", linewidth=1.1, linestyle="--")
    ax_ratio.set_yscale("log")
    ax_ratio.set_xticks(x_ratio, [REGIME_LABELS.get(regime, regime) for regime in ratio_order])
    ax_ratio.set_ylabel("RMSE relative to random CV")
    ax_ratio.text(-0.16, 1.03, "B", transform=ax_ratio.transAxes, fontsize=12, fontweight="bold")
    ratio_top = max(float(np.nanmax(ratio_df["rmse_ratio_vs_random"])), 2.0)
    ax_ratio.set_ylim(0.7, ratio_top * 1.25)
    ratio_ticks = [0.75, 1, 1.5, 2, 4, 8, 16, 32]
    visible_ticks = [tick for tick in ratio_ticks if ax_ratio.get_ylim()[0] <= tick <= ax_ratio.get_ylim()[1]]
    ax_ratio.set_yticks(visible_ticks)
    ax_ratio.set_yticklabels([f"{tick:g}x" for tick in visible_ticks])
    _style_axis(ax_ratio)

    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.10, top=0.94, wspace=0.28)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def plot_validation_hierarchy_r2(summary_df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plot_df = summary_df.loc[summary_df["regime"].isin(MAIN_REGIME_ORDER), ["regime", "model", "r2_mean", "r2_sd"]].copy()
    plot_df["regime"] = pd.Categorical(plot_df["regime"], categories=MAIN_REGIME_ORDER, ordered=True)
    plot_df = plot_df.sort_values(["model", "regime"]).reset_index(drop=True)

    order = [regime for regime in MAIN_REGIME_ORDER if regime in plot_df["regime"].astype(str).unique()]
    x_main = np.arange(len(order))

    in_range_mask = plot_df["r2_mean"] > -3.0
    lower_bounds = (plot_df.loc[in_range_mask, "r2_mean"] - plot_df.loc[in_range_mask, "r2_sd"]).to_numpy(dtype=float)
    if len(lower_bounds) > 0:
        display_bottom = min(float(np.nanmin(lower_bounds)) * 1.08, -0.25)
    else:
        display_bottom = -1.0
    clipped_level = display_bottom * 0.96

    def _format_r2_value(value: float) -> str:
        if abs(value) >= 1000:
            return f"{value:.2e}".replace("e+0", "e").replace("e-0", "e-")
        return f"{value:.2f}"

    annotation_offsets = {
        "lightgbm": (-0.52, 0.92),
        "mlp_regressor": (-0.18, 0.92),
    }

    fig, ax = plt.subplots(figsize=(8.8, 4.8))

    for model in plot_df["model"].drop_duplicates():
        subset = plot_df.loc[plot_df["model"] == model].set_index("regime").reindex(order)
        means = subset["r2_mean"].to_numpy(dtype=float)
        sds = subset["r2_sd"].to_numpy(dtype=float)
        lowers = means - sds
        uppers = means + sds
        color = MODEL_COLORS.get(model, "#4C78A8")
        marker = MODEL_MARKERS.get(model, "o")

        ax.plot(
            x_main,
            np.maximum(means, clipped_level),
            color=color,
            marker=marker,
            linewidth=2.3,
            markersize=7,
            label=MODEL_LABELS.get(model, model),
        )

        for idx, (x_value, mean, sd, lower, upper) in enumerate(zip(x_main, means, sds, lowers, uppers, strict=False)):
            if not np.isfinite(mean):
                continue
            if mean >= display_bottom and lower >= display_bottom:
                ax.vlines(x_value, lower, upper, color=color, linewidth=1.5, alpha=0.8)
                ax.hlines([lower, upper], x_value - 0.05, x_value + 0.05, color=color, linewidth=1.5, alpha=0.8)
            elif mean >= display_bottom and lower < display_bottom:
                ax.vlines(x_value, display_bottom, upper, color=color, linewidth=1.5, alpha=0.8)
                ax.hlines([upper], x_value - 0.05, x_value + 0.05, color=color, linewidth=1.5, alpha=0.8)
            else:
                ax.scatter([x_value], [clipped_level], color=color, s=55, marker=marker, zorder=3)
                x_offset, y_offset = annotation_offsets.get(model, (-0.30, 0.80))
                ax.annotate(
                    f"{_format_r2_value(mean)} +/- {_format_r2_value(sd)}",
                    xy=(x_value, clipped_level),
                    xytext=(x_value + x_offset, display_bottom + y_offset),
                    fontsize=8.5,
                    color=color,
                    ha="left",
                    va="bottom",
                    arrowprops={"arrowstyle": "-|>", "color": color, "lw": 1.1},
                )

    ax.axhline(0.0, color="#7A7A7A", linewidth=1.1, linestyle="--")
    ax.set_xticks(x_main, [REGIME_LABELS.get(regime, regime) for regime in order])
    ax.set_ylim(display_bottom, 1.02)
    ax.set_ylabel("Mean $R^2$")
    ax.set_title("Validation hierarchy on the $R^2$ scale (zoomed near zero)")
    _style_axis(ax)
    ax.legend(frameon=False, ncol=3, loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def plot_mechanism_heatmap(mechanism_df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    filtered = mechanism_df.loc[
        mechanism_df["mechanism_group"] != "null_feature"
    ].copy()
    plot_df = (
        filtered.groupby(["method", "mechanism_group"], as_index=False)["mean_importance_share"]
        .mean()
        .pivot(index="method", columns="mechanism_group", values="mean_importance_share")
        .fillna(0.0)
    )
    plot_df = plot_df.rename(index=METHOD_LABELS)
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
    method_scores = (
        reliability_df.groupby("method", as_index=False)[
            ["mean_null_share", "mean_real_null_gap", "rank_stability", "heuristic_reliability_score"]
        ]
        .mean()
        .sort_values(["heuristic_reliability_score", "rank_stability"], ascending=False)
    )
    method_scores["method"] = method_scores["method"].map(lambda value: METHOD_LABELS.get(value, value))

    fig, axes = plt.subplots(ncols=2, figsize=(10, 4.8), sharey=True)
    y = np.arange(len(method_scores))

    axes[0].barh(y, method_scores["mean_real_null_gap"], color="#5b7db1")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(method_scores["method"])
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Mean real-null importance gap")
    axes[0].grid(axis="x", alpha=0.2)

    axes[1].barh(y, method_scores["rank_stability"], color="#c97b63")
    axes[1].set_xlabel("Mean rank stability")
    axes[1].grid(axis="x", alpha=0.2)

    fig.suptitle("Null-feature benchmarking across importance methods")
    fig.tight_layout(rect=(0, 0.02, 1, 0.94))
    fig.savefig(path, dpi=300)
    plt.close(fig)


def plot_stability_heatmap(stability_df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plot_df = stability_df.pivot(index="method", columns="regime", values="rank_stability").fillna(0.0)
    plot_df = plot_df.reindex(columns=[column for column in REGIME_ORDER if column in plot_df.columns])
    plot_df = plot_df.rename(index=METHOD_LABELS)
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
