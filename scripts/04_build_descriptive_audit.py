from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


STUDY_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = STUDY_ROOT / "data" / "inputs" / "05_modeling_dataset_final.csv"
OUTPUT_DIR = STUDY_ROOT / "results" / "descriptive_audit"
FIG_DIR = OUTPUT_DIR / "figures"
TABLE_DIR = OUTPUT_DIR / "tables"
NOTES_PATH = OUTPUT_DIR / "descriptive_output_notes.md"

METAL_COLORS = {
    "Cd": "#4C78A8",
    "Cr": "#F58518",
    "Hg": "#54A24B",
}
PROFILE_COLORS = {
    "Cd only": "#4C78A8",
    "Cr only": "#F58518",
    "Hg only": "#54A24B",
    "Mixed metals": "#B279A2",
}
POLYMER_ORDER = ["Other", "PA", "PE", "PET", "PLA", "PP", "PS", "PVC"]
METAL_ORDER = ["Cd", "Cr", "Hg"]
AGING_ORDER = ["Virgin", "Aged"]
REPORT_VARS = ["ph", "temp", "ce", "rpm", "sa"]


def ensure_dirs() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)


def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(DATASET_PATH)
    df["metal"] = df[["metal_cd", "metal_cr", "metal_hg"]].idxmax(axis=1).map(
        {"metal_cd": "Cd", "metal_cr": "Cr", "metal_hg": "Hg"}
    )
    df["polymer"] = df[
        ["ret_other", "ret_pa", "ret_pe", "ret_pet", "ret_pla", "ret_pp", "ret_ps", "ret_pvc"]
    ].idxmax(axis=1).map(
        {
            "ret_other": "Other",
            "ret_pa": "PA",
            "ret_pe": "PE",
            "ret_pet": "PET",
            "ret_pla": "PLA",
            "ret_pp": "PP",
            "ret_ps": "PS",
            "ret_pvc": "PVC",
        }
    )
    df["aging"] = np.where(df["ags_aged"].eq(1), "Aged", "Virgin")
    df["log1p_qe"] = np.log1p(df["qe"])
    df["log10_ce_plus"] = np.log10(df["ce"] + 1e-4)
    return df


def write_table(df: pd.DataFrame, stem: str) -> tuple[Path, Path]:
    csv_path = TABLE_DIR / f"{stem}.csv"
    xlsx_path = TABLE_DIR / f"{stem}.xlsx"
    df.to_csv(csv_path, index=False)
    df.to_excel(xlsx_path, index=False)
    return csv_path, xlsx_path


def _fmt_range(series: pd.Series, digits: int = 3) -> str:
    observed = series.dropna()
    if observed.empty:
        return "not reported"
    return f"{observed.min():.{digits}f}-{observed.max():.{digits}f}"


def build_metal_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    metal_study = df.groupby("aut_id")["metal"].nunique()
    for metal in [*METAL_ORDER, "All"]:
        subset = df if metal == "All" else df.loc[df["metal"] == metal].copy()
        label = metal
        records.append(
            {
                "metal": label,
                "rows": int(len(subset)),
                "studies": int(subset["aut_id"].nunique()),
                "experiments": int(subset["exp_id"].nunique()),
                "median_qe": round(float(subset["qe"].median()), 3),
                "iqr_qe": f"{subset['qe'].quantile(0.25):.3f}-{subset['qe'].quantile(0.75):.3f}",
                "p95_qe": round(float(subset["qe"].quantile(0.95)), 3),
                "max_qe": round(float(subset["qe"].max()), 3),
                "pH_reported_pct": round(float(subset["ph"].notna().mean() * 100), 1),
                "temp_reported_pct": round(float(subset["temp"].notna().mean() * 100), 1),
                "ce_reported_pct": round(float(subset["ce"].notna().mean() * 100), 1),
                "rpm_reported_pct": round(float(subset["rpm"].notna().mean() * 100), 1),
                "sa_reported_pct": round(float(subset["sa"].notna().mean() * 100), 1),
                "pH_range": _fmt_range(subset["ph"], digits=2),
                "temp_range": _fmt_range(subset["temp"], digits=1),
                "ce_range": _fmt_range(subset["ce"], digits=3),
                "rpm_range": _fmt_range(subset["rpm"], digits=1),
                "sa_range": _fmt_range(subset["sa"], digits=3),
                "rows_in_multi_metal_studies": int(subset["aut_id"].map(metal_study).gt(1).sum()),
            }
        )
    return pd.DataFrame(records)


def build_coverage_table(df: pd.DataFrame) -> pd.DataFrame:
    full_index = pd.MultiIndex.from_product(
        [METAL_ORDER, POLYMER_ORDER, AGING_ORDER], names=["metal", "polymer", "aging"]
    )
    coverage = (
        df.groupby(["metal", "polymer", "aging"])
        .agg(n_rows=("qe", "size"), n_studies=("aut_id", "nunique"), n_experiments=("exp_id", "nunique"))
        .reindex(full_index, fill_value=0)
        .reset_index()
    )
    metal_totals = coverage.groupby("metal")["n_rows"].transform("sum")
    coverage["within_metal_pct"] = np.where(
        metal_totals.gt(0), (coverage["n_rows"] / metal_totals * 100).round(1), 0.0
    )
    coverage["coverage_flag"] = np.select(
        [
            coverage["n_rows"].eq(0),
            coverage["n_rows"].between(1, 5),
            coverage["n_rows"].between(6, 20),
        ],
        ["absent", "very sparse", "thin"],
        default="moderate or denser",
    )
    return coverage


def build_study_audit_table(df: pd.DataFrame) -> pd.DataFrame:
    metal_presence = (
        df.assign(flag=1)
        .pivot_table(index="aut_id", columns="metal", values="flag", aggfunc="max", fill_value=0)
        .reindex(columns=METAL_ORDER, fill_value=0)
    )
    study_table = df.groupby("aut_id").agg(
        rows=("qe", "size"),
        experiments=("exp_id", "nunique"),
        qe_median=("qe", "median"),
        qe_p95=("qe", lambda s: float(s.quantile(0.95))),
        pH_missing_pct=("ph", lambda s: float(s.isna().mean() * 100)),
        temp_missing_pct=("temp", lambda s: float(s.isna().mean() * 100)),
        ce_missing_pct=("ce", lambda s: float(s.isna().mean() * 100)),
        rpm_missing_pct=("rpm", lambda s: float(s.isna().mean() * 100)),
        sa_missing_pct=("sa", lambda s: float(s.isna().mean() * 100)),
    )
    study_table = study_table.join(metal_presence)
    study_table["metals_present"] = study_table[METAL_ORDER].sum(axis=1).astype(int)
    study_table["metal_profile"] = study_table[METAL_ORDER].apply(
        lambda row: " + ".join(metal for metal in METAL_ORDER if row[metal] == 1), axis=1
    )
    study_table["rows_share_pct"] = study_table["rows"] / len(df) * 100
    study_table = study_table.reset_index().sort_values(["rows", "aut_id"], ascending=[False, True])
    numeric_cols = [
        "qe_median",
        "qe_p95",
        "pH_missing_pct",
        "temp_missing_pct",
        "ce_missing_pct",
        "rpm_missing_pct",
        "sa_missing_pct",
        "rows_share_pct",
    ]
    study_table.loc[:, numeric_cols] = study_table.loc[:, numeric_cols].round(3)
    return study_table


def _style_axis(ax: plt.Axes) -> None:
    ax.grid(axis="y", alpha=0.18, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _save_figure(fig: plt.Figure, stem: str, formats: tuple[str, ...] = ("png",), dpi: int = 300) -> Path:
    primary_path: Path | None = None
    for fmt in formats:
        out = FIG_DIR / f"{stem}.{fmt}"
        fig.savefig(out, dpi=dpi, bbox_inches="tight")
        if primary_path is None:
            primary_path = out
    assert primary_path is not None
    return primary_path


def _populate_qe_distribution_axes(axes: np.ndarray, df: pd.DataFrame) -> None:
    axes[0, 0].hist(df["qe"], bins=40, color="#5B8E7D", edgecolor="white")
    axes[0, 0].set_title("Overall qe distribution")
    axes[0, 0].set_xlabel("qe")
    axes[0, 0].set_ylabel("Rows")
    _style_axis(axes[0, 0])

    axes[0, 1].hist(df["log1p_qe"], bins=40, color="#C97B63", edgecolor="white")
    axes[0, 1].set_title("Overall log1p(qe) distribution")
    axes[0, 1].set_xlabel("log1p(qe)")
    axes[0, 1].set_ylabel("Rows")
    _style_axis(axes[0, 1])

    qe_groups = [df.loc[df["metal"] == metal, "qe"].to_numpy() for metal in METAL_ORDER]
    box1 = axes[1, 0].boxplot(qe_groups, patch_artist=True, labels=METAL_ORDER, showfliers=False)
    for patch, metal in zip(box1["boxes"], METAL_ORDER, strict=False):
        patch.set_facecolor(METAL_COLORS[metal])
        patch.set_alpha(0.7)
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_title("qe by metal (log y-scale)")
    axes[1, 0].set_ylabel("qe")
    _style_axis(axes[1, 0])

    log_groups = [df.loc[df["metal"] == metal, "log1p_qe"].to_numpy() for metal in METAL_ORDER]
    box2 = axes[1, 1].boxplot(log_groups, patch_artist=True, labels=METAL_ORDER, showfliers=False)
    for patch, metal in zip(box2["boxes"], METAL_ORDER, strict=False):
        patch.set_facecolor(METAL_COLORS[metal])
        patch.set_alpha(0.7)
    axes[1, 1].set_title("log1p(qe) by metal")
    axes[1, 1].set_ylabel("log1p(qe)")
    _style_axis(axes[1, 1])


def plot_qe_distributions(df: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.4))
    _populate_qe_distribution_axes(axes, df)
    fig.suptitle("Descriptive audit of adsorption-capacity distributions", fontsize=13)
    fig.tight_layout(rect=(0, 0.01, 1, 0.97))
    out = _save_figure(fig, "fig_descriptive_qe_distributions")
    plt.close(fig)
    return out


def _heatmap_with_labels(
    ax: plt.Axes,
    data: np.ndarray,
    row_labels: list[str],
    col_labels: list[str],
    title: str,
    cmap: str,
    fmt: str = ".1f",
    text_values: np.ndarray | None = None,
) -> None:
    image = ax.imshow(data, aspect="auto", cmap=cmap)
    ax.set_title(title)
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            shown = text_values[i, j] if text_values is not None else format(data[i, j], fmt)
            ax.text(j, i, shown, ha="center", va="center", fontsize=8)
    plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04)


def plot_coverage_heatmap(coverage: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), sharey=True)
    for ax, aging in zip(axes, AGING_ORDER, strict=False):
        subset = coverage.loc[coverage["aging"] == aging].copy()
        values = (
            subset.pivot(index="metal", columns="polymer", values="within_metal_pct")
            .reindex(index=METAL_ORDER, columns=POLYMER_ORDER)
            .fillna(0.0)
        )
        counts = (
            subset.pivot(index="metal", columns="polymer", values="n_rows")
            .reindex(index=METAL_ORDER, columns=POLYMER_ORDER)
            .fillna(0)
            .astype(int)
        )
        _heatmap_with_labels(
            ax,
            values.to_numpy(),
            METAL_ORDER,
            POLYMER_ORDER,
            f"{aging} samples within each metal (%)",
            cmap="YlGnBu",
            fmt=".1f",
            text_values=counts.to_numpy().astype(str),
        )
    fig.suptitle("Coverage audit: metal x polymer x aging", fontsize=13)
    fig.tight_layout(rect=(0, 0.01, 1, 0.95))
    out = _save_figure(fig, "fig_coverage_metal_polymer_aging", formats=("png", "pdf", "tiff"))
    plt.close(fig)
    return out


def _boxplot_by_metal(ax: plt.Axes, df: pd.DataFrame, column: str, title: str, ylabel: str, transform: str | None = None) -> None:
    data = []
    labels = []
    positions = []
    for idx, metal in enumerate(METAL_ORDER, start=1):
        series = df.loc[df["metal"] == metal, column].dropna().copy()
        if transform == "log10_ce_plus":
            series = np.log10(series + 1e-4)
        if series.empty:
            continue
        data.append(series.to_numpy())
        labels.append(metal)
        positions.append(idx)
    if data:
        box = ax.boxplot(data, positions=positions, patch_artist=True, showfliers=False, widths=0.55)
        for patch, label in zip(box["boxes"], labels, strict=False):
            patch.set_facecolor(METAL_COLORS[label])
            patch.set_alpha(0.7)
        for pos, label in zip(positions, labels, strict=False):
            values = df.loc[df["metal"] == label, column].dropna().copy()
            if transform == "log10_ce_plus":
                values = np.log10(values + 1e-4)
            if values.empty:
                continue
            sample = values.sample(min(len(values), 120), random_state=42)
            jitter = np.random.default_rng(42 + pos).uniform(-0.10, 0.10, size=len(sample))
            ax.scatter(np.full(len(sample), pos) + jitter, sample, s=8, alpha=0.22, color="#333333")
    ax.set_xticks([1, 2, 3], METAL_ORDER)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    _style_axis(ax)


def _populate_condition_range_axes(axes: list[plt.Axes] | np.ndarray, df: pd.DataFrame) -> None:
    _boxplot_by_metal(axes[0], df, "ce", "Reported Ce by metal", "log10(Ce + 1e-4)", transform="log10_ce_plus")
    _boxplot_by_metal(axes[1], df, "rpm", "Reported rpm by metal", "rpm")


def plot_condition_ranges(df: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    _populate_condition_range_axes(axes, df)
    fig.suptitle("Comparability audit for reported Ce and agitation speed", fontsize=13)
    fig.tight_layout(rect=(0, 0.01, 1, 0.94))
    out = _save_figure(fig, "fig_condition_ranges_by_metal", formats=("png", "pdf", "tiff"))
    plt.close(fig)
    return out


def plot_qe_and_condition_combo(df: pd.DataFrame) -> Path:
    fig = plt.figure(figsize=(11.8, 11.8))
    grid = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.0, 0.95])

    qe_axes = np.array(
        [
            [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])],
            [fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1])],
        ]
    )
    condition_axes = np.array([fig.add_subplot(grid[2, 0]), fig.add_subplot(grid[2, 1])], dtype=object)

    _populate_qe_distribution_axes(qe_axes, df)
    _populate_condition_range_axes(condition_axes, df)

    fig.suptitle("Descriptive audit of qe distributions and key comparable conditions", fontsize=13)
    fig.tight_layout(rect=(0, 0.01, 1, 0.97))
    out = _save_figure(fig, "fig_qe_and_condition_ranges_combined", formats=("png", "pdf", "tiff"))
    plt.close(fig)
    return out


def plot_missingness_audit(df: pd.DataFrame, study_table: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 6.5), gridspec_kw={"width_ratios": [1.0, 1.55]})

    metal_missing = (
        df.groupby("metal")[REPORT_VARS]
        .apply(lambda g: g.isna().mean() * 100)
        .reindex(METAL_ORDER)
        .round(1)
    )
    _heatmap_with_labels(
        axes[0],
        metal_missing.to_numpy(),
        METAL_ORDER,
        REPORT_VARS,
        "Missingness by metal (%)",
        cmap="OrRd",
        fmt=".1f",
    )

    study_missing = (
        study_table.set_index("aut_id")[
            ["pH_missing_pct", "temp_missing_pct", "ce_missing_pct", "rpm_missing_pct", "sa_missing_pct"]
        ]
        .rename(
            columns={
                "pH_missing_pct": "ph",
                "temp_missing_pct": "temp",
                "ce_missing_pct": "ce",
                "rpm_missing_pct": "rpm",
                "sa_missing_pct": "sa",
            }
        )
    )
    study_labels = [f"{aut} ({rows:.1f}%)" for aut, rows in zip(study_table["aut_id"], study_table["rows_share_pct"], strict=False)]
    _heatmap_with_labels(
        axes[1],
        study_missing.to_numpy(),
        study_labels,
        REPORT_VARS,
        "Missingness by study (% of rows missing)",
        cmap="PuRd",
        fmt=".0f",
    )
    axes[1].tick_params(axis="y", labelsize=7)

    fig.suptitle("Reporting-completeness audit", fontsize=13)
    fig.tight_layout(rect=(0, 0.01, 1, 0.95))
    out = FIG_DIR / "fig_missingness_audit.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    return out


def plot_clustering_audit(df: pd.DataFrame, study_table: pd.DataFrame) -> Path:
    profile_map = {}
    for _, row in study_table.iterrows():
        if row["metals_present"] > 1:
            profile_map[row["aut_id"]] = "Mixed metals"
        elif row["Cd"] == 1:
            profile_map[row["aut_id"]] = "Cd only"
        elif row["Cr"] == 1:
            profile_map[row["aut_id"]] = "Cr only"
        else:
            profile_map[row["aut_id"]] = "Hg only"

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.6))

    ordered = study_table.copy()
    ordered["profile"] = ordered["aut_id"].map(profile_map)
    axes[0].bar(
        np.arange(len(ordered)),
        ordered["rows"],
        color=[PROFILE_COLORS[p] for p in ordered["profile"]],
    )
    axes[0].set_xticks(np.arange(len(ordered)))
    axes[0].set_xticklabels(ordered["aut_id"], rotation=75, ha="right", fontsize=7)
    axes[0].set_ylabel("Rows")
    axes[0].set_title("Rows per study, sorted descending")
    _style_axis(axes[0])

    sorted_rows = ordered["rows"].to_numpy()
    cumulative_share = np.cumsum(sorted_rows) / sorted_rows.sum() * 100
    xvals = np.arange(1, len(sorted_rows) + 1)
    axes[1].plot(xvals, cumulative_share, color="#5B8E7D", linewidth=2.4, marker="o", markersize=4)
    axes[1].axhline(50, color="#7A7A7A", linestyle="--", linewidth=1.0)
    axes[1].axhline(80, color="#7A7A7A", linestyle=":", linewidth=1.0)
    for k in [1, 3, 5, 10]:
        if k <= len(xvals):
            axes[1].text(k, cumulative_share[k - 1] + 2, f"{k} studies: {cumulative_share[k - 1]:.1f}%", fontsize=8)
    axes[1].set_xlim(1, len(sorted_rows))
    axes[1].set_ylim(0, 100)
    axes[1].set_xlabel("Studies after sorting by row count")
    axes[1].set_ylabel("Cumulative share of all rows (%)")
    axes[1].set_title("How concentrated are rows across studies?")
    _style_axis(axes[1])

    handles = [
        plt.Line2D([0], [0], color=color, linewidth=8, label=label) for label, color in PROFILE_COLORS.items()
    ]
    axes[0].legend(handles=handles, frameon=False, fontsize=8, loc="upper right")

    fig.suptitle("Cluster and overlap audit for the pooled literature dataset", fontsize=13)
    fig.tight_layout(rect=(0, 0.01, 1, 0.95))
    out = FIG_DIR / "fig_clustering_audit.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    return out


def write_notes() -> Path:
    note = """# Descriptive Coverage Audit Notes

All outputs in this folder are descriptive and should be framed as coverage/comparability audits for pooled literature data. They are not evidence of mechanism or causal effect.

## Figure: fig_descriptive_qe_distributions.png

- Shows: Overall `qe` and `log1p(qe)` distributions plus metal-specific spread for Cd, Cr, and Hg.
- Placement: Main text candidate if a concise visual overview of response heterogeneity is needed; otherwise strong SI figure.
- Caveat: Between-metal contrasts are descriptive only because most studies contribute only one metal.

## Figure: fig_coverage_metal_polymer_aging.png

- Shows: Coverage of `metal x polymer x aging`, with cell labels as sample counts and color as within-metal share.
- Placement: Strong SI figure; useful in main text only if coverage imbalance is central to the framing.
- Caveat: Sparse or absent cells indicate limited literature coverage, not absence of adsorption.

## Figure: fig_condition_ranges_by_metal.png

- Shows: Reported ranges of `Ce` and rpm by metal to audit comparability of the better-covered retained condition variables.
- Placement: SI figure with high interpretive value for reviewers.
- Caveat: `Ce` spans orders of magnitude and rpm coverage remains uneven across metals, so cross-metal comparison is still only partially comparable.

## Figure: fig_qe_and_condition_ranges_combined

- Shows: A combined descriptive overview of response heterogeneity (`qe`, `log1p(qe)`, and metal-specific spread) plus comparability of reported `Ce` and rpm.
- Placement: Main text candidate when a single compact overview figure is preferred.
- Caveat: This combines outcome and condition coverage descriptively only; it does not create matched-condition or causal comparison across metals.

## Figure: fig_missingness_audit.png

- Shows: Missingness by metal and by study for the main continuous reporting variables (`ph`, `temp`, `ce`, `rpm`, `sa`).
- Placement: Strong SI figure; can support main-text statements about structured reporting gaps.
- Caveat: Study-level missingness is highly clustered, so pooled summaries hide substantial between-study reporting differences.

## Figure: fig_clustering_audit.png

- Shows: Study-level row concentration and how quickly cumulative row share accumulates across the largest studies.
- Placement: Main text or SI, especially if grouped structure and portability limits are being emphasized.
- Caveat: This is a clustering audit only; it should not be interpreted as study quality ranking.

## Table: table_metal_descriptive_summary

- Shows: Metal-level counts, grouping depth, `qe` summary statistics, reporting completeness, and observed ranges for key variables.
- Placement: Main text candidate if a compact cross-metal audit table is preferred; otherwise SI table.
- Caveat: Observed ranges mix within-study and between-study heterogeneity and do not imply controlled metal-to-metal contrasts.

## Table: table_coverage_metal_polymer_aging

- Shows: Full count table for each `metal x polymer x aging` combination, with studies, experiments, within-metal share, and a simple sparsity flag.
- Placement: SI table.
- Caveat: Thin cells often reflect literature availability rather than chemically meaningful rarity.

## Table: table_study_level_audit

- Shows: Rows, experiments, metal profile, `qe` summary, row share, and missingness for each study.
- Placement: SI table for transparency and reviewer-oriented audit.
- Caveat: Study-level rows are not independent and should not be read as replicate evidence of the same underlying process.

## Omitted or down-weighted items

- Surface area (`sa`) was not given a standalone range figure because observed coverage is highly uneven across metals and many studies report none at all; it is retained in the tables and missingness audit instead.
- Direct metal-to-metal descriptive contrasts should be interpreted very cautiously because only two studies contain more than one metal in the locked modeling dataset.
"""
    NOTES_PATH.write_text(note, encoding="utf-8")
    return NOTES_PATH


def main() -> int:
    ensure_dirs()
    df = load_dataset()

    metal_summary = build_metal_summary_table(df)
    coverage = build_coverage_table(df)
    study_audit = build_study_audit_table(df)

    write_table(metal_summary, "table_metal_descriptive_summary")
    write_table(coverage, "table_coverage_metal_polymer_aging")
    write_table(study_audit, "table_study_level_audit")

    plot_qe_distributions(df)
    plot_coverage_heatmap(coverage)
    plot_condition_ranges(df)
    plot_qe_and_condition_combo(df)
    plot_missingness_audit(df, study_audit)
    plot_clustering_audit(df, study_audit)
    write_notes()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
