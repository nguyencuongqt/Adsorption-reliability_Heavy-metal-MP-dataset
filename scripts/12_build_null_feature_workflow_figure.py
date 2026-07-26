from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


STUDY_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = STUDY_ROOT / "results" / "figures"
SI_DIR = STUDY_ROOT / "manuscript_package" / "supplementary_information" / "si_figures"

RESULT_BASENAME = "figure_null_feature_workflow_diagram"
SI_BASENAME = "sifig05_null_feature_workflow_diagram"

COLORS = {
    "ink": "#23313B",
    "muted": "#64727D",
    "gold_fill": "#F4EBC8",
    "gold_edge": "#A7841A",
    "red_fill": "#F8DEDB",
    "red_edge": "#BE4D4A",
    "blue_fill": "#DCE9F5",
    "blue_edge": "#3D6E9E",
    "green_fill": "#DCEDE4",
    "green_edge": "#2D725A",
    "panel_fill": "#F7F5F1",
    "panel_edge": "#C9C3BA",
    "white": "#FFFFFF",
}


def add_round_box(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    facecolor: str,
    edgecolor: str,
    title: str,
    body: str,
    title_size: float = 12.2,
    body_size: float = 9.4,
    rounding: float = 0.18,
) -> None:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.03,rounding_size={rounding}",
        linewidth=1.8,
        facecolor=facecolor,
        edgecolor=edgecolor,
    )
    ax.add_patch(patch)
    ax.text(
        x + 0.22,
        y + h - 0.34,
        title,
        ha="left",
        va="top",
        fontsize=title_size,
        fontweight="bold",
        color=COLORS["ink"],
    )
    ax.text(
        x + 0.22,
        y + h - 0.78,
        body,
        ha="left",
        va="top",
        fontsize=body_size,
        color=COLORS["ink"],
        linespacing=1.25,
    )


def add_small_box(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    facecolor: str,
    edgecolor: str,
    text: str,
    fontsize: float = 9.2,
) -> None:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.12",
        linewidth=1.2,
        facecolor=facecolor,
        edgecolor=edgecolor,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=COLORS["ink"],
        fontweight="bold",
    )


def add_arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = "#6B7780",
    lw: float = 2.2,
    style: str = "-|>",
    mutation_scale: float = 17,
    connectionstyle: str = "arc3",
) -> None:
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle=style,
        mutation_scale=mutation_scale,
        linewidth=lw,
        color=color,
        connectionstyle=connectionstyle,
    )
    ax.add_patch(arrow)


def build_figure() -> plt.Figure:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})

    fig = plt.figure(figsize=(14.0, 8.0), facecolor="white")
    ax = fig.add_axes([0.02, 0.03, 0.96, 0.94])
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis("off")

    ax.text(
        0.45,
        9.65,
        "Null-feature workflow for interpretation reliability",
        ha="left",
        va="top",
        fontsize=19,
        fontweight="bold",
        color=COLORS["ink"],
    )
    ax.text(
        0.45,
        9.23,
        "SI schematic for Text S1.",
        ha="left",
        va="top",
        fontsize=10.8,
        color=COLORS["muted"],
    )

    add_round_box(
        ax,
        0.55,
        6.05,
        2.65,
        2.2,
        facecolor=COLORS["gold_fill"],
        edgecolor=COLORS["gold_edge"],
        title="1. Locked data",
        body=(
            "Adsorption data\n"
            "Target: qe\n"
            "Real predictors\n"
            "Groups: aut_id, exp_id"
        ),
    )

    add_round_box(
        ax,
        3.7,
        5.35,
        3.4,
        3.0,
        facecolor=COLORS["red_fill"],
        edgecolor=COLORS["red_edge"],
        title="2. Create synthetic\nnull features",
        body=(
            "Negative controls\n"
            "with no link to qe"
        ),
        body_size=9.3,
    )
    add_small_box(
        ax,
        4.05,
        6.78,
        2.7,
        0.48,
        facecolor=COLORS["white"],
        edgecolor=COLORS["red_edge"],
        text="Random noise",
    )
    add_small_box(
        ax,
        4.05,
        6.13,
        2.7,
        0.48,
        facecolor=COLORS["white"],
        edgecolor=COLORS["red_edge"],
        text="Permuted features",
    )
    add_small_box(
        ax,
        4.05,
        5.48,
        2.7,
        0.48,
        facecolor=COLORS["white"],
        edgecolor=COLORS["red_edge"],
        text="Grouped effects",
    )

    add_round_box(
        ax,
        7.65,
        6.05,
        3.0,
        2.2,
        facecolor=COLORS["panel_fill"],
        edgecolor=COLORS["panel_edge"],
        title="3. Append null\ncontrols",
        body=(
            "Add seven null variables.\n"
            "Rank real and null features\n"
            "in one shared pipeline."
        ),
        body_size=9.3,
    )

    add_round_box(
        ax,
        11.15,
        5.8,
        4.15,
        2.45,
        facecolor=COLORS["blue_fill"],
        edgecolor=COLORS["blue_edge"],
        title="4. Audit importance\nmethods",
        body=(
            "EN coefficient\n"
            "EN permutation\n"
            "LGBM gain, permutation, SHAP"
        ),
        body_size=9.2,
    )
    add_small_box(
        ax,
        11.48,
        5.2,
        0.95,
        0.46,
        facecolor=COLORS["white"],
        edgecolor=COLORS["blue_edge"],
        text="RRCV",
        fontsize=8.9,
    )
    add_small_box(
        ax,
        12.52,
        5.2,
        0.95,
        0.46,
        facecolor=COLORS["white"],
        edgecolor=COLORS["blue_edge"],
        text="EGCV",
        fontsize=8.9,
    )
    add_small_box(
        ax,
        13.56,
        5.2,
        0.95,
        0.46,
        facecolor=COLORS["white"],
        edgecolor=COLORS["blue_edge"],
        text="SGCV",
        fontsize=8.9,
    )

    add_round_box(
        ax,
        2.0,
        1.45,
        4.05,
        2.35,
        facecolor=COLORS["green_fill"],
        edgecolor=COLORS["green_edge"],
        title="5. Compare real\nvs null signals",
        body=(
            "Null share\n"
            "Real-null separation\n"
            "Top-rank intrusion\n"
            "Rank stability"
        ),
        body_size=9.4,
    )

    add_round_box(
        ax,
        6.55,
        1.45,
        3.35,
        2.35,
        facecolor=COLORS["green_fill"],
        edgecolor=COLORS["green_edge"],
        title="6. Test transfer\nrobustness",
        body=(
            "Do patterns stay similar\n"
            "from interpolation\n"
            "to harder transfer?"
        ),
        body_size=9.4,
    )

    add_round_box(
        ax,
        10.4,
        1.45,
        4.9,
        2.35,
        facecolor=COLORS["panel_fill"],
        edgecolor=COLORS["panel_edge"],
        title="7. Read results\nconservatively",
        body=(
            "More reliable = low null influence\n"
            "+ stable real-feature ranks.\n"
            "Null intrusion is a warning."
        ),
        body_size=9.3,
    )

    add_arrow(ax, (3.22, 7.12), (3.68, 7.12), color=COLORS["gold_edge"])
    add_arrow(ax, (7.12, 7.12), (7.63, 7.12), color=COLORS["red_edge"])
    add_arrow(ax, (10.68, 7.12), (11.12, 7.12), color=COLORS["blue_edge"])
    add_arrow(ax, (13.23, 5.8), (8.23, 3.84), color=COLORS["blue_edge"], connectionstyle="arc3,rad=0.0")
    add_arrow(ax, (6.08, 2.62), (6.5, 2.62), color=COLORS["green_edge"])
    add_arrow(ax, (9.95, 2.62), (10.36, 2.62), color=COLORS["green_edge"])

    ax.text(
        0.58,
        0.78,
        "Null features act as negative controls inside the same grouped validation workflow.",
        ha="left",
        va="bottom",
        fontsize=9.8,
        color=COLORS["muted"],
    )
    ax.text(
        0.58,
        0.42,
        "This tests spurious attribution, not causality.",
        ha="left",
        va="bottom",
        fontsize=9.8,
        color=COLORS["muted"],
    )

    return fig


def save_variants(fig: plt.Figure, output_dir: Path, basename: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in [".png", ".pdf", ".tiff"]:
        fig.savefig(
            output_dir / f"{basename}{suffix}",
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
        )


def main() -> None:
    fig = build_figure()
    save_variants(fig, RESULTS_DIR, RESULT_BASENAME)
    save_variants(fig, SI_DIR, SI_BASENAME)
    plt.close(fig)


if __name__ == "__main__":
    main()
