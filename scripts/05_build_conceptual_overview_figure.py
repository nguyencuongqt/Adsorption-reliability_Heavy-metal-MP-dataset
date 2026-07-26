from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch


STUDY_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = STUDY_ROOT / "results" / "figures"
PACKAGE_DIR = STUDY_ROOT / "manuscript_package" / "figures"

RESULT_BASENAME = "figure_conceptual_research_workflow"
PACKAGE_BASENAME = "fig01_candidate_conceptual_workflow_revised"

COLORS = {
    "ink": "#23313B",
    "muted": "#64727D",
    "panel": "#F5F1EA",
    "blue_fill": "#DCE9F5",
    "blue_edge": "#3D6E9E",
    "orange_fill": "#F7E2CF",
    "orange_edge": "#C8752D",
    "red_fill": "#F8DEDB",
    "red_edge": "#BE4D4A",
    "green_fill": "#DCEDE4",
    "green_edge": "#2D725A",
    "gold_fill": "#F4EBC8",
    "gold_edge": "#A7841A",
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
    title_size: float = 14,
    body_size: float = 10.1,
    rounding: float = 0.2,
) -> None:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.02,rounding_size={rounding}",
        linewidth=1.8,
        facecolor=facecolor,
        edgecolor=edgecolor,
    )
    ax.add_patch(patch)
    ax.text(
        x + 0.22,
        y + h - 0.38,
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
        linespacing=1.22,
    )


def add_badge(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    text: str,
    facecolor: str,
    edgecolor: str,
    text_color: str = "#23313B",
    fontsize: float = 11,
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
        color=text_color,
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
    mutation_scale: float = 16,
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


def add_count_circle(
    ax: plt.Axes,
    x: float,
    y: float,
    *,
    number: str,
    label: str,
    facecolor: str,
    edgecolor: str,
) -> None:
    circle = Circle((x, y), radius=0.48, facecolor=facecolor, edgecolor=edgecolor, linewidth=1.6)
    ax.add_patch(circle)
    ax.text(x, y + 0.05, number, ha="center", va="center", fontsize=13, fontweight="bold", color=COLORS["ink"])
    ax.text(x, y - 0.68, label, ha="center", va="top", fontsize=9.5, color=COLORS["muted"])


def build_figure() -> plt.Figure:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
        }
    )

    fig = plt.figure(figsize=(13.5, 7.0), facecolor="white")
    ax = fig.add_axes([0.02, 0.02, 0.96, 0.96])
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9.2)
    ax.axis("off")

    ax.text(
        0.45,
        8.9,
        "Study concept and workflow",
        ha="left",
        va="top",
        fontsize=19,
        fontweight="bold",
        color=COLORS["ink"],
    )
    ax.text(
        0.45,
        8.53,
        "Reliability assessment for literature-derived heavy-metal adsorption by microplastics",
        ha="left",
        va="top",
        fontsize=11.2,
        color=COLORS["muted"],
    )

    add_round_box(
        ax,
        0.45,
        5.55,
        3.15,
        2.15,
        facecolor=COLORS["blue_fill"],
        edgecolor=COLORS["blue_edge"],
        title="1. Research questions",
        body=(
            "RQ1  Validation design under grouped data?\n"
            "RQ2  Interpretation reliability under\n"
            "     method and null-feature stress tests"
        ),
        body_size=10.0,
    )

    add_badge(
        ax,
        0.78,
        5.92,
        1.2,
        0.42,
        text="RQ1",
        facecolor="white",
        edgecolor=COLORS["blue_edge"],
        fontsize=10,
    )
    add_badge(
        ax,
        2.08,
        5.92,
        1.2,
        0.42,
        text="RQ2",
        facecolor="white",
        edgecolor=COLORS["blue_edge"],
        fontsize=10,
    )

    add_round_box(
        ax,
        4.2,
        5.55,
        3.25,
        2.15,
        facecolor=COLORS["gold_fill"],
        edgecolor=COLORS["gold_edge"],
        title="2. Literature-derived data",
        body=(
            "Heavy-metal adsorption by microplastics\n"
            "Target: qe\n"
            "Predictors: 25\n"
            "Structure: grouped + heterogeneous"
        ),
        body_size=10.0,
    )
    add_badge(
        ax,
        4.62,
        5.94,
        0.82,
        0.58,
        text="23\nstudies",
        facecolor="white",
        edgecolor=COLORS["gold_edge"],
        fontsize=10,
    )
    add_badge(
        ax,
        5.4,
        5.94,
        0.96,
        0.58,
        text="149\nexperiments",
        facecolor="white",
        edgecolor=COLORS["gold_edge"],
        fontsize=9.5,
    )
    add_badge(
        ax,
        6.32,
        5.94,
        0.86,
        0.58,
        text="1009\nrows",
        facecolor="white",
        edgecolor=COLORS["gold_edge"],
        fontsize=10,
    )

    add_round_box(
        ax,
        7.8,
        5.55,
        4.2,
        2.15,
        facecolor=COLORS["panel"],
        edgecolor="#B1A79C",
        title="3. Methods",
        body="",
        body_size=9.6,
    )
    ax.text(8.12, 6.92, "train-fold preprocessing only", ha="left", va="top", fontsize=10.0, color=COLORS["muted"])

    add_round_box(
        ax,
        8.12,
        5.8,
        1.76,
        1.0,
        facecolor=COLORS["blue_fill"],
        edgecolor=COLORS["blue_edge"],
        title="RQ1",
        body="EN | LGBM | MLP\n3 validation regimes",
        title_size=11.0,
        body_size=8.6,
        rounding=0.12,
    )
    ax.text(9.97, 6.3, "&", ha="center", va="center", fontsize=15, color=COLORS["muted"], fontweight="bold")
    add_round_box(
        ax,
        10.18,
        5.8,
        1.5,
        1.0,
        facecolor=COLORS["orange_fill"],
        edgecolor=COLORS["orange_edge"],
        title="RQ2",
        body="5 methods\n7 null features",
        title_size=11.0,
        body_size=8.6,
        rounding=0.12,
    )

    add_round_box(
        ax,
        12.4,
        5.55,
        3.05,
        2.15,
        facecolor=COLORS["green_fill"],
        edgecolor=COLORS["green_edge"],
        title="4. Expected outputs",
        body="",
        body_size=9.8,
    )
    ax.text(
        12.72,
        6.78,
        "Generalization audit\nInterpretation audit\nGuidance for reliable ML",
        ha="left",
        va="top",
        fontsize=10.4,
        color=COLORS["ink"],
        linespacing=1.28,
    )

    add_arrow(ax, (3.65, 6.62), (4.05, 6.62), color="#7A8791", lw=2.1)
    add_arrow(ax, (7.4, 6.62), (7.8, 6.62), color="#7A8791", lw=2.1)
    add_arrow(ax, (12.05, 6.62), (12.3, 6.62), color="#7A8791", lw=2.1)

    add_round_box(
        ax,
        1.4,
        2.0,
        13.2,
        1.6,
        facecolor="#FBFBF8",
        edgecolor="#CFC9BF",
        title="Conceptual logic",
        body=(
            "Grouped literature data -> reliability-focused modeling -> two complementary audits:\n"
            "prediction under grouped validation and interpretation under null-feature stress testing"
        ),
        title_size=11.8,
        body_size=10.2,
        rounding=0.12,
    )
    add_arrow(ax, (9.25, 5.52), (8.35, 3.67), color=COLORS["blue_edge"], lw=1.9, connectionstyle="arc3,rad=0.06")
    add_arrow(ax, (10.95, 5.52), (12.6, 3.67), color=COLORS["orange_edge"], lw=1.9, connectionstyle="arc3,rad=-0.06")

    return fig


def save_outputs(fig: plt.Figure) -> list[Path]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PACKAGE_DIR.mkdir(parents=True, exist_ok=True)

    outputs = [
        RESULTS_DIR / f"{RESULT_BASENAME}.png",
        RESULTS_DIR / f"{RESULT_BASENAME}.pdf",
        RESULTS_DIR / f"{RESULT_BASENAME}.tiff",
        PACKAGE_DIR / f"{PACKAGE_BASENAME}.png",
        PACKAGE_DIR / f"{PACKAGE_BASENAME}.pdf",
        PACKAGE_DIR / f"{PACKAGE_BASENAME}.tiff",
    ]

    for path in outputs:
        save_kwargs: dict[str, object] = {"dpi": 300, "bbox_inches": "tight", "facecolor": "white"}
        if path.suffix.lower() in {".tif", ".tiff"}:
            save_kwargs["pil_kwargs"] = {"compression": "tiff_lzw"}
        fig.savefig(path, **save_kwargs)
    return outputs


def main() -> None:
    fig = build_figure()
    try:
        outputs = save_outputs(fig)
    finally:
        plt.close(fig)

    print("Saved conceptual workflow figure:")
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
