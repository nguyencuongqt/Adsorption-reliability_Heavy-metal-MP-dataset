from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd
from PIL import Image


STUDY_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = STUDY_ROOT / "results"
PACKAGE_DIR = STUDY_ROOT / "manuscript_package"


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def copy_table(src: Path, dst_csv: Path, dst_xlsx: Path) -> None:
    ensure_parent(dst_csv)
    shutil.copy2(src, dst_csv)
    df = pd.read_csv(src)
    df.to_excel(dst_xlsx, index=False)


def copy_figure_set(src_png: Path, dst_stem: Path) -> None:
    ensure_parent(dst_stem.with_suffix(".png"))
    shutil.copy2(src_png, dst_stem.with_suffix(".png"))
    with Image.open(src_png) as image:
        rgb = image.convert("RGB")
        rgb.save(dst_stem.with_suffix(".pdf"), "PDF", resolution=300.0)
        rgb.save(dst_stem.with_suffix(".tiff"), "TIFF", compression="tiff_lzw")


def main() -> int:
    main_tables = PACKAGE_DIR / "tables"
    si_tables = PACKAGE_DIR / "supplementary_information" / "si_tables"
    main_figures = PACKAGE_DIR / "figures"
    si_figures = PACKAGE_DIR / "supplementary_information" / "si_figures"
    for path in [main_tables, si_tables, main_figures, si_figures]:
        path.mkdir(parents=True, exist_ok=True)

    copy_table(
        RESULTS_DIR / "tables" / "table_dataset_grouped_structure.csv",
        main_tables / "table01_dataset_grouped_structure.csv",
        main_tables / "table01_dataset_grouped_structure.xlsx",
    )
    copy_table(
        RESULTS_DIR / "tables" / "table_performance_across_validation_regimes.csv",
        main_tables / "table02_performance_across_validation_regimes.csv",
        main_tables / "table02_performance_across_validation_regimes.xlsx",
    )
    copy_table(
        RESULTS_DIR / "tables" / "table_generalization_gaps.csv",
        main_tables / "table03_generalization_gaps.csv",
        main_tables / "table03_generalization_gaps.xlsx",
    )
    copy_table(
        RESULTS_DIR / "tables" / "table_feature_importance_method_reliability.csv",
        si_tables / "tableS1_feature_importance_method_reliability.csv",
        si_tables / "tableS1_feature_importance_method_reliability.xlsx",
    )
    copy_table(
        RESULTS_DIR / "tables" / "rq2_mechanism_group_importance.csv",
        si_tables / "tableS2_mechanism_group_importance.csv",
        si_tables / "tableS2_mechanism_group_importance.xlsx",
    )
    copy_table(
        RESULTS_DIR / "tables" / "rq2_feature_importance_summary.csv",
        si_tables / "tableS3_feature_importance_summary.csv",
        si_tables / "tableS3_feature_importance_summary.xlsx",
    )

    copy_figure_set(
        RESULTS_DIR / "figures" / "figure_validation_hierarchy_performance_drop.png",
        main_figures / "fig01_validation_hierarchy_performance_drop",
    )
    copy_figure_set(
        RESULTS_DIR / "figures" / "figure_mechanism_group_importance_across_methods.png",
        main_figures / "fig02_mechanism_group_importance_across_methods",
    )
    copy_figure_set(
        RESULTS_DIR / "figures" / "figure_null_feature_benchmarking.png",
        main_figures / "fig03_null_feature_benchmarking",
    )
    copy_figure_set(
        RESULTS_DIR / "figures" / "figure_importance_stability_heatmap.png",
        si_figures / "sifig01_importance_stability_heatmap",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
