from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from .config import StudyConfig


def load_dataset(config: StudyConfig) -> pd.DataFrame:
    df = pd.read_csv(config.dataset_path)
    required = {config.target_column, *config.id_columns, *config.drop_columns}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    return df


def get_real_feature_columns(df: pd.DataFrame, config: StudyConfig) -> list[str]:
    excluded = {config.target_column, *config.id_columns, *config.drop_columns}
    return [column for column in df.columns if column not in excluded]


def build_model_features(df: pd.DataFrame, null_features_df: pd.DataFrame | None = None) -> pd.DataFrame:
    import sys

    project_root = Path(__file__).resolve().parents[3]
    legacy_src = project_root / "src"
    if str(legacy_src) not in sys.path:
        sys.path.insert(0, str(legacy_src))

    from ml_benchmark import data as legacy_data

    features = legacy_data.engineer_features(df).copy()
    if null_features_df is not None and not null_features_df.empty:
        for column in null_features_df.columns:
            features[column] = pd.to_numeric(null_features_df[column], errors="coerce")
    legacy_data.NUMERIC_FEATURES = features.columns.tolist()
    legacy_data.CATEGORICAL_FEATURES = []
    return features


def feature_group(feature_name: str) -> str:
    if feature_name.startswith(("num__", "cat__")):
        feature_name = feature_name.split("__", 1)[1]
    if feature_name.startswith("null_"):
        return "null_feature"
    if feature_name.startswith(("ph", "temp", "ce", "rpm")):
        return "solution_chemistry"
    if feature_name in {"ph_missing", "temp_missing", "rpm_missing"}:
        return "solution_chemistry"
    if feature_name.startswith("sa") or feature_name == "sa_missing":
        return "surface_area_porosity"
    if feature_name.startswith("metal_"):
        return "metal_identity"
    if feature_name.startswith("fg_"):
        return "surface_functionality"
    if feature_name.startswith("ret_"):
        return "polymer_identity"
    if feature_name.startswith("ags_"):
        return "aging_state"
    return "other"


def null_feature_type(feature_name: str) -> str:
    if feature_name.startswith(("num__", "cat__")):
        feature_name = feature_name.split("__", 1)[1]
    if not feature_name.startswith("null_"):
        return "real"
    if "perm_" in feature_name:
        return "permuted_real"
    if "study" in feature_name or "exp" in feature_name:
        return "group_aware_random"
    return "random_noise"


def add_null_features(df: pd.DataFrame, config: StudyConfig) -> tuple[pd.DataFrame, list[str]]:
    rng = np.random.default_rng(config.null_feature_seed)
    working = df.copy()

    working["null_random_normal"] = rng.normal(loc=0.0, scale=1.0, size=len(working))
    working["null_random_uniform"] = rng.uniform(low=-1.0, high=1.0, size=len(working))
    working["null_random_binary"] = rng.integers(low=0, high=2, size=len(working))

    for column in ["ph", "ce"]:
        values = working[column].to_numpy(copy=True)
        permuted = values.copy()
        rng.shuffle(permuted)
        working[f"null_perm_{column}"] = permuted

    study_noise = {study: float(rng.normal()) for study in sorted(working["aut_id"].dropna().unique())}
    exp_noise = {exp: float(rng.normal()) for exp in sorted(working["exp_id"].dropna().unique())}
    working["null_study_random_effect"] = working["aut_id"].map(study_noise).astype(float)
    working["null_exp_random_effect"] = working["exp_id"].map(exp_noise).astype(float)

    null_columns = [column for column in working.columns if column.startswith("null_")]
    return working, null_columns


def dataset_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(8192)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def build_dataset_structure_table(df: pd.DataFrame, real_features: list[str]) -> pd.DataFrame:
    rows_per_study = df.groupby("aut_id").size()
    rows_per_experiment = df.groupby("exp_id").size()
    total_rows = int(len(df))
    coverage_cells_total = 3 * 8 * 2

    metal_frames = {
        "Cd(II)": df[df["metal_cd"] == 1],
        "Cr(VI)": df[df["metal_cr"] == 1],
        "Hg(II)": df[df["metal_hg"] == 1],
    }

    def count_pct(column: str) -> str:
        count = int(df.get(column, pd.Series(dtype=float)).fillna(0).sum())
        return f"{count} ({count / total_rows * 100:.1f}%)"

    def pct_only(column: str) -> str:
        count = int(df.get(column, pd.Series(dtype=float)).fillna(0).sum())
        return f"{count / total_rows * 100:.1f}"

    def pct_missing(column: str) -> str:
        return f"{df[column].isna().mean() * 100:.1f}%"

    def pct_missing_value(column: str) -> str:
        return f"{df[column].isna().mean() * 100:.1f}"

    def qe_summary(frame: pd.DataFrame) -> str:
        return (
            f"{frame['qe'].median():.3f} "
            f"({frame['qe'].quantile(0.25):.3f}-{frame['qe'].quantile(0.75):.3f}) / "
            f"{frame['qe'].quantile(0.95):.3f} / {frame['qe'].max():.3f}"
        )

    polymer_columns = ["ret_ps", "ret_pe", "ret_pa", "ret_pet", "ret_pp", "ret_pvc", "ret_pla", "ret_other"]
    populated_cells = 0
    sparse_or_thin_cells = 0
    for metal_column in ["metal_cd", "metal_cr", "metal_hg"]:
        metal_df = df[df[metal_column] == 1]
        for polymer_column in polymer_columns:
            polymer_df = metal_df[metal_df[polymer_column] == 1]
            for aging_column in ["ags_aged", "ags_virgin"]:
                n_rows = int(polymer_df[polymer_df[aging_column] == 1].shape[0])
                if n_rows > 0:
                    populated_cells += 1
                    if 1 <= n_rows <= 20:
                        sparse_or_thin_cells += 1

    records = [
        {
            "Section": "Corpus structure",
            "Metric": "Observations / studies / experiments",
            "Value": f"{total_rows} / {int(df['aut_id'].nunique())} / {int(df['exp_id'].nunique())}",
            "Unit/Note": "rows / studies / experiments",
        },
        {
            "Section": "Corpus structure",
            "Metric": "Rows per study: median (IQR) / maximum",
            "Value": f"{rows_per_study.median():.0f} ({rows_per_study.quantile(0.25):.0f}-{rows_per_study.quantile(0.75):.0f}) / {int(rows_per_study.max())}",
            "Unit/Note": "rows",
        },
        {
            "Section": "Corpus structure",
            "Metric": "Top-3 studies share",
            "Value": f"{rows_per_study.sort_values(ascending=False).head(3).sum() / total_rows * 100:.1f}",
            "Unit/Note": "% of rows",
        },
        {
            "Section": "Corpus structure",
            "Metric": "Rows per experiment: median (IQR) / maximum",
            "Value": f"{rows_per_experiment.median():.0f} ({rows_per_experiment.quantile(0.25):.0f}-{rows_per_experiment.quantile(0.75):.0f}) / {int(rows_per_experiment.max())}",
            "Unit/Note": "rows",
        },
        {
            "Section": "Adsorption-capacity distribution",
            "Metric": "Cd(II) qe: median (IQR) / 95th percentile / maximum",
            "Value": qe_summary(metal_frames["Cd(II)"]),
            "Unit/Note": "mg g^-1",
        },
        {
            "Section": "Adsorption-capacity distribution",
            "Metric": "Cr(VI) qe: median (IQR) / 95th percentile / maximum",
            "Value": qe_summary(metal_frames["Cr(VI)"]),
            "Unit/Note": "mg g^-1",
        },
        {
            "Section": "Adsorption-capacity distribution",
            "Metric": "Hg(II) qe: median (IQR) / 95th percentile / maximum",
            "Value": qe_summary(metal_frames["Hg(II)"]),
            "Unit/Note": "mg g^-1",
        },
        {
            "Section": "Composition",
            "Metric": "Cd(II) / Cr(VI) / Hg(II)",
            "Value": f"{pct_only('metal_cd')} / {pct_only('metal_cr')} / {pct_only('metal_hg')}",
            "Unit/Note": "% of rows",
        },
        {
            "Section": "Composition",
            "Metric": "PS / PE / PA / PET / PP / PVC / PLA / Other",
            "Value": (
                f"{pct_only('ret_ps')} / {pct_only('ret_pe')} / {pct_only('ret_pa')} / {pct_only('ret_pet')} / "
                f"{pct_only('ret_pp')} / {pct_only('ret_pvc')} / {pct_only('ret_pla')} / {pct_only('ret_other')}"
            ),
            "Unit/Note": "% of rows",
        },
        {
            "Section": "Composition",
            "Metric": "Aged / virgin",
            "Value": f"{pct_only('ags_aged')} / {pct_only('ags_virgin')}",
            "Unit/Note": "% of rows",
        },
        {
            "Section": "Coverage",
            "Metric": "Metal x polymer x aging cells populated / total",
            "Value": f"{populated_cells} / {coverage_cells_total}",
            "Unit/Note": "cells",
        },
        {
            "Section": "Coverage",
            "Metric": "Absent cells",
            "Value": f"{coverage_cells_total - populated_cells} / {coverage_cells_total}",
            "Unit/Note": "cells",
        },
        {
            "Section": "Coverage",
            "Metric": "Populated cells still very sparse or thin",
            "Value": f"{sparse_or_thin_cells} / {populated_cells}",
            "Unit/Note": "cells",
        },
        {
            "Section": "Reporting completeness",
            "Metric": "Missing pH / sa / temp / rpm",
            "Value": f"{pct_missing_value('ph')} / {pct_missing_value('sa')} / {pct_missing_value('temp')} / {pct_missing_value('rpm')}",
            "Unit/Note": "% of rows",
        },
    ]
    return pd.DataFrame(records)


def build_feature_dictionary(real_features: list[str], feature_inventory_path: Path, null_features: list[str] | None = None) -> pd.DataFrame:
    inventory = pd.read_csv(feature_inventory_path)
    inventory = inventory.rename(columns={"feature": "feature_name"})
    records = []
    inventory_lookup = inventory.set_index("feature_name").to_dict(orient="index")
    for feature in real_features:
        source = inventory_lookup.get(feature, {})
        records.append(
            {
                "feature_name": feature,
                "feature_kind": "real",
                "role": source.get("role", "unknown"),
                "missing_fraction_original": source.get("missing_fraction_original"),
                "mechanism_group": feature_group(feature),
                "null_feature_type": "real",
            }
        )
    for feature in null_features or []:
        records.append(
            {
                "feature_name": feature,
                "feature_kind": "null",
                "role": "synthetic_null",
                "missing_fraction_original": 0.0,
                "mechanism_group": "null_feature",
                "null_feature_type": null_feature_type(feature),
            }
        )
    return pd.DataFrame(records)
