from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .config import StudyConfig

PROJECT_ROOT = Path(__file__).resolve().parents[3]
LEGACY_SRC = PROJECT_ROOT / "src"
if str(LEGACY_SRC) not in sys.path:
    sys.path.insert(0, str(LEGACY_SRC))

from ml_benchmark import data as legacy_data  # noqa: E402


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
    metal_counts = {
        "rows_metal_cd": int(df.get("metal_cd", pd.Series(dtype=float)).fillna(0).sum()),
        "rows_metal_cr": int(df.get("metal_cr", pd.Series(dtype=float)).fillna(0).sum()),
        "rows_metal_hg": int(df.get("metal_hg", pd.Series(dtype=float)).fillna(0).sum()),
    }
    records = [
        {"metric": "n_rows", "value": int(len(df))},
        {"metric": "n_real_features", "value": int(len(real_features))},
        {"metric": "n_studies", "value": int(df["aut_id"].nunique())},
        {"metric": "n_experiments", "value": int(df["exp_id"].nunique())},
        {"metric": "median_rows_per_study", "value": float(rows_per_study.median())},
        {"metric": "max_rows_per_study", "value": int(rows_per_study.max())},
        {"metric": "median_rows_per_experiment", "value": float(rows_per_experiment.median())},
        {"metric": "max_rows_per_experiment", "value": int(rows_per_experiment.max())},
        {"metric": "median_qe", "value": float(df["qe"].median())},
        {"metric": "p95_qe", "value": float(df["qe"].quantile(0.95))},
        {"metric": "max_qe", "value": float(df["qe"].max())},
        {"metric": "missing_fraction_ph", "value": float(df["ph"].isna().mean())},
        {"metric": "missing_fraction_sa", "value": float(df["sa"].isna().mean())},
    ]
    for metric, value in metal_counts.items():
        records.append({"metric": metric, "value": value})
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
