from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


STUDY_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = STUDY_ROOT.parent
CONFIG_PATH = STUDY_ROOT / "configs" / "study_config.json"


@dataclass(frozen=True)
class StudyConfig:
    raw: dict[str, Any]

    @property
    def dataset_path(self) -> Path:
        return STUDY_ROOT / self.raw["dataset_path"]

    @property
    def feature_inventory_path(self) -> Path:
        return STUDY_ROOT / self.raw["feature_inventory_path"]

    @property
    def author_registry_path(self) -> Path:
        return STUDY_ROOT / self.raw["author_registry_path"]

    @property
    def target_column(self) -> str:
        return self.raw["target_column"]

    @property
    def id_columns(self) -> list[str]:
        return list(self.raw["id_columns"])

    @property
    def drop_columns(self) -> list[str]:
        return list(self.raw["drop_columns"])

    @property
    def random_state(self) -> int:
        return int(self.raw["random_state"])

    @property
    def random_cv(self) -> dict[str, Any]:
        return dict(self.raw["random_cv"])

    @property
    def group_cv(self) -> dict[str, Any]:
        return dict(self.raw["group_cv"])

    @property
    def rq1_models(self) -> list[str]:
        return list(self.raw["rq1_models"])

    @property
    def rq2_models(self) -> list[str]:
        return list(self.raw["rq2_models"])

    @property
    def mechanism_groups(self) -> list[str]:
        return list(self.raw["mechanism_groups"])

    @property
    def null_feature_seed(self) -> int:
        return int(self.raw.get("null_feature_seed", self.random_state + 100))


def load_config(path: Path | None = None) -> StudyConfig:
    config_path = path or CONFIG_PATH
    with config_path.open("r", encoding="utf-8") as handle:
        return StudyConfig(json.load(handle))
