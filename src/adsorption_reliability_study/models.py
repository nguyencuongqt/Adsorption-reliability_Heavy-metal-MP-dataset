from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.pipeline import Pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[3]
LEGACY_SRC = PROJECT_ROOT / "src"
if str(LEGACY_SRC) not in sys.path:
    sys.path.insert(0, str(LEGACY_SRC))

from ml_benchmark.models import build_preprocessor, get_model_spec  # noqa: E402


class SafeClippedLogTargetRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, regressor, clip_quantile: float = 0.995, max_log_prediction: float = 4.0):
        self.regressor = regressor
        self.clip_quantile = clip_quantile
        self.max_log_prediction = max_log_prediction

    def fit(self, X, y):
        y_array = np.asarray(y, dtype=float).reshape(-1)
        upper = float(np.quantile(y_array, self.clip_quantile)) if len(y_array) else 0.0
        self.target_clip_upper_ = max(upper, 0.0)
        y_transformed = np.log1p(np.clip(y_array, 0.0, self.target_clip_upper_))
        self.regressor_ = clone(self.regressor)
        self.regressor_.fit(X, y_transformed)
        return self

    def predict(self, X):
        pred = np.asarray(self.regressor_.predict(X), dtype=float).reshape(-1)
        pred = np.nan_to_num(pred, nan=0.0, posinf=self.max_log_prediction, neginf=0.0)
        pred = np.clip(pred, a_min=0.0, a_max=self.max_log_prediction)
        return np.clip(np.expm1(pred), a_min=0.0, a_max=None)

    @property
    def named_steps(self):
        return self.regressor_.named_steps


@dataclass
class FittedModel:
    name: str
    estimator: object
    feature_names: list[str]

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        return self

    def predict_log(self, X: pd.DataFrame) -> np.ndarray:
        preds = np.asarray(self.estimator.predict(X), dtype=float).reshape(-1)
        preds = np.nan_to_num(preds, nan=0.0, posinf=np.expm1(4.0), neginf=0.0)
        return np.log1p(np.clip(preds, a_min=0.0, a_max=None))

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        preds = np.asarray(self.estimator.predict(X), dtype=float).reshape(-1)
        preds = np.nan_to_num(preds, nan=0.0, posinf=np.expm1(4.0), neginf=0.0)
        return np.clip(preds, a_min=0.0, a_max=np.expm1(4.0))

    def transformed_validation_frame(self, X: pd.DataFrame) -> pd.DataFrame:
        preprocessor = self.estimator.named_steps["preprocessor"]
        transformed = preprocessor.transform(X[self.feature_names])
        feature_names = list(preprocessor.get_feature_names_out())
        return pd.DataFrame(transformed, columns=feature_names, index=X.index)

    @property
    def model_step(self):
        return self.estimator.named_steps["model"]


def fit_model(model_name: str, X_train: pd.DataFrame, y_train: np.ndarray, random_state: int) -> FittedModel:
    spec = get_model_spec(model_name, random_state=random_state, params=None)
    scale_numeric = model_name in {"linear_regression", "elastic_net", "mlp_regressor", "tabnet"}
    regressor = Pipeline(steps=[("preprocessor", build_preprocessor(scale_numeric=scale_numeric)), ("model", spec.estimator)])
    estimator = SafeClippedLogTargetRegressor(regressor=regressor)
    estimator.fit(X_train, y_train)
    return FittedModel(model_name, estimator, list(X_train.columns))
