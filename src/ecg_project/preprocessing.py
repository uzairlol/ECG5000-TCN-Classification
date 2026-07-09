from __future__ import annotations

from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

from .data import feature_frame, to_binary_labels


@dataclass
class ScaledArrays:
    train_features: np.ndarray
    validation_features: np.ndarray
    test_features: np.ndarray
    train_labels: np.ndarray
    validation_labels: np.ndarray
    test_labels: np.ndarray


def fit_scaler(train_frame: pd.DataFrame, scaler: RobustScaler | None = None) -> RobustScaler:
    scaler = scaler or RobustScaler()
    if not hasattr(scaler, "center_"):
        scaler.fit(feature_frame(train_frame))
    return scaler


def transform_features(frame: pd.DataFrame, scaler: RobustScaler) -> np.ndarray:
    transformed = scaler.transform(feature_frame(frame))
    return transformed.astype(np.float32)


def prepare_arrays(
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    scaler: RobustScaler | None = None,
) -> tuple[ScaledArrays, RobustScaler]:
    fitted_scaler = fit_scaler(train_frame, scaler=scaler)

    arrays = ScaledArrays(
        train_features=transform_features(train_frame, fitted_scaler),
        validation_features=transform_features(validation_frame, fitted_scaler),
        test_features=transform_features(test_frame, fitted_scaler),
        train_labels=to_binary_labels(train_frame["label"]),
        validation_labels=to_binary_labels(validation_frame["label"]),
        test_labels=to_binary_labels(test_frame["label"]),
    )
    return arrays, fitted_scaler


def save_scaler(scaler: RobustScaler, path) -> None:
    joblib.dump(scaler, path)


def load_scaler(path) -> RobustScaler:
    return joblib.load(path)
