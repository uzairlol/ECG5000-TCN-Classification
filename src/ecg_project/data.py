from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from .config import DataConfig


ECG_FEATURE_COUNT = 140


def _feature_columns() -> list[str]:
    return [f"x_{index:03d}" for index in range(1, ECG_FEATURE_COUNT + 1)]


def load_ecg_file(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, sep=r"\s+", header=None)
    expected_columns = ECG_FEATURE_COUNT + 1
    if frame.shape[1] != expected_columns:
        raise ValueError(
            f"Expected {expected_columns} columns in {path}, found {frame.shape[1]}."
        )

    frame.columns = ["label", *_feature_columns()]
    frame["label"] = frame["label"].astype(int)
    return frame


def load_dataset(data_config: DataConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_path = data_config.data_dir / data_config.train_filename
    test_path = data_config.data_dir / data_config.test_filename
    return load_ecg_file(train_path), load_ecg_file(test_path)


def to_binary_labels(labels: pd.Series | np.ndarray) -> np.ndarray:
    values = np.asarray(labels)
    return (values != 1).astype(np.int64)


def split_train_validation(
    train_frame: pd.DataFrame,
    validation_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    stratify_labels = to_binary_labels(train_frame["label"])
    train_split, validation_split = train_test_split(
        train_frame,
        test_size=validation_size,
        random_state=random_state,
        stratify=stratify_labels,
    )
    return train_split.reset_index(drop=True), validation_split.reset_index(drop=True)


def feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.drop(columns=["label"])


def build_tensor_dataset(features: np.ndarray, labels: np.ndarray) -> TensorDataset:
    feature_tensor = torch.as_tensor(features, dtype=torch.float32).unsqueeze(1)
    label_tensor = torch.as_tensor(labels, dtype=torch.long)
    return TensorDataset(feature_tensor, label_tensor)


@dataclass
class ECGDatasetBundle:
    train_frame: pd.DataFrame
    validation_frame: pd.DataFrame
    test_frame: pd.DataFrame
    scaler: object
    train_dataset: TensorDataset
    validation_dataset: TensorDataset
    test_dataset: TensorDataset


def make_dataloaders(
    train_dataset: TensorDataset,
    validation_dataset: TensorDataset,
    test_dataset: TensorDataset,
    batch_size: int,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    return train_loader, validation_loader, test_loader
