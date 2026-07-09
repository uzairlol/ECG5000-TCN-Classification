from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class DataConfig:
    data_dir: Path = Path("data") / "ECG5000"
    train_filename: str = "ECG5000_TRAIN.txt"
    test_filename: str = "ECG5000_TEST.txt"
    label_column: str = "label"
    validation_size: float = 0.2
    random_state: int = 42


@dataclass(frozen=True)
class ModelConfig:
    input_length: int = 140
    channels: tuple[int, ...] = (32, 64, 64)
    kernel_size: int = 5
    dropout: float = 0.2
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 64
    max_epochs: int = 25


@dataclass(frozen=True)
class ProjectConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    artifacts_dir: Path = Path("artifacts")
    checkpoint_name: str = "ecg_tcn.ckpt"
    scaler_name: str = "scaler.joblib"
    evaluation_summary_name: str = "evaluation_summary.json"
    predictions_name: str = "evaluation_predictions.csv"
    confusion_matrix_name: str = "confusion_matrix.csv"
    reconstruction_checkpoint_name: str = "ecg_reconstruction.ckpt"
    reconstruction_summary_name: str = "ecg_reconstruction_summary.json"
    reconstruction_predictions_name: str = "ecg_reconstruction_predictions.csv"
    reconstruction_confusion_matrix_name: str = "ecg_reconstruction_confusion_matrix.csv"
