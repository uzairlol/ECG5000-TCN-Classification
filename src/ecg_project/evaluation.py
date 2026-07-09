from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    return {
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "average_precision": float(average_precision_score(y_true, y_score)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def best_threshold_from_validation(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, float]:
    candidate_thresholds = np.unique(y_score)
    if candidate_thresholds.size == 0:
        raise ValueError("Cannot calibrate a threshold from empty scores.")

    best_threshold = float(candidate_thresholds[0])
    best_f1 = -1.0
    for threshold in candidate_thresholds:
        predictions = (y_score >= threshold).astype(int)
        score = f1_score(y_true, predictions, zero_division=0)
        if score > best_f1:
            best_f1 = float(score)
            best_threshold = float(threshold)
    return best_threshold, best_f1


def anomaly_predictions(y_score: np.ndarray, threshold: float) -> np.ndarray:
    return (y_score >= threshold).astype(int)


def confusion(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return confusion_matrix(y_true, y_pred)


def save_predictions(
    path: Path,
    y_true: np.ndarray,
    y_score: np.ndarray,
    y_pred: np.ndarray,
    threshold: float,
) -> None:
    frame = pd.DataFrame(
        {
            "sample_index": np.arange(len(y_true)),
            "y_true": y_true.astype(int),
            "y_score": y_score.astype(float),
            "y_pred": y_pred.astype(int),
            "threshold": np.full(len(y_true), float(threshold)),
        }
    )
    frame.to_csv(path, index=False)


def save_confusion_matrix(path: Path, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    matrix = confusion_matrix(y_true, y_pred)
    frame = pd.DataFrame(matrix, index=["true_0", "true_1"], columns=["pred_0", "pred_1"])
    frame.to_csv(path)


def save_evaluation_summary(path: Path, metrics: dict[str, float], threshold: float) -> None:
    payload = {
        "threshold": float(threshold),
        "metrics": {name: float(value) for name, value in metrics.items()},
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
