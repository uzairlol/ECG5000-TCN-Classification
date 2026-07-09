from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
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


def save_confusion_matrix_figure(path: Path, y_true: np.ndarray, y_pred: np.ndarray, title: str) -> None:
    matrix = confusion_matrix(y_true, y_pred)
    figure, axis = plt.subplots(figsize=(5.5, 4.5))
    image = axis.imshow(matrix, cmap="Blues")
    axis.set_title(title)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("True")
    axis.set_xticks([0, 1], labels=["Normal", "Anomaly"])
    axis.set_yticks([0, 1], labels=["Normal", "Anomaly"])

    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            axis.text(col, row, int(matrix[row, col]), ha="center", va="center", color="black")

    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def save_score_distribution_figure(
    path: Path,
    y_true: np.ndarray,
    y_score: np.ndarray,
    title: str,
    threshold: float | None = None,
) -> None:
    figure, axis = plt.subplots(figsize=(7, 4.5))
    normal_scores = y_score[y_true == 0]
    anomaly_scores = y_score[y_true == 1]
    axis.hist(normal_scores, bins=30, alpha=0.7, label="Normal", color="#4c78a8", density=True)
    axis.hist(anomaly_scores, bins=30, alpha=0.7, label="Anomaly", color="#e45756", density=True)
    if threshold is not None:
        axis.axvline(threshold, color="black", linestyle="--", label=f"Threshold = {threshold:.3f}")
    axis.set_title(title)
    axis.set_xlabel("Score")
    axis.set_ylabel("Density")
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def save_reconstruction_examples_figure(
    path: Path,
    originals: np.ndarray,
    reconstructions: np.ndarray,
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    max_examples: int = 3,
) -> None:
    example_count = min(max_examples, len(originals))
    figure, axes = plt.subplots(example_count, 1, figsize=(10, 3.5 * example_count), sharex=True)
    if example_count == 1:
        axes = [axes]

    for index in range(example_count):
        axis = axes[index]
        axis.plot(originals[index], label="Original", color="#4c78a8", linewidth=1.5)
        axis.plot(reconstructions[index], label="Reconstruction", color="#f58518", linewidth=1.2)
        axis.set_title(
            f"Sample {index} | label={int(labels[index])} | score={scores[index]:.4f} | threshold={threshold:.4f}"
        )
        axis.legend(loc="upper right")
        axis.grid(alpha=0.2)

    axes[-1].set_xlabel("Time step")
    figure.tight_layout()
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)
