from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def plot_training_history(history: dict[str, list[float]]) -> None:
    epochs = range(1, len(history.get("train_loss", [])) + 1)
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, history.get("train_loss", []), label="Train loss")
    plt.plot(epochs, history.get("val_loss", []), label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training History")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_ecg_sample(signal: np.ndarray, title: str = "ECG sample") -> None:
    plt.figure(figsize=(12, 4))
    plt.plot(signal)
    plt.title(title)
    plt.xlabel("Sample")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.show()


def plot_scores(y_score: np.ndarray, y_pred: np.ndarray | None = None, title: str = "Anomaly scores") -> None:
    plt.figure(figsize=(12, 4))
    plt.plot(y_score, label="Score")
    if y_pred is not None:
        anomaly_idx = np.where(y_pred == 1)[0]
        plt.scatter(anomaly_idx, y_score[anomaly_idx], color="red", label="Predicted anomaly")
    plt.title(title)
    plt.xlabel("Index")
    plt.ylabel("Probability")
    plt.legend()
    plt.tight_layout()
    plt.show()
