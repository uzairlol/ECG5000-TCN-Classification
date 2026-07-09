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


def save_latent_space_figure(features: np.ndarray, labels: np.ndarray, save_path, title: str = "SSL Latent Space Projection") -> None:
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    projected = pca.fit_transform(features)
    
    plt.figure(figsize=(8, 6))
    for label, color, name in [(1, "#1f77b4", "Normal"), (0, "#d62728", "Anomaly")]:
        mask = (labels == label)
        plt.scatter(projected[mask, 0], projected[mask, 1], c=color, label=name, alpha=0.6, edgecolors='none', s=25)
        
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel("PCA Component 1", fontsize=11)
    plt.ylabel("PCA Component 2", fontsize=11)
    plt.legend(frameon=True, facecolor='white', edgecolor='none')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def save_data_efficiency_figure(results: dict[str, dict[str, float]], save_path) -> None:
    regimes = list(results.keys())
    accuracies = [metrics["accuracy"] * 100 for metrics in results.values()]
    f1_scores = [metrics["f1_score"] * 100 for metrics in results.values()]
    
    plt.figure(figsize=(7, 5))
    plt.plot(regimes, accuracies, marker='o', linewidth=2.5, color='#2ca02c', label='Accuracy (%)')
    plt.plot(regimes, f1_scores, marker='s', linewidth=2.5, color='#ff7f0e', label='F1-Score (%)')
    
    plt.title("Downstream Probing vs. Label Fraction", fontsize=14, fontweight='bold')
    plt.xlabel("Percentage of Labeled Training Samples", fontsize=11)
    plt.ylabel("Performance (%)", fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.ylim(60, 105)
    plt.legend(loc='lower right', frameon=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

