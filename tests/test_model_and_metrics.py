import numpy as np
import torch

from ecg_project.evaluation import anomaly_predictions, best_threshold_from_validation, classification_metrics
from ecg_project.model import ECGTCNClassifier


def test_model_forward_shape():
    model = ECGTCNClassifier()
    inputs = torch.randn(4, 1, 140)

    logits = model(inputs)

    assert logits.shape == (4, 2)


def test_threshold_and_metrics_pipeline():
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.2, 0.8, 0.9])

    threshold, f1 = best_threshold_from_validation(y_true, y_score)
    y_pred = anomaly_predictions(y_score, threshold)
    metrics = classification_metrics(y_true, y_pred, y_score)

    assert 0.1 <= threshold <= 0.9
    assert f1 >= 0.0
    assert set(metrics) == {"roc_auc", "average_precision", "precision", "recall", "f1"}