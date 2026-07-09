from __future__ import annotations

import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

from .ssl_model import ECGSimCLR

def extract_features(model: ECGSimCLR, loader: torch.utils.data.DataLoader) -> tuple[np.ndarray, np.ndarray]:
    """Passes all data through the frozen encoder to extract representations."""
    model.eval()
    device = next(model.parameters()).device
    
    all_features = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            # Forward pass through backbone + pool
            features = model(inputs)
            all_features.append(features.cpu().numpy())
            all_labels.append(labels.numpy())
            
    return np.concatenate(all_features, axis=0), np.concatenate(all_labels, axis=0)


def run_linear_probing(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    test_features: np.ndarray,
    test_labels: np.ndarray,
    fractions: list[float] = [0.01, 0.1, 1.0]
) -> dict[str, dict[str, float]]:
    """
    Trains logistic regression on varying fractions of the training set
    to evaluate representation robustness in low-data regimes.
    """
    results = {}
    
    for frac in fractions:
        if frac < 1.0:
            # Stratified split to keep representation of classes
            try:
                X_sub, _, y_sub, _ = train_test_split(
                    train_features,
                    train_labels,
                    train_size=frac,
                    random_state=42,
                    stratify=train_labels
                )
            except ValueError:
                # Handle cases where frac is too small for stratification
                X_sub, _, y_sub, _ = train_test_split(
                    train_features,
                    train_labels,
                    train_size=frac,
                    random_state=42
                )
        else:
            X_sub, y_sub = train_features, train_labels
            
        # Fit linear classifier (Logistic Regression)
        clf = LogisticRegression(max_iter=1000, C=1.0)
        clf.fit(X_sub, y_sub)
        
        preds = clf.predict(test_features)
        probs = clf.predict_proba(test_features)[:, 1]
        
        acc = accuracy_score(test_labels, preds)
        f1 = f1_score(test_labels, preds)
        auc = roc_auc_score(test_labels, probs)
        
        results[f"{int(frac * 100)}%"] = {
            "samples": len(X_sub),
            "accuracy": acc,
            "f1_score": f1,
            "roc_auc": auc
        }
        
    return results
