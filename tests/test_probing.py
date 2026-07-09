import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from ecg_project.ssl_model import ECGSimCLR
from ecg_project.probing import extract_features, run_linear_probing

def test_extract_features():
    model = ECGSimCLR(channels=(16, 32))
    x = torch.randn(10, 1, 140)
    y = torch.randint(0, 2, (10,))
    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size=4)
    
    feats, labels = extract_features(model, loader)
    assert feats.shape == (10, 32)
    assert labels.shape == (10,)

def test_run_linear_probing():
    train_feats = np.random.randn(50, 32)
    train_labels = np.random.randint(0, 2, (50,))
    test_feats = np.random.randn(20, 32)
    test_labels = np.random.randint(0, 2, (20,))
    
    results = run_linear_probing(
        train_feats, train_labels,
        test_feats, test_labels,
        fractions=[0.1, 1.0]
    )
    
    assert "10%" in results
    assert "100%" in results
    assert results["100%"]["samples"] == 50
    assert 0.0 <= results["100%"]["accuracy"] <= 1.0
