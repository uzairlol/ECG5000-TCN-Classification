import torch
from ecg_project.augmentations import (
    add_jitter,
    scale,
    permute,
    time_warp,
    ContrastiveECGDataset,
)

def test_augmentations_preserve_shape():
    x = torch.randn(1, 140)
    
    x_jit = add_jitter(x)
    assert x_jit.shape == (1, 140)
    assert not torch.equal(x, x_jit)
    
    x_sc = scale(x)
    assert x_sc.shape == (1, 140)
    assert not torch.equal(x, x_sc)
    
    x_perm = permute(x, max_segments=3)
    assert x_perm.shape == (1, 140)
    
    x_warp = time_warp(x)
    assert x_warp.shape == (1, 140)

def test_contrastive_dataset_produces_two_views():
    features = torch.randn(10, 140)
    labels = torch.randint(0, 2, (10,))
    
    dataset = ContrastiveECGDataset(features, labels)
    assert len(dataset) == 10
    
    view1, view2, label = dataset[0]
    assert view1.shape == (1, 140)
    assert view2.shape == (1, 140)
    assert not torch.equal(view1, view2)
    assert label == labels[0]
