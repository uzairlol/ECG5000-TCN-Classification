import torch
from ecg_project.ssl_model import ECGSimCLR, nt_xent_loss

def test_nt_xent_loss_zeros_on_identical_pairs():
    # If representations are perfectly aligned, positive similarities are high
    # negative similarities are lower. Let's check loss decreases as representations align.
    zi = torch.randn(8, 16)
    zj = zi.clone() # Identical positive views
    
    loss_identical = nt_xent_loss(zi, zj, temperature=0.1)
    
    # Random positive views
    zj_random = torch.randn(8, 16)
    loss_random = nt_xent_loss(zi, zj_random, temperature=0.1)
    
    # Aligned views should have much lower loss than random views
    assert loss_identical < loss_random

def test_ecg_simclr_shapes():
    model = ECGSimCLR(
        input_length=140,
        channels=(16, 32),
        projection_dim=16
    )
    
    # Input batch: 4 samples, 1 channel, 140 sequence length
    x = torch.randn(4, 1, 140)
    
    # Encoder output (representation)
    h = model(x)
    assert h.shape == (4, 32)
    
    # Projection head output
    z = model.project(h)
    assert z.shape == (4, 16)

def test_ecg_simclr_training_step():
    model = ECGSimCLR(
        input_length=140,
        channels=(16, 32),
        projection_dim=16
    )
    xi = torch.randn(4, 1, 140)
    xj = torch.randn(4, 1, 140)
    labels = torch.zeros(4, dtype=torch.long)
    
    loss = model.training_step((xi, xj, labels), 0)
    assert loss.dim() == 0 # scalar loss
    assert not torch.isnan(loss)
