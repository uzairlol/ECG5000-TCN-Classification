from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

try:
    from pytorch_lightning import LightningModule
except ImportError:
    from lightning.pytorch import LightningModule

from .model import ResidualBlock

def nt_xent_loss(zi: torch.Tensor, zj: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    """
    Computes NT-Xent loss (Normalized Temperature-scaled Cross Entropy Loss)
    for contrastive learning.
    
    Args:
        zi: representations of view 1, shape [batch_size, projection_dim]
        zj: representations of view 2, shape [batch_size, projection_dim]
        temperature: scaling parameter
    """
    zi = F.normalize(zi, dim=1)
    zj = F.normalize(zj, dim=1)
    
    batch_size = zi.shape[0]
    # Shape: [2 * batch_size, projection_dim]
    representations = torch.cat([zi, zj], dim=0)
    
    # Cosine similarity matrix: [2 * batch_size, 2 * batch_size]
    similarity_matrix = torch.matmul(representations, representations.T)
    similarity_matrix = similarity_matrix / temperature
    
    # Exclude self-similarity matches
    mask = torch.eye(2 * batch_size, dtype=torch.bool, device=zi.device)
    
    # The positive matches are zi <-> zj
    # Which are located at offset batch_size and -batch_size
    diag_pos_1 = torch.diag(similarity_matrix, batch_size)
    diag_pos_2 = torch.diag(similarity_matrix, -batch_size)
    positives = torch.cat([diag_pos_1, diag_pos_2], dim=0) # [2 * batch_size]
    
    # Mask out the self-similarity values from denominators
    similarity_matrix.masked_fill_(mask, -9e15)
    
    # Negative loss calculation (log-sum-exp over all non-self elements)
    loss = -positives + torch.logsumexp(similarity_matrix, dim=1)
    return loss.mean()


class ECGSimCLR(LightningModule):
    def __init__(
        self,
        input_length: int = 140,
        channels: tuple[int, ...] = (32, 64, 64),
        kernel_size: int = 5,
        dropout: float = 0.2,
        projection_dim: int = 32,
        temperature: float = 0.1,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.temperature = temperature
        
        # Build TCN Backbone
        layers: list[nn.Module] = []
        in_channels = 1
        for out_channels in channels:
            layers.append(ResidualBlock(in_channels, out_channels, kernel_size, dropout))
            in_channels = out_channels
            
        self.backbone = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        
        # SimCLR Projection Head (MLP)
        hidden_dim = in_channels
        self.projection_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, projection_dim)
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # Extract features (encoder forward pass)
        features = self.backbone(inputs)
        pooled = self.pool(features).squeeze(-1) # [batch_size, hidden_dim]
        return pooled

    def project(self, features: torch.Tensor) -> torch.Tensor:
        # Project representation to lower dimensional space for contrastive loss
        return self.projection_head(features)

    def training_step(self, batch, batch_idx):
        # batch contains: (view_i, view_j, labels)
        xi, xj, _ = batch
        
        # Encode
        hi = self(xi)
        hj = self(xj)
        
        # Project
        zi = self.project(hi)
        zj = self.project(hj)
        
        # Loss
        loss = nt_xent_loss(zi, zj, temperature=self.temperature)
        self.log("train_loss", loss, prog_bar=True, on_epoch=True, on_step=True)
        return loss

    def validation_step(self, batch, batch_idx):
        xi, xj, _ = batch
        hi = self(xi)
        hj = self(xj)
        zi = self.project(hi)
        zj = self.project(hj)
        loss = nt_xent_loss(zi, zj, temperature=self.temperature)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
