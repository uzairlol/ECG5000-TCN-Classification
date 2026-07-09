from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

try:
    from pytorch_lightning import LightningModule
except ImportError:  # pragma: no cover - fallback for newer lightning packaging
    from lightning.pytorch import LightningModule

class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dropout: float):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding)
        self.norm1 = nn.BatchNorm1d(out_channels)
        self.norm2 = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.shortcut = nn.Identity() if in_channels == out_channels else nn.Conv1d(in_channels, out_channels, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(inputs)
        outputs = self.conv1(inputs)
        outputs = self.norm1(outputs)
        outputs = F.relu(outputs)
        outputs = self.dropout(outputs)
        outputs = self.conv2(outputs)
        outputs = self.norm2(outputs)
        outputs = outputs + residual
        return F.relu(outputs)


class ECGTCNClassifier(LightningModule):
    def __init__(
        self,
        input_length: int = 140,
        channels: tuple[int, ...] = (32, 64, 64),
        kernel_size: int = 5,
        dropout: float = 0.2,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.input_length = input_length
        self.channels = channels
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        layers: list[nn.Module] = []
        in_channels = 1
        for out_channels in channels:
            layers.append(ResidualBlock(in_channels, out_channels, kernel_size, dropout))
            in_channels = out_channels

        self.backbone = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(in_channels, 2)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.backbone(inputs)
        pooled = self.pool(features).squeeze(-1)
        return self.classifier(pooled)

    def _shared_step(self, batch, stage: str) -> torch.Tensor:
        inputs, targets = batch
        logits = self(inputs)
        loss = F.cross_entropy(logits, targets)
        predictions = torch.argmax(logits, dim=1)
        accuracy = (predictions == targets).float().mean()
        self.log(f"{stage}_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log(f"{stage}_acc", accuracy, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        self._shared_step(batch, "test")

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

    @torch.no_grad()
    def predict_proba(self, inputs: torch.Tensor) -> torch.Tensor:
        logits = self(inputs)
        return torch.softmax(logits, dim=1)
