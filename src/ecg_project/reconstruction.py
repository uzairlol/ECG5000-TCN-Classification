from __future__ import annotations

import torch
from torch import nn

try:
    from pytorch_lightning import LightningModule
except ImportError:  # pragma: no cover - fallback for newer lightning packaging
    from lightning.pytorch import LightningModule

from .model import ResidualBlock


class ECGDenoisingReconstructionModel(LightningModule):
    def __init__(
        self,
        input_length: int = 140,
        channels: tuple[int, ...] = (32, 64, 64),
        kernel_size: int = 5,
        dropout: float = 0.2,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        mask_ratio: float = 0.25,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.input_length = input_length
        self.channels = channels
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.mask_ratio = mask_ratio

        layers: list[nn.Module] = []
        in_channels = 1
        for out_channels in channels:
            layers.append(ResidualBlock(in_channels, out_channels, kernel_size, dropout))
            in_channels = out_channels

        self.backbone = nn.Sequential(*layers)
        self.reconstruction_head = nn.Conv1d(in_channels, 1, kernel_size=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.backbone(inputs)
        return self.reconstruction_head(features)

    def _random_mask(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.rand_like(inputs) > self.mask_ratio

    def _reconstruction_loss(self, inputs: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        reconstructions = self(inputs)
        missing_positions = (~mask).float()
        squared_error = (reconstructions - targets) ** 2 * missing_positions
        per_sample_denom = missing_positions.sum(dim=(1, 2)).clamp(min=1.0)
        per_sample_loss = squared_error.sum(dim=(1, 2)) / per_sample_denom
        return per_sample_loss.mean()

    def _shared_step(self, batch, stage: str) -> torch.Tensor:
        inputs, targets = batch
        mask = self._random_mask(inputs)
        masked_inputs = inputs * mask.float()
        loss = self._reconstruction_loss(masked_inputs, targets, mask)
        self.log(f"{stage}_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
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
    def reconstruction_error(self, inputs: torch.Tensor) -> torch.Tensor:
        reconstructions = self(inputs)
        return ((reconstructions - inputs) ** 2).mean(dim=(1, 2))