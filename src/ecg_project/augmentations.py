import random
import torch
import numpy as np
from torch.utils.data import Dataset

def add_jitter(x: torch.Tensor, sigma: float = 0.03) -> torch.Tensor:
    """Adds random Gaussian noise to the ECG signal."""
    noise = torch.randn_like(x) * sigma
    return x + noise

def scale(x: torch.Tensor, low: float = 0.8, high: float = 1.2) -> torch.Tensor:
    """Scales the amplitude of the ECG signal by a random factor."""
    factor = random.uniform(low, high)
    return x * factor

def permute(x: torch.Tensor, max_segments: int = 4) -> torch.Tensor:
    """Splits the signal into random number of segments and shuffles them."""
    if max_segments <= 1:
        return x.clone()
    num_segments = random.randint(2, max_segments)
    c, l = x.shape
    segment_length = l // num_segments
    
    segments = []
    for i in range(num_segments):
        start = i * segment_length
        end = (i + 1) * segment_length if i < num_segments - 1 else l
        segments.append(x[:, start:end])
        
    random.shuffle(segments)
    return torch.cat(segments, dim=1)

def time_warp(x: torch.Tensor, warp_factor: float = 0.15) -> torch.Tensor:
    """Distorts the time steps of the signal via interpolation."""
    c, l = x.shape
    # Generate random anchor points for warping
    orig_steps = np.linspace(0, 1, l)
    
    # Perturb intermediate steps slightly
    num_anchors = 4
    anchor_steps = np.linspace(0, 1, num_anchors)
    perturbed_anchors = anchor_steps + np.random.uniform(-warp_factor/num_anchors, warp_factor/num_anchors, num_anchors)
    perturbed_anchors[0] = 0.0
    perturbed_anchors[-1] = 1.0
    perturbed_anchors = np.sort(perturbed_anchors)
    
    # Interpolate to find new step indices
    new_steps = np.interp(orig_steps, perturbed_anchors, anchor_steps)
    new_indices = (new_steps * (l - 1)).astype(int)
    
    return x[:, new_indices]

class ContrastiveECGDataset(Dataset):
    """
    Dataset wrapper that returns two different augmented views of the same ECG signal.
    """
    def __init__(self, features: torch.Tensor, labels: torch.Tensor | None = None):
        # features shape: [N, 1, 140] or [N, 140]
        if features.dim() == 2:
            self.features = features.unsqueeze(1)
        else:
            self.features = features
            
        if labels is not None:
            self.labels = torch.as_tensor(labels, dtype=torch.long)
        else:
            self.labels = torch.zeros(len(features), dtype=torch.long)

    def __len__(self) -> int:
        return len(self.features)

    def _apply_random_augmentations(self, x: torch.Tensor) -> torch.Tensor:
        aug_funcs = [
            lambda t: add_jitter(t, sigma=random.uniform(0.01, 0.05)),
            lambda t: scale(t, low=0.85, high=1.15),
            lambda t: permute(t, max_segments=3),
            lambda t: time_warp(t, warp_factor=0.1)
        ]
        # Randomly choose 2 distinct augmentations to apply
        selected_augs = random.sample(aug_funcs, 2)
        x_aug = x.clone()
        for aug in selected_augs:
            x_aug = aug(x_aug)
        return x_aug

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.features[idx] # Shape: [1, 140]
        x_i = self._apply_random_augmentations(x)
        x_j = self._apply_random_augmentations(x)
        return x_i, x_j, self.labels[idx]
