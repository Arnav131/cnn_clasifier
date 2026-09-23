"""Configurable CNN whose architecture knobs form part of the MOGEO search space."""
from __future__ import annotations

import torch
import torch.nn as nn

from .config import HParams, IMAGE_SIZE, NUM_CLASSES


class ConfigurableCNN(nn.Module):
    """A conv-block CNN generalizing the historical architecture (3 blocks,
    32/64/128 channels) into a search-space-driven design:

    - `num_blocks` Conv-BN-ReLU-MaxPool blocks, channels doubling each block
      starting at `base_channels` (capped at 512 to bound memory).
    - Global average pooling (replaces the historical Flatten(), which tied
      the classifier head's size to the input resolution).
    - Dropout before the final linear classifier.
    """

    def __init__(self, hparams: HParams, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.hparams_used = hparams

        layers = []
        in_channels = 3
        out_channels = hparams.base_channels
        for _ in range(hparams.num_blocks):
            out_channels = min(out_channels, 512)
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.MaxPool2d(2))
            in_channels = out_channels
            out_channels = out_channels * 2

        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=hparams.dropout)
        self.classifier = nn.Linear(in_channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.classifier(x)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def build_model(hparams: HParams, num_classes: int = NUM_CLASSES) -> ConfigurableCNN:
    return ConfigurableCNN(hparams, num_classes=num_classes)


if __name__ == "__main__":
    # quick sanity check
    hp = HParams()
    m = build_model(hp)
    x = torch.randn(2, 3, IMAGE_SIZE, IMAGE_SIZE)
    out = m(x)
    print("output shape:", out.shape, "| params:", m.num_parameters())
