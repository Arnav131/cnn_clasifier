"""Configurable CNN with hand-crafted feature fusion and MOGEO optimization support.

Directly implements the 'CNN Layers (Model Training and Classification)' stage of
the flowchart, fusing deep visual features learned by Conv blocks with the
MOGEO-selected hand-crafted features (Shape, Size, Colour, Vein, Texture).
"""
from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn

from .config import HParams, IMAGE_SIZE, NUM_CLASSES
from .features import FEATURE_NAMES


def get_selected_feature_indices(hparams: HParams) -> List[int]:
    """Returns indices of selected hand-crafted features based on HParams flags."""
    indices = []
    for idx, name in enumerate(FEATURE_NAMES):
        if name.startswith("shape_") and hparams.select_shape:
            indices.append(idx)
        elif name.startswith("size_") and hparams.select_size:
            indices.append(idx)
        elif name.startswith("colour_") and hparams.select_colour:
            indices.append(idx)
        elif name.startswith("vein_") and hparams.select_vein:
            indices.append(idx)
        elif name.startswith("texture_") and hparams.select_texture:
            indices.append(idx)
    return indices


class ConfigurableCNN(nn.Module):
    """A hybrid CNN that fuses learned visual features with hand-crafted features:

    - `num_blocks` Conv-BN-ReLU-MaxPool blocks, channels doubling each block
      starting at `base_channels` (capped at 512 to bound memory).
    - Global average pooling to extract dense spatial representations.
    - Hand-crafted feature branch: selects MOGEO-optimized feature subsets
      (Shape, Size, Colour, Vein, Texture), projects through BN-Linear-ReLU.
    - Feature fusion: concatenates visual representations with projected features.
    - Dropout before final linear classification into plant species classes.
    """

    def __init__(self, hparams: HParams, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.hparams_used = hparams

        # 1. Visual feature extractor
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

        # 2. Hand-crafted feature selection and projection branch
        self.selected_indices = get_selected_feature_indices(hparams) if hparams.use_features else []
        self.num_selected_features = len(self.selected_indices)

        if hparams.use_features and self.num_selected_features > 0:
            self.register_buffer(
                "feature_index_tensor",
                torch.tensor(self.selected_indices, dtype=torch.long)
            )
            self.feature_proj = nn.Sequential(
                nn.BatchNorm1d(self.num_selected_features),
                nn.Linear(self.num_selected_features, hparams.feature_proj_dim),
                nn.ReLU(inplace=True),
            )
            classifier_in_features = in_channels + hparams.feature_proj_dim
        else:
            self.feature_index_tensor = None
            self.feature_proj = None
            classifier_in_features = in_channels

        self.dropout = nn.Dropout(p=hparams.dropout)
        self.classifier = nn.Linear(classifier_in_features, num_classes)

    def forward(self, x: torch.Tensor, features: Optional[torch.Tensor] = None) -> torch.Tensor:
        vis = self.features(x)
        vis = self.pool(vis)
        vis = torch.flatten(vis, 1)

        if self.feature_proj is not None:
            if features is not None:
                # Handle single sample or batch
                if features.dim() == 1:
                    features = features.unsqueeze(0)
                sub_feats = features[:, self.feature_index_tensor]
                proj = self.feature_proj(sub_feats)
            else:
                # Fallback to zero vector if features omitted during inference
                proj = torch.zeros(
                    vis.size(0),
                    self.hparams_used.feature_proj_dim,
                    device=vis.device,
                    dtype=vis.dtype
                )
            fused = torch.cat([vis, proj], dim=1)
        else:
            fused = vis

        fused = self.dropout(fused)
        return self.classifier(fused)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def build_model(hparams: HParams, num_classes: int = NUM_CLASSES) -> ConfigurableCNN:
    return ConfigurableCNN(hparams, num_classes=num_classes)


if __name__ == "__main__":
    hp = HParams()
    m = build_model(hp)
    x = torch.randn(2, 3, IMAGE_SIZE, IMAGE_SIZE)
    f = torch.randn(2, 36)
    out = m(x, f)
    print("output shape:", out.shape, "| params:", m.num_parameters())

