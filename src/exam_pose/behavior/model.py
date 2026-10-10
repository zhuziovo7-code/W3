"""Small MLP baseline for single-frame pose behavior classification."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .pose_features import POSE_FEATURE_DIM


@dataclass(frozen=True)
class PoseMLPConfig:
    input_dim: int = POSE_FEATURE_DIM
    hidden_dim: int = 192
    num_layers: int = 3
    dropout: float = 0.3
    num_classes: int = 2


class ResidualMLPBlock(nn.Module):
    """A small residual block that works well for low-dimensional pose features."""

    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class PoseMLP(nn.Module):
    """A compact baseline classifier for bbox-normalized COCO keypoints."""

    def __init__(self, config: PoseMLPConfig) -> None:
        super().__init__()
        self.config = config
        if config.num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        blocks: list[nn.Module] = [
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        ]
        for _ in range(config.num_layers - 1):
            blocks.append(ResidualMLPBlock(config.hidden_dim, config.dropout))
        blocks.extend(
            [
            nn.LayerNorm(config.hidden_dim),
            nn.Linear(config.hidden_dim, config.num_classes),
            ]
        )
        self.net = nn.Sequential(*blocks)

    def forward(self, pose_features: torch.Tensor) -> torch.Tensor:
        """Classify pose features shaped [B, 51]."""
        return self.net(pose_features)


def build_pose_mlp(
    num_classes: int,
    hidden_dim: int = 192,
    num_layers: int = 3,
    dropout: float = 0.3,
) -> PoseMLP:
    return PoseMLP(
        PoseMLPConfig(
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
    )
