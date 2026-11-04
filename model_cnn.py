"""One-dimensional convolutional activity-recognition model."""

from __future__ import annotations

from typing import Any, Mapping

import torch.nn as nn

class CNN1D(nn.Module):
    """Two convolutional blocks followed by global average pooling."""

    def __init__(
        self,
        num_classes: int,
        conv_channels: tuple[int, int] = (64, 128),
        kernel_size: int = 5,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        first_channels, second_channels = conv_channels
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(
                3,
                first_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            ),
            nn.BatchNorm1d(first_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(
                first_channels,
                second_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            ),
            nn.BatchNorm1d(second_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(second_channels, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, features):
        return self.classifier(self.feature_extractor(features))


def build_model(
    num_classes: int, parameters: Mapping[str, Any]
) -> nn.Module:
    return CNN1D(
        num_classes=num_classes,
        conv_channels=tuple(parameters["conv_channels"]),
        kernel_size=int(parameters["kernel_size"]),
        dropout=float(parameters["dropout"]),
    )

