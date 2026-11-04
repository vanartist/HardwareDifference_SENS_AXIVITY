"""CNN-BiLSTM activity-recognition model."""

from __future__ import annotations

from typing import Any, Mapping

import torch
import torch.nn as nn

class CNNBiLSTM(nn.Module):
    """Convolutional feature extractor plus bidirectional LSTM pooling."""

    def __init__(
        self,
        num_classes: int,
        conv_channels: tuple[int, int] = (64, 128),
        kernel_size: int = 5,
        lstm_hidden: int = 128,
        lstm_layers: int = 1,
        bidirectional: bool = True,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if not bidirectional:
            raise ValueError("CNNBiLSTM requires bidirectional=True")
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
        )
        self.bilstm = nn.LSTM(
            input_size=second_channels,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
            bidirectional=True,
        )
        output_dimensions = lstm_hidden * 2
        self.sequence_dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(output_dimensions, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, features):
        features = self.feature_extractor(features)
        features = features.permute(0, 2, 1)
        sequence, _ = self.bilstm(features)
        sequence = self.sequence_dropout(sequence)
        context = torch.mean(sequence, dim=1)
        return self.classifier(context)


def build_model(
    num_classes: int, parameters: Mapping[str, Any]
) -> nn.Module:
    return CNNBiLSTM(
        num_classes=num_classes,
        conv_channels=tuple(parameters["conv_channels"]),
        kernel_size=int(parameters["kernel_size"]),
        lstm_hidden=int(parameters["lstm_hidden"]),
        lstm_layers=int(parameters["lstm_layers"]),
        bidirectional=bool(parameters["bidirectional"]),
        dropout=float(parameters["dropout"]),
    )

