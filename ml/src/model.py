"""
Lightweight 1D-CNN Model for TinyML Acceleration Classification.

This model is designed for use on microcontrollers with constrained memory.
The architecture is fully parameterizable (inputs, filters, classes) to find
the best accuracy/memory compromise.

Author: nakmuaycoder
Date: 2026/04
"""

import torch
import torch.nn as nn


class VectorNorm(nn.Module):
    """
    Computes the L2 norm of the input acceleration vector.
    Input shape: (B, 3, L) -> Output shape: (B, 1, L)
    """

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies L2 normalization across the channel dimension (dim=1).

        Args:
            x (torch.Tensor): Input batch of shape (Batch, 3, Sequence_Length).

        Returns:
            torch.Tensor: Normalized batch (B, 1, L).
        """
        return torch.linalg.norm(x, ord=2, dim=1, keepdim=True)


class MinMaxNormalize(nn.Module):
    """
    Normalizes data between [0, 1] (or custom range) based on fixed physical limits.
    X_std = (x - min) / (max - min)
    X_scaled = X_std * (max_new - min_new) + min_new
    """

    def __init__(
        self,
        min_val: float = -4.0,
        max_val: float = 4.0,
        range_min: float = 0.0,
        range_max: float = 1.0,
    ):
        """
        Args:
            min_val: Expected minimum value from the sensor (e.g. -4g).
            max_val: Expected maximum value from the sensor (e.g. +4g).
            range_min: Target minimum (e.g. 0.0).
            range_max: Target maximum (e.g. 1.0).
        """
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val
        self.range_min = range_min
        self.range_max = range_max

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Scales the input tensor with clipping for robustness.
        """
        denom = self.max_val - self.min_val
        if denom == 0:
            return torch.full_like(x, self.range_min)
        x_std = (x - self.min_val) / denom
        x_scaled = x_std * (self.range_max - self.range_min) + self.range_min
        return torch.clamp(x_scaled, self.range_min, self.range_max)


class TinyMLConvNet(nn.Module):
    """
    Lean 1D-CNN architecture optimized for Arduino Nano 33 BLE.
    Returns raw Logits (no Softmax).
    """

    def __init__(
        self,
        in_channels: int = 3,
        filters: list[int] | None = None,
        num_classes: int = 3,
        seq_len: int = 10,
        hidden_dim: int = 16,
    ):
        """
        Initializes the 1D-CNN with specific filter counts.

        Args:
            in_channels (int): Input sensor channels (1 or 3).
            filters (List[int]): Filters count for each of the 2 conv layers. Defaults to [8, 16].
            num_classes (int): Prediction classes count.
            seq_len (int): Windows length.
            hidden_dim (int): Hidden layer size for the classifier.
        """
        super().__init__()

        # Use default if none provided (avoiding B006 mutable default)
        if filters is None:
            filters = [8, 16]

        if len(filters) != 2:
            raise ValueError(
                "Currently supporting exactly 2 convolutional layers for TinyML stability."
            )

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.seq_len = seq_len
        self.filter_config = filters

        # Dynamic Feature Extraction Layers
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, filters[0], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(filters[0], filters[1], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )

        # Classifier Setup
        dummy_input = torch.zeros(1, in_channels, seq_len)
        self.flat_size = self._get_flat_size(dummy_input)

        self.classifier = nn.Sequential(
            nn.Linear(self.flat_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes),
        )

    def _get_flat_size(self, x: torch.Tensor) -> int:
        """
        Calculates the number of flattened features dynamically.
        """
        with torch.no_grad():
            features = self.features(x)
            return features.view(1, -1).size(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass. Note: Returns Logits.
        """
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
