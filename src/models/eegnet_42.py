"""Compact EEGNet-4,2 variant described by Hernandez-Ruiz et al. (2021).

The paper changes all three activations to stepped LeakyReLUs, removes batch
normalization and dropout, and uses a first temporal kernel of fs/(2*ds).
This module returns logits for CrossEntropyLoss, rather than softmax scores.
"""

from __future__ import annotations

import argparse

import torch
from torch import nn


class EEGNet42(nn.Module):
    """EEGNet-4,2 for ``(batch, channels, 1, samples)`` input tensors."""

    def __init__(
        self,
        n_channels: int = 64,
        n_samples: int = 480,
        n_classes: int = 4,
        sampling_rate: int = 160,
        downsample_factor: int = 1,
    ) -> None:
        super().__init__()
        if n_channels < 1 or n_samples < 1 or n_classes < 2:
            raise ValueError("Invalid input dimensions or number of classes")
        if downsample_factor not in (1, 2, 3):
            raise ValueError("downsample_factor must be 1, 2, or 3")
        if 6 % downsample_factor:
            raise ValueError("First pooling factor 6/ds must be integral")
        temporal_width = round(sampling_rate / (2 * downsample_factor))
        first_pool = 6 // downsample_factor
        self.n_channels = n_channels
        self.n_samples = n_samples
        self.temporal = nn.Conv2d(1, 4, (1, temporal_width), padding="same", bias=False)
        self.activation1 = nn.LeakyReLU(0.6)
        self.spatial = nn.Conv2d(4, 8, (n_channels, 1), groups=4, bias=False)
        self.activation2 = nn.LeakyReLU(0.5)
        self.pool1 = nn.AvgPool2d((1, first_pool))
        self.separable_depthwise = nn.Conv2d(8, 8, (1, 16), padding="same", groups=8, bias=False)
        self.separable_pointwise = nn.Conv2d(8, 8, (1, 1), bias=False)
        self.activation3 = nn.LeakyReLU(0.4)
        self.pool2 = nn.AvgPool2d((1, 8))
        n_features = 8 * (n_samples // first_pool // 8)
        if n_features < 8:
            raise ValueError("Input window too short for the two pooling stages")
        self.classifier = nn.Linear(n_features, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Classify a batch; the input must already be filtered and normalized."""
        if x.ndim == 3:
            x = x.unsqueeze(2)
        if x.ndim != 4 or x.shape[1:] != (self.n_channels, 1, self.n_samples):
            raise ValueError(
                f"Expected (batch, {self.n_channels}, 1, {self.n_samples}); got {tuple(x.shape)}"
            )
        x = x.transpose(1, 2)
        x = self.activation1(self.temporal(x))
        x = self.pool1(self.activation2(self.spatial(x)))
        x = self.separable_pointwise(self.separable_depthwise(x))
        x = self.pool2(self.activation3(x))
        return self.classifier(x.flatten(1))


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect the Hernandez-Ruiz EEGNet-4,2 configuration.")
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--samples", type=int, default=480)
    parser.add_argument("--classes", type=int, default=4)
    parser.add_argument("--sampling-rate", type=int, default=160)
    parser.add_argument("--downsample-factor", type=int, default=1)
    args = parser.parse_args()
    model = EEGNet42(args.channels, args.samples, args.classes, args.sampling_rate, args.downsample_factor)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Output shape: {tuple(model(torch.zeros(2, args.channels, 1, args.samples)).shape)}")


if __name__ == "__main__":
    main()
