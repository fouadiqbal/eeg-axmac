"""EEGNet model for EEG motor-imagery classification."""

from __future__ import annotations

import argparse

import torch
from torch import nn


class EEGNet(nn.Module):
    """Compact EEGNet with temporal, spatial, and separable convolutions.

    Inputs have shape ``(batch, channels, 1, samples)``. The default filters,
    kernels, pooling, and dropout follow the common EEGNet-8,2 configuration.
    """

    def __init__(
        self,
        n_channels: int = 64,
        n_samples: int = 480,
        n_classes: int = 4,
        f1: int = 8,
        depth_multiplier: int = 2,
        f2: int = 16,
        dropout: float = 0.5,
        temporal_kernel: int = 64,
        first_pool: int = 4,
    ) -> None:
        super().__init__()
        if n_channels <= 0 or n_samples < 64 or n_classes < 2 or first_pool < 1:
            raise ValueError("channels and pool must be positive, samples >= 64, classes >= 2")
        self.n_channels = n_channels
        self.n_samples = n_samples
        self.n_classes = n_classes
        self.temporal = nn.Conv2d(1, f1, (1, temporal_kernel), padding=(0, temporal_kernel // 2), bias=False)
        self.temporal_bn = nn.BatchNorm2d(f1)
        self.spatial = nn.Conv2d(f1, f1 * depth_multiplier, (n_channels, 1), groups=f1, bias=False)
        self.spatial_bn = nn.BatchNorm2d(f1 * depth_multiplier)
        self.activation1 = nn.ELU()
        self.pool1 = nn.AvgPool2d((1, first_pool))
        self.dropout1 = nn.Dropout(dropout)
        self.separable_depthwise = nn.Conv2d(
            f1 * depth_multiplier, f1 * depth_multiplier, (1, 16),
            padding=(0, 8), groups=f1 * depth_multiplier, bias=False,
        )
        self.separable_pointwise = nn.Conv2d(f1 * depth_multiplier, f2, 1, bias=False)
        self.separable_bn = nn.BatchNorm2d(f2)
        self.activation2 = nn.ELU()
        self.pool2 = nn.AvgPool2d((1, 8))
        self.dropout2 = nn.Dropout(dropout)
        feature_samples = self._feature_samples(n_samples, temporal_kernel, first_pool)
        if feature_samples < 1:
            raise ValueError("Input window is too short for the configured pooling layers")
        self.classifier = nn.Linear(f2 * feature_samples, n_classes)

    @staticmethod
    def _feature_samples(n_samples: int, temporal_kernel: int, first_pool: int) -> int:
        """Compute output time length, accounting for even-kernel padding."""
        after_temporal = n_samples + (1 if temporal_kernel % 2 == 0 else 0)
        after_first_pool = after_temporal // first_pool
        after_separable = after_first_pool + 1  # The 16-tap kernel uses symmetric padding 8.
        return after_separable // 8

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return logits for each class."""
        if x.ndim == 3:
            x = x.unsqueeze(2)
        if x.ndim != 4 or x.shape[1] != self.n_channels:
            raise ValueError(f"Expected (batch, {self.n_channels}, 1, samples) input, got {tuple(x.shape)}")
        if x.shape[2] != 1:
            raise ValueError(f"Expected singleton spatial axis, got {tuple(x.shape)}")
        x = x.transpose(1, 2)
        x = self.temporal_bn(self.temporal(x))
        x = self.spatial_bn(self.spatial(x))
        x = self.dropout1(self.pool1(self.activation1(x)))
        x = self.separable_depthwise(x)
        x = self.separable_pointwise(x)
        x = self.dropout2(self.pool2(self.activation2(self.separable_bn(x))))
        return self.classifier(x.flatten(start_dim=1))


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect EEGNet configuration and output shape.")
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--samples", type=int, default=480)
    parser.add_argument("--classes", type=int, default=4)
    parser.add_argument("--temporal-kernel", type=int, default=64)
    parser.add_argument("--first-pool", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.5)
    args = parser.parse_args()
    model = EEGNet(
        args.channels, args.samples, args.classes,
        dropout=args.dropout, temporal_kernel=args.temporal_kernel,
        first_pool=args.first_pool,
    )
    print(f"EEGNet parameters: {sum(p.numel() for p in model.parameters()):,}")
    try:
        output = model(torch.zeros(2, args.channels, 1, args.samples))
    except RuntimeError as exc:
        raise SystemExit(f"Model shape check failed: {exc}") from exc
    print(f"Output shape: {tuple(output.shape)}")


if __name__ == "__main__":
    main()
