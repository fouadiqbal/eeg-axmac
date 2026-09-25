"""Bounded-memory inference wrappers for exact and approximate INT8 MACs.

Only a selected layer is replaced. All subsequent operations in the model
remain PyTorch floating point. The exact-LUT control uses identical input and
weight quantization, making its difference from an approximate LUT attributable
to the multiplier truth table rather than quantization alone.
"""

from __future__ import annotations

import copy

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from src.axm.lut_multiplier import lut_multiply_signed, quantize_symmetric


class LUTConv2d(nn.Module):
    """Emulate one Conv2d (including grouped/depthwise) with int32 sums."""

    def __init__(self, conv: nn.Conv2d, lut: np.ndarray | torch.Tensor, activation_scale: float,
                 positions_per_chunk: int = 8192) -> None:
        super().__init__()
        if activation_scale <= 0 or positions_per_chunk < 1 or conv.padding_mode != "zeros":
            raise ValueError("Expected positive scale/chunk and zero-padded convolution")
        self.conv = conv
        self.activation_scale = float(activation_scale)
        self.positions_per_chunk = positions_per_chunk
        self.register_buffer("lut", torch.as_tensor(lut, dtype=torch.int32))
        weight_max = conv.weight.detach().abs().amax(dim=(1, 2, 3)).clamp(min=1e-12)
        self.register_buffer("weight_scale", weight_max / 127)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        conv = self.conv
        xq = quantize_symmetric(x, self.activation_scale).float()
        wq = quantize_symmetric(conv.weight, self.weight_scale[:, None, None, None])
        kh, kw = conv.kernel_size
        dh, dw = conv.dilation
        sh, sw = conv.stride
        if isinstance(conv.padding, str):
            if conv.padding != "same" or conv.stride != (1, 1):
                raise ValueError("Only stride-one 'same' string padding is supported")
            total_h, total_w = dh * (kh - 1), dw * (kw - 1)
            left_h, left_w = total_h // 2, total_w // 2
            xq = F.pad(xq, (left_w, total_w - left_w, left_h, total_h - left_h))
            padding = (0, 0)
        else:
            padding = conv.padding
        patches = F.unfold(xq, kernel_size=conv.kernel_size, dilation=conv.dilation,
                           padding=padding, stride=conv.stride).to(torch.int16)
        batch, _, locations = patches.shape
        out_h = (xq.shape[2] + 2 * padding[0] - dh * (kh - 1) - 1) // sh + 1
        out_w = (xq.shape[3] + 2 * padding[1] - dw * (kw - 1) - 1) // sw + 1
        if locations != out_h * out_w:
            raise RuntimeError("Convolution patch count differs from output shape")
        out = torch.empty((batch, conv.out_channels, locations), device=x.device, dtype=x.dtype)
        channels_per_group = conv.in_channels // conv.groups
        outputs_per_group = conv.out_channels // conv.groups
        operands_per_filter = channels_per_group * kh * kw
        lut = self.lut.to(x.device)
        for output_channel in range(conv.out_channels):
            group = output_channel // outputs_per_group
            first = group * operands_per_filter
            weight = wq[output_channel].reshape(1, operands_per_filter, 1)
            scale = self.activation_scale * self.weight_scale[output_channel]
            bias = 0 if conv.bias is None else conv.bias[output_channel]
            for start in range(0, locations, self.positions_per_chunk):
                stop = min(start + self.positions_per_chunk, locations)
                products = lut_multiply_signed(
                    patches[:, first:first + operands_per_filter, start:stop], weight, lut
                )
                total = products.sum(dim=1, dtype=torch.int32)
                out[:, output_channel, start:stop] = total.to(x.dtype) * scale + bias
        return out.reshape(batch, conv.out_channels, out_h, out_w)


class LUTLinear(nn.Module):
    """Emulate one dense layer with per-output-channel weight quantization."""

    def __init__(self, linear: nn.Linear, lut: np.ndarray | torch.Tensor, activation_scale: float) -> None:
        super().__init__()
        if activation_scale <= 0:
            raise ValueError("Activation scale must be positive")
        self.linear = linear
        self.activation_scale = float(activation_scale)
        self.register_buffer("lut", torch.as_tensor(lut, dtype=torch.int32))
        self.register_buffer("weight_scale", linear.weight.detach().abs().amax(dim=1).clamp(min=1e-12) / 127)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or x.shape[1] != self.linear.in_features:
            raise ValueError("Expected a batch of flattened dense-layer inputs")
        xq = quantize_symmetric(x, self.activation_scale)
        wq = quantize_symmetric(self.linear.weight, self.weight_scale[:, None])
        lut = self.lut.to(x.device)
        outputs = []
        for channel in range(self.linear.out_features):
            products = lut_multiply_signed(xq, wq[channel][None, :], lut)
            value = products.sum(dim=1, dtype=torch.int32).to(x.dtype)
            bias = 0 if self.linear.bias is None else self.linear.bias[channel]
            outputs.append(value * self.activation_scale * self.weight_scale[channel] + bias)
        return torch.stack(outputs, dim=1)


def with_layer_lut(model: nn.Module, layer: str, lut: np.ndarray | torch.Tensor,
                   activation_scales: dict[str, float]) -> nn.Module:
    """Return a copied model with only the requested MAC layer emulated."""
    separable_names = ("sep_depth", "sep_point") if hasattr(model, "sep_depth") else (
        "separable_depthwise", "separable_pointwise"
    )
    names = {
        "temporal": ("temporal",),
        "spatial": ("spatial",),
        "separable": separable_names,
        "dense": ("classifier",),
    }
    if layer not in names:
        raise ValueError(f"Unknown layer: {layer}")
    copied = copy.deepcopy(model)
    for name in names[layer]:
        original = getattr(copied, name)
        scale = activation_scales[name]
        wrapper = LUTLinear(original, lut, scale) if isinstance(original, nn.Linear) else LUTConv2d(original, lut, scale)
        setattr(copied, name, wrapper)
    return copied.to(next(model.parameters()).device).eval()
