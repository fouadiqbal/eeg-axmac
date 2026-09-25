"""Inference-only approximate multiplication for symmetric INT8 operands."""

from __future__ import annotations

import numpy as np
import torch


def exact_unsigned_lut() -> np.ndarray:
    """Return the exact 8-bit *unsigned* multiplier truth table."""
    values = np.arange(256, dtype=np.uint16)
    return values[:, None] * values[None, :]


def quantize_symmetric(values: torch.Tensor, scale: torch.Tensor | float) -> torch.Tensor:
    """Quantize floating values into the symmetric INT8 range [-127, 127]."""
    return torch.clamp(torch.round(values / scale), -127, 127).to(torch.int16)


def lut_multiply_signed(a: torch.Tensor, b: torch.Tensor, lut: torch.Tensor) -> torch.Tensor:
    """Multiply signed INT8 tensors through an unsigned EvoApprox8b LUT.

    The unsigned circuit receives operand magnitudes. External sign logic
    restores the product sign. This sign-magnitude adapter is an explicit
    design choice; its hardware area, delay and power are not in EvoApprox8b's
    reported circuit values. Accumulation is performed separately in int32.
    """
    if lut.shape != (256, 256):
        raise ValueError(f"Expected a 256x256 unsigned multiplier LUT, got {tuple(lut.shape)}")
    a, b = torch.broadcast_tensors(a, b)
    if a.device.type == "cpu" and (torch.any(a < -127) or torch.any(a > 127) or torch.any(b < -127) or torch.any(b > 127)):
        raise ValueError("Operands must be symmetric INT8 values in [-127, 127]")
    magnitude_a = a.to(torch.int32).abs()
    magnitude_b = b.to(torch.int32).abs()
    index = (magnitude_a * 256 + magnitude_b).to(torch.long)
    product = torch.take(lut.reshape(-1).to(device=a.device, dtype=torch.int32), index)
    sign = torch.where((a < 0) ^ (b < 0), -1, 1)
    return product * sign.to(torch.int32)


def accumulate_products(products: torch.Tensor, dim: int) -> torch.Tensor:
    """Sum approximate or exact INT8 products using an int32 accumulator."""
    return products.to(torch.int32).sum(dim=dim, dtype=torch.int32)
