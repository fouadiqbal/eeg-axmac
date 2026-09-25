"""Exact and official EvoApprox8b LUT checks; execute on Kaggle."""

from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from src.axm.approx_conv import LUTConv2d
from src.axm.evoapprox_loader import build_lut, load_metadata
from src.axm.lut_multiplier import exact_unsigned_lut, lut_multiply_signed
from src.axm.metrics import circuit_error_metrics


def test_exact_unsigned_and_signed_adapter() -> None:
    lut = exact_unsigned_lut()
    expected = np.arange(256, dtype=np.uint16)[:, None] * np.arange(256, dtype=np.uint16)[None, :]
    np.testing.assert_array_equal(lut, expected)
    operands = torch.tensor([-127, -23, -1, 0, 1, 25, 127], dtype=torch.int16)
    actual = lut_multiply_signed(operands[:, None], operands[None, :], torch.from_numpy(lut.astype(np.int32)))
    torch.testing.assert_close(actual, operands[:, None].to(torch.int32) * operands[None, :].to(torch.int32))


def test_official_approximate_metrics(tmp_path: Path) -> None:
    """Compare exhaustive ER/MRED with the archived circuit's rounded values."""
    metadata = load_metadata(tmp_path)
    circuit = "mul8_051"
    lut = build_lut(circuit, tmp_path, metadata)
    result = circuit_error_metrics(lut)
    published = metadata[circuit]["params"]
    assert abs(result["ER"] * 100 - float(published["ep%"])) < 0.15
    assert abs(result["MRED"] * 100 - float(published["mre%"])) < 0.15


def test_exact_lut_grouped_convolution_matches_integer_control() -> None:
    """The exact-LUT control has the same padding, grouping and int32 sums."""
    torch.manual_seed(17)
    conv = torch.nn.Conv2d(2, 2, (1, 3), groups=2, padding=(0, 1), bias=False)
    x = torch.randn(3, 2, 1, 9)
    wrapper = LUTConv2d(conv, exact_unsigned_lut(), activation_scale=0.025)
    actual = wrapper(x)
    aq = torch.clamp(torch.round(x / 0.025), -127, 127)
    wq = torch.clamp(torch.round(conv.weight / wrapper.weight_scale[:, None, None, None]), -127, 127)
    integer = F.conv2d(aq, wq, padding=(0, 1), groups=2)
    expected = integer * (0.025 * wrapper.weight_scale)[None, :, None, None]
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
