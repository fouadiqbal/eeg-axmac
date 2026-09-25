"""Exact and official EvoApprox8b LUT checks; execute on Kaggle."""

from pathlib import Path

import numpy as np
import torch

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
