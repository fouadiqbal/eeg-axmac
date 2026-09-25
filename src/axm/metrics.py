"""Circuit-level error metrics and a clearly labeled data-weighted diagnostic."""

from __future__ import annotations

import numpy as np


def circuit_error_metrics(lut: np.ndarray) -> dict[str, float]:
    """Compute unsigned 8x8 error statistics over every input combination.

    MRED excludes zero-reference products, as the relative error is undefined
    there; their error count is reported separately. ``ME`` is signed mean
    error, not the architecture-dependent AME of Liu et al. (2024).
    """
    if lut.shape != (256, 256):
        raise ValueError("Expected a 256x256 multiplier truth table")
    exact = np.arange(256, dtype=np.int32)[:, None] * np.arange(256, dtype=np.int32)[None, :]
    error = lut.astype(np.int32) - exact
    nonzero = exact != 0
    return {
        "ER": float(np.mean(error != 0)),
        "MRED": float(np.mean(np.abs(error[nonzero]) / exact[nonzero])),
        "MAE": float(np.mean(np.abs(error))),
        "ME": float(np.mean(error)),
        "zero_reference_error_rate": float(np.mean(error[~nonzero] != 0)),
    }


def input_weighted_mean_error(
    lut: np.ndarray, activation_magnitudes: np.ndarray, weight_magnitudes: np.ndarray,
) -> float:
    """Estimate Eq. 4's data-weighted mean error from operand marginals.

    The factorized joint distribution is an approximation. This is a
    layer-specific operand diagnostic, not the full architectural mean error
    (AME, Eq. 24), which additionally models downstream error propagation.
    """
    a = np.asarray(activation_magnitudes, dtype=np.int16).reshape(-1)
    w = np.asarray(weight_magnitudes, dtype=np.int16).reshape(-1)
    if a.size == 0 or w.size == 0 or np.any((a < 0) | (a > 127)) or np.any((w < 0) | (w > 127)):
        raise ValueError("Expected nonempty signed-INT8 operand magnitudes in [0,127]")
    pa = np.bincount(a, minlength=256).astype(np.float64) / a.size
    pw = np.bincount(w, minlength=256).astype(np.float64) / w.size
    exact = np.arange(256, dtype=np.int32)[:, None] * np.arange(256, dtype=np.int32)[None, :]
    return float(np.sum((lut.astype(np.int32) - exact) * pa[:, None] * pw[None, :]))
