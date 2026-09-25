"""Calibrate symmetric INT8 scales from a held-in training subset on Kaggle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn


MAC_LAYERS = ("temporal", "spatial", "separable_depthwise", "separable_pointwise", "classifier")


def calibrate_model(model: nn.Module, train_x: torch.Tensor, seed: int = 42,
                    fraction: float = 0.10, percentile: float = 99.9,
                    batch_size: int = 32) -> dict:
    """Estimate activation and per-output-weight INT8 scales without test data.

    A fixed random 10% subset of training windows drives forward-pre-hooks.
    At most 2,048 input magnitudes per batch/layer are retained to keep memory
    bounded; the resulting 99.9th percentile is an empirical estimator.
    """
    if not 0 < fraction <= 1 or not 0 < percentile <= 100:
        raise ValueError("Invalid calibration fraction or percentile")
    rng = np.random.default_rng(seed)
    subset = np.sort(rng.choice(len(train_x), size=max(1, round(len(train_x) * fraction)), replace=False))
    names = ("temporal", "spatial", "sep_depth", "sep_point", "classifier") if hasattr(model, "sep_depth") else MAC_LAYERS
    records: dict[str, list[np.ndarray]] = {name: [] for name in names}
    handles = []

    def collect(name: str):
        def hook(_module: nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
            values = inputs[0].detach().abs().flatten().cpu().numpy()
            if len(values) > 2048:
                values = values[rng.choice(len(values), 2048, replace=False)]
            records[name].append(values)
        return hook

    for name in names:
        handles.append(getattr(model, name).register_forward_pre_hook(collect(name)))
    device = next(model.parameters()).device
    model.eval()
    try:
        with torch.no_grad():
            for start in range(0, len(subset), batch_size):
                model(train_x[subset[start:start + batch_size]].to(device))
    finally:
        for handle in handles:
            handle.remove()
    layers = {}
    for name in names:
        module = getattr(model, name)
        magnitudes = np.concatenate(records[name])
        p = float(np.percentile(magnitudes, percentile))
        weight = module.weight.detach().abs().reshape(module.out_channels if isinstance(module, nn.Conv2d)
                                                        else module.out_features, -1)
        weight_scales = (weight.amax(dim=1).clamp(min=1e-12) / 127).cpu().tolist()
        layers[name] = {"activation_scale": max(p / 127, 1e-12),
                        "weight_scales": weight_scales,
                        "sampled_activation_values": len(magnitudes)}
    return {"seed": seed, "fraction": fraction, "percentile": percentile,
            "calibration_windows": len(subset), "layers": layers}


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect saved INT8 calibration statistics.")
    parser.add_argument("--input", type=Path, required=True, help="JSON file produced by calibrate_model")
    args = parser.parse_args()
    print(json.dumps(json.loads(args.input.read_text(encoding="utf-8")), indent=2))


if __name__ == "__main__":
    main()
