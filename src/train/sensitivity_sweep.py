"""Inference-only layer-by-circuit sensitivity sweep on held-out EEG windows."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.axm.approx_conv import with_layer_lut
from src.axm.evoapprox_loader import build_lut, load_metadata, select_circuits
from src.axm.lut_multiplier import exact_unsigned_lut
from src.axm.metrics import circuit_error_metrics
from src.eval.metrics import classification_metrics
from src.models.eegnet import PaperAlignedEEGNet82
from src.train.quantize import calibrate_model


LAYERS = ("temporal", "spatial", "separable", "dense")


def evaluate(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor,
             batch_size: int, device: torch.device) -> dict:
    """Evaluate every supplied held-out window exactly once."""
    model.eval()
    outputs = []
    with torch.no_grad():
        for batch in x.split(batch_size):
            outputs.append(model(batch.to(device)).cpu())
    logits = torch.cat(outputs)
    pred = logits.argmax(dim=1).numpy()
    scores = logits.softmax(dim=1).numpy()
    return classification_metrics(y.numpy(), pred, scores=scores, n_classes=logits.shape[1])


def sweep_fold(cache: Path, checkpoint: Path, output_dir: Path, library_dir: Path,
               count: int = 50, batch_size: int = 16, max_windows: int | None = None,
               seed: int = 42, layers: tuple[str, ...] = LAYERS) -> Path:
    """Save one row per selected layer and multiplier for one outer test fold.

    ``delta_*`` compares the approximate circuit with an exact signed-adapter
    LUT at the same INT8 scales and selected layer. It therefore isolates the
    circuit error from quantization. FP32 metrics are recorded separately.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("Run the sensitivity sweep on a Kaggle GPU")
    torch.set_num_threads(4)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    pack = torch.load(cache, map_location="cpu", weights_only=True)
    x = pack["validation_x"]
    y = pack["validation_y4"]
    if max_windows is not None:
        x, y = x[:max_windows], y[:max_windows]
    if len(x) == 0:
        raise ValueError("No held-out windows selected")
    model = PaperAlignedEEGNet82(x.shape[1], x.shape[-1], int(pack["train_y4"].max()) + 1).to(device)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(saved["model"])
    model.eval()
    fp32_metrics = evaluate(model, x, y, batch_size, device)
    calibration_path = output_dir / f"{cache.stem}_calibration.json"
    if calibration_path.exists():
        calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    else:
        calibration = calibrate_model(model, pack["train_x"], seed=seed)
        calibration_path.write_text(json.dumps(calibration, indent=2), encoding="utf-8")
    scales = {name: values["activation_scale"] for name, values in calibration["layers"].items()}
    catalog = load_metadata(library_dir)
    chosen = select_circuits(catalog, count)
    exact_lut = exact_unsigned_lut()
    metrics_path = output_dir / "evoapprox8b_circuit_metrics.json"
    circuit_metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
    result_path = output_dir / f"sensitivity_matrix_{cache.stem}_eegnet82_128.csv"
    rows = pd.read_csv(result_path).to_dict("records") if result_path.exists() else []
    completed = {(str(row["layer"]), str(row["multiplier_id"])) for row in rows}
    for layer in layers:
        exact_model = with_layer_lut(model, layer, exact_lut, scales)
        exact_metrics = evaluate(exact_model, x, y, batch_size, device)
        del exact_model
        torch.cuda.empty_cache()
        for index, name in enumerate(chosen, 1):
            if (layer, name) in completed:
                continue
            lut = build_lut(name, library_dir, catalog)
            if name not in circuit_metrics:
                circuit_metrics[name] = circuit_error_metrics(lut)
                metrics_path.write_text(json.dumps(circuit_metrics, indent=2), encoding="utf-8")
            approximate = with_layer_lut(model, layer, lut, scales)
            measured = evaluate(approximate, x, y, batch_size, device)
            errors = circuit_metrics[name]
            row = {"dataset": pack.get("dataset", "eegmmidb"), "model": "eegnet82_128", "fold": pack.get("fold", 0),
                   "seed": seed, "n_validation": len(y), "layer": layer, "multiplier_id": name,
                   "ER": errors["ER"], "MRED": errors["MRED"], "MAE": errors["MAE"],
                   "ME": errors["ME"], "AME": np.nan,
                   "AME_status": "requires architecture-specific propagation model",
                   "fp32_acc": fp32_metrics["accuracy"], "exact_int8_acc": exact_metrics["accuracy"],
                   "approx_acc": measured["accuracy"],
                   "delta_acc": measured["accuracy"] - exact_metrics["accuracy"],
                   "delta_f1": measured["macro_f1"] - exact_metrics["macro_f1"],
                   "delta_sensitivity": measured["macro_TPR"] - exact_metrics["macro_TPR"]}
            rows.append(row)
            pd.DataFrame(rows).to_csv(result_path, index=False)
            print(f"{layer} [{index}/{len(chosen)}] {name}: ΔACC={row['delta_acc']:+.4f}", flush=True)
            del approximate
            gc.collect(); torch.cuda.empty_cache()
    return result_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--library-dir", type=Path, default=Path("data/evoapprox8b"))
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-windows", type=int, default=None, help="Pilot only; omit for the full held-out fold")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--layers", nargs="+", choices=LAYERS, default=list(LAYERS))
    args = parser.parse_args()
    print(sweep_fold(args.cache, args.checkpoint, args.output_dir, args.library_dir,
                     args.count, args.batch_size, args.max_windows, args.seed, tuple(args.layers)))


if __name__ == "__main__":
    main()
