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
from src.models.eegnet import EEGNet, PaperAlignedEEGNet82
from src.models.eegnet_42 import EEGNet42
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


def _model(name: str, channels: int, samples: int, classes: int) -> torch.nn.Module:
    if name == "eegnet82_table":
        return EEGNet(channels, samples, classes, temporal_kernel=64, first_pool=4)
    if name == "eegnet82_128":
        return PaperAlignedEEGNet82(channels, samples, classes)
    if name in ("eegnet42_ds1", "eegnet42_ds2"):
        return EEGNet42(channels, samples, classes, sampling_rate=160 if channels == 64 else 250,
                        downsample_factor=2 if name.endswith("ds2") else 1)
    raise ValueError(f"Unsupported sweep model: {name}")


def _select_classes(pack: dict, split: str, n_classes: int,
                    model_name: str) -> tuple[torch.Tensor, torch.Tensor]:
    resolution = "_ds2" if model_name.endswith("ds2") else ""
    if n_classes == 2:
        return pack[f"{split}_x2{resolution}"], pack[f"{split}_y2"]
    x, y = pack[f"{split}_x{resolution}"], pack[f"{split}_y4"]
    if n_classes == 3:
        mask = y < 3
        return x[mask], y[mask]
    if n_classes == 4:
        return x, y
    raise ValueError("n_classes must be 2, 3, or 4")


def sweep_fold(cache: Path, checkpoint: Path, output_dir: Path, library_dir: Path,
               count: int = 50, batch_size: int = 16, max_windows: int | None = None,
               seed: int = 42, layers: tuple[str, ...] = LAYERS,
               model_name: str = "eegnet82_table", n_classes: int = 4,
               mode: str = "global", subject: int | None = None,
               subject_fold: int | None = None) -> Path:
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
    validation_x, validation_y = _select_classes(pack, "validation", n_classes, model_name)
    global_train_x, _ = _select_classes(pack, "train", n_classes, model_name)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if (saved.get("model_name") != model_name or int(saved.get("classes", -1)) != n_classes
            or int(saved.get("fold", -1)) != int(pack.get("fold", -2))
            or saved.get("dataset", pack.get("dataset", "eegmmidb")) != pack.get("dataset", "eegmmidb")):
        raise ValueError("Checkpoint model/classes/dataset/fold do not match the selected sweep config")
    checkpoint_mode = saved.get("mode", "global")
    if checkpoint_mode != mode:
        raise ValueError(f"Expected a {mode} checkpoint; received {checkpoint_mode}")
    if mode == "global":
        x, y, train_x = validation_x, validation_y, global_train_x
        mode_tag = "global"
        subject_id = None
        subject_fold_id = None
    elif mode == "sstl":
        if subject is None or subject_fold is None:
            raise ValueError("SS-TL sensitivity requires --subject and --subject-fold")
        subject_id = int(subject)
        subject_fold_id = int(subject_fold)
        subjects = pack["validation_subjects"]
        original_labels = pack["validation_y4"]
        if n_classes == 2:
            subjects = subjects[original_labels < 2]
        elif n_classes == 3:
            subjects = subjects[original_labels < 3]
        selected = subjects == subject_id
        subject_x, subject_y = validation_x[selected], validation_y[selected]
        if int(saved.get("target_subject", -1)) != subject_id or int(saved.get("subject_fold", -1)) != subject_fold_id:
            raise ValueError("SS-TL checkpoint target subject/fold does not match sweep request")
        train_indices = np.asarray(saved.get("fine_tune_train_indices", []), dtype=np.int64)
        test_indices = np.asarray(saved.get("fine_tune_test_indices", []), dtype=np.int64)
        if (not len(train_indices) or not len(test_indices)
                or set(train_indices).intersection(test_indices)
                or sorted(np.concatenate((train_indices, test_indices)).tolist()) != list(range(len(subject_y)))):
            raise ValueError("SS-TL checkpoint trial partitions are missing, overlapping, or incomplete")
        if subject_id in set(map(int, saved.get("global_training_subjects", []))):
            raise ValueError("SS-TL global initialization was trained on the target subject")
        x, y, train_x = subject_x[test_indices], subject_y[test_indices], subject_x[train_indices]
        mode_tag = f"sstl_subject{subject_id:03d}_cv{subject_fold_id}"
    else:
        raise ValueError("mode must be 'global' or 'sstl'")
    if max_windows is not None:
        x, y = x[:max_windows], y[:max_windows]
    if len(x) == 0:
        raise ValueError("No held-out windows selected")
    model = _model(model_name, x.shape[1], x.shape[-1], n_classes).to(device)
    model.load_state_dict(saved["model"])
    model.eval()
    fp32_metrics = evaluate(model, x, y, batch_size, device)
    calibration_path = output_dir / f"{cache.stem}_{model_name}_{n_classes}class_{mode_tag}_calibration.json"
    if calibration_path.exists():
        calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    else:
        calibration = calibrate_model(model, train_x, seed=seed)
        calibration_path.write_text(json.dumps(calibration, indent=2), encoding="utf-8")
    scales = {name: values["activation_scale"] for name, values in calibration["layers"].items()}
    catalog = load_metadata(library_dir)
    chosen = select_circuits(catalog, count)
    exact_lut = exact_unsigned_lut()
    metrics_path = output_dir / "evoapprox8b_circuit_metrics.json"
    circuit_metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
    result_path = output_dir / f"sensitivity_matrix_{cache.stem}_{model_name}_{n_classes}class_{mode_tag}.csv"
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
            row = {"dataset": pack.get("dataset", "eegmmidb"), "model": model_name,
                   "n_classes": n_classes, "mode": mode, "fold": pack.get("fold", 0),
                   "subject": subject_id, "subject_fold": subject_fold_id,
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
    parser.add_argument("--model", choices=("eegnet82_table", "eegnet82_128", "eegnet42_ds1", "eegnet42_ds2"), default="eegnet82_table")
    parser.add_argument("--classes", type=int, choices=(2, 3, 4), default=4)
    parser.add_argument("--mode", choices=("global", "sstl"), default="global")
    parser.add_argument("--subject", type=int, help="Required for --mode sstl")
    parser.add_argument("--subject-fold", type=int, choices=(0, 1, 2, 3), help="Required for --mode sstl")
    args = parser.parse_args()
    print(sweep_fold(args.cache, args.checkpoint, args.output_dir, args.library_dir,
                     args.count, args.batch_size, args.max_windows, args.seed, tuple(args.layers),
                     args.model, args.classes, args.mode, args.subject, args.subject_fold))


if __name__ == "__main__":
    main()
