"""Inference-only EvoApprox8b sweep on the completed Wang full-cohort fold 0.

Run this on Kaggle after attaching the saved canonical full-cohort notebook output.
The same held-out participants and predictions used in the baseline report are
mandatory. Delta metrics compare an approximate LUT with an exact INT8 LUT at
the same calibration scales; FP32 baseline metrics are retained separately.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.axm.approx_conv import with_layer_lut
from src.axm.evoapprox_loader import EVOAPPROX_COMMIT, build_lut, load_metadata, select_circuits
from src.axm.lut_multiplier import exact_unsigned_lut
from src.axm.metrics import circuit_error_metrics
from src.eval.metrics import classification_metrics
from src.models.eegnet import PaperAlignedEEGNet82
from src.train.quantize import calibrate_model


LAYERS = ("temporal", "spatial", "separable", "dense")


def _evaluate(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor,
              batch_size: int, device: torch.device) -> dict:
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch in x.split(batch_size):
            predictions.append(model(batch.to(device)).argmax(dim=1).cpu().numpy())
    pred = np.concatenate(predictions)
    return classification_metrics(y.numpy(), pred, n_classes=4)


def _subject_cache(cache_dir: Path, ids: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
    windows, labels = [], []
    for subject in ids:
        path = cache_dir / f"S{subject:03d}.npz"
        if not path.is_file():
            raise FileNotFoundError(f"Missing full-cohort subject cache: {path}")
        with np.load(path, allow_pickle=False) as data:
            x, y = data["x"], data["y"]
        if x.shape != (84, 64, 1, 480) or not np.array_equal(np.bincount(y, minlength=4), [21] * 4):
            raise ValueError(f"Invalid Wang subject cache: {path}")
        windows.append(torch.from_numpy(x.astype(np.float32, copy=False)))
        labels.append(torch.from_numpy(y.astype(np.int64, copy=False)))
    return torch.cat(windows), torch.cat(labels)


def run(baseline_root: Path, output_dir: Path, library_dir: Path, cache_dir: Path,
        count: int = 50, batch_size: int = 8, fold: int = 0) -> Path:
    if not torch.cuda.is_available():
        raise RuntimeError("Use a Kaggle GPU for the full held-out sensitivity sweep")
    if fold != 0:
        raise ValueError("Fold 0 is the preregistered representative fold for this sweep")
    result_dir = baseline_root / "canonical_full_cohort"
    summary_path = result_dir / "canonical_full_cohort_summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError("The complete five-fold canonical baseline summary is required")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("n_subjects") != 105 or summary.get("folds") != 5 or summary.get("epochs") != 100:
        raise ValueError("Baseline summary does not describe the completed 105-subject protocol")
    for i in range(5):
        if not (result_dir / f"canonical_full_cohort_fold{i}_result.json").is_file():
            raise FileNotFoundError(f"Canonical baseline fold {i} is incomplete")
    row = json.loads((result_dir / f"canonical_full_cohort_fold{fold}_result.json").read_text(encoding="utf-8"))
    if row.get("tag") != "canonical_full_cohort" or row.get("epochs") != 100:
        raise ValueError("Selected fold is not a completed canonical full-cohort result")
    train_ids, test_ids = row["train_subjects"], row["validation_subjects"]
    if len(train_ids) != 84 or len(test_ids) != 21 or set(train_ids) & set(test_ids):
        raise ValueError("Canonical fold subject partition is invalid")
    train_x, _ = _subject_cache(cache_dir, train_ids)
    test_x, test_y = _subject_cache(cache_dir, test_ids)
    checkpoint = torch.load(result_dir / f"canonical_full_cohort_fold{fold}_model.pt",
                            map_location="cpu", weights_only=True)
    if checkpoint.get("fold") != fold or checkpoint.get("run_id") != row["run_id"]:
        raise ValueError("Model checkpoint provenance differs from fold result")
    device = torch.device("cuda")
    model = PaperAlignedEEGNet82(64, 480, 4).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    fp32 = _evaluate(model, test_x, test_y, batch_size, device)
    if abs(fp32["accuracy"] - row["accuracy"]) > 1e-12 or abs(fp32["macro_f1"] - row["macro_f1"]) > 1e-12:
        raise ValueError("Recomputed FP32 metrics disagree with saved canonical fold")
    output_dir.mkdir(parents=True, exist_ok=True)
    calibration_path = output_dir / "fold0_train_only_calibration.json"
    if calibration_path.exists():
        calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    else:
        calibration = calibrate_model(model, train_x, seed=42)
        calibration_path.write_text(json.dumps(calibration, indent=2), encoding="utf-8")
    scales = {name: item["activation_scale"] for name, item in calibration["layers"].items()}
    metadata = load_metadata(library_dir)
    names = select_circuits(metadata, count)
    catalog_rows = []
    for name in names:
        params = metadata[name]["params"]
        catalog_rows.append({"multiplier_id": name, "published_ER_pct": params["ep%"],
                             "published_MRED_pct": params["mre%"], "published_MAE_pct": params["mae%"],
                             "published_power": params["pwr"], "source_commit": EVOAPPROX_COMMIT})
    pd.DataFrame(catalog_rows).to_csv(output_dir / "selected_multiplier_catalog.csv", index=False)
    matrix_path = output_dir / "sensitivity_matrix_eegmmidb.csv"
    rows = pd.read_csv(matrix_path).to_dict("records") if matrix_path.exists() else []
    completed = {(item["layer"], item["multiplier_id"]) for item in rows}
    exact_lut = exact_unsigned_lut()
    for layer in LAYERS:
        exact_model = with_layer_lut(model, layer, exact_lut, scales)
        exact = _evaluate(exact_model, test_x, test_y, batch_size, device)
        del exact_model
        for index, name in enumerate(names, 1):
            if (layer, name) in completed:
                continue
            lut = build_lut(name, library_dir, metadata)
            errors = circuit_error_metrics(lut)
            approximate = with_layer_lut(model, layer, lut, scales)
            measured = _evaluate(approximate, test_x, test_y, batch_size, device)
            del approximate
            rows.append({"dataset": "EEGMMIDB", "model": "Wang EEGNet-8,2", "fold": fold,
                         "n_validation": len(test_y), "layer": layer, "multiplier_id": name,
                         "ER": errors["ER"], "MRED": errors["MRED"], "MAE": errors["MAE"],
                         "ME": errors["ME"], "AME": np.nan,
                         "AME_status": "requires validated architecture propagation model",
                         "fp32_acc": fp32["accuracy"], "exact_int8_acc": exact["accuracy"],
                         "approx_acc": measured["accuracy"],
                         "delta_acc": measured["accuracy"] - exact["accuracy"],
                         "delta_f1": measured["macro_f1"] - exact["macro_f1"],
                         "delta_sensitivity": measured["macro_TPR"] - exact["macro_TPR"]})
            pd.DataFrame(rows).to_csv(matrix_path, index=False)
            print(f"{layer} {index}/{len(names)} {name}: delta_acc={rows[-1]['delta_acc']:+.4f}", flush=True)
            torch.cuda.empty_cache()
    return matrix_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--library-dir", type=Path, default=Path("data/evoapprox8b"))
    parser.add_argument("--cache-dir", type=Path, required=True,
                        help="Recreated 105-subject Wang cache; never calibrate on held-out subjects")
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    print(run(args.baseline_root, args.output_dir, args.library_dir, args.cache_dir,
              args.count, args.batch_size))


if __name__ == "__main__":
    main()
