"""Subject-specific transfer learning (SS-TL) for EEGMMIDB held-out subjects.

Each target subject is absent from the global checkpoint's recorded training
subject set. Its trials are partitioned into four stratified folds; each fold
fine-tunes a fresh copy of the global weights on three parts and evaluates on
its untouched fourth part. Fine-tuning defaults are implementation defaults,
not asserted to reproduce unspecified paper details.
"""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, TensorDataset

from src.train.train_baseline import make_model
from src.models.eegnet import PaperAlignedEEGNet82
from src.train.result_store import persist_baseline_result
from src.train.transfer_protocol import assert_fold_checkpoint_matches, subject_cv_indices


def _reference(model_name: str, n_classes: int, mode: str) -> float | None:
    if model_name.startswith("eegnet42"):
        ds = "ds2" if model_name.endswith("ds2") else "ds1"
        refs = {
            "ds1": {2: {"global": 83.15, "sstl": 87.46},
                    3: {"global": 75.74, "sstl": 83.26},
                    4: {"global": 65.75, "sstl": 74.31}},
            "ds2": {2: {"global": 82.52, "sstl": 93.10},
                    3: {"global": 75.34, "sstl": 93.21},
                    4: {"global": 65.56, "sstl": 89.23}},
        }
        return refs[ds][n_classes][mode]
    return {2: {"global": 82.43, "sstl": 84.32},
            3: {"global": 75.07, "sstl": 80.07},
            4: {"global": 65.07, "sstl": 70.83}}[n_classes][mode]


def run_subject_transfer(cache_path: Path, global_checkpoint: Path, output_dir: Path,
                         model_name: str = "eegnet82_table", n_classes: int = 4,
                         epochs: int = 5, batch_size: int = 16, learning_rate: float = 1e-3,
                         seed: int = 42, device_name: str = "cuda") -> pd.DataFrame:
    """Run four-fold SS-TL for every subject in one global validation fold."""
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Run SS-TL on Kaggle CUDA; local training is disabled by project protocol")
    if n_classes not in (2, 3, 4) or epochs < 1 or batch_size < 1 or learning_rate <= 0:
        raise ValueError("Invalid SS-TL run parameters")
    pack = torch.load(cache_path, map_location="cpu", weights_only=True)
    global_state = torch.load(global_checkpoint, map_location="cpu", weights_only=True)
    target_subjects = sorted(map(int, pack["validation_subjects"].unique().tolist()))
    training_subjects = list(map(int, global_state.get("training_subjects", [])))
    fold_training_subjects = list(map(int, pack["train_subjects"].unique().tolist()))
    assert_fold_checkpoint_matches(training_subjects, fold_training_subjects, target_subjects,
                                   int(global_state.get("fold", -1)), int(pack.get("fold", -2)))
    if global_state.get("model_name") != model_name:
        raise ValueError("Global checkpoint model name does not match requested SS-TL model")
    if int(global_state.get("classes", -1)) != n_classes:
        raise ValueError("Global checkpoint class count does not match requested SS-TL class count")
    resolution = "_ds2" if model_name.endswith("ds2") else ""
    x_all, y_all = pack[f"validation_x{resolution}"], pack["validation_y4"]
    subject_all = pack["validation_subjects"]
    if n_classes == 2:
        keep = y_all < 2
    elif n_classes == 3:
        keep = y_all < 3
    else:
        keep = torch.ones_like(y_all, dtype=torch.bool)
    device = torch.device(device_name)
    rows = []
    output_dir.mkdir(parents=True, exist_ok=True)
    protocol = pack.get("protocol", "subject_cv")
    protocol_suffix = "" if protocol == "subject_cv" else f"_{protocol}"
    for subject in target_subjects:
        selected = (subject_all == subject) & keep
        x, y = x_all[selected], y_all[selected]
        indices = subject_cv_indices(y.numpy(), seed + subject)
        for subject_fold, (train_idx, test_idx) in enumerate(indices):
            checkpoint_run_id = global_state.get("run_id", "legacy-init")
            tag = f"{cache_path.stem}_{model_name}_{n_classes}class{protocol_suffix}_init-{checkpoint_run_id}_subject{subject:03d}_cv{subject_fold}"
            result_path = output_dir / f"{tag}_result.json"
            checkpoint_path = output_dir / f"{tag}.pt"
            if result_path.is_file() and checkpoint_path.is_file():
                row = json.loads(result_path.read_text(encoding="utf-8"))
                persist_baseline_result(row, output_dir.parent / "baseline_accuracy_report.csv")
                rows.append(row)
                continue
            random.seed(seed + subject * 10 + subject_fold)
            np.random.seed(seed + subject * 10 + subject_fold)
            torch.manual_seed(seed + subject * 10 + subject_fold)
            model = make_model(model_name, x.shape[1], x.shape[-1], n_classes).to(device)
            model.load_state_dict(global_state["model"])
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            loader = DataLoader(TensorDataset(x[train_idx], y[train_idx]), batch_size=batch_size,
                                shuffle=True, num_workers=0,
                                generator=torch.Generator().manual_seed(seed + subject_fold))
            criterion = torch.nn.CrossEntropyLoss()
            for _epoch in range(epochs):
                model.train()
                for xb, yb in loader:
                    xb, yb = xb.to(device), yb.to(device)
                    optimizer.zero_grad(set_to_none=True)
                    loss = criterion(model(xb), yb)
                    loss.backward(); optimizer.step()
                    if isinstance(model, PaperAlignedEEGNet82):
                        model.apply_max_norm_constraints()
            model.eval()
            with torch.no_grad():
                pred = torch.cat([model(batch.to(device)).cpu().argmax(1)
                                  for batch in x[test_idx].split(128)]).numpy()
            truth = y[test_idx].numpy()
            accuracy = float(accuracy_score(truth, pred))
            reference = _reference(model_name, n_classes, "sstl")
            row = {"dataset": pack.get("dataset", "eegmmidb"), "model": model_name,
                   "run_id": tag, "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                   "n_classes": n_classes, "mode": "sstl", "fold": f"subject{subject:03d}_cv{subject_fold}",
                   "protocol": protocol,
                   "accuracy": accuracy, "macro_f1": float(f1_score(truth, pred, average="macro", zero_division=0)),
                   "published_reference_accuracy": reference,
                   "delta_vs_published": accuracy * 100 - reference,
                   "within_band": "yes" if abs(accuracy * 100 - reference) <= 3 else "no",
                   "baseline_status": "measured_ss_tl",
                   "notes": "Four-fold stratified per-subject fine-tuning, 5 epochs, Adam LR=1e-3 and batch size 16, following the authors' released SS-TL script; cohort and preprocessing still differ, so this is not an exact reproduction."}
            torch.save({"model": model.state_dict(), "model_name": model_name,
                        "classes": n_classes, "dataset": pack.get("dataset", "eegmmidb"),
                        "fold": int(pack["fold"]), "mode": "sstl", "target_subject": subject,
                        "subject_fold": subject_fold, "global_training_subjects": training_subjects,
                        "seed": seed + subject,
                        "fine_tune_train_indices": train_idx.tolist(),
                        "fine_tune_test_indices": test_idx.tolist()}, checkpoint_path)
            result_path.write_text(json.dumps(row, indent=2), encoding="utf-8")
            pd.DataFrame({"true": truth, "pred": pred}).to_csv(output_dir / f"{tag}_predictions.csv", index=False)
            persist_baseline_result(row, output_dir.parent / "baseline_accuracy_report.csv")
            rows.append(row)
            del model
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True, help="One global fold cache")
    parser.add_argument("--global-checkpoint", type=Path, required=True)
    parser.add_argument("--model", choices=("eegnet82_table", "eegnet82_128", "wang_eegnet82", "eegnet42_ds1", "eegnet42_ds2"), default="eegnet82_table")
    parser.add_argument("--classes", type=int, choices=(2, 3, 4), default=4)
    parser.add_argument("--output-dir", type=Path, default=Path("results/subject_transfer"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    print(run_subject_transfer(args.cache, args.global_checkpoint, args.output_dir, args.model,
                               args.classes, args.epochs, args.batch_size, args.learning_rate,
                               args.seed, args.device).to_string(index=False))


if __name__ == "__main__":
    main()
