"""Train exact-arithmetic EEGNet models on cached subject folds in Kaggle."""

from __future__ import annotations

import argparse
import json
import random
from uuid import uuid4
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, TensorDataset

from src.models.eegnet import EEGNet, PaperAlignedEEGNet82
from src.models.eegnet_42 import EEGNet42


from src.train.result_store import persist_baseline_result


def make_model(name: str, channels: int, samples: int, classes: int) -> torch.nn.Module:
    """Build the requested model for one dataset's input shape."""
    if name == "eegnet82_table":
        return EEGNet(channels, samples, classes, temporal_kernel=64, first_pool=4)
    if name in ("eegnet82_128", "wang_eegnet82"):
        return PaperAlignedEEGNet82(channels, samples, classes)
    if name in ("eegnet42", "eegnet42_ds1", "eegnet42_ds2"):
        downsample = 2 if name == "eegnet42_ds2" else 1
        return EEGNet42(channels, samples, classes,
                        sampling_rate=160 if channels == 64 else 250,
                        downsample_factor=downsample)
    raise ValueError(f"Unknown model: {name}")


def learning_rate(name: str, epoch: int) -> float:
    """Return the source-paper step schedule for 100 fixed epochs."""
    if name.startswith("eegnet42"):
        return .01 if epoch <= 20 else (.002 if epoch <= 40 else (.0002 if epoch <= 60 else (4e-6 if epoch <= 80 else 4e-8)))
    return .01 if epoch <= 20 else (.001 if epoch <= 50 else .0001)


def train_fold(cache: Path, model_name: str, output_dir: Path, seed: int = 42,
               epochs: int = 100, batch_size: int = 16, device_name: str = "cuda",
               n_classes: int = 4, run_id: str | None = None) -> dict:
    """Fit one model on training subjects, then score untouched held-out data."""
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("A Kaggle CUDA accelerator is required for this run")
    if epochs < 1 or batch_size < 1:
        raise ValueError("Epochs and batch size must be positive")
    run_id = run_id or str(uuid4())
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.set_num_threads(4)
    device = torch.device(device_name)
    if n_classes not in (2, 3, 4):
        raise ValueError("n_classes must be 2, 3, or 4")
    pack = torch.load(cache, map_location="cpu", weights_only=True)
    resolution = "_ds2" if model_name.endswith("ds2") else ""
    if n_classes == 2:
        x_train, y_train = pack[f"train_x2{resolution}"], pack["train_y2"]
        x_val, y_val = pack[f"validation_x2{resolution}"], pack["validation_y2"]
        val_subjects = pack["validation_subjects"][((pack["validation_y4"] == 0) | (pack["validation_y4"] == 1))]
    else:
        x_train, y_train = pack[f"train_x{resolution}"], pack["train_y4"]
        x_val, y_val = pack[f"validation_x{resolution}"], pack["validation_y4"]
        val_subjects = pack.get("validation_subjects")
        if n_classes == 3:
            train_mask, val_mask = y_train < 3, y_val < 3
            x_train, y_train = x_train[train_mask], y_train[train_mask]
            x_val, y_val = x_val[val_mask], y_val[val_mask]
            if val_subjects is not None:
                val_subjects = val_subjects[val_mask]
    if pack.get("protocol") != "T_to_E":
        assert set(pack["train_subjects"].tolist()).isdisjoint(set(pack["validation_subjects"].tolist()))
    channels, samples = x_train.shape[1], x_train.shape[-1]
    classes = n_classes
    model = make_model(model_name, channels, samples, classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=.01)
    criterion = torch.nn.CrossEntropyLoss()
    output_dir.mkdir(parents=True, exist_ok=True)
    run_name = f"{cache.stem}_{model_name}_{n_classes}class_{run_id}"
    resume_path = output_dir / f"{run_name}_resume.pt"
    result_path = output_dir / f"{run_name}_result.json"
    if result_path.exists():
        cached_row = json.loads(result_path.read_text(encoding="utf-8"))
        persist_baseline_result(cached_row, output_dir.parent / "baseline_accuracy_report.csv")
        return cached_row
    history = []
    first_epoch = 1
    if resume_path.exists():
        saved = torch.load(resume_path, map_location="cpu", weights_only=False)
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        history = saved["history"]
        first_epoch = saved["epoch"] + 1
    for epoch in range(first_epoch, epochs + 1):
        lr = learning_rate(model_name, epoch)
        for group in optimizer.param_groups:
            group["lr"] = lr
        loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True,
                            num_workers=0, pin_memory=device.type == "cuda",
                            generator=torch.Generator().manual_seed(seed * 1000 + epoch))
        model.train(); loss_sum = 0.0; correct = 0
        for xb, yb in loader:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward(); optimizer.step()
            if isinstance(model, PaperAlignedEEGNet82):
                model.apply_max_norm_constraints()
            loss_sum += loss.item() * len(yb)
            correct += (logits.argmax(1) == yb).sum().item()
        history.append({"epoch": epoch, "lr": lr, "train_loss": loss_sum / len(y_train),
                        "train_accuracy": correct / len(y_train)})
        if epoch == 1 or epoch % 10 == 0:
            print(f"{run_name} epoch {epoch}/{epochs} train_acc={history[-1]['train_accuracy']:.4f}", flush=True)
        if epoch % 10 == 0:
            torch.save({"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                        "history": history}, resume_path)
            pd.DataFrame(history).to_csv(output_dir / f"{run_name}_history.csv", index=False)
    model.eval()
    with torch.no_grad():
        pred = torch.cat([model(batch.to(device)).cpu().argmax(1) for batch in x_val.split(128)]).numpy()
    true = y_val.numpy()
    row = {"dataset": pack["dataset"] if "dataset" in pack else "eegmmidb", "model": model_name,
           "run_id": run_id, "timestamp_utc": datetime.now(timezone.utc).isoformat(),
           "fold": pack.get("fold"), "protocol": pack.get("protocol", "subject_cv"), "seed": seed,
           "n_classes": classes, "mode": "global", "n_validation": len(true),
           "training_subjects": pack.get("train_subjects", torch.empty(0, dtype=torch.int16)).unique().tolist(),
           "validation_subjects": val_subjects.unique().tolist() if val_subjects is not None else [],
           "notes": ("Five-subject local CPU smoke result using raw Wang-repository window construction; not a benchmark estimate."
                     if pack.get("protocol") == "wang_repo_subject_cv" else
                     "Five-subject local CPU smoke result using project filtering/window construction; not a benchmark estimate."),
           "accuracy": float(accuracy_score(true, pred)),
           "macro_f1": float(f1_score(true, pred, average="macro", zero_division=0))}
    torch.save({"model": model.state_dict(), "seed": seed, "model_name": model_name,
                "channels": channels, "samples": samples, "classes": classes,
                "run_id": run_id, "protocol": pack.get("protocol", "subject_cv"),
                "training_subjects": row["training_subjects"], "fold": row["fold"],
                "dataset": row["dataset"], "mode": "global"}, output_dir / f"{run_name}.pt")
    prediction = {"true": true, "pred": pred}
    if "validation_subjects" in pack:
        prediction["subject"] = pack["validation_subjects"].numpy()
    pd.DataFrame(prediction).to_csv(output_dir / f"{run_name}_predictions.csv", index=False)
    reference = None
    if model_name == "eegnet82_table":
        reference = {2: .8243, 3: .7507, 4: .6507}[n_classes]
    elif model_name in ("eegnet42_ds1", "eegnet42_ds2"):
        reference = {"eegnet42_ds1": {2: .8315, 3: .7574, 4: .6575},
                     "eegnet42_ds2": {2: .8252, 3: .7534, 4: .6556}}[model_name][n_classes]
    if reference is not None:
        row["published_reference_accuracy"] = reference
        row["delta_vs_published"] = row["accuracy"] - reference
        row["within_band"] = "yes" if abs(row["delta_vs_published"] * 100) <= 3 else "no"
    result_path.write_text(json.dumps(row, indent=2), encoding="utf-8")
    persist_baseline_result(row, output_dir.parent / "baseline_accuracy_report.csv")
    print("OUTER RESULT:", row, flush=True)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("eegmmidb", "bciiv2a"), required=True)
    parser.add_argument("--model", choices=("eegnet82_table", "eegnet82_128", "wang_eegnet82", "eegnet42", "eegnet42_ds1", "eegnet42_ds2"), required=True)
    parser.add_argument("--run-id", default=None, help="Optional unique run ID; repeated ID is idempotent.")
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results/baselines"))
    parser.add_argument("--folds", type=int, nargs="+", default=list(range(5)))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--classes", type=int, choices=(2, 3, 4), default=4)
    args = parser.parse_args()
    for fold in args.folds:
        cache = args.cache_dir / f"{args.dataset}_fold{fold}.pt"
        run_id = f"{args.run_id}-fold{fold}" if args.run_id else None
        train_fold(cache, args.model, args.output_dir, args.seed + fold,
                   args.epochs, args.batch_size, args.device, args.classes, run_id)


if __name__ == "__main__":
    main()
