"""Train exact-arithmetic EEGNet models on cached subject folds in Kaggle."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, TensorDataset

from src.models.eegnet import EEGNet, PaperAlignedEEGNet82
from src.models.eegnet_42 import EEGNet42


def make_model(name: str, channels: int, samples: int, classes: int) -> torch.nn.Module:
    """Build the requested model for one dataset's input shape."""
    if name == "eegnet82_table":
        return EEGNet(channels, samples, classes, temporal_kernel=64, first_pool=4)
    if name == "eegnet82_128":
        return PaperAlignedEEGNet82(channels, samples, classes)
    if name == "eegnet42":
        return EEGNet42(channels, samples, classes, sampling_rate=160 if channels == 64 else 250)
    raise ValueError(f"Unknown model: {name}")


def learning_rate(name: str, epoch: int) -> float:
    """Return the source-paper step schedule for 100 fixed epochs."""
    if name == "eegnet42":
        return .01 if epoch <= 20 else (.002 if epoch <= 40 else (.0002 if epoch <= 60 else (4e-6 if epoch <= 80 else 4e-8)))
    return .01 if epoch <= 20 else (.001 if epoch <= 50 else .0001)


def train_fold(cache: Path, model_name: str, output_dir: Path, seed: int = 42,
               epochs: int = 100, batch_size: int = 16, device_name: str = "cuda") -> dict:
    """Fit one model on training subjects, then score untouched held-out data."""
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("A Kaggle CUDA accelerator is required for this run")
    if epochs < 1 or batch_size < 1:
        raise ValueError("Epochs and batch size must be positive")
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.set_num_threads(4)
    device = torch.device(device_name)
    pack = torch.load(cache, map_location="cpu", weights_only=True)
    x_train, y_train = pack["train_x"], pack["train_y4"]
    x_val, y_val = pack["validation_x"], pack["validation_y4"]
    if pack.get("protocol") != "T_to_E":
        assert set(pack["train_subjects"].tolist()).isdisjoint(set(pack["validation_subjects"].tolist()))
    channels, samples = x_train.shape[1], x_train.shape[-1]
    classes = int(max(y_train.max(), y_val.max())) + 1
    model = make_model(model_name, channels, samples, classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=.01)
    criterion = torch.nn.CrossEntropyLoss()
    output_dir.mkdir(parents=True, exist_ok=True)
    run_name = f"{cache.stem}_{model_name}"
    resume_path = output_dir / f"{run_name}_resume.pt"
    result_path = output_dir / f"{run_name}_result.json"
    if result_path.exists():
        return json.loads(result_path.read_text(encoding="utf-8"))
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
           "fold": pack.get("fold"), "protocol": pack.get("protocol", "subject_cv"), "seed": seed,
           "n_validation": len(true), "accuracy": float(accuracy_score(true, pred)),
           "macro_f1": float(f1_score(true, pred, average="macro", zero_division=0))}
    torch.save({"model": model.state_dict(), "seed": seed, "model_name": model_name,
                "channels": channels, "samples": samples, "classes": classes}, output_dir / f"{run_name}.pt")
    prediction = {"true": true, "pred": pred}
    if "validation_subjects" in pack:
        prediction["subject"] = pack["validation_subjects"].numpy()
    pd.DataFrame(prediction).to_csv(output_dir / f"{run_name}_predictions.csv", index=False)
    result_path.write_text(json.dumps(row, indent=2), encoding="utf-8")
    print("OUTER RESULT:", row, flush=True)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("eegmmidb", "bciiv2a"), required=True)
    parser.add_argument("--model", choices=("eegnet82_table", "eegnet82_128", "eegnet42"), required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results/baselines"))
    parser.add_argument("--folds", type=int, nargs="+", default=list(range(5)))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    for fold in args.folds:
        cache = args.cache_dir / f"{args.dataset}_fold{fold}.pt"
        train_fold(cache, args.model, args.output_dir, args.seed + fold,
                   args.epochs, args.batch_size, args.device)


if __name__ == "__main__":
    main()
