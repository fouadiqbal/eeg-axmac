"""Checks for the released Wang EEGNet settings and run persistence behavior."""

from pathlib import Path

import pytest
import torch

from src.models.eegnet import PaperAlignedEEGNet82
from src.train.result_store import persist_baseline_result


def test_wang_max_norm_constraints_match_released_limits() -> None:
    model = PaperAlignedEEGNet82(n_channels=4, n_samples=480, n_classes=4)
    with torch.no_grad():
        model.spatial.weight.fill_(3.0)
        model.classifier.weight.fill_(2.0)
    model.apply_max_norm_constraints()
    spatial_norms = torch.linalg.vector_norm(model.spatial.weight, dim=(1, 2, 3))
    dense_norms = torch.linalg.vector_norm(model.classifier.weight, dim=1)
    assert torch.all(spatial_norms <= 1.0 + 1e-6)
    assert torch.all(dense_norms <= 0.25 + 1e-6)


def test_wang_model_forward_has_expected_four_class_shape() -> None:
    model = PaperAlignedEEGNet82(n_channels=64, n_samples=480, n_classes=4)
    with torch.no_grad():
        assert model(torch.zeros(2, 64, 1, 480)).shape == (2, 4)


def test_csv_records_distinct_run_ids_and_idempotent_retry(tmp_path: Path) -> None:
    report = tmp_path / "baseline.csv"
    common = {"dataset": "EEGMMIDB", "model": "wang_eegnet82", "n_classes": 4,
              "mode": "global", "fold": 0, "accuracy": .62}
    persist_baseline_result({**common, "run_id": "run-a"}, report)
    persist_baseline_result({**common, "run_id": "run-b"}, report)
    persist_baseline_result({**common, "run_id": "run-b"}, report)
    import csv
    with report.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert [row["run_id"] for row in rows] == ["run-a", "run-b"]
    assert all(row["timestamp_utc"] for row in rows)
