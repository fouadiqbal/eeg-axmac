"""Optional leakage audit for a downloaded real EEGMMIDB smoke run."""

import csv
import glob
import os
from pathlib import Path

import pytest


def test_real_eegmmidb_cache_and_sstl_artifacts_are_leakage_safe():
    cache_name = os.environ.get("EEG_REAL_CACHE")
    transfer_name = os.environ.get("EEG_REAL_TRANSFER_DIR")
    report_name = os.environ.get("EEG_REAL_REPORT")
    if not (cache_name and transfer_name and report_name):
        pytest.skip("Set EEG_REAL_CACHE, EEG_REAL_TRANSFER_DIR, and EEG_REAL_REPORT for a real-data audit")
    import torch

    pack = torch.load(cache_name, map_location="cpu", weights_only=True)
    train_subjects = set(map(int, pack["train_subjects"].unique().tolist()))
    validation_subjects = set(map(int, pack["validation_subjects"].unique().tolist()))
    assert train_subjects.isdisjoint(validation_subjects)
    assert set(pack["train_y4"].tolist()) == {0, 1, 2, 3}
    assert set(pack["validation_y4"].tolist()) == {0, 1, 2, 3}
    if pack.get("protocol") == "wang_repo_subject_cv":
        # The source loader's default normalization is off.
        assert float(pack["train_x"].abs().max()) > 1e-8
    else:
        assert torch.allclose(pack["train_x"].mean(dim=(0, 2, 3)), torch.zeros(64), atol=2e-4)
        assert torch.allclose(pack["train_x"].std(dim=(0, 2, 3), unbiased=False), torch.ones(64), atol=2e-3)

    transfer_dir = Path(transfer_name)
    checkpoints = [Path(file) for file in glob.glob(str(transfer_dir / "*wang_repo_subject_cv_init-wangcompat-real-bnaxis*subject*_cv*.pt"))]
    assert len(checkpoints) == 4
    for checkpoint in checkpoints:
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        assert state["target_subject"] not in set(state["global_training_subjects"])
        train_indices = set(state["fine_tune_train_indices"])
        test_indices = set(state["fine_tune_test_indices"])
        assert train_indices.isdisjoint(test_indices)
        assert train_indices | test_indices == set(range(len(train_indices | test_indices)))

    with Path(report_name).open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    real_runs = [row for row in rows if row.get("run_id", "").startswith("wangcompat-real-bnaxis-fold0")]
    assert len(real_runs) == 2
    assert len({row["run_id"] for row in real_runs}) == 2
    assert all(row["timestamp_utc"] for row in real_runs)
