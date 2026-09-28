"""Leakage guards for subject splits, preprocessing, and epoch construction."""

import numpy as np
import pytest

from src.data.preprocess import (
    _assert_non_overlapping_epochs,
    _normalization_stats,
    _non_overlapping_windows,
    _subject_folds,
)


@pytest.mark.parametrize("subjects", [list(range(1, 106)), list(range(1, 10))])
def test_global_subject_folds_are_disjoint_and_cover_validation_once(subjects):
    folds = _subject_folds(subjects, seed=42)
    validation_subjects = []
    for fold in folds:
        train = set(fold["train_subjects"])
        validation = set(fold["validation_subjects"])
        assert train.isdisjoint(validation)
        assert train | validation == set(subjects)
        validation_subjects.extend(validation)
    assert sorted(validation_subjects) == sorted(subjects)


def test_normalization_uses_only_training_arrays():
    train = np.ones((3, 2, 1, 8), dtype=np.float32) * 2.0
    validation_extreme = np.full((2, 2, 1, 8), 1e6, dtype=np.float32)
    mean, std = _normalization_stats([train])
    assert np.allclose(mean, 2.0)
    assert np.allclose(std, 1e-6)
    normalized_validation = (validation_extreme - mean) / std
    assert np.all(normalized_validation > 1e10)


def test_contiguous_windows_are_non_overlapping_and_drop_only_tail():
    signal = np.arange(2 * 25, dtype=np.float32).reshape(2, 25)
    windows = _non_overlapping_windows(signal, n_samples=10)
    assert windows.shape == (2, 2, 10)
    assert np.array_equal(windows[0], signal[:, :10])
    assert np.array_equal(windows[1], signal[:, 10:20])
    assert not np.intersect1d(windows[0], windows[1]).size


def test_epoch_interval_guard_accepts_disjoint_and_rejects_overlap():
    disjoint = np.array([[100, 0, 1], [580, 0, 1], [1060, 0, 1]])
    _assert_non_overlapping_epochs(disjoint, n_samples=480, source="fixture")
    overlapping = np.array([[100, 0, 1], [579, 0, 1]])
    with pytest.raises(ValueError, match="Overlapping"):
        _assert_non_overlapping_epochs(overlapping, n_samples=480, source="fixture")


def test_subject_specific_initialization_is_excluded_and_4folds_are_disjoint(tmp_path):
    """Each target is excluded from global init and held-out from its fine-tune fold."""
    import json
    from src.train.transfer_protocol import (assert_fold_checkpoint_matches, assert_global_excludes_target, subject_cv_indices)
    from src.data.preprocess import _fold_cache_is_complete

    training_subjects = list(range(1, 9))
    assert_global_excludes_target(training_subjects, target_subject=9)
    assert_fold_checkpoint_matches(training_subjects, training_subjects, [9], checkpoint_fold=2, cache_fold=2)
    with pytest.raises(ValueError, match="includes target"):
        assert_global_excludes_target(training_subjects + [9], target_subject=9)
    with pytest.raises(ValueError, match="do not match"):
        assert_fold_checkpoint_matches(training_subjects, training_subjects[:-1], [9], checkpoint_fold=2, cache_fold=2)
    with pytest.raises(ValueError, match="fold does not match"):
        assert_fold_checkpoint_matches(training_subjects, training_subjects, [9], checkpoint_fold=1, cache_fold=2)
    labels = np.repeat(np.arange(4), 8)
    folds = subject_cv_indices(labels, seed=7)
    assert len(folds) == 4
    seen = []
    for train, test in folds:
        assert set(train).isdisjoint(set(test))
        assert sorted(np.concatenate([train, test]).tolist()) == list(range(len(labels)))
        seen.extend(test.tolist())
    assert sorted(seen) == list(range(len(labels)))

    subjects = [1, 2, 3, 4, 5]
    output = tmp_path / "processed"
    splits = tmp_path / "splits"
    output.mkdir(); splits.mkdir()
    for fold in range(5):
        (output / f"eegmmidb_fold{fold}.pt").touch()
        val = [subjects[fold]]
        train = [subject for subject in subjects if subject not in val]
        (splits / f"eegmmidb_{fold}.json").write_text(json.dumps({
            "dataset": "eegmmidb", "seed": 42, "fold": fold, "cache_version": 2,
            "train_subjects": train, "validation_subjects": val,
        }), encoding="utf-8")
    assert _fold_cache_is_complete(output, splits, "eegmmidb", 42, subjects)
    assert not _fold_cache_is_complete(output, splits, "eegmmidb", 43, subjects)

    from src.train.result_store import persist_baseline_result
    report = tmp_path / "results" / "baseline_accuracy_report.csv"
    result = {"dataset": "EEGMMIDB", "model": "EEGNet-8,2", "n_classes": 4,
              "mode": "global", "fold": 0, "accuracy": 0.623, "macro_f1": 0.624,
              "published_reference_accuracy": 0.6507}
    persist_baseline_result(result, report)
    result2 = {**result, "run_id": "independent-run-2", "timestamp_utc": "2026-09-26T00:00:00+00:00"}
    persist_baseline_result(result2, report)
    persist_baseline_result(result2, report)
    import csv
    with report.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 2
    assert rows[0]["accuracy"] == "62.3"
    assert rows[0]["run_id"] != rows[1]["run_id"]
    assert rows[1]["run_id"] == "independent-run-2"
    assert rows[1]["timestamp_utc"] == "2026-09-26T00:00:00+00:00"
    assert float(rows[0]["published_reference_accuracy"]) == pytest.approx(65.07)
    assert float(rows[0]["delta_vs_published"]) == pytest.approx(-2.77)
