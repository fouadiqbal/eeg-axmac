"""Checks for processed EEGMMIDB and BCI-IV-2a cache structure."""

import os
from pathlib import Path

import pytest
import numpy as np

from src.data.preprocess import _normalization_stats, _subject_folds

try:
    import torch
except ImportError:
    torch = None


@pytest.mark.parametrize("fold", range(5))
def test_eegmmidb_fold_shapes_and_dtypes(fold: int) -> None:
    """Cached folds contain aligned float epochs and integer class labels."""
    if torch is None:
        pytest.skip("Install PyTorch to check serialized tensor caches.")
    path = Path(os.environ.get("EEG_CACHE_DIR", "data/processed/eegmmidb")) / f"eegmmidb_fold{fold}.pt"
    if not path.exists():
        pytest.skip("Run preprocessing before checking generated fold caches.")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    for split in ("train", "validation"):
        x = payload[f"{split}_x"]
        y = payload[f"{split}_y4"]
        assert x.ndim == 4 and x.shape[1:] == (64, 1, 480)
        assert x.dtype == torch.float32
        assert y.ndim == 1 and y.dtype == torch.int64
        assert len(x) == len(y) and len(y) > 0
        assert set(y.tolist()).issubset({0, 1, 2, 3})
        x2, y2 = payload[f"{split}_x2"], payload[f"{split}_y2"]
        assert len(x2) == len(y2)
        assert set(y2.tolist()).issubset({0, 1})


@pytest.mark.parametrize("fold", range(5))
def test_bciiv2a_fold_shapes_and_dtypes(fold: int) -> None:
    """Every processed BCI outer fold has 22 channels and 1,125 samples."""
    if torch is None:
        pytest.skip("Install PyTorch to check serialized tensor caches.")
    path = Path(os.environ.get("BCI_CACHE_DIR", "data/processed/bciiv2a")) / f"bciiv2a_fold{fold}.pt"
    if not path.exists():
        pytest.skip("Run BCI-IV-2a preprocessing before checking its fold caches.")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    assert set(payload["train_subjects"].tolist()).isdisjoint(set(payload["validation_subjects"].tolist()))
    for split in ("train", "validation"):
        x, y = payload[f"{split}_x"], payload[f"{split}_y4"]
        assert x.shape[1:] == (22, 1, 1125)
        assert len(x) == len(y) and x.dtype == torch.float32 and y.dtype == torch.int64
        assert torch.isfinite(x).all()
        assert set(y.tolist()) == {0, 1, 2, 3}


def test_subject_folds_cover_each_subject_once_per_validation() -> None:
    subjects = list(range(1, 106))
    folds = _subject_folds(subjects, seed=42)
    validation = [subject for fold in folds for subject in fold["validation_subjects"]]
    assert sorted(validation) == subjects
    for fold in folds:
        assert len(fold["train_subjects"]) == 84
        assert len(fold["validation_subjects"]) == 21
        assert set(fold["train_subjects"]).isdisjoint(fold["validation_subjects"])


def test_normalization_stats_are_channelwise_and_train_only() -> None:
    train = np.ones((2, 3, 1, 4), dtype=np.float32)
    train[:, 1, :, :] *= 3
    mean, std = _normalization_stats([train])
    assert mean.shape == (1, 3, 1, 1)
    assert np.allclose(mean.ravel(), [1, 3, 1])
    assert np.all(std > 0)  # zero-variance channels receive the stable epsilon floor
