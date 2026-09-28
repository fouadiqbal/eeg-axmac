"""Pure protocol helpers for leakage-audited subject-specific transfer."""

from __future__ import annotations

import numpy as np


def assert_global_excludes_target(training_subjects: list[int] | tuple[int, ...], target_subject: int) -> None:
    """Raise when the target participant contributed to global initialization."""
    if target_subject in set(map(int, training_subjects)):
        raise ValueError(f"Global initialization includes target subject {target_subject}")
    if not training_subjects:
        raise ValueError("Global checkpoint has no training-subject provenance")


def assert_fold_checkpoint_matches(training_subjects: list[int] | tuple[int, ...],
                                  fold_training_subjects: list[int] | tuple[int, ...],
                                  target_subjects: list[int] | tuple[int, ...],
                                  checkpoint_fold: int, cache_fold: int) -> None:
    """Require the SS-TL initializer to be the matching outer-fold model."""
    if set(map(int, training_subjects)) != set(map(int, fold_training_subjects)):
        raise ValueError("Global checkpoint training subjects do not match the cache fold")
    if checkpoint_fold != cache_fold:
        raise ValueError("Global checkpoint fold does not match the target cache fold")
    for subject in target_subjects:
        assert_global_excludes_target(training_subjects, int(subject))


def subject_cv_indices(labels: np.ndarray, seed: int = 42) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return four stratified train/test partitions for one subject."""
    labels = np.asarray(labels)
    counts = np.unique(labels, return_counts=True)[1]
    if len(counts) < 2 or counts.min() < 4:
        raise ValueError("Each class needs at least four trials for per-subject 4-fold CV")
    rng = np.random.default_rng(seed)
    fold_parts: list[list[np.ndarray]] = [[] for _ in range(4)]
    for label in np.unique(labels):
        members = np.flatnonzero(labels == label)
        rng.shuffle(members)
        for index, part in enumerate(np.array_split(members, 4)):
            fold_parts[index].append(part)
    all_indices = np.arange(len(labels))
    partitions = []
    for parts in fold_parts:
        test = np.sort(np.concatenate(parts))
        train = np.setdiff1d(all_indices, test, assume_unique=True)
        partitions.append((train, test))
    return partitions
