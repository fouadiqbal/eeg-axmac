"""Build EEGMMIDB four-class epochs, fold manifests, and normalized tensor caches."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np

from src.data.download import EEGMMIDB_RUNS, eligible_subjects

EXCLUDED_SUBJECTS = {88, 89, 92, 100}
CLASS_NAMES = ("left_hand", "right_hand", "rest", "feet")
CLASS_IDS = {name: index for index, name in enumerate(CLASS_NAMES)}
SFREQ = 160
N_SAMPLES = 3 * SFREQ


def _read_subject(data_dir: Path, subject: int) -> tuple[np.ndarray, np.ndarray]:
    """Read, filter, epoch and label one subject using the run-specific EDF events.

    Run 1 supplies continuous eyes-open rest, split into non-overlapping windows.
    Runs 4/8/12 contain left/right fists (T1/T2); runs 6/10/14 contain both
    fists/feet (T1/T2), of which T2 is the imagined-feet class. Cue-locked task
    epochs start at cue onset, matching the project prompt's 3-second window.
    """
    try:
        import mne
    except ImportError as exc:
        raise RuntimeError("MNE is required; install requirements.txt.") from exc

    root = data_dir / "raw" / "MNE-eegbci-data" / "files" / "eegmmidb" / "1.0.0" / f"S{subject:03d}"
    trials: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for run in EEGMMIDB_RUNS:
        path = root / f"S{subject:03d}R{run:02d}.edf"
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}; run `python -m src.data.download --subjects {subject}` first.")
        raw = mne.io.read_raw_edf(path, preload=True, verbose="ERROR")
        raw.pick("eeg")
        if raw.info["sfreq"] != SFREQ or len(raw.ch_names) != 64:
            raise ValueError(f"Unexpected recording metadata in {path}: {len(raw.ch_names)} EEG channels at {raw.info['sfreq']} Hz")
        raw.filter(0.5, 40.0, verbose="ERROR")

        if run == 1:
            data = raw.get_data(picks="eeg")
            data = data[:, : data.shape[1] // N_SAMPLES * N_SAMPLES]
            rest = data.reshape(64, -1, N_SAMPLES).transpose(1, 0, 2)
            trials.append(rest.astype(np.float32))
            labels.append(np.full(len(rest), CLASS_IDS["rest"], dtype=np.int64))
            continue

        # MNE annotations are run-relative: task sets differ by run family.
        event_id: dict[str, int] = {"T1": 1, "T2": 2} if run in (4, 8, 12) else {"T2": 2}
        events, _ = mne.events_from_annotations(raw, event_id=event_id, verbose="ERROR")
        epochs = mne.Epochs(
            raw, events, event_id=event_id, tmin=0.0, tmax=(N_SAMPLES - 1) / SFREQ,
            baseline=None, preload=True, picks="eeg", reject_by_annotation=True, verbose="ERROR",
        )
        data = epochs.get_data()[:, :, :N_SAMPLES].astype(np.float32)
        if run in (4, 8, 12):
            y = np.where(epochs.events[:, -1] == event_id["T1"], CLASS_IDS["left_hand"], CLASS_IDS["right_hand"])
        else:
            y = np.full(len(data), CLASS_IDS["feet"], dtype=np.int64)
        trials.append(data)
        labels.append(y.astype(np.int64))

    x = np.concatenate(trials, axis=0)[:, :, None, :]
    y = np.concatenate(labels)
    if x.shape[1:] != (64, 1, N_SAMPLES) or set(np.unique(y)) != set(range(4)):
        raise ValueError(f"Subject {subject} failed shape/class check: X={x.shape}, class counts={np.bincount(y,minlength=4)}")
    return x, y


def _subject_folds(subjects: list[int], seed: int) -> list[dict[str, list[int]]]:
    """Create deterministic, subject-level five-fold train/validation splits."""
    if len(subjects) < 5:
        raise ValueError("At least five subjects are required for subject-level five-fold CV.")
    shuffled = list(subjects)
    random.Random(seed).shuffle(shuffled)
    base, remainder = divmod(len(shuffled), 5)
    folds: list[dict[str, list[int]]] = []
    start = 0
    for fold_index in range(5):
        size = base + (1 if fold_index < remainder else 0)
        validation = shuffled[start : start + size]
        validation_set = set(validation)
        train = [subject for subject in subjects if subject not in validation_set]
        folds.append({"train_subjects": train, "validation_subjects": validation})
        start += size
    return folds


def _normalization_stats(xs: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-channel mean/std over training trials and time only."""
    joined = np.concatenate(xs, axis=0)
    mean = joined.mean(axis=(0, 2, 3), keepdims=True)
    std = joined.std(axis=(0, 2, 3), keepdims=True).clip(min=1e-6)
    return mean.astype(np.float32), std.astype(np.float32)


def _write_manifest(path: Path, seed: int, fold: int, split: dict[str, list[int]]) -> None:
    path.write_text(json.dumps({"dataset": "eegmmidb", "seed": seed, "fold": fold, **split}, indent=2), encoding="utf-8")


def preprocess_eegmmidb(
    data_dir: Path, output_dir: Path, seed: int = 42, subjects: list[int] | None = None,
) -> Path:
    """Create five normalized `.pt` caches and reproducible JSON fold manifests."""
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required; install the CPU or CUDA build.") from exc

    selected = eligible_subjects() if subjects is None else sorted(set(subjects))
    if not selected or set(selected) - set(eligible_subjects()):
        raise ValueError("Subjects must be eligible EEGMMIDB IDs (1–109, excluding 88, 89, 92, 100).")
    folds = _subject_folds(selected, seed)
    print(f"Loading and filtering {len(selected)} subjects; expected class order: {CLASS_NAMES}", flush=True)
    per_subject: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for index, subject in enumerate(selected, 1):
        per_subject[subject] = _read_subject(data_dir, subject)
        print(f"[{index}/{len(selected)}] subject {subject:03d}: X={per_subject[subject][0].shape}; counts={np.bincount(per_subject[subject][1],minlength=4).tolist()}", flush=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    split_dir = data_dir / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    for fold_id, fold in enumerate(folds):
        _write_manifest(split_dir / f"eegmmidb_{fold_id}.json", seed, fold_id, fold)
        train_xs = [per_subject[s][0] for s in fold["train_subjects"]]
        mean, std = _normalization_stats(train_xs)
        payload: dict[str, Any] = {
            "seed": seed, "fold": fold_id, "class_names": CLASS_NAMES,
            "mean": torch.from_numpy(mean), "std": torch.from_numpy(std),
        }
        for split_name, ids in (("train", fold["train_subjects"]), ("validation", fold["validation_subjects"])):
            xs: list[np.ndarray] = []
            ys: list[np.ndarray] = []
            subject_ids: list[np.ndarray] = []
            for subject in ids:
                x, y = per_subject[subject]
                xs.append(((x - mean) / std).astype(np.float32))
                ys.append(y)
                subject_ids.append(np.full(len(y), subject, dtype=np.int16))
            x_tensor = torch.from_numpy(np.concatenate(xs, axis=0))
            y4 = torch.from_numpy(np.concatenate(ys).astype(np.int64))
            y2_mask = (y4 == CLASS_IDS["left_hand"]) | (y4 == CLASS_IDS["right_hand"])
            payload[f"{split_name}_x"] = x_tensor
            payload[f"{split_name}_y4"] = y4
            payload[f"{split_name}_x2"] = x_tensor[y2_mask]
            payload[f"{split_name}_y2"] = y4[y2_mask]
            payload[f"{split_name}_subjects"] = torch.from_numpy(np.concatenate(subject_ids))
            counts = torch.bincount(y4, minlength=4).tolist()
            print(f"fold {fold_id} {split_name}: shape={tuple(x_tensor.shape)}, counts={counts}", flush=True)
        torch.save(payload, output_dir / f"eegmmidb_fold{fold_id}.pt")
    print(f"Saved five fold caches to {output_dir} and manifests to {split_dir}", flush=True)
    return output_dir


def _read_bciiv2a_session(raw_dir: Path, subject: int, session: str) -> tuple[np.ndarray, np.ndarray]:
    """Read one official GDF session with 22 EEG channels and cue-locked labels."""
    import mne
    from scipy.io import loadmat

    if session not in ("T", "E") or subject not in range(1, 10):
        raise ValueError("Expected subject 1-9 and session T or E")
    path = raw_dir / f"A{subject:02d}{session}.gdf"
    raw = mne.io.read_raw_gdf(path, preload=True, verbose="ERROR")
    raw.pick_types(eeg=True, eog=False, stim=False)
    if len(raw.ch_names) != 22 or raw.info["sfreq"] != 250:
        raise ValueError(f"Unexpected BCI-IV-2a metadata for {path}: {len(raw.ch_names)} EEG channels")
    raw.filter(0.5, 100.0, verbose="ERROR")
    if session == "T":
        event_id = {str(code): code for code in range(769, 773)}
        events, _ = mne.events_from_annotations(raw, event_id=event_id, verbose="ERROR")
        y = (events[:, -1] - 769).astype(np.int64)
    else:
        events, _ = mne.events_from_annotations(raw, event_id={"783": 783}, verbose="ERROR")
        label_path = raw_dir / f"A{subject:02d}E.mat"
        y = np.asarray(loadmat(label_path)["classlabel"]).reshape(-1).astype(np.int64) - 1
    if len(events) != 288 or len(y) != 288 or set(y.tolist()) != {0, 1, 2, 3}:
        raise ValueError(f"Expected 288 labeled four-class cues in {path}; got {len(events)} events and {len(y)} labels")
    # The cue is t=0: retain 0.5 s before it and 3.996 s afterwards.
    epochs = mne.Epochs(
        raw, events, event_id=event_id if session == "T" else {"783": 783},
        tmin=-0.5, tmax=(1125 - 1) / 250 - 0.5, baseline=None,
        preload=True, picks="eeg", reject_by_annotation=False, verbose="ERROR",
    )
    x = epochs.get_data().astype(np.float32)[:, :, None, :]
    if x.shape != (288, 22, 1, 1125) or not np.isfinite(x).all():
        raise ValueError(f"Invalid BCI-IV-2a epochs for {path}: {x.shape}")
    return x, y


def preprocess_bciiv2a(data_dir: Path, output_dir: Path, seed: int = 42) -> Path:
    """Cache subject CV on training sessions and a separate session-E test set.

    Every outer fold uses only session T. The session-E cache uses statistics
    fitted on all T sessions and is reserved for a separate cross-session test.
    """
    import torch

    raw_dir = data_dir / "raw" / "bciiv2a"
    subjects = list(range(1, 10))
    training = {subject: _read_bciiv2a_session(raw_dir, subject, "T") for subject in subjects}
    evaluation = {subject: _read_bciiv2a_session(raw_dir, subject, "E") for subject in subjects}
    folds = _subject_folds(subjects, seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    split_dir = data_dir / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    for fold_id, fold in enumerate(folds):
        (split_dir / f"bciiv2a_{fold_id}.json").write_text(
            json.dumps({"dataset": "bciiv2a", "source_session": "T", "seed": seed, "fold": fold_id, **fold}, indent=2),
            encoding="utf-8",
        )
        mean, std = _normalization_stats([training[s][0] for s in fold["train_subjects"]])
        payload: dict[str, Any] = {
            "dataset": "bciiv2a", "seed": seed, "fold": fold_id,
            "mean": torch.from_numpy(mean), "std": torch.from_numpy(std),
            "class_names": ("left_hand", "right_hand", "feet", "tongue"),
        }
        for split_name, ids in (("train", fold["train_subjects"]), ("validation", fold["validation_subjects"])):
            x = np.concatenate([((training[s][0] - mean) / std).astype(np.float32) for s in ids])
            y = np.concatenate([training[s][1] for s in ids])
            payload[f"{split_name}_x"] = torch.from_numpy(x)
            payload[f"{split_name}_y4"] = torch.from_numpy(y)
            mask = y < 2
            payload[f"{split_name}_x2"] = torch.from_numpy(x[mask])
            payload[f"{split_name}_y2"] = torch.from_numpy(y[mask])
            payload[f"{split_name}_subjects"] = torch.from_numpy(
                np.concatenate([np.full(len(training[s][1]), s, dtype=np.int16) for s in ids])
            )
            print(f"BCI fold {fold_id} {split_name}: X={x.shape}, classes={np.bincount(y,minlength=4).tolist()}", flush=True)
        torch.save(payload, output_dir / f"bciiv2a_fold{fold_id}.pt")

    # Separate protocol: train on all first sessions, test on all second sessions.
    mean, std = _normalization_stats([training[s][0] for s in subjects])
    session_payload: dict[str, Any] = {"dataset": "bciiv2a", "seed": seed, "protocol": "T_to_E",
                                       "mean": torch.from_numpy(mean), "std": torch.from_numpy(std)}
    for name, source in (("train", training), ("validation", evaluation)):
        x = np.concatenate([((source[s][0] - mean) / std).astype(np.float32) for s in subjects])
        y = np.concatenate([source[s][1] for s in subjects])
        session_payload[f"{name}_x"] = torch.from_numpy(x)
        session_payload[f"{name}_y4"] = torch.from_numpy(y)
        session_payload[f"{name}_subjects"] = torch.from_numpy(
            np.concatenate([np.full(len(source[s][1]), s, dtype=np.int16) for s in subjects])
        )
        print(f"BCI {name} session: X={x.shape}, classes={np.bincount(y,minlength=4).tolist()}", flush=True)
    torch.save(session_payload, output_dir / "bciiv2a_T_to_E.pt")
    (split_dir / "bciiv2a_T_to_E.json").write_text(
        json.dumps({"dataset": "bciiv2a", "seed": seed, "train_session": "T", "test_session": "E",
                    "subjects": subjects}, indent=2), encoding="utf-8",
    )
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["eegmmidb", "bciiv2a"], default="eegmmidb")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/eegmmidb"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--subjects", type=int, nargs="*", help="Optional eligible subject subset (at least five).")
    args = parser.parse_args()
    if args.dataset == "bciiv2a":
        if args.subjects is not None:
            parser.error("--subjects is currently supported only for EEGMMIDB")
        target = args.output_dir
        if target == Path("data/processed/eegmmidb"):
            target = Path("data/processed/bciiv2a")
        preprocess_bciiv2a(args.data_dir, target, args.seed)
    else:
        preprocess_eegmmidb(args.data_dir, args.output_dir, args.seed, args.subjects)


if __name__ == "__main__":
    main()
