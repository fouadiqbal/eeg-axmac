"""Build EEGMMIDB four-class epochs, fold manifests, and normalized tensor caches."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np

from src.data.download import EEGMMIDB_RUNS, eligible_subjects, wang_eligible_subjects

EXCLUDED_SUBJECTS = {88, 89, 92, 100}
CLASS_NAMES = ("left_hand", "right_hand", "rest", "feet")
WANG_CLASS_NAMES = CLASS_NAMES
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
            rest = _non_overlapping_windows(data, N_SAMPLES)
            trials.append(rest.astype(np.float32))
            labels.append(np.full(len(rest), CLASS_IDS["rest"], dtype=np.int64))
            continue

        # MNE annotations are run-relative: task sets differ by run family.
        event_id: dict[str, int] = {"T1": 1, "T2": 2} if run in (4, 8, 12) else {"T2": 2}
        events, _ = mne.events_from_annotations(raw, event_id=event_id, verbose="ERROR")
        _assert_non_overlapping_epochs(events, N_SAMPLES, f"EEGMMIDB subject {subject} run {run}")
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


def _read_subject_wang_repo(data_dir: Path, subject: int) -> tuple[np.ndarray, np.ndarray]:
    """Reproduce the released get_data.py 4-class EDF extraction."""
    import pyedflib

    root = data_dir / "raw" / "MNE-eegbci-data" / "files" / "eegmmidb" / "1.0.0" / f"S{subject:03d}"
    xs: list[np.ndarray] = []
    ys: list[int] = []
    for run in EEGMMIDB_RUNS:
        path = root / f"S{subject:03d}R{run:02d}.edf"
        if not path.is_file():
            raise FileNotFoundError(f"Missing {path}; download with --protocol wang_repo first")
        reader = pyedflib.EdfReader(str(path))
        try:
            fs = int(reader.getSampleFrequency(0))
            if fs != SFREQ or reader.signals_in_file < 64:
                raise ValueError(f"Unexpected EDF metadata in {path}: fs={fs}, channels={reader.signals_in_file}")
            signal = np.stack([reader.readSignal(ch) for ch in range(64)]).astype(np.float32)
            onsets, _, descriptions = reader.readAnnotations()
        finally:
            reader.close()
        if run == 1:
            # Match get_data.py's exact 20 contiguous windows and extra Python
            # random.randint window. Its np.random.seed(7) does not seed Python's
            # random module, so the latter is not reproducible in the source.
            for start in range(20):
                xs.append(signal[:, start * N_SAMPLES:(start + 1) * N_SAMPLES]); ys.append(2)
            start = random.Random(7).randint(0, 57 * fs)
            if start + N_SAMPLES > signal.shape[1]:
                raise ValueError(f"Wang loader's random rest window exceeds {path}")
            xs.append(signal[:, start:start + N_SAMPLES]); ys.append(2)
            continue
        counts = {"T1": 0, "T2": 0}
        for label, onset in zip(descriptions, onsets):
            start = int(fs * onset)
            epoch = signal[:, start:start + N_SAMPLES]
            if epoch.shape[1] != N_SAMPLES:
                continue
            if run in (4, 8, 12) and label in ("T1", "T2") and counts[label] < 7:
                xs.append(epoch); ys.append(0 if label == "T1" else 1); counts[label] += 1
            elif run in (6, 10, 14) and label == "T2" and counts["T2"] < 7:
                xs.append(epoch); ys.append(3); counts["T2"] += 1
    x, y = np.stack(xs).astype(np.float32)[:, :, None, :], np.asarray(ys, dtype=np.int64)
    if set(np.unique(y).tolist()) != {0, 1, 2, 3} or not np.isfinite(x).all():
        raise ValueError(f"Invalid Wang-compatible subject {subject}: X={x.shape}, counts={np.bincount(y,minlength=4)}")
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


def _non_overlapping_windows(signal: np.ndarray, n_samples: int) -> np.ndarray:
    """Split channels × time signal into contiguous, non-overlapping windows."""
    if signal.ndim != 2 or n_samples <= 0:
        raise ValueError("Expected a channels × time signal and positive window length")
    n_windows = signal.shape[1] // n_samples
    if n_windows == 0:
        raise ValueError("Signal is shorter than one window")
    return signal[:, : n_windows * n_samples].reshape(signal.shape[0], n_windows, n_samples).transpose(1, 0, 2)


def _assert_non_overlapping_epochs(events: np.ndarray, n_samples: int, source: str) -> None:
    """Fail if cue-locked epochs reuse raw samples within one recording."""
    starts = np.asarray(events)[:, 0].astype(np.int64)
    if starts.size < 2:
        return
    starts.sort()
    if np.any(np.diff(starts) < n_samples):
        raise ValueError(f"Overlapping {source} epochs: cue spacing is shorter than {n_samples} samples")



def _fold_cache_is_complete(output_dir: Path, split_dir: Path, dataset: str,
                            seed: int, subjects: list[int]) -> bool:
    """Return true only when all tensor caches and matching fold manifests exist."""
    for fold_id in range(5):
        cache_path = output_dir / f"{dataset}_fold{fold_id}.pt"
        manifest_path = split_dir / f"{dataset}_{fold_id}.json"
        if not cache_path.is_file() or not manifest_path.is_file():
            return False
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        train = set(manifest.get("train_subjects", []))
        validation = set(manifest.get("validation_subjects", []))
        if (manifest.get("dataset") != dataset or manifest.get("seed") != seed
                or manifest.get("fold") != fold_id or manifest.get("cache_version") != 2
                or not train.isdisjoint(validation)
                or train | validation != set(subjects)):
            return False
    return True


def _upgrade_legacy_eegmmidb_cache(output_dir: Path, split_dir: Path,
                                    seed: int, subjects: list[int], torch_module) -> bool:
    """Add ds=2 tensors to legacy fold caches without reopening or filtering EDFs."""
    cache_paths = [output_dir / f"eegmmidb_fold{fold}.pt" for fold in range(5)]
    manifest_paths = [split_dir / f"eegmmidb_{fold}.json" for fold in range(5)]
    if not all(path.is_file() for path in cache_paths + manifest_paths):
        return False
    manifests = []
    payloads = []
    try:
        for fold, (cache_path, manifest_path) in enumerate(zip(cache_paths, manifest_paths)):
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            train_subjects = set(manifest.get("train_subjects", []))
            validation_subjects = set(manifest.get("validation_subjects", []))
            if (manifest.get("dataset") != "eegmmidb" or manifest.get("seed") != seed
                    or manifest.get("fold") != fold or not train_subjects.isdisjoint(validation_subjects)
                    or train_subjects | validation_subjects != set(subjects)
                    or manifest.get("cache_version") not in (None, 2)):
                return False
            payload = torch_module.load(cache_path, map_location="cpu", weights_only=True)
            required = {"mean", "std", "train_x", "train_y4", "validation_x", "validation_y4"}
            if manifest.get("cache_version") == 2:
                required |= {"mean_ds2", "std_ds2", "train_x_ds2", "train_x2_ds2",
                             "validation_x_ds2", "validation_x2_ds2"}
            if not required.issubset(payload):
                return False
            manifests.append(manifest)
            payloads.append(payload)
    except (OSError, ValueError, KeyError, json.JSONDecodeError, RuntimeError):
        return False

    from scipy.signal import resample_poly

    for fold, (manifest, payload, cache_path, manifest_path) in enumerate(
            zip(manifests, payloads, cache_paths, manifest_paths)):
        if manifest.get("cache_version") == 2:
            continue
        mean, std = payload["mean"].numpy(), payload["std"].numpy()
        train_raw = payload["train_x"].numpy() * std + mean
        train_ds2 = resample_poly(train_raw, up=1, down=2, axis=-1).astype(np.float32)
        mean_ds2, std_ds2 = _normalization_stats([train_ds2])
        payload["mean_ds2"] = torch_module.from_numpy(mean_ds2)
        payload["std_ds2"] = torch_module.from_numpy(std_ds2)
        for split_name in ("train", "validation"):
            raw = payload[f"{split_name}_x"].numpy() * std + mean
            downsampled = resample_poly(raw, up=1, down=2, axis=-1).astype(np.float32)
            x_ds2 = torch_module.from_numpy(((downsampled - mean_ds2) / std_ds2).astype(np.float32))
            y4 = payload[f"{split_name}_y4"]
            payload[f"{split_name}_x_ds2"] = x_ds2
            payload[f"{split_name}_x2_ds2"] = x_ds2[(y4 == 0) | (y4 == 1)]
        torch_module.save(payload, cache_path)
        manifest["cache_version"] = 2
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return True


def _write_manifest(path: Path, seed: int, fold: int, split: dict[str, list[int]]) -> None:
    path.write_text(json.dumps({"dataset": "eegmmidb", "seed": seed, "fold": fold,
                                "cache_version": 2, **split}, indent=2), encoding="utf-8")


def preprocess_eegmmidb(
    data_dir: Path, output_dir: Path, seed: int = 42, subjects: list[int] | None = None,
    protocol: str = "project",
) -> Path:
    """Create five normalized `.pt` caches and reproducible JSON fold manifests."""
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required; install the CPU or CUDA build.") from exc

    if protocol not in ("project", "wang_repo"):
        raise ValueError("protocol must be 'project' or 'wang_repo'")
    allowed = eligible_subjects() if protocol == "project" else wang_eligible_subjects()
    selected = allowed if subjects is None else sorted(set(subjects))
    if not selected or set(selected) - set(allowed):
        raise ValueError(f"Invalid {protocol} cohort; expected exclusions {sorted(set(range(1, 110)) - set(allowed))}")
    if protocol == "wang_repo":
        combined_path = output_dir / "eegmmidb_wang_repo_4class.npz"
        cache_paths = [output_dir / f"eegmmidb_fold{fold}.pt" for fold in range(5)]
        if combined_path.is_file() and all(path.is_file() for path in cache_paths):
            try:
                complete = True
                for fold_id, cache_path in enumerate(cache_paths):
                    cached = torch.load(cache_path, map_location="cpu", weights_only=True)
                    train_ids = set(map(int, cached["train_subject_ids"].tolist()))
                    validation_ids = set(map(int, cached["validation_subject_ids"].tolist()))
                    complete &= (cached.get("protocol") == "wang_repo_subject_cv"
                                 and cached.get("cache_version") == 1
                                 and cached.get("seed") == seed and cached.get("fold") == fold_id
                                 and train_ids.isdisjoint(validation_ids)
                                 and train_ids | validation_ids == set(selected)
                                 and set(cached["class_names"]) == set(WANG_CLASS_NAMES))
                if complete:
                    print(f"Using existing Wang-compatible tensor caches in {output_dir}; EDF loading skipped.", flush=True)
                    return output_dir
            except (OSError, KeyError, ValueError, RuntimeError):
                pass
        per_subject = {subject: _read_subject_wang_repo(data_dir, subject) for subject in selected}
        output_dir.mkdir(parents=True, exist_ok=True)
        x = np.concatenate([per_subject[s][0] for s in selected])
        y = np.concatenate([per_subject[s][1] for s in selected])
        subject_ids = np.concatenate([np.full(len(per_subject[s][1]), s, dtype=np.int16) for s in selected])
        np.savez_compressed(output_dir / "eegmmidb_wang_repo_4class.npz", X_Train=x, y_Train=y,
                            subjects=subject_ids, cohort=np.asarray(selected), protocol=np.asarray("wang_repo"))
        # Keep the released loader's raw, unnormalized amplitudes, but split by
        # participant for a leakage-safe generalization estimate.
        folds = _subject_folds(selected, seed)
        import torch
        for fold_id, split in enumerate(folds):
            payload: dict[str, Any] = {"dataset": "eegmmidb", "protocol": "wang_repo_subject_cv",
                                       "cache_version": 1, "seed": seed, "fold": fold_id,
                                       "class_names": WANG_CLASS_NAMES}
            for split_name, ids in (("train", split["train_subjects"]),
                                    ("validation", split["validation_subjects"])):
                split_x = np.concatenate([per_subject[s][0] for s in ids])
                split_y = np.concatenate([per_subject[s][1] for s in ids])
                split_subjects = np.concatenate([np.full(len(per_subject[s][1]), s, dtype=np.int16) for s in ids])
                payload[f"{split_name}_x"] = torch.from_numpy(split_x)
                payload[f"{split_name}_y4"] = torch.from_numpy(split_y)
                payload[f"{split_name}_subjects"] = torch.from_numpy(split_subjects)
                binary = split_y < 2
                payload[f"{split_name}_x2"] = torch.from_numpy(split_x[binary])
                payload[f"{split_name}_y2"] = torch.from_numpy(split_y[binary])
            payload["train_subject_ids"] = torch.from_numpy(np.asarray(split["train_subjects"], dtype=np.int16))
            payload["validation_subject_ids"] = torch.from_numpy(np.asarray(split["validation_subjects"], dtype=np.int16))
            torch.save(payload, output_dir / f"eegmmidb_fold{fold_id}.pt")
        print(f"Wang repo-compatible windows: X={x.shape}, counts={np.bincount(y,minlength=4).tolist()}; saved {output_dir / 'eegmmidb_wang_repo_4class.npz'} and subject-disjoint fold caches in {output_dir}", flush=True)
        return output_dir
    folds = _subject_folds(selected, seed)
    split_dir = data_dir / "splits"
    if _fold_cache_is_complete(output_dir, split_dir, "eegmmidb", seed, selected):
        print(f"Using existing EEGMMIDB tensor and split caches in {output_dir}; raw processing skipped.", flush=True)
        return output_dir
    if _upgrade_legacy_eegmmidb_cache(output_dir, split_dir, seed, selected, torch):
        print(f"Upgraded cached EEGMMIDB folds with ds=2 tensors in {output_dir}; EDF loading/filtering skipped.", flush=True)
        return output_dir
    print(f"Loading and filtering {len(selected)} subjects; expected class order: {CLASS_NAMES}", flush=True)
    per_subject: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for index, subject in enumerate(selected, 1):
        per_subject[subject] = _read_subject(data_dir, subject)
        print(f"[{index}/{len(selected)}] subject {subject:03d}: X={per_subject[subject][0].shape}; counts={np.bincount(per_subject[subject][1],minlength=4).tolist()}", flush=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    split_dir.mkdir(parents=True, exist_ok=True)
    for fold_id, fold in enumerate(folds):
        _write_manifest(split_dir / f"eegmmidb_{fold_id}.json", seed, fold_id, fold)
        from scipy.signal import resample_poly

        train_xs = [per_subject[s][0] for s in fold["train_subjects"]]
        mean, std = _normalization_stats(train_xs)
        train_xs_ds2 = [resample_poly(x, up=1, down=2, axis=-1).astype(np.float32) for x in train_xs]
        mean_ds2, std_ds2 = _normalization_stats(train_xs_ds2)
        payload: dict[str, Any] = {
            "seed": seed, "fold": fold_id, "class_names": CLASS_NAMES,
            "mean": torch.from_numpy(mean), "std": torch.from_numpy(std),
            "mean_ds2": torch.from_numpy(mean_ds2), "std_ds2": torch.from_numpy(std_ds2),
        }
        for split_name, ids in (("train", fold["train_subjects"]), ("validation", fold["validation_subjects"])):
            xs: list[np.ndarray] = []
            xs_ds2: list[np.ndarray] = []
            ys: list[np.ndarray] = []
            subject_ids: list[np.ndarray] = []
            for subject in ids:
                x, y = per_subject[subject]
                x_ds2 = resample_poly(x, up=1, down=2, axis=-1).astype(np.float32)
                xs.append(((x - mean) / std).astype(np.float32))
                xs_ds2.append(((x_ds2 - mean_ds2) / std_ds2).astype(np.float32))
                ys.append(y)
                subject_ids.append(np.full(len(y), subject, dtype=np.int16))
            x_tensor = torch.from_numpy(np.concatenate(xs, axis=0))
            x_tensor_ds2 = torch.from_numpy(np.concatenate(xs_ds2, axis=0))
            y4 = torch.from_numpy(np.concatenate(ys).astype(np.int64))
            y2_mask = (y4 == CLASS_IDS["left_hand"]) | (y4 == CLASS_IDS["right_hand"])
            payload[f"{split_name}_x"] = x_tensor
            payload[f"{split_name}_x_ds2"] = x_tensor_ds2
            payload[f"{split_name}_y4"] = y4
            payload[f"{split_name}_x2"] = x_tensor[y2_mask]
            payload[f"{split_name}_x2_ds2"] = x_tensor_ds2[y2_mask]
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
    # MNE labels the three EOG channels as EEG in these GDF files. Select the
    # 22 channels whose names do not carry the EOG prefix explicitly.
    eeg_names = [name for name in raw.ch_names if not name.upper().startswith("EOG-")]
    raw.pick(eeg_names)
    if len(raw.ch_names) != 22 or raw.info["sfreq"] != 250:
        raise ValueError(f"Unexpected BCI-IV-2a metadata for {path}: {len(raw.ch_names)} EEG channels")
    raw.filter(0.5, 100.0, verbose="ERROR")
    if session == "T":
        event_id = {str(code): code for code in range(769, 773)}
        events, _ = mne.events_from_annotations(raw, event_id=event_id, verbose="ERROR")
        _assert_non_overlapping_epochs(events, 1125, f"BCI-IV-2a subject {subject} session {session}")
        y = (events[:, -1] - 769).astype(np.int64)
    else:
        events, _ = mne.events_from_annotations(raw, event_id={"783": 783}, verbose="ERROR")
        label_path = raw_dir / f"A{subject:02d}E.mat"
        y = np.asarray(loadmat(label_path)["classlabel"]).reshape(-1).astype(np.int64) - 1
    if len(events) != 288 or len(y) != 288 or set(y.tolist()) != {0, 1, 2, 3}:
        raise ValueError(f"Expected 288 labeled four-class cues in {path}; got {len(events)} events and {len(y)} labels")
    _assert_non_overlapping_epochs(events, 1125, f"BCI-IV-2a subject {subject} session {session}")
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
    folds = _subject_folds(subjects, seed)
    split_dir = data_dir / "splits"
    if (_fold_cache_is_complete(output_dir, split_dir, "bciiv2a", seed, subjects)
            and (output_dir / "bciiv2a_T_to_E.pt").is_file()
            and (split_dir / "bciiv2a_T_to_E.json").is_file()):
        print(f"Using existing BCI IV 2a tensor and split caches in {output_dir}; raw processing skipped.", flush=True)
        return output_dir
    training = {subject: _read_bciiv2a_session(raw_dir, subject, "T") for subject in subjects}
    evaluation = {subject: _read_bciiv2a_session(raw_dir, subject, "E") for subject in subjects}
    output_dir.mkdir(parents=True, exist_ok=True)
    split_dir.mkdir(parents=True, exist_ok=True)
    for fold_id, fold in enumerate(folds):
        (split_dir / f"bciiv2a_{fold_id}.json").write_text(
            json.dumps({"dataset": "bciiv2a", "source_session": "T", "seed": seed, "fold": fold_id,
                        "cache_version": 2, **fold}, indent=2),
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
    parser.add_argument("--protocol", choices=("project", "wang_repo"), default="project")
    args = parser.parse_args()
    if args.dataset == "bciiv2a":
        if args.subjects is not None:
            parser.error("--subjects is currently supported only for EEGMMIDB")
        target = args.output_dir
        if target == Path("data/processed/eegmmidb"):
            target = Path("data/processed/bciiv2a")
        preprocess_bciiv2a(args.data_dir, target, args.seed)
    else:
        preprocess_eegmmidb(args.data_dir, args.output_dir, args.seed, args.subjects, args.protocol)


if __name__ == "__main__":
    main()
