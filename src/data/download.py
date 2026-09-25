"""Download the run files required for EEGMMIDB four-class experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

EXCLUDED_SUBJECTS = {88, 89, 92, 100}
EEGMMIDB_RUNS = (1, 4, 6, 8, 10, 12, 14)


def eligible_subjects() -> list[int]:
    """Return the 105 subjects retained by the specified reference protocol."""
    return [subject for subject in range(1, 110) if subject not in EXCLUDED_SUBJECTS]


def download_eegmmidb(data_dir: Path, subjects: list[int] | None = None) -> Path:
    """Fetch baseline and imagery EDFs using MNE's resumable PhysioNet cache.

    The complete default acquisition includes seven runs per eligible subject.
    Pass an explicit subject subset for a smoke run. MNE reports per-file download
    progress and skips cached files on subsequent calls.
    """
    try:
        import mne
    except ImportError as exc:
        raise RuntimeError("MNE is required. Install dependencies with `pip install -r requirements.txt`.") from exc

    selected = eligible_subjects() if subjects is None else sorted(set(subjects))
    invalid = sorted(set(selected) - set(eligible_subjects()))
    if invalid:
        raise ValueError(f"Subjects must be between 1 and 109 and exclude {sorted(EXCLUDED_SUBJECTS)}: {invalid}")
    if not selected:
        raise ValueError("At least one subject must be selected.")

    data_dir.mkdir(parents=True, exist_ok=True)
    downloaded: list[str] = []
    total = len(selected) * len(EEGMMIDB_RUNS)
    for index, subject in enumerate(selected, start=1):
        print(f"\nSubject {subject:03d} ({index}/{len(selected)}): requesting runs {EEGMMIDB_RUNS}", flush=True)
        subject_files = mne.datasets.eegbci.load_data(
            subjects=[subject], runs=list(EEGMMIDB_RUNS), path=str(data_dir), update_path=True
        )
        downloaded.extend(str(path) for path in subject_files)
        print(f"Progress: {len(downloaded)}/{total} run files available", flush=True)
    if len(downloaded) != total:
        raise RuntimeError(f"Expected {total} run files, MNE returned {len(downloaded)}")
    return data_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["eegmmidb"], default="eegmmidb")
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--subjects", type=int, nargs="*", help="Optional subject subset for smoke runs.")
    args = parser.parse_args()
    location = download_eegmmidb(args.data_dir, args.subjects)
    n_subjects = len(args.subjects) if args.subjects is not None else len(eligible_subjects())
    print(f"EEGMMIDB acquisition complete: {n_subjects} subjects, {n_subjects * len(EEGMMIDB_RUNS)} run files; cache: {location}")


if __name__ == "__main__":
    main()
