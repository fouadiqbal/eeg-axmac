"""Download the run files required for EEGMMIDB four-class experiments."""

from __future__ import annotations

import argparse
from pathlib import Path
from urllib.request import urlretrieve
from zipfile import ZipFile

EXCLUDED_SUBJECTS = {88, 89, 92, 100}
EEGMMIDB_RUNS = (1, 4, 6, 8, 10, 12, 14)
BCIIV2A_GDF_URL = "https://www.bbci.de/competition/download/competition_iv/BCICIV_2a_gdf.zip"
BCIIV2A_LABEL_URL = "https://www.bbci.de/competition/iv/results/ds2a/true_labels.zip"


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


def _extract_flat_zip(archive: Path, destination: Path, suffix: str) -> list[Path]:
    """Extract only expected data files by basename, never archive paths."""
    extracted: list[Path] = []
    with ZipFile(archive) as zipped:
        for member in zipped.infolist():
            name = Path(member.filename).name
            if not name.lower().endswith(suffix) or member.is_dir():
                continue
            target = destination / name
            if not target.exists() or target.stat().st_size != member.file_size:
                with zipped.open(member) as source, target.open("wb") as output:
                    while chunk := source.read(1024 * 1024):
                        output.write(chunk)
            extracted.append(target)
    return extracted


def download_bciiv2a(data_dir: Path) -> Path:
    """Fetch official GDF sessions and the organizer's evaluation labels.

    EEG recordings are used only on the requested Kaggle runtime. Keeping
    evaluation labels in a separate archive prevents accidental train leakage.
    """
    destination = data_dir / "bciiv2a"
    destination.mkdir(parents=True, exist_ok=True)
    archives = (
        ("BCICIV_2a_gdf.zip", BCIIV2A_GDF_URL, ".gdf"),
        ("true_labels.zip", BCIIV2A_LABEL_URL, ".mat"),
    )
    for filename, url, suffix in archives:
        archive = destination / filename
        if not archive.exists():
            print(f"Downloading official BCI-IV-2a archive: {url}", flush=True)
            partial = archive.with_suffix(archive.suffix + ".part")
            urlretrieve(url, partial)
            partial.replace(archive)
        files = _extract_flat_zip(archive, destination, suffix)
        print(f"{filename}: {len(files)} {suffix} files available", flush=True)
    training = sorted(destination.glob("A??T.gdf"))
    evaluation = sorted(destination.glob("A??E.gdf"))
    labels = sorted(destination.glob("A??E.mat"))
    if not (len(training) == len(evaluation) == len(labels) == 9):
        raise RuntimeError(
            f"Expected nine training GDFs, nine evaluation GDFs and nine label MATs; "
            f"found {len(training)}, {len(evaluation)}, {len(labels)}"
        )
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["eegmmidb", "bciiv2a"], default="eegmmidb")
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--subjects", type=int, nargs="*", help="Optional subject subset for smoke runs.")
    args = parser.parse_args()
    if args.dataset == "bciiv2a":
        if args.subjects is not None:
            parser.error("--subjects is currently supported only for EEGMMIDB")
        location = download_bciiv2a(args.data_dir)
        print(f"BCI-IV-2a acquisition complete: {location}")
    else:
        location = download_eegmmidb(args.data_dir, args.subjects)
        n_subjects = len(args.subjects) if args.subjects is not None else len(eligible_subjects())
        print(f"EEGMMIDB acquisition complete: {n_subjects} subjects, {n_subjects * len(EEGMMIDB_RUNS)} run files; cache: {location}")


if __name__ == "__main__":
    main()
