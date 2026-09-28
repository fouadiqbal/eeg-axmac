"""Download the run files required for EEGMMIDB four-class experiments."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.request import urlretrieve
from zipfile import ZipFile

EXCLUDED_SUBJECTS = {88, 89, 92, 100}
WANG_EXCLUDED_SUBJECTS = {88, 92, 100, 104}
EEGMMIDB_RUNS = (1, 4, 6, 8, 10, 12, 14)
BCIIV2A_GDF_URL = "https://www.bbci.de/competition/download/competition_iv/BCICIV_2a_gdf.zip"
BCIIV2A_LABEL_URL = "https://www.bbci.de/competition/iv/results/ds2a/true_labels.zip"


def eligible_subjects() -> list[int]:
    """Return the 105 subjects retained by the specified reference protocol."""
    return [subject for subject in range(1, 110) if subject not in EXCLUDED_SUBJECTS]


def wang_eligible_subjects() -> list[int]:
    """Subject set in MHersche et al.'s released ``get_data.py`` loader."""
    return [subject for subject in range(1, 110) if subject not in WANG_EXCLUDED_SUBJECTS]


def download_eegmmidb(data_dir: Path, subjects: list[int] | None = None,
                     workers: int = 16, protocol: str = "project") -> Path:
    """Fetch the selected EDFs from PhysioNet's official public S3 mirror.

    Existing EDFs are checked and reused. Independent files download with a
    bounded worker pool, which reduced the observed Kaggle reacquisition time
    for 735 files from a long serial run to about 12 seconds in this session.
    """
    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config

    if workers < 1 or workers > 32:
        raise ValueError("workers must be between 1 and 32")
    if protocol not in ("project", "wang_repo"):
        raise ValueError("protocol must be 'project' or 'wang_repo'")
    allowed = set(eligible_subjects() if protocol == "project" else wang_eligible_subjects())
    selected = sorted(allowed) if subjects is None else sorted(set(subjects))
    invalid = sorted(set(selected) - allowed)
    if invalid:
        exclusions = sorted(EXCLUDED_SUBJECTS if protocol == "project" else WANG_EXCLUDED_SUBJECTS)
        raise ValueError(f"Subjects must be between 1 and 109 and exclude {exclusions}: {invalid}")
    if not selected:
        raise ValueError("At least one subject must be selected.")

    root = data_dir / "MNE-eegbci-data" / "files" / "eegmmidb" / "1.0.0"
    root.mkdir(parents=True, exist_ok=True)
    client = boto3.client("s3", config=Config(signature_version=UNSIGNED, max_pool_connections=workers + 4,
                                             connect_timeout=20, read_timeout=120,
                                             retries={"max_attempts": 5}))

    def valid_edf(path: Path) -> bool:
        if not path.exists() or path.stat().st_size <= 100_000:
            return False
        with path.open("rb") as stream:
            return stream.read(1) == b"0"

    def fetch(subject: int, run: int) -> Path:
        folder = root / f"S{subject:03d}"
        folder.mkdir(parents=True, exist_ok=True)
        target = folder / f"S{subject:03d}R{run:02d}.edf"
        if valid_edf(target):
            return target
        key = f"eegmmidb/1.0.0/S{subject:03d}/S{subject:03d}R{run:02d}.edf"
        partial = target.with_suffix(".edf.part")
        try:
            client.download_file("physionet-open", key, str(partial))
            if not valid_edf(partial):
                raise IOError(f"Downloaded EDF failed validation: {key}")
            partial.replace(target)
        except Exception:
            partial.unlink(missing_ok=True)
            raise
        return target

    total = len(selected) * len(EEGMMIDB_RUNS)
    jobs = [(subject, run) for subject in selected for run in EEGMMIDB_RUNS]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(fetch, subject, run) for subject, run in jobs]
        for index, future in enumerate(as_completed(futures), 1):
            future.result()
            if index % 50 == 0 or index == total:
                print(f"Verified EDF files: {index}/{total}", flush=True)
    if sum(valid_edf(root / f"S{s:03d}" / f"S{s:03d}R{r:02d}.edf")
           for s, r in jobs) != total:
        raise RuntimeError("Final EDF validation count did not match requested cohort")
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
    parser.add_argument("--workers", type=int, default=16, help="Parallel EDF transfers for EEGMMIDB (1-32).")
    parser.add_argument("--protocol", choices=("project", "wang_repo"), default="project")
    args = parser.parse_args()
    if args.dataset == "bciiv2a":
        if args.subjects is not None:
            parser.error("--subjects is currently supported only for EEGMMIDB")
        location = download_bciiv2a(args.data_dir)
        print(f"BCI-IV-2a acquisition complete: {location}")
    else:
        location = download_eegmmidb(args.data_dir, args.subjects, args.workers, args.protocol)
        n_subjects = len(args.subjects) if args.subjects is not None else len(eligible_subjects() if args.protocol == "project" else wang_eligible_subjects())
        print(f"EEGMMIDB acquisition complete: {n_subjects} subjects, {n_subjects * len(EEGMMIDB_RUNS)} run files; cache: {location}")


if __name__ == "__main__":
    main()
