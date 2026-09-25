"""Copy a Kaggle notebook export into the project with explanatory Markdown.

This edits notebook JSON and extracts existing figure outputs. It does not run
any EEG preprocessing, model training, or inference on the local computer.
"""

import base64
import json
import sys
from pathlib import Path


EXPLANATIONS = {
    0: ("Setup | Confirm the Kaggle runtime", "**Purpose:** Record the Python version and check that the EEG and machine-learning packages are available on Kaggle. **Expected output:** One availability line per package. This is an environment check; it does not process EEG data or train a model."),
    1: ("Phase A, pilot | Inspect EEGMMIDB source recordings", "**Purpose:** Fetch four EDF runs for subject 1 and inspect channels, sampling frequency, recording duration, and event annotations. **Expected output:** Four file names and representative metadata. This small source check establishes how the later full-cohort loader should interpret the data."),
    2: ("Phase A, pilot | Check four-class EEGMMIDB preprocessing", "**Purpose:** Filter one subject at 0.5–40 Hz; create 3-second left-hand, right-hand, eyes-open-rest, and feet-imagery windows; split trials; and fit normalization on training windows only. **Expected output:** A saved pilot cache, class counts, and a checked `(N, 64, 1, 480)` tensor. This is a preprocessing smoke test, not a subject-independent result."),
    3: ("Phase A, pilot | Visualize signal and label diagnostics", "**Purpose:** Plot subject-1 class counts, example C3 waveforms, and class-wise power spectra after preprocessing. **Expected output:** A 300 dpi diagnostic figure. These plots show input characteristics only and must not be cited as cross-validation performance."),
    4: ("Phase B, pilot | Smoke-test EEGNet-8,2", "**Purpose:** Instantiate EEGNet-8,2, check tensor output and parameter count, and train briefly on subject-1 windows. **Expected output:** Five training epochs, pilot accuracy and macro-F1, learning curves, and a pilot confusion matrix. The within-subject trial split checks the code path; it is not the Phase B headline accuracy."),
    5: ("Phase A | Acquire the selected EEGMMIDB cohort", "**Purpose:** Fetch the seven specified EDF runs for each of 105 eligible subjects from PhysioNet's public S3 mirror, reusing valid files. **Expected output:** Progress counts and a final `735/735` EDF validation. Only after that assertion can the complete cohort preprocessing proceed."),
    6: ("Phase A | Preprocess subjects and create five outer folds", "**Purpose:** Filter and epoch every selected subject into four classes, validate finite `(N, 64, 1, 480)` tensors, and create five disjoint held-out-subject splits. Fit each fold's normalization statistics on its training subjects. **Expected output:** Subject caches, split manifests, five fold caches, and per-fold shape and class-count checks."),
    7: ("Phase B | Define EEGNet-8,2 and inspect fold 0", "**Purpose:** Define the exact-arithmetic 3,092-parameter baseline, load the first subject-disjoint fold, and confirm that Kaggle sees a CUDA device. **Expected output:** Model/device information and 84 training versus 21 held-out subjects. This prepares a baseline check before the full sweep."),
    8: ("Phase B | Train and evaluate the fold-0 baseline", "**Purpose:** Reserve an inner group of training subjects to select the epoch by macro-F1, then evaluate the outer held-out group once. **Expected output:** Inner learning progress, selected epoch, outer accuracy, and outer macro-F1. Outer labels do not influence checkpoint choice."),
    9: ("Phase B | Save fold-0 model and diagnostics", "**Purpose:** Save the fold-0 checkpoint and learning history, report class-level metrics, and plot the inner learning curve with the outer confusion matrix. **Expected output:** A checkpoint, CSV history, and 300 dpi fold-0 figure. This single fold is a diagnostic, not the five-fold estimate."),
    10: ("Phase B | Train five subject-independent EEGMMIDB folds", "**Purpose:** Repeat inner epoch selection and outer held-out evaluation for all five subject groups, then aggregate predictions. **Expected output:** Five fold metrics and checkpoints plus one out-of-fold prediction for every processed window. This evaluates the exact-arithmetic EEGNet-8,2 baseline only."),
    12: ("Phase B | Summarize the five-fold baseline", "**Purpose:** Compute fold accuracy and macro-F1, pooled out-of-fold scores, per-class metrics, a subject-cluster bootstrap interval, and a normalized confusion matrix. **Expected output:** Metrics tables, subject-level CSV, and a 300 dpi paper figure. The observed pooled accuracy is 62.30%, below the prompt's 65.07% reference tolerance; this is not an approximate-MAC result."),
    15: ("Phase A + B audit | Compare dataset scale without inventing scores", "**Purpose:** Plot published cohort and sensor specifications alongside the analysis-window sizes used in this study. **Expected output:** A 300 dpi comparison of 105 selected EEGMMIDB versus 9 BCI IV 2a subjects, 64 versus 22 EEG sensors, and the derived input sample counts. BCI IV 2a was not trained here, so the figure is descriptive rather than a performance or hardware-energy comparison. EvoApprox8b is a multiplier library, not an EEG dataset. Sources: [PhysioNet EEGMMIDB](https://physionet.org/content/eegmmidb/1.0.0/), [BCI IV 2a downloads](https://bbci.de/competition/iv/download/), [BNCI description](https://bnci-horizon-2020.eu/database/data-sets), and [EvoApprox8b](https://github.com/ehw-fit/evoapprox8b)."),
}

PARALLEL_EDF_SOURCE = '''# Step 6 — acquire and validate all 735 EDF files from the official PhysioNet S3 mirror.
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import boto3, time
from botocore import UNSIGNED
from botocore.config import Config
ROOT = Path('/kaggle/working/eeg-axmac')
EDF_ROOT = ROOT/'data/raw/MNE-eegbci-data/files/eegmmidb/1.0.0'
EDF_ROOT.mkdir(parents=True, exist_ok=True)
EXCLUDED = {88,89,92,100}
SUBJECTS = [s for s in range(1,110) if s not in EXCLUDED]
RUNS = [1,4,6,8,10,12,14]
EXPECTED = len(SUBJECTS)*len(RUNS)
def valid_edf(path):
    return path.exists() and path.stat().st_size > 100_000 and path.open('rb').read(1) == b'0'
s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED, max_pool_connections=20,
    connect_timeout=20, read_timeout=120, retries={'max_attempts':5}))
jobs = [(subject,run) for subject in SUBJECTS for run in RUNS]
print(f'Official PhysioNet S3 mirror: {EXPECTED} requested; '
      f'{sum(valid_edf(EDF_ROOT/f"S{s:03d}"/f"S{s:03d}R{r:02d}.edf") for s,r in jobs)} already valid.', flush=True)
def fetch_edf(subject,run):
    folder = EDF_ROOT/f'S{subject:03d}'
    folder.mkdir(parents=True, exist_ok=True)
    target = folder/f'S{subject:03d}R{run:02d}.edf'
    if valid_edf(target): return target
    partial = target.with_suffix('.edf.part')
    key = f'eegmmidb/1.0.0/S{subject:03d}/S{subject:03d}R{run:02d}.edf'
    try:
        s3.download_file('physionet-open', key, str(partial))
        if not valid_edf(partial): raise IOError(f'Invalid EDF downloaded: {key}')
        partial.replace(target)
    except Exception:
        partial.unlink(missing_ok=True)
        raise
    return target
started = time.time()
with ThreadPoolExecutor(max_workers=16) as pool:
    futures = [pool.submit(fetch_edf,s,r) for s,r in jobs]
    for done,future in enumerate(as_completed(futures),1):
        future.result()
        if done%100==0 or done==EXPECTED:
            print(f'Validated {done}/{EXPECTED} EDFs; elapsed={time.time()-started:.1f}s',flush=True)
valid_count = sum(valid_edf(EDF_ROOT/f'S{s:03d}'/f'S{s:03d}R{r:02d}.edf') for s,r in jobs)
print(f'Acquisition validation: {valid_count}/{EXPECTED} valid EDFs',flush=True)
assert valid_count == EXPECTED
'''


def main() -> None:
    src, dst, figure_dir = map(Path, sys.argv[1:4])
    notebook = json.loads(src.read_text(encoding="utf-8"))
    cleaned = []
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "markdown" and "## Accuracy audit" in "".join(cell.get("source", [])):
            protocol_note = (
                "\n**Cohort and window caveat.** The supplied execution prompt excludes subjects "
                "88, 89, 92, and 100, which this notebook follows. The authors' released "
                "loader instead excludes 88, 92, 100, and 104; it also caps imagery trials "
                "at seven per class per run and creates 21 rest windows per subject. "
                "This notebook's cached cohort has 9,184 windows, so the published 65.07% "
                "is contextual rather than a matched external test. "
                "[Authors' loader](https://github.com/MHersche/eegnet-based-embedded-bci/blob/master/get_data.py).\n"
            )
            if "**Cohort and window caveat.**" not in "".join(cell.get("source", [])):
                cell["source"] = list(cell.get("source", [])) + [protocol_note]
        if cell["cell_type"] == "code":
            code = "".join(cell.get("source", []))
            if not code.strip():
                continue
            if code.startswith("# Step 6") and "PhysioNet" in code:
                cell["source"] = PARALLEL_EDF_SOURCE.splitlines(keepends=True)
                code = PARALLEL_EDF_SOURCE
            title, description = EXPLANATIONS.get(
                index,
                (
                    "Kaggle experiment | " + code.splitlines()[0].lstrip("# ").strip()[:80],
                    "**Purpose:** Execute the documented experiment on Kaggle and inspect its recorded outputs. "
                    "Interpret results only after the cell finishes successfully.",
                ),
            )
            explanation = {
                "cell_type": "markdown",
                "metadata": {},
                "source": [f"### {title}\n", "\n", description + "\n"],
            }
            if cleaned and cleaned[-1]["cell_type"] == "markdown" and index in (12, 15):
                cleaned[-1] = explanation
            elif not cleaned or cleaned[-1]["cell_type"] != "markdown":
                cleaned.append(explanation)
        cleaned.append(cell)
    notebook["cells"] = cleaned
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

    # Paper figures already generated on Kaggle are copied from notebook output.
    figure_dir.mkdir(parents=True, exist_ok=True)
    wanted = {
        "eegmmidb_5fold_oof_summary.png": "Summarize the five-fold baseline",
        "dataset_scale_comparison.png": "Compare dataset scale without inventing scores",
        "paper_aligned_accuracy_comparison.png": "Accuracy comparison and feature-importance diagnostics",
        "anova_bandpower_and_input_attribution.png": "Train-only ANOVA bandpower map",
        "paper_aligned_confusion_matrix.png": "Class-level error analysis for the paper-aligned variant",
        "three_model_eegmmidb_comparison.png": "Three-model EEGMMIDB comparison",
    }
    for filename, section in wanted.items():
        for position, cell in enumerate(cleaned):
            if cell["cell_type"] != "markdown" or section not in "".join(cell.get("source", [])):
                continue
            for later in cleaned[position + 1 :]:
                if later["cell_type"] == "code":
                    for output in later.get("outputs", []):
                        png = output.get("data", {}).get("image/png")
                        if png:
                            (figure_dir / filename).write_bytes(base64.b64decode("".join(png)))
                    break
            break

    nonempty_code = [i for i, cell in enumerate(cleaned) if cell["cell_type"] == "code" and "".join(cell.get("source", [])).strip()]
    missing = [i for i in nonempty_code if i == 0 or cleaned[i-1]["cell_type"] != "markdown"]
    print(f"Saved {dst}; cells={len(cleaned)}, code={len(nonempty_code)}, code without preceding Markdown={len(missing)}")
    for filename in wanted:
        path = figure_dir / filename
        print(f"{filename}: {'extracted' if path.exists() else 'missing from export'}")


if __name__ == "__main__":
    main()
