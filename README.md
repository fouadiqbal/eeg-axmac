# EEG-CNN Approximate-MAC Research Pipeline

Reproducible EEG motor-imagery baselines and approximate-multiplier sensitivity
experiments. EEGMMIDB is the first dataset with a measured five-fold baseline.
BCI-IV-2a and the approximate-MAC sensitivity sweep remain unverified.

The executed research notebook is
[`notebooks/eeg_cnn_approx_mac_kaggle_run.ipynb`](notebooks/eeg_cnn_approx_mac_kaggle_run.ipynb).
Its training and plots run in [Kaggle](https://www.kaggle.com/code/fouadiqbal/eeg-cnn-approximate-mac-research-phase-a/edit)
with a T4 GPU; the local notebook is an exported copy for inspection in VS Code.

## Experimental status

- EEGMMIDB: 735 EDF files for 105 prompt-selected subjects, 9,184 four-class
  windows, and five disjoint held-out-subject folds. The original 64-tap EEGNet
  baseline reached 62.30% pooled accuracy and 62.36% macro-F1.
- A fixed 100-epoch, 128-tap/8-pool variant is being evaluated on the same five
  folds. The first two completed held-out folds reached 65.52% and 66.88%; a
  five-fold result is not yet established.
- ANOVA bandpower and input-attribution cells are included as diagnostics but
  have not yet executed. Neither method improves classification by itself.
- The published 65.07% four-class figure uses a different subject exclusion and
  trial extraction protocol. The supplied prompt excludes 88, 89, 92, and 100;
  the [authors' loader](https://github.com/MHersche/eegnet-based-embedded-bci/blob/master/get_data.py)
  excludes 88, 92, 100, and 104 and uses 21 windows per class per subject.

The standalone `src/models/eegnet.py` provides configurable 64/4 and 128/8
temporal-kernel/first-pool settings. It has not been run on the local PC.

## Setup

Use Python 3.10–3.12 with a CPU or CUDA PyTorch build appropriate for your
machine, then install `requirements.txt`.

## Phase A: EEGMMIDB

```powershell
python -m src.data.download --dataset eegmmidb --data-dir data/raw
python -m src.data.preprocess --dataset eegmmidb --data-dir data --output-dir data/processed/eegmmidb --seed 42
pytest tests/test_data_shapes.py
```

The preprocessing command caches five subject-level folds under
`data/processed/eegmmidb/` and their split manifests under `data/splits/`.
Raw and processed data are excluded from git.

## Phase B

Phase B commands will be documented here after the model and approximate-MAC
steps are implemented.
