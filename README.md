# EEG-CNN Approximate-MAC Research Pipeline

Reproducible EEG motor-imagery baselines and approximate-multiplier sensitivity
experiments. EEGMMIDB has a measured five-fold exact-arithmetic baseline and
one additional EEGNet training variant. BCI-IV-2a and approximate-MAC
sensitivity experiments remain unverified.
The five-fold EEGNet-4,2 run is also complete in Kaggle Version 2. Further
experiments are paused at the user's request; the saved version contains the
completed notebook output and fold checkpoint files.

The executed research notebook is
[`notebooks/eeg_cnn_approx_mac_kaggle_run.ipynb`](notebooks/eeg_cnn_approx_mac_kaggle_run.ipynb).
Its training and plots run in [Kaggle](https://www.kaggle.com/code/fouadiqbal/eeg-cnn-approximate-mac-research-phase-a/edit)
with a T4 GPU; the local notebook is an exported copy for inspection in VS Code.

## Experimental status

- EEGMMIDB: 735 EDF files for 105 prompt-selected subjects, 9,184 four-class
  windows, and five disjoint held-out-subject folds. The original 64-tap EEGNet
  baseline reached 62.30% pooled accuracy and 62.36% macro-F1.
- A fixed 100-epoch, 128-tap/8-pool, 20%-dropout variant completed the same
  five held-out-subject folds on Kaggle T4. Its pooled accuracy is **65.30%**
  and macro-F1 is **65.42%** over 9,184 windows. Fold accuracies are 64.76%,
  67.16%, 65.83%, 64.55%, and 64.20%. The pooled accuracy is 3.00 percentage
  points above the earlier baseline; the subject-cluster bootstrap 95% interval
  for the variant is 62.83%–67.88%. This variant changes architecture and
  training settings together, so the gain cannot be assigned to one setting.
- Train-only ANOVA bandpower and held-out gradient × input attribution have
  executed as diagnostics. Neither method improves classification by itself.
  The resulting figures, five-fold comparison, and pooled confusion matrix are
  in [`results/figures`](results/figures).
- EEGNet-4,2 completed five held-out-subject folds with accuracies 64.44%,
  66.50%, 62.11%, 63.40%, and 62.63%. Its window-weighted accuracy is
  **63.82%** over 9,184 windows. The pooled macro-F1 and three-model plot
  have not yet been computed, so this is a provisional model comparison.
- The published 65.07% four-class figure uses a different subject exclusion and
  trial extraction protocol. The supplied prompt excludes 88, 89, 92, and 100;
  the [authors' loader](https://github.com/MHersche/eegnet-based-embedded-bci/blob/master/get_data.py)
  excludes 88, 92, 100, and 104 and uses 21 windows per class per subject.

EEGMMIDB is the only EEG dataset trained so far. BCI Competition IV 2a is
compared descriptively in the notebook but has no measured accuracy here.
EvoApprox8b is an approximate arithmetic library, not an EEG dataset, and no
multiplier has been applied to a trained model yet. The published 65.07%
figure is contextual because the cohort and window construction differ.

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
