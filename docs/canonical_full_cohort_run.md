# Canonical EEGMMIDB full-cohort run

The dedicated [Kaggle notebook](https://www.kaggle.com/code/fouadiqbal/eegmmidb-wang-full-cohort-canonical?scriptVersionId=353070628) is the canonical Wang-compatible baseline. Version 353070628 is a committed GPU T4 x2 run. The older Phase A/B notebook contains historical exploratory results and is not the source of this baseline score.

## Protocol and verified acquisition

- Participants: IDs 1–109 excluding 88, 92, 100, and 104, matching the released loader's cohort exclusions.
- Runs: 1, 4, 6, 8, 10, 12, 14; 735 EDF files validated, 1,749,996,960 total bytes.
- Windows: 84 per subject; 8,820 total; 2,205 in each of four classes.
- Splits: five subject-disjoint folds, each with 84 training and 21 held-out participants.
- Model: Wang-compatible EEGNet-8,2; spatial L2 max norm 1.0; classifier L2 max norm 0.25; batch size 16; 100 epochs; fixed learning-rate schedule.
- Documented departure: the released extra rest window uses an unseeded Python RNG. This run fixes its seed at 7 for repeatability. Subject-disjoint CV also deliberately differs from the released window-wise KFold because the latter mixes participants.

## Persistence

Fold checkpoints (every 10 epochs), final models, predictions, and report files are written to `/kaggle/working/canonical_full_cohort`. All 105 subject caches are saved under `/kaggle/working/eeg_axmac_wang_full_cohort/subject_cache`. The raw EDF files remain under `/kaggle/temp/eeg_axmac_wang_full_cohort/raw`; a later session can reuse the subject caches without reacquiring the EDFs. The AXM sweep runner accepts this saved `--cache-dir` directly.

## Outcome

Kaggle version 353070628 completed successfully in 2,260.2 seconds on GPU T4 x2. Its saved output lists 135 files, including all five final fold checkpoints/models, predictions, fold metrics, two figures, the summary and report, and 105 subject caches.

| Held-out fold | Accuracy | Macro-F1 | Windows |
|---:|---:|---:|---:|
| 0 | 60.71% | 60.76% | 1,764 |
| 1 | 69.56% | 69.51% | 1,764 |
| 2 | 65.36% | 65.43% | 1,764 |
| 3 | 63.32% | 63.44% | 1,764 |
| 4 | 66.55% | 66.49% | 1,764 |
| **Mean ± sample SD** | **65.10 ± 3.33%** | **65.13 ± 3.28%** | **8,820 total** |

The mean differs from the 65.07% published reference by +0.03 percentage points. This meets the specified ±3 pp acceptance band and is tagged `verified_within_band` in the Kaggle report. This is a protocol-aligned comparison, not an exact reproduction: subject-disjoint validation and a seeded extra rest window are documented departures from the released code.

The full-cohort fold-0 approximate-MAC sweep has since completed in the separate [Kaggle sensitivity notebook Version 6](https://www.kaggle.com/code/fouadiqbal/eegmmidb-canonical-approximate-mac-sensitivity?scriptVersionId=353306324). It measured 50 multipliers in four layer types on fold 0; it did not retrain the baseline and did not sweep all five folds. See [`eegmmidb_axm_sensitivity_results.md`](eegmmidb_axm_sensitivity_results.md) and the task-by-task [`targeted_results_audit.md`](targeted_results_audit.md).
