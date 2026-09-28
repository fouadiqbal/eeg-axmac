# Methods audit for the EEG approximate-MAC pipeline

This audit separates methods supported by the supplied papers from results
actually measured in this repository. The PDFs remain with the researcher;
their file names are given so a reader can locate the source. Any accuracy
claim in the notebook must come from its recorded held-out predictions.

| Source supplied by the researcher | Relevant method | Decision for this pipeline |
| --- | --- | --- |
| `2004.00077v3.pdf` (Wang et al., 2020) | Subject-level five-fold EEGMMIDB evaluation, 3 s windows, compact EEGNet, epoch-wise learning-rate schedule | Keep five disjoint outer subject folds. Treat its 65.07% as a within-band reference because our subject exclusions/window construction differ from its released loader; exact training duration behind the reported score is not verified from the current public script default. |
| `hernandez-ruiz2021.pdf` | EEGNet-4,2 on PhysioNet EEGMMIDB, ds=1/ds=2, global and SS-TL settings; LeakyReLU slopes 0.6/0.5/0.4, no batch normalization/dropout | Use the corrected EEGMMIDB-only target matrix below. These figures are context until cohort, preprocessing, and transfer protocol are matched. |
| `2006.00622v1.pdf` (Ingolfsson et al., EEG-TCNet) | BCI IV 2a 4-class EEG-TCNet: 77.35% fixed hyperparameters and 83.84% with subject-specific hyperparameter search | BCI IV 2a benchmark reproduction is deferred until EEG-TCNet and its train/test protocol are implemented; do not compare EEGNet outcomes to these scores as matched baselines. |
| `SmartEEG_An_End-to-End_Framework_for_the_Analysis_and_Classification_of_EEG_signals.pdf` | Patient-level validation and augmentation choices in seizure EEG | The patient-split lesson supports subject-disjoint folds. Its seizure augmentation outcomes cannot be transferred as motor-imagery accuracy claims. Any overlapping-window augmentation must stay inside training subjects. |
| `An_End-to-End_Automated_Pipeline_for_EEG_Classification_on_TinyML_Platforms_From_Signal_to_On-Device_Inference.pdf` | Reaction-Diffusion Transform and INT8 TinyML deployment on seizure/emotion datasets | RDT is a candidate deployment ablation, not a demonstrated motor-imagery improvement. Its preprocessing cost and resulting EEGNet accuracy would need measurement. |
| `2408.12836v1.pdf` (architectural error metric) | Circuit error matrix, data-weighted mean error (Eq. 4), architecture-dependent AME (Eq. 24) | Compute ER/MRED/MAE/ME once per circuit. Do not call circuit mean error AME: full AME requires layer distributions and downstream propagation and cannot be reused across all layers. |
| `EvoApprox8bnbspnbspLibrary_of_Approximate_Adders_and_Multipliers_for_Circuit_Design_and_Benchmarking_of_Approximation_Methods.pdf` and [official archived library](https://github.com/ehw-fit/evoapprox8b) | Pareto collection of unsigned eight-bit multiplier circuits with C models and circuit metrics | Use pinned official C models to generate exhaustive 256×256 `uint16` LUTs on Kaggle. Symmetric signed INT8 inference adds sign-magnitude logic, whose hardware cost is not covered by the unsigned circuit's published PPA. |

## Preprocessing decisions to verify experimentally

1. Keep the prompt's 0.5–40 Hz EEGMMIDB and 0.5–100 Hz BCI IV 2a
   baselines fixed while comparing models. Changing filters, epoch length,
   channel count, and architecture together would make a gain uninterpretable.
2. Fit z-score statistics, INT8 calibration, and any ANOVA-based channel
   selection on each outer fold's training subjects only. Use inner training
   subjects for selecting hyperparameters. The held-out fold is scored once.
3. Consider mu/beta-focused filtering, artifact handling, a shorter post-cue
   window, and anti-aliased ds=2 as explicit ablations. Quantify both accuracy
   and operation/input-memory changes. The current ANOVA heatmap is a
   diagnostic, not evidence that channel pruning improves accuracy.
4. BCI IV 2a training (`T`) and evaluation (`E`) sessions are kept separate.
   Five-fold subject CV on `T` and `T`→`E` evaluation answer different
   generalization questions; they must have separate result tables.
5. SHAP or gradient attribution explains a trained model but does not update
   its weights or directly improve accuracy. Select a smaller channel set only
   if a train-only nested comparison justifies it.

No measured hardware area, power, energy, or latency result exists yet.
Circuit catalog figures cannot be attributed to a complete signed EEG
accelerator without synthesis and the sign/control/buffering overhead.


## Catch-up audit addendum (2026-09-26)

### Leakage verification

- **Global subject splits:** Passed by `_subject_folds` source inspection and
  `tests/test_no_leakage.py`; EEGMMIDB and BCI IV 2a use mutually exclusive
  subject sets in each outer fold. Existing cache tensors/manifests are absent
  from this local checkout, so Kaggle fold artifacts were not independently
  reopened for this audit.
- **Normalization:** Passed by source trace and a synthetic isolation test.
  Outer-fold means and standard deviations use only that fold's training
  subjects (and BCI T-to-E uses session T only). Inner validation carved out
  during EEGMMIDB epoch selection may still be included in the outer-training
  normalization; this does not contaminate the outer held-out fold, but is not
  a fully nested preprocessing fit.
- **Epoch overlap:** Explicit guards now verify cue spacing is at least the
  epoch length for EEGMMIDB task cues and BCI T/E cues. EEGMMIDB rest uses
  contiguous, non-overlapping 480-sample windows. Subjects are held out as
  whole groups. Raw interval manifests are not stored, so this is a source-level
  and runtime-construction guarantee, not a separate interval audit of old
  Kaggle caches.
- **SS-TL initialization:** The training implementation requires checkpoint fold
  and training-subject provenance to match the global fold cache, then asserts
  each target participant was excluded. Four-way stratified subject splits are
  checked for disjoint, exhaustive coverage. At the time of this original
  addendum, SS-TL scores had not yet been run; see the real-data verification
  section below for the subsequent five-subject end-to-end run.
- The current leakage suite reported **6 passed, 0 skipped** at the time of
  this original addendum. Python source compilation passed. No model training
  had run on the PC at that point; see the real-data verification below.

### Protocol correction and remaining work

The supplied Hernandez-Ruiz et al. study reports its 2/3/4-class ds=1 and ds=2
results on PhysioNet EEGMMIDB; it is not a source for the earlier BCI IV 2a
target matrix. The target figures should be used only for the dataset/protocol named in the
corrected table. Training, SS-TL, and sensitivity-sweep code now accept the
EEGMMIDB class/model variants, and preprocessing caches both ds=1 and ds=2
inputs. These configs have not yet been executed; BCI IV 2a matched reproduction
remains deferred pending EEG-TCNet.


## Corrected reference target matrix (2026-09-26)

| Dataset | Model/protocol | Classes | Published accuracy | Use in this project |
|---|---|---:|---:|---|
| PhysioNet EEGMMIDB | Wang et al. standard EEGNet, global | 2 | 82.43% | Reference only; project reproduction candidate |
| PhysioNet EEGMMIDB | Wang et al. standard EEGNet, global | 3 | 75.07% | Reference only; project reproduction candidate |
| PhysioNet EEGMMIDB | Wang et al. standard EEGNet, global | 4 | 65.07% | Primary project reference; 3 percentage-point acceptance band |
| PhysioNet EEGMMIDB | Hernandez-Ruiz EEGNet-4,2, ds=1, global / SS-TL | 2 | 83.15% / 87.46% | Secondary model reference |
| PhysioNet EEGMMIDB | Hernandez-Ruiz EEGNet-4,2, ds=1, global / SS-TL | 3 | 75.74% / 83.26% | Secondary model reference |
| PhysioNet EEGMMIDB | Hernandez-Ruiz EEGNet-4,2, ds=1, global / SS-TL | 4 | 65.75% / 74.31% | Secondary model reference |
| PhysioNet EEGMMIDB | Hernandez-Ruiz EEGNet-4,2, ds=2, global / SS-TL | 2 | 82.52% / 93.10% | Secondary model reference |
| PhysioNet EEGMMIDB | Hernandez-Ruiz EEGNet-4,2, ds=2, global / SS-TL | 3 | 75.34% / 93.21% | Secondary model reference |
| PhysioNet EEGMMIDB | Hernandez-Ruiz EEGNet-4,2, ds=2, global / SS-TL | 4 | 65.56% / 89.23% | Secondary model reference |
| BCI Competition IV 2a | Ingolfsson et al. EEG-TCNet, fixed hyperparameters | 4 | 77.35% | BCI reproduction deferred; protocol/data artifacts differ from current subject-CV |
| BCI Competition IV 2a | Ingolfsson et al. EEG-TCNet, subject-specific hyperparameter search | 4 | 83.84% | Not a simple SS-TL target; requires faithful per-subject CV/grid-search protocol |

The corrected Hernandez-Ruiz figures are not BCI IV 2a targets. EEG-TCNet figures
are BCI IV 2a results but should not be used to score the current EEGNet model.


## Baseline reconciliation (2026-09-26)

The historical EEGNet-8,2 project-protocol result (62.30% accuracy,
62.36% macro-F1; pooled OOF) remains in
`results/baseline_accuracy_report.csv` with status `superseded`. It was within
2.77 points of Wang et al.'s 65.07% reference under the project's earlier
acceptance rule, but its exclusions, epoch construction, architecture, and
training procedure differ from Wang. Keep it for audit history only.

The Wang-compatible EEGMMIDB path is now the **sole canonical baseline path**
for Wang-EEGNet comparisons going forward. It uses the corrected architecture
and author-loader-compatible default four-class data construction. The
five-subject result is explicitly a smoke check, not a benchmark or full-scale
reproduction; see the medium-scale check when completed.

The 65.30% run is explicitly excluded as an exact reproduction and is not in
the baseline report. It is a separate model/training run, not an interchangeable
estimate. Its architecture resembles the Wang authors' released global model
configuration (8 temporal filters, depth multiplier 2, 16 pointwise filters,
128-sample temporal kernel, pool 8 then 8, ELU, batch normalization, dropout
0.2, Adam with an epoch schedule of 1e-2/1e-3/1e-4, and batch size 16). However,
the exported Kaggle model omits the authors' max-norm constraints (spatial
depthwise max-norm 1 and classifier max-norm 0.25) and uses a fixed 100 epochs.
The current public script sets its runnable `n_epochs` default to 2, so the saved
Kaggle record does not establish that its training duration matches the run
behind the published table. The cohort/window construction also differs from
the Wang repository's data loader. The model is therefore excluded from the
verified baseline report, despite partial architecture alignment. Therefore it is not promoted to a
verified reproduction based on accuracy proximity.

The superseded 62.30% run used the common EEGNet-8,2 architecture: temporal
F1=8, D=2 spatial filters per temporal filter, F2=16 pointwise filters,
64-tap temporal convolution, pool 4 then 8, ELU, batch normalization, and
0.5 dropout. Training used Adam at LR=1e-3 with weight decay 1e-4, batch size
128, inner-subject macro-F1 epoch selection up to 60 epochs with patience 8,
then a fresh fit on all outer-training subjects for the selected epoch count.
Its score-band status is historical and must not be interpreted as the
canonical Wang comparison.

## Data construction diff against the released Wang repository

Sources inspected line-by-line: the authors' [`get_data.py`](https://github.com/MHersche/eegnet-based-embedded-bci/blob/master/get_data.py),
[`eeg_reduction.py`](https://github.com/MHersche/eegnet-based-embedded-bci/blob/master/eeg_reduction.py),
[`models.py`](https://github.com/MHersche/eegnet-based-embedded-bci/blob/master/models.py),
and [`main_global.py`](https://github.com/MHersche/eegnet-based-embedded-bci/blob/master/main_global.py).

| Aspect | Released Wang code | Prior project preprocessing | Action/status |
|---|---|---|---|
| Cohort | Subject IDs 1–109 excluding 88, 92, 100, 104 (`get_data.py:58–76`) | Excludes 88, 89, 92, 100 | Added explicit `wang_repo` cohort (89 is restored; 104 excluded). Existing project cohort remains unchanged for prior subject-CV results. |
| Runs/classes | Runs 1, 4, 6, 8, 10, 12, 14; T1/T2 map to left/right for runs 4/8/12; T2 maps feet for 6/10/14 | Same task-run family and intended labels | Matched in the opt-in Wang extraction. |
| Task epoch timing | EDF annotation onset through onset + 480 samples (3 s) | MNE events, t=0..2.99375 s | Sample-equivalent for the same annotation onsets; no causal cue delay is added in either implementation. |
| Task trial counts | Up to 7 T1 and 7 T2 per task run, selected in annotation order | Keeps all accepted task annotations | Wang-compatible reader caps each event type at 7. |
| Rest | Run 1: 20 adjacent 480-sample windows plus an additional 480-sample window at Python `random.randint(0, 57*fs)`; task-run T0 code is unreachable because it follows `continue` (`get_data.py:187–227`) | All complete, non-overlapping run-1 windows; no random extra | Wang-compatible reader mirrors 20+1. Source code calls NumPy seed but uses Python's unseeded RNG, so the extra window is intrinsically nondeterministic in the original. Our compatibility path uses `random.Random(7)` for repeatability and documents this one deliberate reproducibility deviation. |
| Signal filtering | No filtering in `get_data.py` | MNE zero-phase 0.5–40 Hz filter | Wang path reads the raw EDF signal with no filter. Existing pipeline remains filtered. |
| Normalization | Optional per-trial/per-channel standardization; default off (`get_data.py:35–87`) | Outer-training-subject channel mean/std | Wang path stores raw windows and leaves normalization off by default. Project pipeline retains training-only normalization to protect held-out subjects. |
| Downsampling/channel/time reduction | `eeg_reduction.py`: when `n_ds>1`, select specified channels then `scipy.signal.decimate`; truncate to first `T*160` samples. For standard run, n_ds=1/n_ch=64/T=3 | Project caches ds=1 and anti-aliased `resample_poly` ds=2; full 64 channels and 3 s | Exact Wang-compatible dataset is the unfiltered 64-channel, 480-sample base; reduction is still separately parameterized in the official experiments and must use their channel lists/decimator for nondefault settings. |
| Fold split | `KFold(n_splits=5)` over ordered windows, without shuffle (`main_global.py:115–134`) | Subject-disjoint five-fold split | Not adopted as primary evaluation: it allows the same participant in train and validation and is subject leakage. Exact source split is documented but rejected for valid generalization claims. |

Remaining scope for exact compatibility: the Wang extraction implements the
released default four-class setting (ds=1, all 64 channels, 3 seconds). Its
nondefault channel-subset/decimation sweep is not yet wired into the
compatibility CLI. The one extra run-1 rest window is made deterministic with
`random.Random(7)` because the released code seeds NumPy but draws it through
Python's separately unseeded `random` module. The main Phase A/B Kaggle notebook
is now run in the focused [EEGMMIDB Wang Full Cohort Canonical notebook](https://www.kaggle.com/code/fouadiqbal/eegmmidb-wang-full-cohort-canonical).
The earlier Phase A/B notebook retains its historical exploratory cells.

The compatibility extraction is available via `--protocol wang_repo` on the
download and preprocessing commands. It produces `eegmmidb_wang_repo_4class.npz`;
the subject-disjoint project cache remains the default. No accuracy is claimed
in this section; subsequent smoke measurements are reported below. The previous
62.30% project-protocol result remains only as a superseded audit record, not an
exact Wang-code reproduction or a canonical Wang comparison.

### Architecture and training corrections

The released `models.py` applies `depthwise_constraint=max_norm(1.)` on the
spatial depthwise convolution and `kernel_constraint=max_norm(regRate)` on the
classifier, with `regRate=0.25` passed from `main_global.py`. The
`PaperAlignedEEGNet82` training path now projects those spatial kernels to L2
norm ≤1 and each classifier output kernel to L2 norm ≤0.25 after every update.
All three released `BatchNormalization(axis=1)` placements are also represented
in the Torch port by permuting to the corresponding axes before/after each
normalization layer; this was corrected after an initial five-subject smoke
run, whose scores are superseded and must not be interpreted as corrected
results.
Its exact released configuration is temporal kernel 128, first pooling 8,
second pooling 8, F1=8, D=2, F2=16, dropout 0.2, batch 16, and the existing
step schedule. The train CLI and fold function default to 100 epochs as
requested. Note: the inspected checked-in `main_global.py` currently says
`n_epochs=2` at line 84; therefore 100 epochs follows the user's explicit
paper-protocol correction and is not a value found in that script's present
default. An exact released-script run would use 2 epochs. This distinction must
remain in any results table.

The original official task-run rest behavior is a code defect/quirk and its
window-level KFold is leakage-prone. These are reproduced only in the named
compatibility extraction where noted; all performance evaluation in this
project must use subject-disjoint folds. Thus no exact numerical reproduction
can be claimed while preserving leakage-safe evaluation.

### Local real-data verification (2026-09-26)

- Clean venv: `tmp/step2-venv` installed every entry in `requirements.txt`
  including `pyyaml`, `pyedflib`, CPU PyTorch, MNE, and MOABB. `pip check`
  reported no broken requirements. All three YAML configs parsed successfully.
- Acquisition: five actual EEGMMIDB subjects (1–5), 35 EDFs, 83,543,520
  aggregate bytes under `data/raw/MNE-eegbci-data/files/eegmmidb/1.0.0/`.
  One initial invocation used `data/` rather than `data/raw/`; preprocessing
  rejected the missing expected path. The corrected download and preprocessing
  completed successfully.
- Project cache path: first preprocessing wrote five fold caches under
  `data/processed/eegmmidb/` (122,402,005 bytes each). The second invocation
  logged `Using existing ... caches ... raw processing skipped`, took 2.92 s,
  and left cache timestamps unchanged.
- Wang loader path: `data/processed/eegmmidb_wang/` contains the raw
  420-window compatibility dataset (105 examples per class) and subject-disjoint
  fold caches. The compressed dataset is 17,620,555 bytes and each fold cache
  is 77,425,021 bytes. A repeat call logged that EDF loading was skipped, took
  4.47 s, and left cache timestamps unchanged. This small subset has one
  validation subject per fold; all local scores below are smoke measurements,
  not benchmark estimates.
- Corrected global model: fold 0 trained on subjects 1, 2, 3, 5 for 100 epochs
  and evaluated once on subject 4 (84 windows). An intermediate model version
  scored 33.33% accuracy / 31.64% macro-F1, but still used feature-map
  BatchNorm axes rather than the released `axis=1` settings; it is marked
  `superseded_architecture` in the CSV and is not the final result.
- Final axis-aligned global model: fold 0 again trained on subjects 1, 2, 3, 5
  for 100 epochs and evaluated once on subject 4 (84 windows). Accuracy =
  34.52%, macro-F1 = 33.22%. A second independent run with the same
  initialization/configuration produced the same score. The two final rows are
  preserved in
  `results/baseline_accuracy_report.csv` with distinct run IDs and UTC times.
  The score is 30.55 percentage points below the 65.07% reference; the tiny
  five-subject subset cannot support a Wang benchmark comparison.
- SS-TL ran end-to-end from the corrected global checkpoint on held-out subject
  4, using four stratified within-subject folds, 5 fine-tuning epochs, Adam
  1e-3 and batch size 16. Four result rows, prediction files and checkpoints
  are in `results/subject_transfer/`. With the final axis-aligned checkpoint,
  fold accuracies were 58.33%, 50.00%, 40.00%, and 60.00% (mean 52.08%); these
  are exploratory smoke values only.
- The real-cache audit verifies disjoint global subject sets, all four labels,
  Wang-path raw (unnormalized) data, target-subject exclusion from each global
  initialization, disjoint/exhaustive SS-TL train/test indices, and distinct
  global run history: **10 passed** across the leakage, real-data integration,
  and Wang-constraint checks. Compilation passed. The broader shape suite has
  10 dataset-dependent skips because BCI IV 2a data are not present here.
  One approximate-circuit metrics test was not executable on this Windows host:
  its exhaustive LUT builder requires `gcc`, which is not installed. The
  environment and source did not fail that check; its external compiler
  prerequisite is open.
- **Superseding status update (2026-09-27):** the full 105-subject, subject-disjoint,
  five-fold Wang-compatible Kaggle baseline has since completed at 65.10 ± 3.33%
  accuracy and 65.13 ± 3.28% macro-F1. This resolves the older recommendation
  above, which described the pre-Kaggle snapshot. GCC was then available on the
  Kaggle T4 and the approximate-circuit correctness tests passed 3/3. The
  EEGMMIDB representative-fold approximate-MAC sweep and BCI IV 2a five-fold
  approximate sweep are also recorded. See
  [`canonical_full_cohort_run.md`](canonical_full_cohort_run.md),
  [`eegmmidb_axm_sensitivity_results.md`](eegmmidb_axm_sensitivity_results.md),
  and [`targeted_results_audit.md`](targeted_results_audit.md) for current
  status. Formal AME validation and hardware measurements remain open.


The corrected target sources are Wang et al., [arXiv:2004.00077](https://arxiv.org/abs/2004.00077),
Hernandez-Ruiz et al., [IEEE Sensors 2021, DOI 10.1109/SENSORS47087.2021.9639747](https://doi.org/10.1109/SENSORS47087.2021.9639747),
the Hernandez-Ruiz numerical target table in the authors' [2021 research poster](https://eneriz-daniel.com/assets/pdf/JJIQYF2021-slides.pdf); and Ingolfsson et al., [arXiv:2006.00622](https://arxiv.org/abs/2006.00622).
The authors' released Wang training script is [main_global.py](https://github.com/MHersche/eegnet-based-embedded-bci/blob/master/main_global.py),
and the SS-TL procedure is [main_ss.py](https://github.com/MHersche/eegnet-based-embedded-bci/blob/master/main_ss.py).
