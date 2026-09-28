# EEGNet approximate-multiplier study

Reproducible code and compact validated results for subject-disjoint motor-imagery EEG evaluation, layer-wise approximate-multiplier inference, and SKY130 multiplier-cell comparison. Full-cohort model training and sensitivity evaluation ran on Kaggle; physical-design flow ran in Colab. Raw EEG and large intermediate tool outputs are not stored here.

## Validated status

- EEGMMIDB Wang-compatible EEGNet-8,2: **65.10 ± 3.33%** four-class accuracy over five subject-disjoint folds (105 eligible participants), versus Wang et al.'s 65.07%. This is a near reproduction with documented protocol departures; see [canonical run](docs/canonical_full_cohort_run.md) and [methods audit](docs/methods_audit.md).
- EEGMMIDB approximate-multiplier screen: 50 EvoApprox8b circuits × four layer types × five folds, 1,000 inference conditions. Full held-out-set effects were subsequently measured for selected circuits; see [software closure report](docs/software_closure_report.md) and compact CSVs in `results/software_closure/`.
- BCI IV 2a session EEGNet-8,2 baseline is validated within the predeclared ±3 percentage-point band. The older BCI multiplier sweep used incompatible checkpoints and remains **unvalidated**; do not cite it as a finding.
- Exact and selected approximate signed INT8 multiplier cells passed exhaustive RTL/LUT comparison, routing, LVS, Magic DRC, and KLayout DRC. Matched SKY130 cell PPA estimates are in [physical-design report](docs/rtl_physical_design_report.md) and `results/rtl_sky130/evidence/results.csv`. These are cell estimates, not whole-network energy measurements.
- An attempted EEGNet-adapted AME propagation metric failed held-out-circuit validation and is excluded from predictive claims.

## Reproduce the work

- Install Python dependencies: `pip install -r requirements.txt`.
- Baseline preparation and training: `src/data/` and `src/train/train_baseline.py`, with configs in `configs/` and the [canonical Kaggle notebook](notebooks/eegmmidb_wang_full_cohort_canonical.ipynb).
- Approximate inference and analysis: `src/axm/`, `src/train/sweep_canonical_eegmmidb.py`, `src/train/sensitivity_sweep.py`, `scripts/software_closure_statistics.py`, and the Kaggle runners in `notebooks/`.
- RTL and cell physical design: `rtl/generate_designs.py`, `rtl/colab_physical_run.py`, and the selected source models under `rtl/evoapprox8b/`.

The original EDF data, training checkpoints, and full OpenLane run trees are excluded from this repository. Compact result tables and method reports are retained for traceability. Superseded and unvalidated historical rows remain labeled in their source CSVs and are excluded from validated conclusions.
