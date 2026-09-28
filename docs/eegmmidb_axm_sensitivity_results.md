# EEGMMIDB Approximate-MAC Sensitivity Sweep — Results

## Scope and evidence

The full inference-only sweep completed on Kaggle using a Tesla T4 and gcc 11.4.0. It used the accepted 105-subject Wang-compatible, five-fold baseline and a representative fold fixed before sensitivity evaluation (fold 0; 84 training subjects, 21 held-out subjects, 1,764 held-out windows). No model was retrained during the sweep. Exact FP32 replay matched the saved reference at 60.7143% accuracy and 60.7575% macro-F1. All 105 subject caches/checkpoints were found. The official approximate-circuit tests passed (3/3), and three circuit truth tables/ER/MRED values were spot-checked.

The selected set contains 50 EvoApprox8b unsigned 8-bit multipliers, producing 200 one-layer-at-a-time evaluations across temporal, spatial/depthwise, separable, and dense layers. The measured CSV has 200 rows. Training-only INT8 calibration was used, and the approximate layer was compared against the exact INT8 LUT at matching scales. The main sweep took 1,643.5 seconds on T4.

## Layer sensitivity

Mean absolute changes are in percentage points (pp), averaged across the 50 circuits.

| EEGNet layer | Mean absolute Δaccuracy (pp) | Mean absolute Δmacro-F1 (pp) | Mean absolute Δsensitivity (pp) | Δaccuracy range |
|---|---:|---:|---:|---:|
| Dense | 0.387 | 0.361 | 0.387 | −1.020 to +1.077 pp |
| Temporal | 0.956 | 0.928 | 0.956 | −1.871 to +0.227 pp |
| Spatial/depthwise | 1.260 | 1.309 | 1.260 | −3.968 to +0.170 pp |
| Separable | 1.279 | 1.315 | 1.279 | −3.458 to +0.340 pp |

Spatial and separable substitutions were most sensitive on average; dense was least sensitive. Positive changes for some circuits reflect quantization/approximation interactions and do not imply general accuracy improvement.

## Correlation findings

Pooled correlations with Δaccuracy across all 200 layer/circuit rows:

| Error metric | Pearson r | Spearman ρ |
|---|---:|---:|
| ER | −0.312 | −0.469 |
| MRED | −0.461 | −0.446 |
| MAE | −0.420 | −0.441 |
| Signed mean error (ME) | +0.315 | +0.390 |
| Eq. 4 DWM, factorized operand marginals | +0.192 | +0.293 |

Per-layer relationships varied materially. Examples: ER’s dense Pearson r was +0.033 while pooled r was −0.312; MAE within separable was −0.777/−0.836 (Pearson/Spearman), compared with pooled −0.420/−0.441. Training-only DWM’s temporal correlation was +0.764/+0.704 while pooled DWM correlation was +0.192/+0.293. These associations are descriptive across 50 selected circuits per layer and should not be read as independent subject-level evidence. The result supports layer-stratified analysis: vision-derived scalar error metrics do not rank EEG accuracy loss uniformly across layers.

## Completion status and limitation

The requested circuit tests, 50-circuit curation, 200-row layer sensitivity matrix, Δaccuracy/ΔF1/Δsensitivity analysis, pooled and per-layer Pearson/Spearman tables, and 15 plots are complete. The executed Kaggle notebook was saved as Version 6 and its validated artifact ZIP was retrieved from the live Kaggle session. The extracted evidence is saved under [`results/axm_eegmmidb_fold0/`](../results/axm_eegmmidb_fold0/), including the matrix, 50-circuit catalog, metrics, tables, and figures. An EEGNet-adapted AME propagation fit was attempted and **failed held-out-circuit validation** for the convolutional layers (r=.151/.342/.056; NRMSE≈1); the dense fit was tautological. Accordingly, all 200 AME values are blank, all 15 AME correlation entries have n=0, and no AME plots are included. The Eq. 4 factorized DWM remains a workload-weighted diagnostic, not formal AME. A further reproducibility limitation is that this sweep evaluates one preselected fold, not all five folds. No GitHub push was made.

## Notebook

Canonical Kaggle notebook: https://www.kaggle.com/code/fouadiqbal/eegmmidb-canonical-approximate-mac-sensitivity?scriptVersionId=353306324 (Version 6 Quick Save, successful; about 10 seconds, no rerun of the 27-minute sweep). The local notebook export is being synchronized to this version for VS Code inspection.
