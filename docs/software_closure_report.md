# Software-side closure report — 2026-09-28

## 1. Baseline validation

| Dataset and model | Protocol | This work | Published comparator | Verdict |
|---|---|---:|---:|---|
| EEGMMIDB, Wang-compatible EEGNet | 105 subjects, subject-disjoint five folds, 100 epochs | 65.10 ± 3.33% | 65.07% | Accepted canonical baseline |
| BCI-IV-2a, fixed EEGNet-8,2 | Nine session T→E models, 750 epochs, training-only scaling | 71.10 ± 11.73%; macro-F1 71.02% | 72.40% fixed EEGNet in EEG-TCNet Table III | Within ±3 percentage points (−1.30 pp); near-reproduction because comparator training code and checkpoints were not released |
| BCI-IV-2a, released fixed EEG-TCNet | Nine session T→E released checkpoints | 77.346 ± 11.577%; macro-F1 77.190% | 77.35% | Reproduced released reference; does not validate EEGNet sweep |

The nine EEGNet checkpoints, predictions, per-subject CSV, and config are archived in `results/bciiv2a_eegnet82/bciiv2a_eegnet82_validated_models.zip` (SHA-256 `575f7da7ecb04aedde533d6813f7f678b388bf1ad009a6f36189f7f31ab2adb6`). The old 1,000-evaluation BCI subject-disjoint multiplier sweep used different checkpoints and a different protocol; its result file remains tagged `unvalidated_baseline` and is **excluded** from sensitivity findings and BCI significance tests. Validation of the new T→E baseline cannot retroactively validate that sweep.

## 2. EEGMMIDB five-fold sensitivity

Archived results contain 1,000 unique cases (5 folds × 4 layers × 50 multipliers), 21,000 subject-paired records, 105 unique held-out subjects, and 420 balanced held-out windows per fold. Per-fold ranking by mean absolute accuracy change: separable was most sensitive in folds 0, 1, 2, and 4; spatial was most sensitive in fold 3. Dense was least sensitive in folds 0, 1, and 4; temporal was least sensitive in folds 2 and 3. Thus the ranking **varies across folds**. The pooled mean absolute change across circuit means was separable 1.594 pp, spatial 1.035 pp, temporal 0.375 pp, dense 0.352 pp.

The per-fold matrices and the per-(layer, multiplier) five-fold mean ± SD are in `results/axm_eegmmidb_5fold_recovered/` and `results/software_closure/sensitivity_mean_std_5fold.csv`. These are coarse-screen results. Seeds were `20260927 + fold`.

## 3. Paired tests and error-metric correlations

An independent recheck used each subject's **integer difference in correct predictions** to prevent floating-point tie misranking, two-sided asymptotic Wilcoxon with zero differences omitted, then Benjamini–Hochberg over 200 layer–multiplier tests. **28 significant accuracy drops** remain: 18 separable, 10 spatial, zero temporal, zero dense. This supersedes the notebook's earlier floating-rank count of 20. The tests are exploratory because the five training folds overlap; absence of a significant drop is not an equivalence finding. The full table is `results/software_closure/wilcoxon_integer_ties.csv`.

On five-fold circuit means, pooled Pearson correlations between error metrics and Δaccuracy were ER −0.234, MRED −0.472, MAE −0.437; Spearman values were −0.430, −0.425, −0.415. The fold-and-circuit bootstrap table, including per-layer estimates and macro-F1, is `results/software_closure/correlations_fold_multiplier_bootstrap.csv`. Dense-layer ER changes sign versus the pooled estimate (+0.205 versus −0.234), but its interval includes zero; this is evidence of layer context, not a confirmed positive relationship. AME is excluded: the earlier EEGNet propagation fit failed held-out-circuit validation and is retained as a negative result, not repaired or reinterpreted.

No BCI multiplier significance result is claimed because the historical BCI sweep is unvalidated. A new sweep on the saved session EEGNet checkpoints would be a separate experiment.

## 4. Full held-out-set finalist confirmation and recommendation

The three candidates were subsequently evaluated without retraining on **all 1,764 held-out windows per EEGMMIDB fold** at all four layers: 60 cases and 1,260 paired subject records. `results/software_closure/eegmmidb_finalists_full_test.zip` (SHA-256 `49b2f408eed87d333c713ae1358938cf2e2df23fffee3fed20589f846aed43e8`) contains the raw journal and final CSVs. The extracted tables are in `results/software_closure/finalists_full_test/`.

| Candidate tier | Circuit | Spatial Δaccuracy, mean ± fold SD | Separable Δaccuracy, mean ± fold SD | EvoApprox published power proxy |
|---|---|---:|---:|---:|
| Near-negligible impact | mul8_348 | −0.034 ± 0.111 pp | −0.034 ± 0.051 pp | 0.423 |
| Moderate tradeoff | mul8_112 | −1.542 ± 0.298 pp | −0.805 ± 0.803 pp | 0.202 |
| High-saving, high-impact stress case | mul8_424 | −3.265 ± 0.861 pp | −1.338 ± 0.812 pp | 0.122 |

Recommend carrying **mul8_348, mul8_112, and mul8_424** as three distinct sensitivity tiers for multiplier-cell RTL comparison. The coarse-screen BH test found spatial mul8_424 and separable mul8_424 significant drops; mul8_348 and mul8_112 were not significant under that 200-test screen. The full-test confirmation changes the effect estimates above but is not an equivalence test. Published EvoApprox power values are circuit-catalog proxies, not measured SKY130 PPA or demonstrated system savings.

## Gate

The EEGMMIDB baseline, 50-circuit five-fold screen, corrected exploratory statistics, and three full-test finalists are complete and archived. The BCI **baseline** is validated within band, while the historical BCI **sensitivity sweep** remains explicitly excluded. This is sufficient to review EEGMMIDB-specific finalist selection; it is **not** a cross-dataset BCI sensitivity claim. No RTL, physical design, or GitHub push was performed.
