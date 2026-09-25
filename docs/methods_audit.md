# Methods audit for the EEG approximate-MAC pipeline

This audit separates methods supported by the supplied papers from results
actually measured in this repository. The PDFs remain with the researcher;
their file names are given so a reader can locate the source. Any accuracy
claim in the notebook must come from its recorded held-out predictions.

| Source supplied by the researcher | Relevant method | Decision for this pipeline |
| --- | --- | --- |
| `2004.00077v3.pdf` (Wang et al., 2020) | Subject-level five-fold EEGMMIDB evaluation, 3 s windows, compact EEGNet, fixed 100-epoch schedule | Keep five disjoint outer subject folds. Treat its 65.07% as context because our excluded subject and window counts differ from its released loader. |
| `hernandez-ruiz2021.pdf` | EEGNet-4,2 with LeakyReLU slopes 0.6/0.5/0.4, no batch normalization/dropout; ds and window-length resource trade-offs | Measure the compact model on the *same* cached EEGMMIDB folds. Evaluate ds=2 and shorter windows only as separate, declared ablations; the published 65.75% is contextual. |
| `2006.00622v1.pdf` (EEG-TCNet) | BCI IV 2a motor-imagery preprocessing and compact temporal network comparison | Use its BCI IV 2a discussion as dataset context. A TCN would be a new model family and is not silently substituted for EEGNet. |
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
