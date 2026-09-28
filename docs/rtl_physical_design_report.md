# SKY130 multiplier-cell physical-design report

## Scope and provenance

Four registered signed INT8 multiplier cells were compared: an exact multiplier and the full-fold software finalists `mul8_348`, `mul8_112`, and `mul8_424`. The three approximate cores came from EvoApprox8b commit `ad6bb819bed0c641e26c882f06796efcfe406960`; their original C and Verilog sources are archived in `rtl/evoapprox8b/`. All four designs use the same sign/magnitude adapter and output register. The original Verilog files were kept intact; the generated synthesis files add compatibility definitions for missing primitive names. This is a **multiplier-cell** comparison, not an EEGNet accelerator or on-device measurement.

The Windows host did not have working Yosys, OpenROAD/OpenLane, or SKY130 tools. Runs used OpenLane 2.1.11 in Google Colab with SKY130A revision `bdc9412b3e468c102d01b7cf6337be06ec6e9c9a` and `sky130_fd_sc_hd`. The official `spm` example passed routing, Magic DRC, and LVS. The in-flow KLayout step initially failed with `SRE module mismatch`, so `RUN_KLAYOUT_DRC=false` was used during implementation. This was resolved without rerunning place-and-route: the same SKY130A KLayout DRC runset was applied to each saved final GDS using `env -u PYTHONPATH klayout -b -rd input=<GDS> -rd report=<report> -r sky130A.lydrc`. Clearing the Nix Python path avoided the interpreter/standard-library mismatch. All four direct KLayout runs exited 0 with **zero violation items** in their report databases. The saved [Colab notebook](https://colab.research.google.com/drive/10OtY6HqmbdSbIS_C6xK-yHnmQ7xlUTdh) contains the successful Python dependency install and physical run outputs; separate KLayout evidence is saved locally below.

## Verification and measurement

Every design passed exhaustive RTL simulation against its exact or official EvoApprox C/LUT behavior for all 65,536 signed operand pairs. The four OpenLane final metrics report zero setup and hold violations, zero final routing and Magic DRC errors, and zero LVS errors. Separate KLayout DRC report databases for the same four final GDS files each contain zero violation items. The timing target was 50 MHz (20 ns period); a preliminary 100 MHz exact run did not close timing, so every comparison below was rerun at the same 50 MHz target.

Area is placed standard-cell instance area. Total power is OpenLane's estimated dynamic plus leakage power at the stated activity and 50 MHz; it is not measured silicon power. Delay is `20 ns − final worst setup slack`, a setup-derived cycle-path estimate, and `fmax = 1/delay`; it is not a separately characterized combinational propagation delay. Energy/result assumes one result per clock (`power × 20 ns`), and EDP is this energy multiplied by the setup-derived delay. Values are rounded from the preserved metrics in `results/rtl_sky130/evidence/`.

| Design | Cell area (µm²) | Delay (ns) | fmax (MHz) | Dynamic (mW) | Static (mW) | Total (mW) | EDP (pJ·ns) | Area reduction | Power reduction | EDP reduction |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Exact | 3238.11 | 14.48 | 69.05 | 0.371937 | 0.00000424 | 0.371941 | 107.73 | — | — | — |
| mul8_348 | 3235.60 | 17.16 | 58.27 | 0.372423 | 0.00000415 | 0.372428 | 127.83 | 0.08% | −0.13% | −18.66% |
| mul8_112 | 2160.82 | 14.14 | 70.74 | 0.232506 | 0.00000289 | 0.232509 | 65.74 | 33.27% | 37.49% | 38.98% |
| mul8_424 | 1714.14 | 13.54 | 73.88 | 0.189163 | 0.00000237 | 0.189166 | 51.21 | 47.06% | 49.14% | 52.46% |

A public [SKY130 8-bit serial-parallel multiplier example](https://github.com/bvbhavana1/SPM) reports 3711.06 µm² standard-cell area. Our exact cell's 3238.11 µm² is the same order of magnitude, a plausibility check only: the architecture, register boundary, tool settings, and activity are different. Its published 1.06 mW power is not a matched power benchmark.

## Accuracy versus cell PPA

The software figures below are EEGMMIDB five-fold **full-held-out-set** ΔAccuracy, mean ± fold SD, when each multiplier is applied to one layer at a time. They do not imply the entire model achieves the cell-level power savings.

| Multiplier | Spatial ΔAccuracy (pp) | Separable ΔAccuracy (pp) | Cell EDP reduction |
|---|---:|---:|---:|
| mul8_348 | −0.034 ± 0.111 | −0.034 ± 0.051 | −18.66% |
| mul8_112 | −1.542 ± 0.298 | −0.805 ± 0.803 | 38.98% |
| mul8_424 | −3.265 ± 0.861 | −1.338 ± 0.812 | 52.46% |

`mul8_112` is the best balanced candidate: substantial cell area, power, and EDP reductions with smaller observed accuracy loss than `mul8_424`. `mul8_424` has the best raw cell PPA but the largest accuracy penalty, including statistically significant drops in the earlier multiple-comparison-corrected coarse screen. `mul8_348` has negligible accuracy impact but no useful cell PPA gain. The full-test ΔAccuracy estimates are not equivalence claims; the earlier coarse screen and its significance analysis are in `docs/software_closure_report.md`. BCI-IV-2a's historical unvalidated sweep is excluded.

## Evidence and limits

- Reproducible runner: `rtl/colab_physical_run.py`; design generator: `rtl/generate_designs.py`.
- Machine-readable metrics: `results/rtl_sky130/evidence/results.csv`; per-design `final_metrics.json`, simulation logs, OpenLane logs, GDS and netlists are in the same evidence bundle.
- Saved archive: `results/rtl_sky130/evidence.zip`, SHA-256 `51cbed8433e48c4d65826f073fd0495abfbcba31aef0de4237015f56aff2bb4e`.
- KLayout DRC summary, logs, and report databases: `results/rtl_sky130/klayout_drc_evidence/`. Archive: `results/rtl_sky130/klayout_drc_evidence.zip`, SHA-256 `0574f0b0da2a1c94dd670235c903f6edbba30098fc1451d52646429eaf638b28`. The saved Colab notebook reran the four DRC checks and produced this archive; the reproducible runner now invokes the same direct check after future physical runs.
- Open items before fabrication or deployment claims: independent foundry sign-off, matched gate-level or post-layout power vectors, and full-system hardware evaluation. Passing this open-source KLayout deck is not foundry sign-off.

No RTL for a full accelerator was attempted. No GitHub push was made. This report awaits review before manuscript consolidation.
