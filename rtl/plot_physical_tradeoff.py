"""Plot measured multiplier-cell EDP against full-test EEG accuracy effects."""
import csv
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "rtl_sky130"
rows = {r["design"]: r for r in csv.DictReader((OUT / "evidence" / "results.csv").open())}
effects = {
    "mul8_348": {"spatial": (-0.034, 0.111), "separable": (-0.034, 0.051)},
    "mul8_112": {"spatial": (-1.542, 0.298), "separable": (-0.805, 0.803)},
    "mul8_424": {"spatial": (-3.265, 0.861), "separable": (-1.338, 0.812)},
}
with (OUT / "accuracy_ppa.csv").open("w", newline="") as handle:
    writer = csv.writer(handle)
    writer.writerow(["multiplier_id", "layer", "delta_accuracy_pp_mean", "delta_accuracy_pp_sd", "cell_edp_pj_ns", "cell_edp_reduction_pct"])
    for design, layers in effects.items():
        for layer, (mean, sd) in layers.items():
            writer.writerow([design, layer, mean, sd, rows[design]["edp_pj_ns"], rows[design]["edp_pj_ns_reduction_pct"]])

fig, ax = plt.subplots(figsize=(7.2, 4.5), layout="constrained")
colors = {"mul8_348": "#137c80", "mul8_112": "#df8f25", "mul8_424": "#b44655"}
markers = {"spatial": "o", "separable": "s"}
for design, layers in effects.items():
    x = float(rows[design]["edp_pj_ns_reduction_pct"])
    for layer, (mean, sd) in layers.items():
        ax.errorbar(x, -mean, yerr=sd, marker=markers[layer], color=colors[design],
                    capsize=3, linestyle="none", markersize=7,
                    label=f"{design} / {layer}")
ax.axhline(0, color="#666666", lw=0.8)
ax.axvline(0, color="#666666", lw=0.8)
ax.set_xlabel("Multiplier-cell EDP reduction vs exact (%)")
ax.set_ylabel("EEGMMIDB accuracy drop (percentage points)")
ax.set_title("Accuracy cost versus SKY130 cell EDP")
ax.legend(fontsize=8, ncol=2, frameon=False)
ax.grid(alpha=0.25)
fig.savefig(OUT / "accuracy_vs_edp.png", dpi=240)
