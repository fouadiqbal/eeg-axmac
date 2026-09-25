"""Correlate circuit error metrics with measured EEG accuracy changes."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


ERROR_METRICS = ("ER", "MRED", "MAE", "ME", "AME")
TARGETS = ("delta_acc", "delta_f1", "delta_sensitivity")


def analyze(matrix: Path, output_dir: Path) -> Path:
    """Write global/per-layer coefficients and 300 dpi metric-pair plots."""
    data = pd.read_csv(matrix)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for layer, subset in [("all", data), *list(data.groupby("layer"))]:
        for metric in ERROR_METRICS:
            for target in TARGETS:
                valid = subset[[metric, target]].dropna()
                if len(valid) < 3 or valid[metric].nunique() < 2 or valid[target].nunique() < 2:
                    rows.append({"layer": layer, "error_metric": metric, "target": target,
                                 "n": len(valid), "pearson_r": np.nan, "pearson_p": np.nan,
                                 "spearman_rho": np.nan, "spearman_p": np.nan,
                                 "status": "insufficient measured variation"})
                    continue
                pearson = pearsonr(valid[metric], valid[target])
                spearman = spearmanr(valid[metric], valid[target])
                rows.append({"layer": layer, "error_metric": metric, "target": target,
                             "n": len(valid), "pearson_r": pearson.statistic,
                             "pearson_p": pearson.pvalue, "spearman_rho": spearman.statistic,
                             "spearman_p": spearman.pvalue, "status": "measured"})
    table = pd.DataFrame(rows)
    output = output_dir / f"{matrix.stem}_correlations.csv"
    table.to_csv(output, index=False)
    for metric in ERROR_METRICS:
        if metric not in data or data[metric].notna().sum() < 3:
            continue
        for target in TARGETS:
            fig, ax = plt.subplots(figsize=(7.5, 5.4), constrained_layout=True)
            for layer, subset in data.groupby("layer"):
                ax.scatter(subset[metric], subset[target], s=25, alpha=.75, label=layer)
            ax.axhline(0, color="black", lw=.7)
            ax.set(xlabel=metric, ylabel=target, title=f"Circuit {metric} vs held-out {target}")
            ax.legend(frameon=False)
            ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
            fig.savefig(output_dir / f"{metric}_vs_{target}.png", dpi=300, bbox_inches="tight")
            plt.close(fig)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results/plots"))
    args = parser.parse_args()
    print(analyze(args.matrix, args.output_dir))


if __name__ == "__main__":
    main()
