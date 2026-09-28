"""Summarize a completed EEGMMIDB single-fold AXM sweep and draw 12 scatter plots.

Only measured ER, MRED, MAE, and ME are analyzed. AME is left unmeasured
because the EvoApprox circuit catalog has no architecture propagation model.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


LAYERS = ("temporal", "spatial", "separable", "dense")
ERRORS = ("ER", "MRED", "MAE", "ME", "AME")
OUTCOMES = ("delta_acc", "delta_f1", "delta_sensitivity")


def analyze(matrix_path: Path, plots_dir: Path) -> pd.DataFrame:
    matrix = pd.read_csv(matrix_path)
    required = {"layer", "multiplier_id", *ERRORS, *OUTCOMES}
    if not required.issubset(matrix.columns):
        raise ValueError(f"Missing matrix columns: {sorted(required - set(matrix.columns))}")
    if set(matrix.layer) != set(LAYERS) or len(matrix) != 4 * 50:
        raise ValueError("Expected a completed 50-circuit, four-layer sweep")
    if matrix.duplicated(["layer", "multiplier_id"]).any():
        raise ValueError("Duplicate layer/circuit combinations")
    if matrix.groupby("layer").multiplier_id.nunique().to_dict() != {layer: 50 for layer in LAYERS}:
        raise ValueError("Each layer must contain the same 50 distinct circuits")
    plots_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for layer, part in [("pooled", matrix), *list(matrix.groupby("layer"))]:
        for error in ERRORS:
            for outcome in OUTCOMES:
                valid = part[[error, outcome]].replace([np.inf, -np.inf], np.nan).dropna()
                status = "measured"
                values = [np.nan] * 4
                if error == "AME":
                    status = "AME requires a validated architecture propagation model"
                elif len(valid) < 3 or valid[error].nunique() < 2 or valid[outcome].nunique() < 2:
                    status = "insufficient variation"
                else:
                    p = pearsonr(valid[error], valid[outcome])
                    s = spearmanr(valid[error], valid[outcome])
                    values = [p.statistic, p.pvalue, s.statistic, s.pvalue]
                rows.append({"layer": layer, "error_metric": error, "outcome": outcome,
                             "n": len(valid), "pearson_r": values[0], "pearson_p": values[1],
                             "spearman_rho": values[2], "spearman_p": values[3], "status": status})
    result = pd.DataFrame(rows)
    result.to_csv(plots_dir / "eegmmidb_axm_correlations.csv", index=False)
    for error in ERRORS:
        if error == "AME":
            continue
        for outcome in OUTCOMES:
            fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
            for layer, part in matrix.groupby("layer"):
                ax.scatter(part[error], part[outcome] * 100, s=24, alpha=.7, label=layer)
            ax.axhline(0, color="black", lw=.75)
            ax.set(xlabel=error, ylabel=f"{outcome} (percentage points)",
                   title=f"EEGMMIDB fold 0: {error} versus {outcome}")
            ax.grid(alpha=.2)
            ax.legend(frameon=False)
            fig.savefig(plots_dir / f"{error}_vs_{outcome}.png", dpi=300)
            plt.close(fig)
    summary = matrix.groupby("layer")[list(OUTCOMES)].agg(["mean", "median", "min", "max"])
    summary.to_csv(plots_dir / "eegmmidb_layer_sensitivity_summary.csv")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--plots-dir", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.matrix, args.plots_dir)
    print(result[result.outcome == "delta_acc"].to_string(index=False))


if __name__ == "__main__":
    main()
