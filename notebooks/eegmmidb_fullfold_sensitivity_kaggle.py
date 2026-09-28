"""Five-fold EEGMMIDB multiplier sweep; execute only in Kaggle with the attached baseline output."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr, wilcoxon
from statsmodels.stats.multitest import multipletests

assert torch.cuda.is_available(), "Select Kaggle GPU T4 x2; do not run this on the local PC."
device = torch.device("cuda:0")
print("Accelerators:", torch.cuda.device_count(), [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())], flush=True)
torch.set_num_threads(4)


def read_case_journal(path):
    """Load complete per-case records and truncate a crash-damaged final line."""
    if not path.exists():
        return []
    raw = path.read_bytes()
    lines = raw.splitlines(keepends=True)
    records, valid_bytes = [], 0
    for index, line in enumerate(lines):
        if not line.endswith((b"\n", b"\r")):
            if index == len(lines) - 1:
                break
            raise ValueError(f"Truncated journal record before final line: {path}")
        record = json.loads(line)
        if not {"layer", "multiplier_id", "result", "subject_rows"} <= record.keys():
            raise ValueError(f"Journal record missing required fields: {path}")
        records.append(record)
        valid_bytes += len(line)
    if valid_bytes != len(raw):
        with path.open("r+b") as stream:
            stream.truncate(valid_bytes)
            stream.flush()
            os.fsync(stream.fileno())
    keys = [(str(r["layer"]), str(r["multiplier_id"])) for r in records]
    if len(keys) != len(set(keys)):
        raise ValueError(f"Duplicate case keys in journal: {path}")
    return records


def append_case_journal(path, record):
    """Fsync one result and its per-subject rows before moving to the next case."""
    required = {"layer", "multiplier_id", "result", "subject_rows"}
    if not required <= record.keys():
        raise ValueError(f"Journal record missing fields: {sorted(required - record.keys())}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(record, separators=(",", ":"), allow_nan=True) + "\n").encode("utf-8")
    with path.open("ab") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())

# Locate the fixed, previously verified baseline artifacts attached as a Kaggle input.
summaries = list(Path("/kaggle/input").rglob("canonical_full_cohort_summary.json"))
assert len(summaries) == 1, f"Expected one canonical baseline summary, found: {summaries}"
summary_path = summaries[0]
result_dir = summary_path.parent
input_root = result_dir.parent
cache_dir = input_root / "eeg_axmac_wang_full_cohort" / "subject_cache"
baseline = json.loads(summary_path.read_text())
assert baseline["n_subjects"] == 105 and baseline["folds"] == 5 and baseline["epochs"] == 100
assert baseline["verdict"] == "verified_within_band"
assert len(list(cache_dir.glob("S*.npz"))) == 105
print("Baseline:", summary_path, "| subject caches:", len(list(cache_dir.glob('S*.npz'))), flush=True)

# Pin the inference implementation, then patch its batch-normalization axis to the
# exact full-cohort checkpoint architecture recorded by the accepted baseline.
work = Path("/kaggle/working/eeg_axmac_fullfold")
work.mkdir(parents=True, exist_ok=True)
repo = work / "source"
commit = "7a2d0215b48dc449add75fe59226948f7080b3b5"
if not repo.exists():
    subprocess.run(["git", "clone", "-q", "https://github.com/fouadiqbal/eeg-axmac.git", str(repo)], check=True)
subprocess.run(["git", "-C", str(repo), "checkout", "-q", commit], check=True)
model_path = repo / "src/models/eegnet.py"
source = model_path.read_text()
head, tail = source.split("class PaperAlignedEEGNet82", 1)
for pattern, replacement in (
    (r"self\.bn1\s*=\s*nn\.BatchNorm2d\(\d+\)", "self.bn1 = nn.BatchNorm2d(n_channels)"),
    (r"self\.bn2\s*=\s*nn\.BatchNorm2d\(\d+\)", "self.bn2 = nn.BatchNorm2d(1)"),
    (r"self\.bn3\s*=\s*nn\.BatchNorm2d\(\d+\)", "self.bn3 = nn.BatchNorm2d(1)"),
):
    tail, count = re.subn(pattern, replacement, tail, count=1)
    assert count in (0, 1), (pattern, count)  # zero means a prior resumed run already patched it
start = tail.index("    def forward(self, x: torch.Tensor) -> torch.Tensor:")
end = tail.find("\n    def ", start + 10)
if end < 0:
    end = len(tail)
forward = """    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = self.temporal(x)
        x = self.bn1(x.transpose(1, 2)).transpose(1, 2)
        x = self.spatial(x)
        x = self.bn2(x.transpose(1, 2)).transpose(1, 2)
        x = self.drop1(self.pool1(torch.nn.functional.elu(x)))
        x = self.sep_point(self.sep_depth(x))
        x = self.bn3(x.transpose(1, 2)).transpose(1, 2)
        x = self.drop2(self.pool2(torch.nn.functional.elu(x)))
        return self.classifier(x.flatten(1))
"""
source = head + "class PaperAlignedEEGNet82" + tail[:start] + forward + tail[end:]
model_path.write_text(source)
sys.path.insert(0, str(repo))

from src.axm.approx_conv import with_layer_lut
from src.axm.evoapprox_loader import EVOAPPROX_COMMIT, build_lut, load_metadata, select_circuits
from src.axm.lut_multiplier import exact_unsigned_lut
from src.axm.metrics import circuit_error_metrics
from src.eval.metrics import classification_metrics
from src.models.eegnet import PaperAlignedEEGNet82
from src.train.quantize import calibrate_model


def score(model, x, y, batch_size=128):
    outputs = []
    with torch.no_grad():
        for batch in x.split(batch_size):
            outputs.append(model(batch.to(device)).cpu())
    logits = torch.cat(outputs)
    predicted = logits.argmax(dim=1).numpy()
    probabilities = logits.softmax(dim=1).numpy()
    metrics = classification_metrics(y.numpy(), predicted, scores=probabilities, n_classes=4)
    return metrics, predicted


def load_subjects(ids):
    xs, ys, groups = [], [], []
    for subject in ids:
        with np.load(cache_dir / f"S{int(subject):03d}.npz", allow_pickle=False) as data:
            x, y = data["x"], data["y"]
        assert x.shape == (84, 64, 1, 480)
        assert np.array_equal(np.bincount(y, minlength=4), [21, 21, 21, 21])
        xs.append(torch.from_numpy(x.astype(np.float32, copy=False)))
        ys.append(torch.from_numpy(y.astype(np.int64, copy=False)))
        groups.extend([int(subject)] * len(y))
    return torch.cat(xs), torch.cat(ys), np.asarray(groups)


metadata = load_metadata(work / "evoapprox8b")
ids = select_circuits(metadata, 50)
assert len(metadata) == 500 and len(ids) == 50 and len(set(ids)) == 50
luts = {name: build_lut(name, work / "evoapprox8b", metadata) for name in ids}
circuit_metrics_by_id = {name: circuit_error_metrics(lut) for name, lut in luts.items()}
catalog_rows = []
for name in ids:
    params = metadata[name]["params"]
    catalog_rows.append({"multiplier_id": name, "published_ER_pct": params["ep%"],
                         "published_MRED_pct": params["mre%"], "published_MAE_pct": params["mae%"],
                         "published_power": params["pwr"], "source_commit": EVOAPPROX_COMMIT})
catalog = pd.DataFrame(catalog_rows)
catalog.to_csv(work / "selected_multiplier_catalog.csv", index=False)
for name in ("mul8_051", ids[0], ids[-1]):
    errors = circuit_metrics_by_id[name]
    params = metadata[name]["params"]
    assert abs(errors["ER"] * 100 - float(params["ep%"])) < .15
    assert abs(errors["MRED"] * 100 - float(params["mre%"])) < .15
    print("Circuit metric check:", name, errors["ER"], errors["MRED"], flush=True)

all_rows, all_subject_rows = [], []
fold_summaries = []
layers = ("temporal", "spatial", "separable", "dense")
overall_started = time.time()
for fold in range(5):
    prior_matrix = work / f"sensitivity_matrix_eegmmidb_fold{fold}.csv"
    prior_subjects = work / f"per_subject_sensitivity_eegmmidb_fold{fold}.csv"
    journal_path = work / f"sensitivity_matrix_eegmmidb_fold{fold}.journal.jsonl"
    if prior_matrix.is_file() and prior_subjects.is_file():
        prior = pd.read_csv(prior_matrix)
        prior_sub = pd.read_csv(prior_subjects)
        if (len(prior) == 200 and len(prior_sub) == 21 * 4 * 50
                and set(prior.fold.astype(int)) == {fold}
                and prior.groupby("layer").multiplier_id.nunique().eq(50).all()):
            all_rows.extend(prior.to_dict("records"))
            all_subject_rows.extend(prior_sub.to_dict("records"))
            fold_summaries.append({"fold": fold, "status": "reused_completed_fold",
                                   "test_subjects": 21, "n_validation": 1764,
                                   "elapsed_minutes_total": (time.time() - overall_started) / 60})
            print(f"Reusing verified fold {fold} CSVs; no GPU evaluations repeated.", flush=True)
            continue
        print(f"Fold {fold} output is incomplete; recomputing this fold.", flush=True)
    fold_result_path = result_dir / f"canonical_full_cohort_fold{fold}_result.json"
    fold_model_path = result_dir / f"canonical_full_cohort_fold{fold}_model.pt"
    assert fold_result_path.is_file() and fold_model_path.is_file()
    fold_result = json.loads(fold_result_path.read_text())
    train_ids = fold_result["train_subjects"]
    test_ids = fold_result["validation_subjects"]
    assert len(train_ids) == 84 and len(test_ids) == 21 and not set(train_ids) & set(test_ids)
    train_x, _, _ = load_subjects(train_ids)
    test_x, test_y, test_groups = load_subjects(test_ids)
    saved = torch.load(fold_model_path, map_location="cpu", weights_only=True)
    assert saved["run_id"] == fold_result["run_id"] and saved["fold"] == fold
    model = PaperAlignedEEGNet82(64, 480, 4).to(device).eval()
    model.load_state_dict(saved["model"], strict=True)
    fp32, _ = score(model, test_x, test_y)
    assert abs(fp32["accuracy"] - fold_result["accuracy"]) < 1e-12
    calibration = calibrate_model(model, train_x, seed=42)
    scales = {key: value["activation_scale"] for key, value in calibration["layers"].items()}
    journal_records = read_case_journal(journal_path)
    fold_rows = [record["result"] for record in journal_records]
    fold_subject_rows = [subject_row for record in journal_records for subject_row in record["subject_rows"]]
    completed = {(str(record["layer"]), str(record["multiplier_id"])) for record in journal_records}
    for layer in layers:
        exact_model = with_layer_lut(model, layer, exact_unsigned_lut(), scales)
        exact_metrics, exact_pred = score(exact_model, test_x, test_y)
        exact_subject = {
            int(sid): float(np.mean(exact_pred[test_groups == sid] == test_y.numpy()[test_groups == sid]))
            for sid in test_ids
        }
        del exact_model
        torch.cuda.empty_cache()
        for index, name in enumerate(ids, 1):
            if (layer, name) in completed:
                continue
            lut = luts[name]
            approx_model = with_layer_lut(model, layer, lut, scales)
            approx_metrics, approx_pred = score(approx_model, test_x, test_y)
            circuit_errors = circuit_metrics_by_id[name]
            row = {"dataset": "eegmmidb", "model": "wang_eegnet82_128", "fold": fold,
                   "seed": 42, "n_subjects_test": len(test_ids), "n_validation": len(test_y),
                   "layer": layer, "multiplier_id": name, "ER": circuit_errors["ER"],
                   "MRED": circuit_errors["MRED"], "MAE": circuit_errors["MAE"],
                   "delta_acc": approx_metrics["accuracy"] - exact_metrics["accuracy"],
                   "delta_f1": approx_metrics["macro_f1"] - exact_metrics["macro_f1"],
                   "delta_sensitivity": approx_metrics["macro_TPR"] - exact_metrics["macro_TPR"],
                   "exact_int8_acc": exact_metrics["accuracy"], "approx_acc": approx_metrics["accuracy"],
                   "baseline_status": "canonical_full_cohort"}
            case_subject_rows = []
            for sid in test_ids:
                mask = test_groups == int(sid)
                approx_acc = float(np.mean(approx_pred[mask] == test_y.numpy()[mask]))
                case_subject_rows.append({"fold": fold, "subject": int(sid), "layer": layer,
                                          "multiplier_id": name, "exact_acc": exact_subject[int(sid)],
                                          "approx_acc": approx_acc,
                                          "delta_acc": approx_acc - exact_subject[int(sid)]})
            append_case_journal(journal_path, {"layer": layer, "multiplier_id": name,
                                               "result": row, "subject_rows": case_subject_rows})
            fold_rows.append(row)
            fold_subject_rows.extend(case_subject_rows)
            completed.add((layer, name))
            if index % 10 == 0 or index == 50:
                print(f"Fold {fold} {layer}: {index}/50 circuits", flush=True)
            del approx_model
            torch.cuda.empty_cache()
    frame = pd.DataFrame(fold_rows)
    subject_frame = pd.DataFrame(fold_subject_rows)
    frame.to_csv(work / f"sensitivity_matrix_eegmmidb_fold{fold}.csv", index=False)
    subject_frame.to_csv(work / f"per_subject_sensitivity_eegmmidb_fold{fold}.csv", index=False)
    all_rows.extend(fold_rows)
    all_subject_rows.extend(fold_subject_rows)
    fold_summaries.append({"fold": fold, "test_subjects": len(test_ids), "n_validation": len(test_y),
                           "fp32_accuracy": fp32["accuracy"], "exact_layer_mean_accuracy": frame.exact_int8_acc.mean(),
                           "elapsed_minutes_total": (time.time() - overall_started) / 60})
    pd.DataFrame(all_rows).to_csv(work / "sensitivity_matrix_eegmmidb_all_folds.partial.csv", index=False)
    pd.DataFrame(all_subject_rows).to_csv(work / "per_subject_sensitivity_eegmmidb_all_folds.partial.csv", index=False)
    print(f"Saved fold {fold}: 200 evaluations; elapsed {(time.time()-overall_started)/60:.1f} min", flush=True)
    del model, train_x, test_x, test_y
    torch.cuda.empty_cache()

matrix = pd.DataFrame(all_rows)
subjects = pd.DataFrame(all_subject_rows)
assert len(matrix) == 1000 and matrix.groupby("fold").size().to_dict() == {i: 200 for i in range(5)}
assert matrix.groupby(["fold", "layer"]).multiplier_id.nunique().eq(50).all()
matrix.to_csv(work / "sensitivity_matrix_eegmmidb_all_folds.csv", index=False)
subjects.to_csv(work / "per_subject_sensitivity_eegmmidb_all_folds.csv", index=False)

# Fold-level mean ± SD per circuit/layer is descriptive; paired subject-level deltas
# are used for Wilcoxon tests to avoid treating trials as independent observations.
aggregate = matrix.groupby(["layer", "multiplier_id"], as_index=False).agg(
    mean_delta_acc=("delta_acc", "mean"), std_delta_acc=("delta_acc", "std"),
    mean_delta_f1=("delta_f1", "mean"), std_delta_f1=("delta_f1", "std"),
    mean_delta_sensitivity=("delta_sensitivity", "mean"), std_delta_sensitivity=("delta_sensitivity", "std"),
    ER=("ER", "mean"), MRED=("MRED", "mean"), MAE=("MAE", "mean"))
aggregate.to_csv(work / "sensitivity_aggregate_mean_std_eegmmidb.csv", index=False)
fold_layer = matrix.assign(abs_delta_acc=lambda d: d.delta_acc.abs()).groupby(["fold", "layer"], as_index=False).agg(
    mean_abs_delta_acc=("abs_delta_acc", "mean"), mean_delta_acc=("delta_acc", "mean"))
fold_layer.to_csv(work / "fold_layer_sensitivity_ranking.csv", index=False)
ranking = fold_layer.pivot(index="fold", columns="layer", values="mean_abs_delta_acc")
ranking.to_csv(work / "fold_layer_ranking.csv")

# Subject-level paired exact-versus-approx tests. Apply BH-FDR over all 200 planned tests.
tests = []
for (layer, name), part in subjects.groupby(["layer", "multiplier_id"]):
    delta = part.delta_acc.to_numpy()
    if np.allclose(delta, 0):
        stat, p_value = 0.0, 1.0
    else:
        result = wilcoxon(delta, alternative="two-sided", zero_method="wilcox", method="auto")
        stat, p_value = float(result.statistic), float(result.pvalue)
    tests.append({"layer": layer, "multiplier_id": name, "n_subjects": len(delta),
                  "median_subject_delta_acc": float(np.median(delta)), "wilcoxon_statistic": stat,
                  "wilcoxon_p": p_value})
tests = pd.DataFrame(tests)
tests["wilcoxon_q_bh"] = multipletests(tests.wilcoxon_p, method="fdr_bh")[1]
tests["significant_fdr_0_05"] = tests.wilcoxon_q_bh < .05
tests.to_csv(work / "wilcoxon_eegmmidb_subject_paired.csv", index=False)

# Correlations use 200 distinct layer/circuit means, not 1,000 fold rows as independent circuits.
# Two-way cluster bootstrap resamples the five subject folds and 50 circuits, then
# recomputes the means; this carries both fold and circuit variability into intervals.
correlations = []
bootstrap_rng = np.random.default_rng(20260927)
fold_ids = np.sort(matrix.fold.unique())
circuit_ids = np.sort(matrix.multiplier_id.unique())
layer_ids = sorted(matrix.layer.unique())
assert len(fold_ids) == 5 and len(circuit_ids) == 50 and len(layer_ids) == 4
for metric in ("ER", "MRED", "MAE"):
    for outcome in ("mean_delta_acc", "mean_delta_f1"):
        x = aggregate[metric].to_numpy()
        y = aggregate[outcome].to_numpy()
        r = pearsonr(x, y)
        rho = spearmanr(x, y)
        scopes = ["pooled", *layer_ids]
        for scope in scopes:
            part = aggregate if scope == "pooled" else aggregate[aggregate.layer == scope]
            x_part, y_part = part[metric].to_numpy(), part[outcome].to_numpy()
            pr, sr = pearsonr(x_part, y_part), spearmanr(x_part, y_part)
            boot = []
            for _ in range(2000):
                sampled_folds = bootstrap_rng.choice(fold_ids, size=len(fold_ids), replace=True)
                sampled_circuits = bootstrap_rng.choice(circuit_ids, size=len(circuit_ids), replace=True)
                chunks = []
                for circuit_position, circuit_id in enumerate(sampled_circuits):
                    for fold_id in sampled_folds:
                        chunk = matrix.loc[(matrix.fold == fold_id) & (matrix.multiplier_id == circuit_id),
                                           ["layer", metric, outcome.replace("mean_", "")]].copy()
                        chunk["bootstrap_circuit_position"] = circuit_position
                        chunks.append(chunk)
                fold_circuit = pd.concat(chunks, ignore_index=True)
                draw = fold_circuit.groupby(["bootstrap_circuit_position", "layer"], as_index=False).agg(
                    metric_value=(metric, "mean"), outcome_value=(outcome.replace("mean_", ""), "mean"))
                if scope != "pooled":
                    draw = draw[draw.layer == scope]
                if draw.metric_value.nunique() > 1 and draw.outcome_value.nunique() > 1:
                    boot.append((pearsonr(draw.metric_value, draw.outcome_value).statistic,
                                 spearmanr(draw.metric_value, draw.outcome_value).statistic))
            boot = np.asarray(boot)
            correlations.append({"scope": scope, "metric": metric, "outcome": outcome, "n": len(x_part),
                                 "pearson_r": pr.statistic, "pearson_p": pr.pvalue,
                                 "pearson_ci_low": float(np.quantile(boot[:, 0], .025)),
                                 "pearson_ci_high": float(np.quantile(boot[:, 0], .975)),
                                 "spearman_rho": sr.statistic, "spearman_p": sr.pvalue,
                                 "spearman_ci_low": float(np.quantile(boot[:, 1], .025)),
                                 "spearman_ci_high": float(np.quantile(boot[:, 1], .975)),
                                 "bootstrap_draws": len(boot), "bootstrap_unit": "fold_and_circuit"})
correlations = pd.DataFrame(correlations)
correlations.to_csv(work / "error_metric_correlations_pooled_5fold.csv", index=False)

plots = work / "plots"
plots.mkdir(exist_ok=True)
fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
for layer, part in fold_layer.groupby("layer"):
    ax.plot(part.fold, part.mean_abs_delta_acc * 100, marker="o", label=layer)
ax.set(xlabel="Held-out subject fold", ylabel="Mean absolute accuracy change (pp)",
       title="EEGMMIDB approximate-MAC layer sensitivity by fold")
ax.grid(alpha=.25); ax.legend(frameon=False)
fig.savefig(plots / "layer_sensitivity_by_fold.png", dpi=300); plt.close(fig)

heat = aggregate.pivot(index="multiplier_id", columns="layer", values="mean_delta_acc") * 100
fig, ax = plt.subplots(figsize=(8, 15), constrained_layout=True)
im = ax.imshow(heat.to_numpy(), aspect="auto", cmap="coolwarm", vmin=-4, vmax=4)
ax.set(xticks=range(len(heat.columns)), xticklabels=heat.columns, yticks=range(len(heat.index)),
       yticklabels=heat.index, title="Pooled 5-fold mean ΔAccuracy by layer and multiplier", ylabel="Multiplier")
fig.colorbar(im, ax=ax, label="ΔAccuracy (percentage points)")
fig.savefig(plots / "pooled_5fold_sensitivity_heatmap.png", dpi=300); plt.close(fig)

summary_layer = aggregate.assign(abs_mean=lambda d: d.mean_delta_acc.abs()).groupby("layer").agg(
    mean_abs_delta_acc=("abs_mean", "mean"), mean_delta_acc=("mean_delta_acc", "mean"),
    std_fold_delta_acc=("std_delta_acc", "mean")).sort_values("mean_abs_delta_acc", ascending=False)
summary_layer.to_csv(work / "pooled_layer_sensitivity_summary.csv")
fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
ax.bar(summary_layer.index, summary_layer.mean_abs_delta_acc * 100,
       yerr=summary_layer.std_fold_delta_acc * 100, capsize=4, color="#2878B5")
ax.set(ylabel="Mean |ΔAccuracy| (pp)", title="Pooled layer sensitivity; error bars show fold SD")
ax.grid(axis="y", alpha=.25)
fig.savefig(plots / "pooled_layer_sensitivity_summary.png", dpi=300); plt.close(fig)

for metric in ("ER", "MRED", "MAE"):
    for outcome, label in (("mean_delta_acc", "Mean ΔAccuracy"), ("mean_delta_f1", "Mean ΔMacro-F1")):
        fig, ax = plt.subplots(figsize=(6, 4.5), constrained_layout=True)
        for layer, part in aggregate.groupby("layer"):
            ax.scatter(part[metric], part[outcome] * 100, s=20, alpha=.7, label=layer)
        ax.axhline(0, color="black", linewidth=.8)
        ax.set(xlabel=metric, ylabel=f"{label} (pp)", title=f"Pooled five-fold: {metric} vs {label}")
        ax.grid(alpha=.2); ax.legend(frameon=False)
        fig.savefig(plots / f"{metric}_vs_{outcome}.png", dpi=300); plt.close(fig)

pd.DataFrame(fold_summaries).to_csv(work / "fold_execution_audit.csv", index=False)
print("FOLD RANKING (mean absolute Δaccuracy, percentage points):")
print((ranking * 100).round(3).to_string())
print("POOLED LAYER SUMMARY:")
print((summary_layer * 100).round(3).to_string())
print("Wilcoxon significant after BH-FDR, by layer:")
print(tests.groupby("layer").significant_fdr_0_05.sum().to_string())
print("CORRELATIONS:")
print(correlations.query("scope == 'pooled_5fold_circuit_layer_means'").round(4).to_string(index=False))
print("Artifacts:", work, "| total elapsed minutes:", round((time.time()-overall_started)/60, 2), flush=True)
