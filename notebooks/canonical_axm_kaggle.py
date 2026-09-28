"""Kaggle-only inference sweep using attached canonical full-cohort output."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

assert torch.cuda.is_available(), "Select a Kaggle T4 GPU"
print("GPU:", torch.cuda.get_device_name(0), "gcc:", subprocess.check_output(["gcc", "--version"], text=True).splitlines()[0], flush=True)
roots = list(Path("/kaggle/input").rglob("canonical_full_cohort_summary.json"))
assert len(roots) == 1, f"Expected one attached canonical baseline, found {roots}"
result_dir = roots[0].parent
input_root = result_dir.parent
cache_dir = input_root / "eeg_axmac_wang_full_cohort" / "subject_cache"
summary = json.loads(roots[0].read_text())
assert summary["n_subjects"] == 105 and summary["folds"] == 5 and summary["epochs"] == 100
assert summary["verdict"] == "verified_within_band"
assert len(list(cache_dir.glob("S*.npz"))) == 105
for fold in range(5):
    assert (result_dir / f"canonical_full_cohort_fold{fold}_model.pt").is_file()
    assert (result_dir / f"canonical_full_cohort_fold{fold}_result.json").is_file()
print("Accepted source:", roots[0], "cached subjects:", len(list(cache_dir.glob("S*.npz"))), flush=True)

work = Path("/kaggle/working/eeg_axmac_sweep")
work.mkdir(parents=True, exist_ok=True)
repo = work / "source"
commit = "7a2d0215b48dc449add75fe59226948f7080b3b5"
if not repo.exists():
    subprocess.run(["git", "clone", "-q", "https://github.com/fouadiqbal/eeg-axmac.git", str(repo)], check=True)
subprocess.run(["git", "-C", str(repo), "checkout", "-q", commit], check=True)
print("Source commit:", subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip(), flush=True)

# The published repository still has the superseded BN layout. Patch only the
# inference architecture to match the executed canonical checkpoint exactly.
model_path = repo / "src/models/eegnet.py"
source = model_path.read_text()
old = """        self.bn1 = nn.BatchNorm2d(8)"""
assert source.count(old) == 1
source = source.replace(old, "        self.bn1 = nn.BatchNorm2d(n_channels)")
for channels in ("bn2", "bn3"):
    old = f"        self.{channels} = nn.BatchNorm2d(16)"
    assert source.count(old) == 1
    source = source.replace(old, f"        self.{channels} = nn.BatchNorm2d(1)")
old = """        x = self.bn1(self.temporal(x))
        x = self.drop1(self.pool1(torch.nn.functional.elu(self.bn2(self.spatial(x)))))
        x = self.sep_point(self.sep_depth(x))
        x = self.drop2(self.pool2(torch.nn.functional.elu(self.bn3(x))))"""
new = """        x = self.temporal(x)
        x = self.bn1(x.transpose(1, 2)).transpose(1, 2)
        x = self.spatial(x)
        x = self.bn2(x.transpose(1, 2)).transpose(1, 2)
        x = self.drop1(self.pool1(torch.nn.functional.elu(x)))
        x = self.sep_point(self.sep_depth(x))
        x = self.bn3(x.transpose(1, 2)).transpose(1, 2)
        x = self.drop2(self.pool2(torch.nn.functional.elu(x)))"""
assert source.count(old) == 1
model_path.write_text(source.replace(old, new))
subprocess.run([sys.executable, "-m", "pytest", "-q", "tests/test_axm_correctness.py"], cwd=repo, check=True)
sys.path.insert(0, str(repo))

from src.axm.evoapprox_loader import EVOAPPROX_COMMIT, build_lut, load_metadata, select_circuits
from src.axm.metrics import circuit_error_metrics
from src.models.eegnet import PaperAlignedEEGNet82

# The pinned repository revision predates the model/mode keyword parameters
# in the local helper. Adapt only the call signature; its pack/checkpoint
# already bind the selected EEGNet, class count, fold, and global protocol.
from src.train import sensitivity_sweep as sweep_module

# Adapt the older pinned helper without replacing Python's global import hook.
# The marker makes rerunning an interrupted notebook cell safe in the same kernel.
_original_sweep_fold = getattr(
    sweep_module.sweep_fold, "_compat_original", sweep_module.sweep_fold
)
def _compatible_sweep_fold(*args, model_name=None, n_classes=None, mode=None,
                           subject=None, subject_fold=None, **kwargs):
    kwargs["batch_size"] = 32
    return _original_sweep_fold(*args, **kwargs)
_compatible_sweep_fold._compat_original = _original_sweep_fold
sweep_module.sweep_fold = _compatible_sweep_fold
evaluate, sweep_fold = sweep_module.evaluate, sweep_module.sweep_fold

lib = work / "evoapprox8b"
metadata = load_metadata(lib)
names = select_circuits(metadata, 50)
assert len(metadata) == 500 and len(names) == 50 and len(set(names)) == 50
published = []
for name in names:
    p = metadata[name]["params"]
    published.append({"multiplier_id": name, "published_ER_pct": p["ep%"],
                      "published_MRED_pct": p["mre%"], "published_MAE_pct": p["mae%"],
                      "published_power": p["pwr"], "source_commit": EVOAPPROX_COMMIT})
pd.DataFrame(published).to_csv(work / "selected_multiplier_catalog.csv", index=False)
for name in ("mul8_051", names[0], names[-1]):
    errors = circuit_error_metrics(build_lut(name, lib, metadata))
    p = metadata[name]["params"]
    assert abs(errors["ER"] * 100 - float(p["ep%"])) < .15
    assert abs(errors["MRED"] * 100 - float(p["mre%"])) < .15
    print("Circuit spot check:", name, errors["ER"], errors["MRED"], errors["MAE"], flush=True)

fold = 0  # fixed representative fold; no model selection on test accuracy
row = json.loads((result_dir / f"canonical_full_cohort_fold{fold}_result.json").read_text())
train_ids, test_ids = row["train_subjects"], row["validation_subjects"]
assert len(train_ids) == 84 and len(test_ids) == 21 and not set(train_ids) & set(test_ids)

def load_subjects(ids):
    xx, yy = [], []
    for sid in ids:
        with np.load(cache_dir / f"S{sid:03d}.npz", allow_pickle=False) as d:
            x, y = d["x"], d["y"]
        assert x.shape == (84, 64, 1, 480) and np.array_equal(np.bincount(y, minlength=4), [21]*4)
        xx.append(torch.from_numpy(x.astype(np.float32, copy=False)))
        yy.append(torch.from_numpy(y.astype(np.int64, copy=False)))
    return torch.cat(xx), torch.cat(yy)

train_x, train_y = load_subjects(train_ids)
test_x, test_y = load_subjects(test_ids)
saved = torch.load(result_dir / f"canonical_full_cohort_fold{fold}_model.pt", map_location="cpu", weights_only=True)
assert saved["run_id"] == row["run_id"] and saved["fold"] == fold
model = PaperAlignedEEGNet82(64, 480, 4).cuda()
model.load_state_dict(saved["model"], strict=True)
replay = evaluate(model, test_x, test_y, 32, torch.device("cuda"))
assert abs(replay["accuracy"] - row["accuracy"]) < 1e-12
assert abs(replay["macro_f1"] - row["macro_f1"]) < 1e-12
print("Exact fold-0 replay:", replay["accuracy"], replay["macro_f1"], "held-out windows:", len(test_y), flush=True)

Path("/kaggle/temp").mkdir(parents=True, exist_ok=True)
pack_path = Path("/kaggle/temp/canonical_fold0_pack.pt")
torch.save({"fold": 0, "dataset": "eegmmidb", "train_x": train_x,
            "train_y4": train_y, "validation_x": test_x, "validation_y4": test_y}, pack_path)
checkpoint_path = Path("/kaggle/temp/canonical_fold0_checkpoint.pt")
torch.save({"model": saved["model"], "fold": 0, "dataset": "eegmmidb",
            "model_name": "eegnet82_128", "classes": 4, "mode": "global"}, checkpoint_path)
del train_x, train_y, test_x, test_y, model
torch.cuda.empty_cache()
matrix = sweep_fold(pack_path, checkpoint_path, work, lib, count=50, batch_size=8,
                    model_name="eegnet82_128", n_classes=4, mode="global")
df = pd.read_csv(matrix)
assert len(df) == 200 and df.groupby("layer").multiplier_id.nunique().to_dict() == {
    "temporal": 50, "spatial": 50, "separable": 50, "dense": 50}
df.to_csv(work / "sensitivity_matrix_eegmmidb.csv", index=False)
print("Complete sensitivity matrix:", matrix, "rows:", len(df), flush=True)

import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
plots = work / "plots"
plots.mkdir(exist_ok=True)
outcomes = ("delta_acc", "delta_f1", "delta_sensitivity")
errors = ("ER", "MRED", "MAE", "ME", "AME")
correlations = []
for layer, part in [("pooled", df), *list(df.groupby("layer"))]:
    for error in errors:
        for outcome in outcomes:
            valid = part[[error, outcome]].dropna()
            status = "measured"
            r = p = rho = sp = np.nan
            if error == "AME":
                status = "architecture propagation model unavailable"
            elif len(valid) < 3 or valid[error].nunique() < 2 or valid[outcome].nunique() < 2:
                status = "insufficient variation"
            else:
                pr = pearsonr(valid[error], valid[outcome])
                sr = spearmanr(valid[error], valid[outcome])
                r, p, rho, sp = pr.statistic, pr.pvalue, sr.statistic, sr.pvalue
            correlations.append({"layer": layer, "error_metric": error, "outcome": outcome,
                                 "n": len(valid), "pearson_r": r, "pearson_p": p,
                                 "spearman_rho": rho, "spearman_p": sp, "status": status})
pd.DataFrame(correlations).to_csv(work / "correlations_eegmmidb.csv", index=False)
for error in errors[:-1]:
    for outcome in outcomes:
        fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
        for layer, part in df.groupby("layer"):
            ax.scatter(part[error], part[outcome]*100, s=25, alpha=.7, label=layer)
        ax.axhline(0, color="black", lw=.8)
        ax.set(xlabel=error, ylabel=outcome + " change (percentage points)",
               title="EEGMMIDB fold 0: " + error + " vs " + outcome)
        ax.legend(frameon=False)
        ax.grid(alpha=.2)
        fig.savefig(plots / f"{error}_vs_{outcome}.png", dpi=300)
        plt.close(fig)
print("Pooled accuracy correlations:")
print(pd.DataFrame(correlations).query("layer == 'pooled' and outcome == 'delta_acc'").to_string(index=False))
print("Layer accuracy changes (percentage points):")
print((df.groupby("layer").delta_acc.agg(["mean", "median", "min", "max"])*100).to_string())
