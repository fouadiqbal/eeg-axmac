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


def score(model, x, y, batch_size=32):
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


# Full held-out-set confirmation of three finalists selected from the coarse five-fold screen.
metadata = load_metadata(work / "evoapprox8b")
ids = ("mul8_348", "mul8_112", "mul8_424")
luts = {name: build_lut(name, work / "evoapprox8b", metadata) for name in ids}
assert all(lut.shape == (256,256) for lut in luts.values())
final_work = work / "finalists_full_test"
final_work.mkdir(parents=True, exist_ok=True)
journal = final_work / "case_journal.jsonl"
prior = read_case_journal(journal)
completed = {(int(rec["result"]["fold"]),rec["layer"],rec["multiplier_id"]) for rec in prior}
print("RESUME",len(completed),"of",5*4*len(ids),flush=True)
for fold in range(5):
    result_path = result_dir / f"canonical_full_cohort_fold{fold}_result.json"
    model_path = result_dir / f"canonical_full_cohort_fold{fold}_model.pt"
    info = json.loads(result_path.read_text())
    train_ids,test_ids = info["train_subjects"],info["validation_subjects"]
    assert len(train_ids)==84 and len(test_ids)==21 and not set(train_ids)&set(test_ids)
    train_x,_,_=load_subjects(train_ids)
    test_x,test_y,test_groups=load_subjects(test_ids)
    saved=torch.load(model_path,map_location="cpu",weights_only=True)
    assert saved["fold"]==fold and saved["run_id"]==info["run_id"]
    model=PaperAlignedEEGNet82(64,480,4).to(device).eval()
    model.load_state_dict(saved["model"],strict=True)
    fp32,_=score(model,test_x,test_y)
    assert abs(fp32["accuracy"]-info["accuracy"])<1e-12
    calibration=calibrate_model(model,train_x,seed=42)
    scales={key:value["activation_scale"] for key,value in calibration["layers"].items()}
    for layer in ("temporal","spatial","separable","dense"):
        exact_model=with_layer_lut(model,layer,exact_unsigned_lut(),scales)
        exact,exact_pred=score(exact_model,test_x,test_y)
        del exact_model
        torch.cuda.empty_cache()
        for name in ids:
            if (fold,layer,name) in completed: continue
            approx_model=with_layer_lut(model,layer,luts[name],scales)
            approx,pred=score(approx_model,test_x,test_y)
            err=circuit_error_metrics(luts[name])
            row={"dataset":"eegmmidb","scope":"full_heldout_test","fold":fold,"layer":layer,
                 "multiplier_id":name,"n_test":len(test_y),"n_subjects_test":len(test_ids),
                 "exact_int8_acc":exact["accuracy"],"approx_acc":approx["accuracy"],
                 "delta_acc":approx["accuracy"]-exact["accuracy"],
                 "delta_f1":approx["macro_f1"]-exact["macro_f1"],
                 "delta_sensitivity":approx["macro_TPR"]-exact["macro_TPR"],
                 "ER":err["ER"],"MRED":err["MRED"],"MAE":err["MAE"],
                 "baseline_status":"canonical_full_cohort","run_id":info["run_id"]}
            paired=[]
            y=test_y.numpy()
            for sid in test_ids:
                mask=test_groups==int(sid)
                paired.append({"fold":fold,"subject":int(sid),"layer":layer,"multiplier_id":name,
                               "exact_correct":int(np.sum(exact_pred[mask]==y[mask])),
                               "approx_correct":int(np.sum(pred[mask]==y[mask])),"n_test":int(mask.sum())})
            append_case_journal(journal,{"layer":layer,"multiplier_id":name,"result":row,"subject_rows":paired})
            completed.add((fold,layer,name))
            print("FULL",fold,layer,name,"DONE",len(completed),"of",60,"DELTA_ACC",round(row["delta_acc"],6),flush=True)
            del approx_model
            torch.cuda.empty_cache()
    del model,train_x,test_x
    torch.cuda.empty_cache()
records=read_case_journal(journal)
assert len(records)==60 and len({(r["result"]["fold"],r["layer"],r["multiplier_id"]) for r in records})==60
matrix=pd.DataFrame([r["result"] for r in records])
subjects=pd.DataFrame([s for r in records for s in r["subject_rows"]])
assert len(subjects)==60*21 and matrix.groupby(["fold","layer"]).size().eq(3).all()
matrix.to_csv(final_work/"sensitivity_matrix_finalists_full_test.csv",index=False)
subjects.to_csv(final_work/"paired_subjects_finalists_full_test.csv",index=False)
aggregate=matrix.groupby(["layer","multiplier_id"],as_index=False).agg(
    mean_delta_acc=("delta_acc","mean"),std_delta_acc=("delta_acc","std"),
    mean_delta_f1=("delta_f1","mean"),std_delta_f1=("delta_f1","std"))
aggregate.to_csv(final_work/"finalists_full_test_mean_std.csv",index=False)
print("FULL_TEST_FINALISTS_COMPLETE",len(matrix),"cases",len(subjects),"paired subjects")
print(aggregate.to_string(index=False),flush=True)
