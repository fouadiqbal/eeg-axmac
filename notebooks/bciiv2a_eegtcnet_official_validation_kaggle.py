"""Validate the released fixed EEG-TCNet pipeline on BCI IV 2a T->E sessions.

Kaggle-only inference script. Attach a BCI IV 2a dataset containing the official
A01T.mat/A01E.mat ... A09T.mat/A09E.mat files before running. This does not fit
or fine-tune the released classifiers.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import hashlib
import h5py
import tf_keras as keras
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score

assert Path("/kaggle").exists(), "Run only in Kaggle; do not execute model inference on this PC."

work = Path("/kaggle/working/bciiv2a_eegtcnet_validation")
work.mkdir(parents=True, exist_ok=True)
repo = work / "eeg-tcnet"
if not repo.exists():
    subprocess.run(["git", "clone", "-q", "https://github.com/iis-eth-zurich/eeg-tcnet.git", str(repo)], check=True)
assert subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip() == "363c7e52d19285b0bac9a8846bbdbd6aac5aa851"
sys.path.insert(0, str(repo))
from utils.data_loading import prepare_features  # noqa: E402

# Resolve all official competition files from attached Kaggle inputs and stage
# them in the directory layout expected by the released loader.
input_roots = list(Path("/kaggle/input").glob("*")) + [work / "data"]
source_files = {}
for subject in range(1, 10):
    for session in ("T", "E"):
        filename = f"A{subject:02d}{session}.mat"
        matches = [path for root in input_roots for path in root.rglob(filename)]
        assert len(matches) == 1, f"Expected one attached {filename}, found {matches}"
        source_files[filename] = matches[0]

data_root = work / "data"
for subject in range(1, 10):
    subject_dir = data_root / f"s{subject}"
    subject_dir.mkdir(parents=True, exist_ok=True)
    for session in ("T", "E"):
        filename = f"A{subject:02d}{session}.mat"
        target = subject_dir / filename
        if not target.exists():
            target.symlink_to(source_files[filename])

class SliceChannel(keras.layers.Layer):
    def call(self, x):
        return x[:, :, -1, :]

class LastTime(keras.layers.Layer):
    def call(self, x):
        return x[:, -1, :]

def load_released_model(path):
    # Replace only Python-version-specific serialized Lambda bytecode with
    # the two identical slice operations in official utils/models.py.
    with h5py.File(path) as h:
        config = json.loads(h.attrs["model_config"])
    lambdas = [l for l in config["config"]["layers"] if l["class_name"] == "Lambda"]
    assert len(lambdas) == 2
    for layer, cls in zip(lambdas, ["SliceChannel", "LastTime"]):
        layer["class_name"] = cls
        layer["config"] = {k: v for k, v in layer["config"].items() if k in ["name", "trainable", "dtype"]}
    model = keras.models.model_from_json(json.dumps(config), custom_objects={"SliceChannel": SliceChannel, "LastTime": LastTime})
    model.load_weights(str(path))
    return model

records = []
for subject in range(1, 10):
    pipeline_path = repo / "models" / "EEG-TCNet" / f"S{subject}" / "model_fixed.h5"
    assert pipeline_path.is_file(), f"Missing released fixed pipeline: {pipeline_path}"
    pipeline = load_released_model(pipeline_path)
    data_path = str(data_root / f"s{subject}") + "/"
    X_train, _, _, X_test, _, y_test_onehot = prepare_features(data_path, subject - 1, False)
    y_true = np.argmax(y_test_onehot, axis=1)
    # Official Scaler fits each channel/time feature over training trials only.
    for channel in range(22):
        scaler = StandardScaler().fit(X_train[:, 0, channel, :])
        X_test[:, 0, channel, :] = scaler.transform(X_test[:, 0, channel, :])
    y_pred = pipeline.predict(X_test, batch_size=32, verbose=0).argmax(axis=1)
    records.append({
        "subject": subject,
        "n_train_trials": int(len(X_train)),
        "n_test_trials": int(len(X_test)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
        "model": "released EEG-TCNet fixed",
        "protocol": "official BCI IV 2a training session T -> evaluation session E",
        "checkpoint": str(pipeline_path),
        "checkpoint_sha256": hashlib.sha256(pipeline_path.read_bytes()).hexdigest(),
    })
    pd.DataFrame(records).to_csv(work / "bciiv2a_eegtcnet_fixed_subject_results.partial.csv", index=False)
    print(records[-1], flush=True)
    keras.backend.clear_session()

results = pd.DataFrame(records)
mean_accuracy = float(results.accuracy.mean())
reference = 0.7735
summary = {
    "dataset": "BCI Competition IV 2a",
    "model": "released EEG-TCNet fixed",
    "protocol": "official BCI IV 2a T -> E session split, nine subjects",
    "n_subjects": int(len(results)),
    "mean_accuracy": mean_accuracy,
    "std_accuracy_subjects": float(results.accuracy.std(ddof=1)),
    "published_reference_accuracy": reference,
    "difference_percentage_points": (mean_accuracy - reference) * 100,
    "within_plus_minus_3pp": abs(mean_accuracy - reference) <= 0.03,
    "status": "verified_within_band" if abs(mean_accuracy - reference) <= 0.03 else "outside_reference_band",
    "interpretation": "Fixed released model only; subject-tuned 83.84% reference is not targeted.",
}
results.to_csv(work / "bciiv2a_eegtcnet_fixed_subject_results.csv", index=False)
(work / "bciiv2a_eegtcnet_fixed_summary.json").write_text(json.dumps(summary, indent=2))

fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
ax.bar(results.subject.astype(str), results.accuracy * 100, color="#3478A8")
ax.axhline(reference * 100, color="#C44E52", linestyle="--", label="Published fixed reference: 77.35%")
ax.axhspan((reference - .03) * 100, (reference + .03) * 100, color="#55A868", alpha=.12,
           label="±3 pp acceptance band")
ax.set(xlabel="BCI IV 2a subject", ylabel="Test-session accuracy (%)",
       title="Released EEG-TCNet fixed model: training session T → evaluation session E")
ax.grid(axis="y", alpha=.25)
ax.legend(frameon=False)
fig.savefig(work / "bciiv2a_eegtcnet_fixed_subject_accuracy.png", dpi=300)
plt.close(fig)

print(results.round(4).to_string(index=False))
print(json.dumps(summary, indent=2))
print("Saved files:", sorted(path.name for path in work.iterdir() if path.is_file()))
