"""Training-only signed DWM (Liu et al. 2024, Eq. 4), correlations, figures."""
from __future__ import annotations
import json, sys, zipfile
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr
from torch import nn

WORK = Path('/kaggle/working/eeg_axmac_sweep')
sys.path.insert(0, str(WORK / 'source'))
from src.axm.evoapprox_loader import build_lut, load_metadata
from src.axm.lut_multiplier import quantize_symmetric
from src.models.eegnet import PaperAlignedEEGNet82

matrix_path, = WORK.glob('sensitivity_matrix_*.csv')
matrix = pd.read_csv(matrix_path)
assert len(matrix) == 200 and matrix.multiplier_id.nunique() == 50
pack = torch.load('/kaggle/temp/canonical_fold0_pack.pt', map_location='cpu', weights_only=True)
saved = torch.load('/kaggle/temp/canonical_fold0_checkpoint.pt', map_location='cpu', weights_only=True)
cal_path, = WORK.glob('*calibration*.json')
cal = json.loads(cal_path.read_text())
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = PaperAlignedEEGNet82(64, 480, 4).to(device).eval()
model.load_state_dict(saved['model'], strict=True)
mods = {'temporal': ('temporal',), 'spatial': ('spatial',),
        'separable': ('sep_depth', 'sep_point'), 'dense': ('classifier',)}
rng = np.random.default_rng(42)
train_x = pack['train_x']
n_cal = max(1, round(len(train_x) * .10))
sample = train_x[np.sort(rng.choice(len(train_x), n_cal, replace=False))]
records = {name: [] for names in mods.values() for name in names}
mac_totals = {name: 0 for name in records}
handles = []
def collect(name):
    scale = cal['layers'][name]['activation_scale']
    def hook(_module, inputs):
        values = quantize_symmetric(inputs[0].detach(), scale).reshape(-1).cpu().numpy()
        if len(values) > 4096:
            values = values[rng.choice(len(values), 4096, replace=False)]
        records[name].append(values.astype(np.int16, copy=False))
    return hook
def count_macs(name):
    def hook(module, _inputs, output):
        if isinstance(module, nn.Conv2d):
            per_output = (module.in_channels // module.groups) * int(np.prod(module.kernel_size))
        else:
            per_output = module.in_features
        mac_totals[name] += int(output.numel()) * per_output
    return hook
for name in records:
    handles.append(getattr(model, name).register_forward_pre_hook(collect(name)))
    handles.append(getattr(model, name).register_forward_hook(count_macs(name)))
with torch.no_grad():
    for batch in sample.split(32): model(batch.to(device))
for handle in handles: handle.remove()

q = np.arange(-127, 128, dtype=np.int16)
mag = np.abs(q.astype(np.int32)); sign = np.where(q < 0, -1, 1).astype(np.int32)
pmfs = {}
for name, values in records.items():
    a = np.concatenate(values)
    p = np.bincount(a + 127, minlength=255).astype(float); p /= p.sum()
    layer = getattr(model, name)
    scales = torch.as_tensor(cal['layers'][name]['weight_scales'], device=layer.weight.device)
    view = (len(scales),) + (1,) * (layer.weight.ndim - 1)
    w = quantize_symmetric(layer.weight.detach(), scales.view(view)).reshape(-1).cpu().numpy()
    f = np.bincount(w + 127, minlength=255).astype(float); f /= f.sum()
    nmac = mac_totals[name] / n_cal
    pmfs[name] = (p, f, nmac)

metadata = load_metadata(WORK / 'evoapprox8b')
ids = list(dict.fromkeys(matrix.multiplier_id.tolist()))
assert len(ids) == 50 and all(i in metadata for i in ids)
exact = np.arange(256, dtype=np.int32)[:, None] * np.arange(256, dtype=np.int32)[None, :]
rows = []
for circuit in ids:
    lut = build_lut(circuit, WORK / 'evoapprox8b', metadata).astype(np.int32)
    err = lut - exact
    delta = err[mag[:, None], mag[None, :]] * (sign[:, None] * sign[None, :])
    for layer, names in mods.items():
        score = sum(n * float(p @ delta @ f) for name in names for p, f, n in [pmfs[name]])
        rows.append({'layer': layer, 'multiplier_id': circuit, 'DWM_Eq4_signed_factorized': score,
                     'DWM_calibration_windows': n_cal,
                     'DWM_note': 'training-only operand marginals; independence approximation'})
matrix = matrix.drop(columns=['DWM_Eq4_signed_factorized'], errors='ignore').merge(
    pd.DataFrame(rows), on=['layer', 'multiplier_id'], validate='one_to_one')
matrix.to_csv(matrix_path, index=False)

metrics = ('ER', 'MRED', 'MAE', 'ME', 'DWM_Eq4_signed_factorized', 'AME')
outcomes = ('delta_acc', 'delta_f1', 'delta_sensitivity')
corr = []
for layer, part in [('pooled', matrix), *list(matrix.groupby('layer'))]:
    for metric in metrics:
        for outcome in outcomes:
            v = part[[metric, outcome]].replace([np.inf, -np.inf], np.nan).dropna()
            status, stats = 'measured', [np.nan] * 4
            if metric == 'AME': status = 'requires downstream propagation factors from Eq. 17'
            elif len(v) < 3 or v[metric].nunique() < 2 or v[outcome].nunique() < 2: status = 'insufficient variation'
            else:
                pr, sr = pearsonr(v[metric], v[outcome]), spearmanr(v[metric], v[outcome])
                stats = [pr.statistic, pr.pvalue, sr.statistic, sr.pvalue]
            corr.append({'layer': layer, 'error_metric': metric, 'outcome': outcome,
                         'n': len(v), 'pearson_r': stats[0], 'pearson_p': stats[1],
                         'spearman_rho': stats[2], 'spearman_p': stats[3], 'status': status})
plots = WORK / 'plots'; plots.mkdir(exist_ok=True)
pd.DataFrame(corr).to_csv(plots / 'eegmmidb_axm_correlations.csv', index=False)
for metric in metrics[:-1]:
    for outcome in outcomes:
        fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
        for layer, part in matrix.groupby('layer'):
            ax.scatter(part[metric], part[outcome] * 100, s=24, alpha=.7, label=layer)
        ax.axhline(0, color='black', lw=.75)
        ax.set(xlabel=metric, ylabel=f'{outcome} change (percentage points)',
               title=f'EEGMMIDB fold 0: {metric} versus {outcome}')
        ax.grid(alpha=.2); ax.legend(frameon=False)
        fig.savefig(plots / f'{metric}_vs_{outcome}.png', dpi=300); plt.close(fig)
summary = matrix.groupby('layer').agg(
    mean_abs_delta_acc=('delta_acc', lambda x: np.abs(x).mean()*100),
    mean_abs_delta_f1=('delta_f1', lambda x: np.abs(x).mean()*100),
    mean_abs_delta_sensitivity=('delta_sensitivity', lambda x: np.abs(x).mean()*100),
    worst_delta_acc=('delta_acc', lambda x: x.min()*100), best_delta_acc=('delta_acc', lambda x: x.max()*100))
summary.to_csv(plots / 'eegmmidb_layer_sensitivity_summary.csv')
print('Layer sensitivity (% absolute mean changes):\n', summary.to_string())
print('Pooled correlations:\n', pd.DataFrame(corr).query("layer == 'pooled'").to_string(index=False))
archive = WORK / 'canonical_sensitivity_artifacts.zip'
files = [matrix_path, WORK/'selected_multiplier_catalog.csv', WORK/'evoapprox8b_circuit_metrics.json',
         plots/'eegmmidb_axm_correlations.csv', plots/'eegmmidb_layer_sensitivity_summary.csv', *sorted(plots.glob('*.png'))]
files = [path for path in files if path.exists()]
with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as zf:
    for path in files: zf.write(path, arcname=path.relative_to(WORK))
print('Artifact archive:', archive, 'bytes:', archive.stat().st_size, 'plots:', len(list(plots.glob('*.png'))))
from IPython.display import Image, display
for path in sorted(plots.glob('*.png')):
    display(Image(filename=str(path), width=700))
