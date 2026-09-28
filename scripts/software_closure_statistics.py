"""Analyze recovered research CSVs; no training or inference."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[1]
source = ROOT / 'results/axm_eegmmidb_5fold_recovered'
out = ROOT / 'results/software_closure'
out.mkdir(exist_ok=True)
d = pd.read_csv(source / 'sensitivity_matrix_eegmmidb_all_folds.csv')
s = pd.read_csv(source / 'per_subject_sensitivity_eegmmidb_all_folds.csv')
keys = ['fold', 'layer', 'multiplier_id']
assert len(d) == 1000 and not d.duplicated(keys).any()
assert (d.groupby(['fold', 'layer']).size() == 50).all()
assert not s.duplicated(keys + ['subject']).any()
assert (s.groupby(keys).size() == 21).all()
assert s.subject.nunique() == 105 and s.groupby('subject').fold.nunique().max() == 1
assert set(d.eval_windows) == {420} and (d.subset_seed == 20260927 + d.fold).all()
paired = s.groupby(keys).delta_acc.mean().reset_index().merge(d, on=keys)
assert np.allclose(paired.delta_acc_x, paired.delta_acc_y, atol=1e-12)
# Each subject contributes exactly 20 balanced windows. Integer correct-count
# differences preserve true ties, unlike subtracting binary floating accuracies.
counts = (s.approx_acc - s.exact_acc) * 20
assert np.allclose(counts, np.rint(counts), atol=1e-10)
s['delta_correct'] = np.rint(counts).astype(int)
def bh(p):
    p = np.asarray(p); order = np.argsort(p)
    q = np.minimum.accumulate((p[order]*len(p)/np.arange(1,len(p)+1))[::-1])[::-1]
    result = np.empty(len(p)); result[order] = np.minimum(1,q)
    return result
rows=[]
for (layer,mul),g in s.groupby(['layer','multiplier_id']):
    delta=g.delta_correct.to_numpy()
    test=wilcoxon(delta, zero_method='wilcox', alternative='two-sided', method='approx') if np.any(delta) else None
    rows.append(dict(layer=layer,multiplier_id=mul,n_subjects=len(g),nonzero_pairs=int(np.count_nonzero(delta)),mean_delta_acc=delta.mean()/20,p=1. if test is None else test.pvalue,method='two-sided asymptotic Wilcoxon; integer correct-count differences; zeros omitted'))
tests=pd.DataFrame(rows);tests['q_bh_200']=bh(tests.p)
tests['significant_drop']=(tests.q_bh_200<.05)&(tests.mean_delta_acc<0)
tests.to_csv(out/'wilcoxon_integer_ties.csv',index=False)
agg=d.groupby(['layer','multiplier_id']).agg(mean_delta_acc=('delta_acc','mean'),std_delta_acc=('delta_acc','std'),mean_delta_f1=('delta_f1','mean'),std_delta_f1=('delta_f1','std'),ER=('ER','first'),MRED=('MRED','first'),MAE=('MAE','first')).reset_index()
agg.to_csv(out/'sensitivity_mean_std_5fold.csv',index=False)
ranking=d.assign(abs_delta=d.delta_acc.abs()).groupby(['fold','layer']).abs_delta.mean().unstack()
ranking.to_csv(out/'layer_ranking_by_fold.csv')
# Means are rounded only to remove CSV floating-point artifacts in equal ranks.
# Bootstrap resamples folds and multipliers independently; all layers of each
# sampled multiplier remain together. This is exploratory with only five folds.
rng=np.random.default_rng(20260928); correlations=[]
for scope in ['pooled']+sorted(d.layer.unique()):
    part=d if scope=='pooled' else d[d.layer==scope]
    layers=sorted(part.layer.unique()); ids=sorted(part.multiplier_id.unique())
    for metric in ['ER','MRED','MAE']:
        x=part.groupby('multiplier_id')[metric].first().reindex(ids).to_numpy()
        for outcome in ['delta_acc','delta_f1']:
            arr=np.array([part[part.fold==f].pivot(index='multiplier_id',columns='layer',values=outcome).reindex(index=ids,columns=layers).to_numpy() for f in range(5)])
            yy=np.round(arr.mean(axis=0).ravel(),12); xx=np.repeat(x,len(layers))
            pr=pearsonr(xx,yy);sr=spearmanr(xx,yy)
            boots=[]
            for _ in range(1000):
                folds=rng.integers(0,5,5); ix=rng.integers(0,len(ids),len(ids))
                by=np.round(arr[folds].mean(axis=0)[ix].ravel(),12);bx=np.repeat(x[ix],len(layers))
                boots.append([pearsonr(bx,by).statistic,spearmanr(bx,by).statistic])
            lo,hi=np.nanpercentile(boots,[2.5,97.5],axis=0)
            correlations.append(dict(scope=scope,metric=metric,outcome=outcome,pearson_r=pr.statistic,spearman_rho=sr.statistic,pearson_p_naive=pr.pvalue,spearman_p_naive=sr.pvalue,pearson_ci_low=lo[0],pearson_ci_high=hi[0],spearman_ci_low=lo[1],spearman_ci_high=hi[1],bootstrap='1000 draws; independently resample 5 folds and 50 multiplier clusters; seed20260928'))
pd.DataFrame(correlations).to_csv(out/'correlations_fold_multiplier_bootstrap.csv',index=False)
cat=pd.read_csv(source/'selected_multiplier_catalog.csv')
selected=agg[agg.multiplier_id.isin(['mul8_348','mul8_112','mul8_424'])].merge(tests[['layer','multiplier_id','q_bh_200','significant_drop']],on=['layer','multiplier_id']).merge(cat[['multiplier_id','power']],on='multiplier_id')
selected['status']='EEGMMIDB coarse-screen candidate; BCI transfer and full-test confirmation pending'
selected.to_csv(out/'finalist_candidates.csv',index=False)
summary=dict(verified_cases=len(d),paired_records=len(s),unique_subjects=105,most_sensitive_by_fold=ranking.idxmax(axis=1).to_dict(),least_sensitive_by_fold=ranking.idxmin(axis=1).to_dict(),significant_drops=tests.groupby('layer').significant_drop.sum().to_dict(),historical_bci_sweep='unvalidated_baseline',physical_design_gate='NO-GO until validated BCI sensitivity and review')
(out/'verification_summary.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
