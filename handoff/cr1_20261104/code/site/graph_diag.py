"""EVALUATOR-ONLY relation diagnostics (separate process; outputs never re-enter training).

Flickr: FIT annotation truth = explicit NR1 audit join (caption karpathy_img_id -> image karpathy_img_id),
cross-checked against the clean Flickr00 observed table; never record//5. CC152K_RECOVERED: no truth/noise
mask exists -> label-free statistics only (no correctness is certified from the model's own scores).
"""
from __future__ import annotations
import argparse, csv, json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from cr1.io import Bank, read_json, write_json, sha256      # noqa: E402
from cr1.diagnostics import relation_truth_report           # noqa: E402

ARMS = ('BASE_ROW', 'BASE_CAP', 'J_ROW', 'J_CAP', 'BCE_CAP', 'NCE_CAP')
NR1 = '/home/infres/yinwang/CS_QMI/outputs_nr1/NR1_20261104'


def flickr_truth(bank):
    with open(f'{NR1}/audit/ID_MAP_fit_images.tsv') as f: im = {r['karpathy_img_id']: r['opaque_id'] for r in csv.DictReader(f, delimiter='\t')}
    with open(f'{NR1}/audit/ID_MAP_fit_texts.tsv') as f: tx = {r['opaque_id']: r['karpathy_img_id'] for r in csv.DictReader(f, delimiter='\t')}
    row = {x: i for i, x in enumerate(bank.image_id)}
    truth = np.array([row[im[tx[t]]] for t in bank.text_id], np.int64)
    with open(f'{NR1}/data/observed_Flickr00.tsv') as f: clean = {r['text_id']: r['image_id'] for r in csv.DictReader(f, delimiter='\t')}
    agree = float(np.mean([clean[t] == bank.image_id[truth[j]] for j, t in enumerate(bank.text_id)]))
    return truth, {'source': 'NR1 audit ID_MAP explicit karpathy join', 'agreement_with_Flickr00_observed': agree,
                   'files_sha256': {p: sha256(f'{NR1}/{p}') for p in ('audit/ID_MAP_fit_images.tsv', 'audit/ID_MAP_fit_texts.tsv')}}


def curves(c, w, good, grid=(.1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0)):
    """Read-only precision-mass and coverage curves (not used to choose anything)."""
    n = len(c); flat_w = w.ravel(); flat_g = good.ravel(); o = np.argsort(-flat_w, kind='stable')
    cm = np.cumsum(flat_w[o]); ct = np.cumsum(flat_w[o]*flat_g[o]); total = cm[-1]
    pm = []
    for q in grid:
        k = min(np.searchsorted(cm, q*total), len(cm)-1); pm.append([q, float(cm[k]/n), float(ct[k]/cm[k])])
    rmax = w.max(1); ro = np.argsort(-rmax, kind='stable'); top = good[np.arange(n), w.argmax(1)]
    cov = [[q, float(top[ro[:max(1, int(q*n))]].mean())] for q in grid]
    return {'precision_vs_accepted_mass': pm, 'columns_pm': ['fraction_of_graph_mass', 'mass_per_text', 'true_mass_share'],
            'top1_precision_vs_row_coverage': cov}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--condition', required=True); ap.add_argument('--work', required=True)
    ap.add_argument('--banks', required=True); ap.add_argument('--out', required=True)
    a = ap.parse_args(); c_ = a.condition
    bank = Bank(Path(a.banks)/c_/'fit_bank.npz', formal=True); graph = np.load(Path(a.work)/c_/'prepared'/'graph.npz')
    cand = graph['candidates']; obs = bank.observed_image_for_text; base_top1 = cand[:, 0]
    truth, tmeta = (flickr_truth(bank) if c_.startswith('Flickr') else (None, {'source': 'none (CC152K_RECOVERED has no truth/noise mask)'}))
    res = {'condition': c_, 'truth': tmeta, 'arms': {}}
    if truth is not None:
        res['observed_noise_rate'] = float((obs != truth).mean())
        res['candidate_true_coverage'] = float((cand == truth[:, None]).any(1).mean())
        res['base_top1_correct'] = float((base_top1 == truth).mean())
        oo = (cand == obs[:, None]).astype(float)
        res['arms']['OBS_restricted_to_candidates'] = relation_truth_report(cand, oo, truth, obs, base_top1)
    for arm in ARMS:
        p = Path(a.work)/c_/'relations'/f'{arm}.npz'
        if not p.exists(): res['arms'][arm] = 'MISSING'; continue
        z = np.load(p); meta = read_json(str(p)+'.json'); w = z['weights']
        d = {'label_free': meta['label_free_stats'], 'relation': meta['relation'], 'solver': meta['solver'], 'coupling_sha256': sha256(p)}
        if truth is not None:
            good = (z['candidates'] == truth[:, None]); onobs = (z['candidates'] == obs[:, None]); new = ~onobs
            d['truth'] = relation_truth_report(z['candidates'], w, truth, obs, base_top1)
            nm = (w*new).sum()
            d['truth'].update({
                'new_edge_mass_per_text': float(nm/len(obs)),
                'new_edge_true_mass_share': float((w*new*good).sum()/nm) if nm > 0 else None,
                'wrong_obs_rows_true_mass_per_row': float((w*good).sum(1)[obs != truth].mean()),
                'correct_obs_rows_kept_obs_mass_per_row': float((w*onobs).sum(1)[obs == truth].mean()),
                'base_wrong_rows_true_mass_per_row': float((w*good).sum(1)[base_top1 != truth].mean()),
                'top1_weight_correct': float((z['candidates'][np.arange(len(obs)), w.argmax(1)] == truth).mean()),
                'true_mass_per_text': float((w*good).sum()/len(obs))})
            d['curves'] = curves(z['candidates'], w, good)
        res['arms'][arm] = d
    write_json(a.out, res); print(json.dumps({k: (v if k != 'arms' else list(v)) for k, v in res.items()})[:2000])


if __name__ == '__main__': main()
