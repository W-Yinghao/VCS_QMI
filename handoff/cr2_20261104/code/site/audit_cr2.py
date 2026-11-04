"""CR2 read-only relation audit on frozen CR1 artifacts (no training, no new fits).

--mode labelfree : deployable-side statistics; reads bank, CR1 prepared graph, CR1 score exports and
                   CR1 graphs only. No truth.
--mode truth     : EVALUATOR ONLY (separate process). Flickr annotation truth via the explicit NR1 audit
                   join (as CR1 graph_diag). Refuses CC152K (no truth exists).
--graphs-root    : optional; also audit CR2 CSR graphs (post-build) in the same way.
"""
from __future__ import annotations
import argparse, csv, glob, json, sys
from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix, load_npz

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/'reference'))
sys.path.insert(0, '/home/infres/yinwang/CS_QMI/handoff/cr1_20261104/code/src')
from graph_diagnostics import summarize           # noqa: E402  (package reference, byte copy)
from cr1.io import Bank, read_json, sha256         # noqa: E402  (frozen CR1, read-only import)
from cr1.folds import EXCLUSIONS, edge_experts     # noqa: E402

CR1 = Path('/projects/EEG-foundation-model/yinghao/outputs_cr1/CR1_20261104')
NR1 = '/home/infres/yinwang/CS_QMI/outputs_nr1/NR1_20261104'
SOURCES = ('BASE', 'J', 'BCE', 'NCE'); CR1_GRAPHS = ('BASE_ROW', 'BASE_CAP', 'J_ROW', 'J_CAP', 'BCE_CAP', 'NCE_CAP')
Q = (0, .01, .1, .25, .5, .75, .9, .99, 1)


def qs(x): return [float(v) for v in np.quantile(np.asarray(x, float), Q)] if len(x) else []


def cr1_csr(path, n_image):
    z = np.load(path); c, w = z['candidates'], z['weights']
    return csr_matrix((w.ravel(), (np.repeat(np.arange(len(c)), c.shape[1]), c.ravel())), shape=(len(c), n_image)), z['slack']


def row_stats(a, obs, h, cap, slack=None):
    s = summarize(a, obs); m = s['row_mass']; mm = np.maximum(m, 1e-300)
    a = a.tocsr(); row = np.repeat(np.arange(a.shape[0]), np.diff(a.indptr))
    maxc = np.zeros(len(m)); np.maximum.at(maxc, row, a.data/mm[row])
    load = s['image_load']
    out = {'row_mass': qs(m), 'row_mass_mean': float(m.mean()), 'conditional_entropy': qs(s['conditional_entropy']),
           'conditional_entropy_mean': float(s['conditional_entropy'].mean()), 'max_conditional_mass': qs(maxc),
           'max_conditional_mass_mean': float(maxc.mean()), 'row_ess': qs(s['conditional_ess']), 'row_ess_mean': float(s['conditional_ess'].mean()),
           'global_edge_ess': s['global_edge_ess'], 'nnz': int(a.nnz),
           'observed_mass_per_text': float(s['observed_mass'].mean()),
           'supported_rows_observed_mass_mean': float(s['observed_mass'][h].mean()) if h.any() else None,
           'supported_rows_observed_fraction_of_row_mass': float((s['observed_mass'][h]/mm[h]).mean()) if h.any() else None,
           'unsupported_rows_observed_mass_mean': float(s['observed_mass'][~h].mean()) if (~h).any() else None,
           'image_load': qs(load), 'load_over_CR1_cap': qs(load/cap), 'images_over_CR1_cap': int((load > cap*(1+1e-6)).sum()),
           'max_load_over_CR1_cap': float((load/cap).max())}
    if slack is not None: out['slack'] = qs(slack); out['slack_mean'] = float(slack.mean())
    return out, s


def labelfree(cond, bank, prep, graph, extra_graphs):
    c = graph['candidates']; obs = bank.observed_image_for_text; n = len(obs)
    h = (c == obs[:, None]).any(1); base_top1 = c[:, 0]
    cap = 2.0*n*bank.p_image/bank.p_image.sum()
    res = {'condition': cond, 'n_text': int(n), 'n_image': int(len(bank.image_global)), 'K': int(c.shape[1]),
           'support_crosstab_label_free': {
               'H1_obs_is_base_top1': int((h & (base_top1 == obs)).sum()), 'H1_obs_not_base_top1': int((h & (base_top1 != obs)).sum()),
               'H0': int((~h).sum()), 'H_fraction': float(h.mean())},
           'observed_rank_in_C_base': qs((np.argmax(c == obs[:, None], 1)+1)[h]), 'scores': {}, 'graphs': {}}
    tr = np.repeat(np.arange(n), c.shape[1]); ir = c.ravel()
    own = edge_experts(graph['image_folds'], graph['text_folds'], tr, ir).reshape(c.shape)
    nexp = np.array([len(np.unique(r)) for r in own])
    res['rows_with_k_experts'] = {int(k): int((nexp == k).sum()) for k in np.unique(nexp)}
    for s_ in SOURCES:
        meta = read_json(CR1/cond/'scores'/f'{s_}.npz.json'); sc = np.load(CR1/cond/'scores'/f'{s_}.npz')['scores']
        raw = np.empty_like(sc)
        for e, ex in enumerate(EXCLUSIONS):
            cal = meta['lineage'][f'E{ex[0]}{ex[1]}']['calibration']; mk = own == e
            raw[mk] = (sc[mk]-cal['b'])/cal['a']
        o_raw = np.argsort(-raw, 1, kind='stable'); o_cal = np.argsort(-sc, 1, kind='stable')
        rk_raw = np.empty_like(o_raw); rk_cal = np.empty_like(o_cal)
        np.put_along_axis(rk_raw, o_raw, np.arange(c.shape[1])[None].repeat(n, 0), 1)
        np.put_along_axis(rk_cal, o_cal, np.arange(c.shape[1])[None].repeat(n, 0), 1)
        diff = np.abs(rk_raw-rk_cal); multi = nexp > 1
        obs_slot = np.argmax(c == obs[:, None], 1)
        res['scores'][s_] = {
            'score_sha256': meta['score_sha256'],
            'per_expert_calibration': {k: {'a': v['calibration']['a'], 'b': v['calibration']['b'], 'edges': v['edges'],
                                           'calibrated_q': v['calibrated_quantiles_0_1_10_25_50_75_90_99_100'], 'raw_q_0_50_100': v['raw_quantiles_0_50_100']}
                                       for k, v in meta['lineage'].items()},
            'rows_top1_changed_raw_vs_calibrated': float((o_raw[:, 0] != o_cal[:, 0]).mean()),
            'rows_top1_changed_among_multi_expert_rows': float((o_raw[multi, 0] != o_cal[multi, 0]).mean()) if multi.any() else None,
            'mean_abs_rank_change': float(diff.mean()), 'rows_any_rank_change': float((diff > 0).any(1).mean()),
            'top1_equals_base_top1_calibrated': float((o_cal[:, 0] == 0).mean()),
            'observed_rank_in_C_raw_H_rows': qs(rk_raw[np.arange(n), obs_slot][h]+1),
            'observed_rank_in_C_calibrated_H_rows': qs(rk_cal[np.arange(n), obs_slot][h]+1),
            'observed_is_calibrated_top1_H_rows': float((rk_cal[np.arange(n), obs_slot][h] == 0).mean()),
            'seam_note': 'within one expert calibration is a positive affine map; rank changes arise only across experts in a row'}
    for g in CR1_GRAPHS:
        a, slack = cr1_csr(CR1/cond/'relations'/f'{g}.npz', len(bank.image_global))
        res['graphs'][f'CR1_{g}'], _ = row_stats(a, obs, h, cap, slack)
    for p in extra_graphs:
        a = load_npz(p); res['graphs'][Path(p).stem], _ = row_stats(a, obs, h, cap)
        rc = read_json(str(p)+'.json'); res['graphs'][Path(p).stem]['receipt_row_mass_max_change'] = rc.get('row_mass_max_change')
    return res


def flickr_truth(bank):
    with open(f'{NR1}/audit/ID_MAP_fit_images.tsv') as f: im = {r['karpathy_img_id']: r['opaque_id'] for r in csv.DictReader(f, delimiter='\t')}
    with open(f'{NR1}/audit/ID_MAP_fit_texts.tsv') as f: tx = {r['opaque_id']: r['karpathy_img_id'] for r in csv.DictReader(f, delimiter='\t')}
    row = {x: i for i, x in enumerate(bank.image_id)}
    return np.array([row[im[tx[t]]] for t in bank.text_id], np.int64)


def truth_part(cond, bank, graph, extra_graphs):
    if not cond.startswith('Flickr'): raise ValueError('No truth exists for this condition; refusing.')
    c = graph['candidates']; obs = bank.observed_image_for_text; t = flickr_truth(bank); n = len(obs)
    h = (c == obs[:, None]).any(1); ok = obs == t; b1 = c[:, 0] == t; tin = (c == t[:, None]).any(1)
    xt = {}
    for hv in (1, 0):
        for okv in (1, 0):
            for b1v in (1, 0):
                for tv in (1, 0):
                    xt[f'H{hv}_obsCorrect{okv}_baseTop1Correct{b1v}_truthInC{tv}'] = int(((h == hv) & (ok == okv) & (b1 == b1v) & (tin == tv)).sum())
    res = {'condition': cond, 'truth_source': 'NR1 audit ID_MAP explicit karpathy join (evaluator only)',
           'crosstab': xt, 'H_rows_obs_correct': float(ok[h].mean()), 'H0_rows_obs_correct': float(ok[~h].mean()),
           'H0_rows_truth_in_C': float(tin[~h].mean()), 'graphs': {}}
    paths = [(f'CR1_{g}', CR1/cond/'relations'/f'{g}.npz', 'cr1') for g in CR1_GRAPHS] + [(Path(p).stem, Path(p), 'csr') for p in extra_graphs]
    for name, p, kind in paths:
        a = cr1_csr(p, len(bank.image_global))[0] if kind == 'cr1' else load_npz(p)
        a = a.tocsr(); m = np.asarray(a.sum(1)).ravel()
        tm = np.asarray(a[np.arange(n), t]).ravel(); om = np.asarray(a[np.arange(n), obs]).ravel()
        newm = m - om
        new_true = np.where(ok, 0.0, tm)   # true mass on edges other than the observed one
        res['graphs'][name] = {
            'true_mass_share': float(tm.sum()/m.sum()), 'true_mass_per_text': float(tm.mean()),
            'H1_true_mass_share': float(tm[h].sum()/m[h].sum()), 'H0_true_mass_share': float(tm[~h].sum()/m[~h].sum()),
            'correct_obs_rows_obs_mass_mean': float(om[ok].mean()) if ok.any() else None,
            'correct_obs_rows_obs_mass_lost_vs_1': float((1-om[ok]).mean()) if ok.any() else None,
            'wrong_obs_rows_true_mass_mean': float(tm[~ok].mean()) if (~ok).any() else None,
            'wrong_obs_rows_obs_mass_mean': float(om[~ok].mean()) if (~ok).any() else None,
            'new_edge_mass_per_text': float(newm.mean()), 'new_edge_true_share': float(new_true.sum()/newm.sum()) if newm.sum() > 0 else None,
            'base_wrong_rows_true_mass_mean': float(tm[~b1].mean())}
    return res


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--mode', choices=['labelfree', 'truth'], required=True)
    ap.add_argument('--condition', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--graphs-glob', default=None)
    a = ap.parse_args(); cond = a.condition
    bank = Bank(CR1/'banks'/cond/'fit_bank.npz', formal=True)
    graph = dict(np.load(CR1/cond/'prepared'/'graph.npz'))
    extra = sorted(p for p in glob.glob(a.graphs_glob) if p.endswith('.npz')) if a.graphs_glob else []
    if a.mode == 'truth': extra = [p for p in extra]  # oracle graphs allowed here (evaluator process)
    else: extra = [p for p in extra if 'ORACLE' not in p]
    res = labelfree(cond, bank, prep=None, graph=graph, extra_graphs=extra) if a.mode == 'labelfree' else truth_part(cond, bank, graph, extra)
    res['inputs_sha256'] = {'bank': bank.digest, 'prepared_graph': sha256(CR1/cond/'prepared'/'graph.npz')}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1)+'\n'); print('wrote', a.out)


if __name__ == '__main__': main()
