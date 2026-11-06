#!/usr/bin/env python3
"""CDR1 one (dataset, data_seed) group. Each --stage runs in its own process.

Stages (in order): curate fit predict evaluate d1 oracle_fit oracle_eval rho_fit rho_predict
rho_eval integrity done. fit/predict/rho_fit/rho_predict run with evaluator_only/ chmod 000.
"""
import argparse, csv, glob, os, subprocess, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (H, PKG, DATA, BANKS, PY, M_SIZES, PIS, TAUS, LAMBDAS, RHO_GRID, EPSILONS,
                    NULL_FAR, dump, load, sha, group_dirs, lock_truth, check_frozen)

CASES = list(csv.DictReader((PKG / 'configs/DATA_CASES.csv').open()))
FITS = list(csv.DictReader((PKG / 'configs/CORE_FITS.csv').open()))
VIEWS = list(csv.DictReader((PKG / 'configs/PREDICTION_VIEWS.csv').open()))


def cases_of(ds, seed):
    return [c for c in CASES if c['dataset'] == ds and int(c['data_seed']) == seed]


def galleries(d, split='EVAL_OPEN'):
    return [g for g in load(d['data'] / 'GALLERIES.json') if g.get('split') == split and 'id' in g]


def sh(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    sys.stdout.write(r.stdout); sys.stderr.write(r.stderr)
    if r.returncode:
        raise RuntimeError(f'command failed rc={r.returncode}: {cmd}')
    return r.stdout


# ---------------------------------------------------------------- stages
def st_curate(ds, seed, d):
    from cdr1.curator import prepare_controlled
    if (d['data'] / 'CURATION_DONE.json').exists():
        return {'status': 'REUSED'}
    bank = BANKS / ds / 'bank_clean.npz'; meta = BANKS / ds / 'bank_meta.json'
    assert load(meta)['bank_sha256'] == sha(bank), 'bank hash mismatch'
    prepare_controlled(bank, meta, d['data'], dataset=ds, seed=seed, n_scores=20000)
    for c in cases_of(ds, seed):
        assert (DATA / c['raw_fit_dir'] / 'fit_scores.npz').exists(), c['case_id']
    return {'status': 'CURATED', 'curation': load(d['data'] / 'CURATION_DONE.json')}


def st_fit(ds, seed, d):
    out = {}
    d['models'].mkdir(parents=True, exist_ok=True)
    for c in cases_of(ds, seed):
        sh([PY, '-I', '-B', str(PKG / 'tools/run_core.py'), '--data-root', str(DATA),
            '--output-root', str(d['models']), '--case', c['case_id'], '--execute'])
        rr = d['models'] / f"RUN_RECEIPT_{c['case_id']}.json"
        os.replace(d['models'] / 'RUN_RECEIPT.json', rr)
        out[c['case_id']] = load(rr)
    return out


def st_readouts(ds, seed, d, stage):
    out = {}
    for c in cases_of(ds, seed):
        sh([PY, '-I', '-B', str(PKG / 'tools/run_readouts.py'), '--stage', stage, '--execute',
            '--data-root', str(DATA), '--models', str(d['models']), '--out', str(d['read']),
            '--source-case', c['case_id']])
        out[c['case_id']] = load(d['read'] / f"{c['case_id']}__to_{ds}.{stage}.receipt.json")['n_readout_views']
    return out


def st_d1(ds, seed, d):
    from cdr1.calibrate import Calibrator
    from cdr1.math_core import post_correct_log_ratio, mechanism_probability
    from cdr1.metrics import binary_metrics, prior_weights
    with np.load(d['data'] / 'evaluator_only/d1_clean_mechanism.npz', allow_pickle=False) as f:
        score, label, ann = f['score'], f['mechanism_label'], f['annotation_match']
    n = 0
    for c in cases_of(ds, seed):
        for v in [v for v in VIEWS if v['case_id'] == c['case_id']]:
            mp = d['models'] / (v['fit_id'] + '.json')
            m = Calibrator.from_dict(load(mp))
            ell = m.clean_log_ratio(score); proj = np.zeros(score.shape, bool)
            if v['post_rho'] != '':
                assert m.rho == 0
                ell, proj = post_correct_log_ratio(ell, float(v['post_rho']))
            for lam in LAMBDAS:
                prob = mechanism_probability(ell, lam)
                w = prior_weights(label.astype(bool), lam)
                dump(d['d1'] / f"{v['view_id']}__lam{int(lam*100):02d}.json",
                     {'view_id': v['view_id'], 'case_id': c['case_id'], 'method': v['method'],
                      'fit_id': v['fit_id'], 'model_sha256': sha(mp), 'post_rho': v['post_rho'],
                      'target': 'clean-P versus independent-Q GENERATING SOURCE', 'source_prior': lam,
                      'metrics': binary_metrics(prob, label, w),
                      'post_projection_fraction': float(proj.mean()),
                      'reference_collision_fraction': float(np.mean(ann[label == 0])),
                      'extra_clean_information': v['extra_clean_information'] == 'True'})
                n += 1
    return {'d1_files': n}


def st_oracle_fit(ds, seed, d):
    """EXTRA CLEAN INFORMATION: reads THRESH_CAL labels only; predicts on EVAL scores only."""
    from cdr1.io import load_gallery_scores
    from cdr1.metrics import prior_weights
    from cdr1.oracle_baselines import fit_clean_top1_platt, predict_clean_top1_platt, null_max_quantile
    tc = {g['M']: g for g in galleries(d, 'THRESH_CAL')}
    out = {}
    for g in galleries(d):
        m = g['M']; t = tc[m]
        s, qid, gid = load_gallery_scores(d['data'] / t['scores'])
        with np.load(d['data'] / t['labels_evaluator_only'], allow_pickle=False) as z:
            assert np.array_equal(z['query_ids'], qid); ti = z['true_index']
        top = np.argmax(s, 1); ts = s[np.arange(len(s)), top]
        present = ti >= 0; correct = present & (top == ti)
        se, qe, ge = load_gallery_scores(d['data'] / g['scores'])
        tope = np.argmax(se, 1); tse = se[np.arange(len(se)), tope]
        arrays = {'query_ids': qe, 'top_index': tope}
        models = {}
        for pi in PIS:
            pm = fit_clean_top1_platt(ts, correct, weights=prior_weights(present, pi))
            models[f'platt_pi{int(pi*100)}'] = pm
            arrays[f'platt_p_pi{int(pi*100)}'] = predict_clean_top1_platt(pm, tse)
        for far in NULL_FAR:
            nm = null_max_quantile(ts[~present], far)
            models[f'nullmax_far{int(far*100):02d}'] = nm
            arrays[f'null_accept_far{int(far*100):02d}'] = (np.zeros(len(tse), bool) if nm['reject_all']
                                                           else tse > nm['threshold'])
        pp = d['oracle'] / f"ORACLE__{g['id']}.npz"; pp.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(pp, **arrays)
        dump(str(pp) + '.json', {'thresh_cal_gallery': t['id'], 'eval_gallery': g['id'], 'models': models,
                                 'thresh_cal_scores_sha256': sha(d['data'] / t['scores']),
                                 'eval_scores_sha256': sha(d['data'] / g['scores']),
                                 'information': 'extra clean THRESH_CAL correctness/absence labels',
                                 'eval_labels_read': False})
        out[g['id']] = 'ORACLE_PREDICTED'
    return out


def st_oracle_eval(ds, seed, d):
    from readout_lib import decision_metrics
    from cdr1.metrics import prior_weights, selective_metrics
    rows = []
    for g in galleries(d):
        pp = d['oracle'] / f"ORACLE__{g['id']}.npz"
        with np.load(pp, allow_pickle=False) as z: p = {k: z[k] for k in z.files}
        with np.load(d['data'] / g['labels_evaluator_only'], allow_pickle=False) as z:
            assert np.array_equal(z['query_ids'], p['query_ids']); ti = z['true_index']
        present = ti >= 0; correct = present & (p['top_index'] == ti)
        for pi in PIS:
            w = prior_weights(present, pi)
            rows.append({'gallery': g['id'], 'M': g['M'], 'pi': pi, 'method': 'CLEAN_TOP1_PLATT_TARGET',
                         'selected': selective_metrics(p[f'platt_p_pi{int(pi*100)}'], correct, present, weights=w)})
            for far, tau in zip(NULL_FAR, (.95, .90)):
                rows.append({'gallery': g['id'], 'M': g['M'], 'pi': pi, 'method': 'MATCHED_NULL_MAX_TARGET',
                             'target_FAR': far, 'paired_tau_cost': tau,
                             'operating_point': decision_metrics(p[f'null_accept_far{int(far*100):02d}'],
                                                                 correct, present, w, 1 - tau)})
    dump(d['oracle'] / 'ORACLE_METRICS.json', rows)
    return {'rows': len(rows)}


def _independent_rho_cases(ds, seed):
    return [c for c in cases_of(ds, seed) if c['noise_kind'] == 'independent' and float(c['rho_event']) > 0]


def st_rho_fit(ds, seed, d):
    from cdr1.io import load_score_fit
    from cdr1.rho import anchor_intervals
    from cdr1.calibrate import fit_calibrator
    out = {}
    for c in _independent_rho_cases(ds, seed):
        cid = c['case_id']; cdir = DATA / c['raw_fit_dir']; rd = d['rho'] / cid; rd.mkdir(parents=True, exist_ok=True)
        with np.load(cdir / 'rho_discovery.npz', allow_pickle=False) as z:
            assert set(z.files) == {'q_score'}; qd = z['q_score']
        pc, qc = load_score_fit(cdir / 'rho_count.npz')
        inter = {}
        for e in EPSILONS:
            r = anchor_intervals(qd, pc, qc, epsilon=None if e == 'none' else float(e))
            r['inputs'] = {'discovery_sha256': sha(cdir / 'rho_discovery.npz'), 'counts_sha256': sha(cdir / 'rho_count.npz')}
            r['rho_supplied_to_fitter'] = False
            dump(rd / f'RHO_INTERVAL_eps{e}.json', r); inter[e] = r
        fs = cdir / 'fit_scores.npz'; digest = sha(fs); ps, qs = load_score_fit(fs)
        true_rho = float(c['rho_event'])
        prof = []
        for rho in RHO_GRID:
            for obj in ('J', 'BCE'):
                if rho == 0.0:
                    src = d['models'] / f'{cid}_NAIVE_{obj}.json'; status = 'REUSE_CORE_NAIVE'
                elif abs(rho - true_rho) < 1e-12:
                    src = d['models'] / f'{cid}_FWD_{obj}.json'; status = 'REUSE_CORE_FORWARD'
                else:
                    src = rd / f'PROF_{obj}_rho{rho:.1f}.json'
                    if src.exists():
                        assert load(src)['fit_diagnostics']['scores_sha256'] == digest; status = 'REUSED'
                    else:
                        t0 = time.time()
                        mdl = fit_calibrator(ps, qs, rho=rho, objective=obj)
                        mdl.fit_diagnostics.update({'scores_sha256': digest, 'rho_information_regime': 'sensitivity',
                                                    'registry_unit': f'{cid}_PROF_{obj}_rho{rho:.1f}',
                                                    'wall_seconds': time.time() - t0})
                        dump(src, mdl.to_dict()); status = 'NEW_PROFILE_FIT'
                prof.append({'rho': rho, 'objective': obj, 'model': str(src), 'status': status,
                             'fit_loss': load(src)['fit_diagnostics']['best_loss'],
                             'interpretation': 'fixed-rho sensitivity; NOT rho selected by loss'})
        anchors = []
        r0 = inter['0']
        if r0['status'] == 'CONDITIONAL_INTERVAL' and r0['rho_point'] is not None:
            rp = float(r0['rho_point'])
            for obj in ('J', 'BCE'):
                if rp >= 1:
                    anchors.append({'objective': obj, 'status': 'ANCHOR_POINT_AT_ONE_NOT_FITTABLE', 'rho_point': rp}); continue
                if rp == 0:
                    src = d['models'] / f'{cid}_NAIVE_{obj}.json'; status = 'REUSE_CORE_NAIVE'
                else:
                    src = rd / f'ANCHOR_EPS0_{obj}.json'
                    if not src.exists():
                        t0 = time.time()
                        mdl = fit_calibrator(ps, qs, rho=rp, objective=obj)
                        mdl.fit_diagnostics.update({'scores_sha256': digest, 'rho_information_regime': 'anchor-assumption',
                                                    'registry_unit': f'{cid}_ANCHOR_EPS0_{obj}', 'wall_seconds': time.time() - t0})
                        dump(src, mdl.to_dict())
                    status = 'ANCHOR_FIT'
                anchors.append({'objective': obj, 'rho_point': rp, 'model': str(src), 'status': status,
                                'interval_eps0': r0['rho_interval']})
        else:
            anchors.append({'status': 'NO_ANCHOR_POINT', 'eps0_status': r0['status']})
        dump(rd / 'PROFILE.json', prof); dump(rd / 'ANCHOR.json', anchors)
        out[cid] = {'new_profile_fits': sum(p['status'] == 'NEW_PROFILE_FIT' for p in prof),
                    'anchors': [a['status'] for a in anchors]}
    return out


def st_rho_predict(ds, seed, d):
    from readout_lib import predict_one
    n = 0
    for c in _independent_rho_cases(ds, seed):
        cid = c['case_id']; rd = d['rho'] / cid
        beta = load(DATA / c['raw_fit_dir'] / 'CASE.json')['collision_beta_fit']
        models = [(f"PROF_{p['objective']}_rho{p['rho']:.1f}", p['model']) for p in load(rd / 'PROFILE.json')]
        models += [(f"ANCHOR_EPS0_{a['objective']}", a['model']) for a in load(rd / 'ANCHOR.json') if 'model' in a]
        for name, mp in models:
            for g in galleries(d):
                for pi in PIS:
                    predict_one(mp, d['data'] / g['scores'], beta, pi, '', rd / 'pred' / f"{name}__{g['id']}__pi{int(pi*100)}.npz")
                    n += 1
    return {'predictions': n}


def st_rho_eval(ds, seed, d):
    from readout_lib import load_pred_labels, evaluate_one, decision_metrics
    from cdr1.calibrate import Calibrator
    from cdr1.math_core import mechanism_probability
    from cdr1.metrics import binary_metrics, prior_weights
    with np.load(d['data'] / 'evaluator_only/d1_clean_mechanism.npz', allow_pickle=False) as f:
        score, label = f['score'], f['mechanism_label']
    for c in _independent_rho_cases(ds, seed):
        cid = c['case_id']; rd = d['rho'] / cid
        prof = load(rd / 'PROFILE.json'); anchors = [a for a in load(rd / 'ANCHOR.json') if 'model' in a]
        res = {'case_id': cid, 'true_rho_evaluator_only': float(c['rho_event']), 'd1': [], 'd1_envelope': {},
               'd2_grid': [], 'd2_lower_envelope': [], 'anchor_d1': [], 'anchor_d2': []}
        w = prior_weights(label.astype(bool), .5)
        for obj in ('J', 'BCE'):
            probs = []
            for p in [p for p in prof if p['objective'] == obj]:
                m = Calibrator.from_dict(load(p['model'])); pr = mechanism_probability(m.clean_log_ratio(score), .5)
                probs.append(pr)
                res['d1'].append({'objective': obj, 'rho': p['rho'], 'metrics': {k: v for k, v in binary_metrics(pr, label, w).items() if k != 'bins'}})
            P = np.array(probs)
            res['d1_envelope'][obj] = {'mean_width': float((P.max(0) - P.min(0)).mean()),
                                       'max_width': float((P.max(0) - P.min(0)).max()), 'n_grid': len(P),
                                       'interpretation': 'finite-grid sensitivity only'}
        for a in anchors:
            m = Calibrator.from_dict(load(a['model'])); pr = mechanism_probability(m.clean_log_ratio(score), .5)
            res['anchor_d1'].append({'objective': a['objective'], 'rho_point': a['rho_point'],
                                     'metrics': {k: v for k, v in binary_metrics(pr, label, w).items() if k != 'bins'}})
        for g in galleries(d):
            for pi in PIS:
                labs = d['data'] / g['labels_evaluator_only']
                for obj in ('J', 'BCE'):
                    pb = []
                    for p in [p for p in prof if p['objective'] == obj]:
                        pp = rd / 'pred' / f"PROF_{obj}_rho{p['rho']:.1f}__{g['id']}__pi{int(pi*100)}.npz"
                        pr, lab, present, correct = load_pred_labels(pp, labs); pb.append(pr['p_best_correct'])
                        wt = prior_weights(present, pi)
                        res['d2_grid'].append({'objective': obj, 'rho': p['rho'], 'M': g['M'], 'pi': pi, 'gallery': g['id'],
                                               'ops': [dict(tau=t, **decision_metrics(pr['p_best_correct'] >= t, correct, present, wt, 1 - t)) for t in TAUS]})
                    lo = np.min(pb, 0)
                    res['d2_lower_envelope'].append({'objective': obj, 'M': g['M'], 'pi': pi, 'gallery': g['id'],
                                                     'mean_envelope_width': float((np.max(pb, 0) - lo).mean()),
                                                     'ops': [dict(tau=t, **decision_metrics(lo >= t, correct, present, wt, 1 - t)) for t in TAUS]})
                for a in anchors:
                    pp = rd / 'pred' / f"ANCHOR_EPS0_{a['objective']}__{g['id']}__pi{int(pi*100)}.npz"
                    r = evaluate_one(pp, labs, pi, pi, pp.name); r['rho_point'] = a['rho_point']; r['objective'] = a['objective']
                    res['anchor_d2'].append(r)
        dump(rd / 'RHO_EVAL.json', res)
    return {'cases': len(_independent_rho_cases(ds, seed))}


def st_integrity(ds, seed, d):
    """Ranking integrity: every prediction's top_index equals base argmax; R@K per gallery."""
    from cdr1.io import load_gallery_scores
    out = {}
    for g in galleries(d):
        s, qid, gid = load_gallery_scores(d['data'] / g['scores'])
        base = np.argmax(s, 1)
        files = (glob.glob(str(d['read'] / f"*__{g['id']}__pi*.npz")) + glob.glob(str(d['rho'] / '*' / 'pred' / f"*__{g['id']}__pi*.npz"))
                 + [str(d['oracle'] / f"ORACLE__{g['id']}.npz")])
        bad = 0
        for f in files:
            with np.load(f, allow_pickle=False) as z:
                if not (np.array_equal(z['top_index'], base) and np.array_equal(z['query_ids'], qid)):
                    bad += 1
        with np.load(d['data'] / g['labels_evaluator_only'], allow_pickle=False) as z:
            ti = z['true_index']
        pres = ti >= 0; order = np.argsort(-s, 1, kind='stable')
        rk = {f'R@{k}': float(np.mean([ti[i] in order[i, :k] for i in np.flatnonzero(pres)])) for k in (1, 5, 10)}
        out[g['id']] = {'n_prediction_files': len(files), 'top_index_mismatch_files': bad, 'base_recall_present': rk}
        assert bad == 0, f'ranking integrity failure {g["id"]}'
    dump(d['receipts'] / 'INTEGRITY.json', out)
    return {'galleries': len(out), 'all_identical': True}


STAGES = dict(curate=st_curate, fit=st_fit, predict=lambda *a: st_readouts(*a, 'predict'),
              evaluate=lambda *a: st_readouts(*a, 'evaluate'), d1=st_d1, oracle_fit=st_oracle_fit,
              oracle_eval=st_oracle_eval, rho_fit=st_rho_fit, rho_predict=st_rho_predict,
              rho_eval=st_rho_eval, integrity=st_integrity)
CLOSED = {'fit', 'predict', 'rho_fit', 'rho_predict'}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--dataset', required=True); ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--stage', required=True, choices=list(STAGES) + ['done'])
    a = ap.parse_args()
    fz = check_frozen()
    d = group_dirs(a.dataset, a.seed)
    for k in ('models', 'read', 'd1', 'oracle', 'rho', 'receipts'):
        d[k].mkdir(parents=True, exist_ok=True)
    if a.stage == 'done':
        st = {s: load(d['receipts'] / f'stage_{s}.json') for s in STAGES}
        dump(d['receipts'] / 'GROUP_DONE.json', {'dataset': a.dataset, 'seed': a.seed, 'frozen_sha256': sha(H / 'PROTOCOL_FROZEN_CDR1.json') if (H / 'PROTOCOL_FROZEN_CDR1.json').exists() else None,
             'stage_seconds': {s: v['seconds'] for s, v in st.items()}, 'host': os.uname().nodename,
             'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())})
        print('GROUP_DONE', a.dataset, a.seed); return
    if (d['data'] / 'evaluator_only').exists():
        lock_truth(d['data'], open_=a.stage not in CLOSED)
    t0 = time.time()
    try:
        res = STAGES[a.stage](a.dataset, a.seed, d)
    finally:
        if (d['data'] / 'evaluator_only').exists():
            lock_truth(d['data'], open_=True)
    dump(d['receipts'] / f'stage_{a.stage}.json', {'stage': a.stage, 'seconds': time.time() - t0, 'result': res,
         'host': os.uname().nodename, 'truth_dir_closed': a.stage in CLOSED})
    print('STAGE_OK', a.dataset, a.seed, a.stage, round(time.time() - t0, 1))


if __name__ == '__main__':
    main()
