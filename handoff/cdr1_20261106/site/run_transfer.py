#!/usr/bin/env python3
"""CDR1 S4 transfer for one data seed (needs both Flickr30K and COCO groups done).

Stages: cc_build (curator role: CC DEV galleries + CC local rho intervals), predict
(Flickr<->COCO via package run_readouts; ->CC via readout_lib), evaluate (separate process;
includes wrong-pi=0.5 sensitivity and source-domain oracle transfer), done.
CC: frozen source calibrators only; no CC fit, no CC rho chosen, no clean oracle on CC.
"""
import argparse, csv, glob, hashlib, json, os, subprocess, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (H, PKG, DATA, OUT, PY, DATASETS, M_SIZES, PIS, TAUS, NULL_FAR, EPSILONS, CC_NAME,
                    dump, load, sha, group_dirs, lock_truth, check_frozen)

CASES = list(csv.DictReader((PKG / 'configs/DATA_CASES.csv').open()))
VIEWS = list(csv.DictReader((PKG / 'configs/PREDICTION_VIEWS.csv').open()))
CCR = Path('/projects/EEG-foundation-model/yinghao/data/cc152k_raw/nr1/CC152K_RECOVERED_20261104')


def ho(ids, salt):
    keys = [hashlib.sha256((salt + '\0' + v).encode()).digest() for v in ids]
    return np.array(sorted(range(len(ids)), key=lambda k: keys[k]), int)


def l2(x):
    x = np.asarray(x, np.float64); return x / np.linalg.norm(x, axis=1, keepdims=True)


def cc_dir(seed): return DATA / CC_NAME / f'd{seed}'


def st_cc_build(seed):
    out = cc_dir(seed)
    if (out / 'CC_BUILD_DONE.json').exists():
        return {'status': 'REUSED'}
    man = load(CCR / 'cache/dev/manifest.json')
    for f in ('image_global.npy', 'text_global.npy', 'image_ids.json', 'text_ids.json'):
        assert sha(CCR / 'cache/dev' / f) == man['files'][f], f
    X = l2(np.load(CCR / 'cache/dev/image_global.npy')); Y = l2(np.load(CCR / 'cache/dev/text_global.npy'))
    iid = json.load(open(CCR / 'cache/dev/image_ids.json')); tid = json.load(open(CCR / 'cache/dev/text_ids.json'))
    pos = load(CCR / 'evaluator_only/dev_positives.json')['positives']
    assert len(pos) == 693 == len(iid) == len(tid)
    ti = {t: k for k, t in enumerate(tid)}; ii = {i: k for k, i in enumerate(iid)}
    cap_of = {}
    for im, tx in pos:
        assert im not in cap_of; cap_of[im] = ti[tx]
    ids = np.array(iid)
    gd = out / 'gallery_inputs'; ev = out / 'evaluator_only'; gd.mkdir(parents=True, exist_ok=True); ev.mkdir(exist_ok=True)
    man_g = []
    for m in M_SIZES:
        order = ho(list(iid), f'CDR1_CC_G_{seed}_{m}')
        g = order[:m]; absent = order[m:m + min(3 * m, len(iid) - m)]
        qg = np.r_[g, absent]; t = np.array([cap_of[iid[k]] for k in qg])
        s = Y[t] @ X[g].T; tl = np.r_[np.arange(m), np.full(len(absent), -1)]
        perm = ho([tid[k] for k in t], f'CDR1_CC_QUERY_ORDER_{seed}_{m}')
        s, t, qg, tl = s[perm], t[perm], qg[perm], tl[perm]
        pre = f'EVAL_OPEN_M{m}_rep{seed}'
        np.savez_compressed(gd / f'{pre}.npz', scores=s, query_ids=np.array(tid)[t], gallery_ids=ids[g])
        np.savez_compressed(ev / f'{pre}_labels.npz', true_index=tl, query_ids=np.array(tid)[t], group_ids=ids[qg])
        man_g.append({'id': pre, 'split': 'EVAL_OPEN', 'M': m, 'n_queries': int(len(t)), 'n_present_groups': m,
                      'n_absent_groups': int(len(absent)), 'scores': f'gallery_inputs/{pre}.npz',
                      'labels_evaluator_only': f'evaluator_only/{pre}_labels.npz',
                      'sampling': 'CC DEV one-caption-per-image; hash gallery; secondary development transfer'})
    dump(out / 'GALLERIES.json', man_g)
    # CC local rho intervals (secondary; observed FIT; no clean labels exist). Model-conditional sensitivity.
    fm = load(CCR / 'cache/fit/manifest.json')
    for f in ('image_global.npy', 'text_global.npy', 'image_ids.json', 'text_ids.json'):
        assert sha(CCR / 'cache/fit' / f) == fm['files'][f], f
    FX = np.load(CCR / 'cache/fit/image_global.npy', mmap_mode='r'); FY = np.load(CCR / 'cache/fit/text_global.npy', mmap_mode='r')
    fi = {v: k for k, v in enumerate(json.load(open(CCR / 'cache/fit/image_ids.json')))}
    ft = {v: k for k, v in enumerate(json.load(open(CCR / 'cache/fit/text_ids.json')))}
    rec = [l.rstrip('\n').split('\t') for l in open(CCR / 'data/observed_CC152K_RECOVERED_20261104.tsv')][1:]
    ri = np.array([fi[r[1]] for r in rec]); rt = np.array([ft[r[2]] for r in rec])
    assert len(rec) == 98212
    fids = json.load(open(CCR / 'cache/fit/image_ids.json'))  # row order
    hk = np.zeros(len(fids), bool); hk[ho(fids, 'CDR1_CC_RHO_DISC')[:len(fids) // 2]] = True  # DISC image half
    rng = np.random.default_rng(seed + 70000)
    def sc(a, b): return np.einsum('ij,ij->i', l2(FX[a]), l2(FY[b]))
    rec_disc = np.flatnonzero(hk[ri]); rec_cnt = np.flatnonzero(~hk[ri])
    def qdraw(pool_recs, n):  # independent image (uniform over pool images) x text (observed text marginal)
        imgs = np.unique(ri[pool_recs]); a = rng.choice(imgs, n); b = rt[rng.choice(pool_recs, n)]
        return sc(a, b)
    n = 20000
    qd = qdraw(rec_disc, n)
    pr = rng.choice(rec_cnt, n); pc = sc(ri[pr], rt[pr]); qc = qdraw(rec_cnt, n)
    from cdr1.rho import anchor_intervals
    rr = {}
    for e in EPSILONS:
        r = anchor_intervals(qd, pc, qc, epsilon=None if e == 'none' else float(e))
        r['scope'] = 'CC observed FIT vs independent reference; model-conditional sensitivity, NOT a clean-probability certificate; no true rho exists'
        rr[e] = r
    dump(OUT / 'cc_rho' / f'd{seed}' / 'CC_RHO_INTERVALS.json', rr)
    dump(out / 'CC_BUILD_DONE.json', {'seed': seed, 'galleries': man_g, 'n_dev_pairs': 693,
         'dev_cache_manifest_sha256': sha(CCR / 'cache/dev/manifest.json'), 'fit_records': 98212})
    return {'status': 'BUILT', 'galleries': [g['id'] for g in man_g]}


def src_cases(ds, seed):
    return [c for c in CASES if c['dataset'] == ds and int(c['data_seed']) == seed]


def sh(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True); sys.stdout.write(r.stdout); sys.stderr.write(r.stderr)
    if r.returncode: raise RuntimeError(f'rc={r.returncode}: {cmd}')


def tdir(seed, src, tgt): return OUT / 'transfer' / f'd{seed}' / f'{src}_to_{tgt}'


def st_predict(seed):
    from readout_lib import predict_one
    n = 0
    for src in DATASETS:
        tgt = [x for x in DATASETS if x != src][0]; sd = group_dirs(src, seed)
        for c in src_cases(src, seed):
            sh([PY, '-I', '-B', str(PKG / 'tools/run_readouts.py'), '--stage', 'predict', '--execute', '--data-root', str(DATA),
                '--models', str(sd['models']), '--out', str(tdir(seed, src, tgt)), '--source-case', c['case_id'], '--target-dataset', tgt])
            beta = load(DATA / c['raw_fit_dir'] / 'CASE.json')['collision_beta_fit']
            for v in [v for v in VIEWS if v['case_id'] == c['case_id']]:
                for g in load(cc_dir(seed) / 'GALLERIES.json'):
                    for pi in PIS:
                        predict_one(sd['models'] / (v['fit_id'] + '.json'), cc_dir(seed) / g['scores'], beta, pi, v['post_rho'],
                                    tdir(seed, src, 'CC') / f"{v['view_id']}__to_CC__{g['id']}__pi{int(pi*100)}.npz"); n += 1
    return {'cc_predictions': n}


def _oracle_transfer(seed, src, tgt_dir, gal, rows, tag):
    """Source-domain THRESH_CAL Platt / null-max applied unchanged to target top scores."""
    from cdr1.io import load_gallery_scores
    from cdr1.oracle_baselines import predict_clean_top1_platt
    from cdr1.metrics import prior_weights, selective_metrics
    from readout_lib import decision_metrics
    sd = group_dirs(src, seed)
    for g in gal:
        om = load(sd['oracle'] / f"ORACLE__EVAL_OPEN_M{g['M']}_rep{seed}.npz.json")['models']
        s, qid, _ = load_gallery_scores(tgt_dir / g['scores'])
        with np.load(tgt_dir / g['labels_evaluator_only'], allow_pickle=False) as z:
            assert np.array_equal(z['query_ids'], qid); ti = z['true_index']
        top = np.argmax(s, 1); ts = s[np.arange(len(s)), top]; present = ti >= 0; correct = present & (top == ti)
        for pi in PIS:
            w = prior_weights(present, pi)
            rows.append({'transfer': tag, 'gallery': g['id'], 'M': g['M'], 'pi': pi, 'method': 'CLEAN_TOP1_PLATT_SOURCE',
                         'selected': selective_metrics(predict_clean_top1_platt(om[f'platt_pi{int(pi*100)}'], ts), correct, present, weights=w)})
            for far, tau in zip(NULL_FAR, (.95, .90)):
                nm = om[f'nullmax_far{int(far*100):02d}']
                acc = np.zeros(len(ts), bool) if nm['reject_all'] else ts > nm['threshold']
                rows.append({'transfer': tag, 'gallery': g['id'], 'M': g['M'], 'pi': pi, 'method': 'MATCHED_NULL_MAX_SOURCE',
                             'target_FAR': far, 'operating_point': decision_metrics(acc, correct, present, w, 1 - tau)})


def st_evaluate(seed):
    from readout_lib import evaluate_one
    rows_or = []; nwrong = 0; ncc = 0
    for src in DATASETS:
        tgt = [x for x in DATASETS if x != src][0]
        for c in src_cases(src, seed):
            sh([PY, '-I', '-B', str(PKG / 'tools/run_readouts.py'), '--stage', 'evaluate', '--execute', '--data-root', str(DATA),
                '--models', str(group_dirs(src, seed)['models']), '--out', str(tdir(seed, src, tgt)), '--source-case', c['case_id'], '--target-dataset', tgt])
        for tag, tg_dir, gal in ((f'{src}_to_{tgt}', DATA / tgt / f'd{seed}', [g for g in load(DATA / tgt / f'd{seed}' / 'GALLERIES.json') if g.get('split') == 'EVAL_OPEN' and 'id' in g]),
                                 (f'{src}_to_CC', cc_dir(seed), load(cc_dir(seed) / 'GALLERIES.json'))):
            td = tdir(seed, src, tag.split('_to_')[1])
            for c in src_cases(src, seed):
                for v in [v for v in VIEWS if v['case_id'] == c['case_id']]:
                    for g in gal:
                        lab = tg_dir / g['labels_evaluator_only']
                        stem = f"{v['view_id']}__to_{tag.split('_to_')[1]}__{g['id']}"
                        if tag.endswith('_CC'):
                            for pi in PIS:
                                pp = td / f'{stem}__pi{int(pi*100)}.npz'
                                dump(td / f'{stem}__pi{int(pi*100)}.metrics.json', evaluate_one(pp, lab, pi, pi, pp.name)); ncc += 1
                        pp = td / f'{stem}__pi50.npz'
                        for pt in (.25, .75):  # wrong pi: predictor fixed pi=0.5, target cohort prior differs
                            dump(td / f'{stem}__WRONGPI_used50_target{int(pt*100)}.metrics.json', evaluate_one(pp, lab, pt, .5, pp.name)); nwrong += 1
            _oracle_transfer(seed, src, tg_dir, gal, rows_or, tag)
    dump(OUT / 'transfer' / f'd{seed}' / 'ORACLE_SOURCE_TRANSFER_METRICS.json', rows_or)
    return {'cc_metrics': ncc, 'wrong_pi_metrics': nwrong, 'oracle_rows': len(rows_or)}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--stage', required=True, choices=['cc_build', 'predict', 'evaluate', 'done'])
    a = ap.parse_args(); check_frozen()
    rc = OUT / 'receipts' / 'transfer' / f'd{a.seed}'; rc.mkdir(parents=True, exist_ok=True)
    if a.stage == 'done':
        st = {s: load(rc / f'stage_{s}.json') for s in ('cc_build', 'predict', 'evaluate')}
        dump(rc / 'TRANSFER_DONE.json', {'seed': a.seed, 'stage_seconds': {k: v['seconds'] for k, v in st.items()},
                                         'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())})
        print('TRANSFER_DONE', a.seed); return
    for ds in DATASETS:
        assert (OUT / 'receipts' / ds / f'd{a.seed}' / 'GROUP_DONE.json').exists(), f'group {ds} d{a.seed} not done'
    dirs = [DATA / ds / f'd{a.seed}' for ds in DATASETS] + [cc_dir(a.seed)]
    closed = a.stage == 'predict'
    for dd in dirs:
        if (dd / 'evaluator_only').exists(): lock_truth(dd, open_=not closed)
    t0 = time.time()
    try:
        res = {'cc_build': st_cc_build, 'predict': st_predict, 'evaluate': st_evaluate}[a.stage](a.seed)
    finally:
        for dd in dirs:
            if (dd / 'evaluator_only').exists(): lock_truth(dd, open_=True)
    dump(rc / f'stage_{a.stage}.json', {'stage': a.stage, 'seconds': time.time() - t0, 'result': res, 'host': os.uname().nodename,
                                        'truth_dir_closed': closed})
    print('STAGE_OK transfer', a.seed, a.stage, round(time.time() - t0, 1))


if __name__ == '__main__':
    main()
