"""R26 summary paired with R25 H0 (native IN2R-code), full DEV only. Usage: summarize.py flickr|cross OUT.json
Reads each run's EPOCHS.json (cotrain list) and DONE.json; checks DONE best == max cotrain rsum and 40 cotrain epochs.
Flickr H0 comes from R25 RESULTS_PER_RUN_CORE.json (same native code, same prefix, same noise lock); COCO/CC152K H0 are R26 runs/<cell>/H0."""
import json, sys, os, glob
import numpy as np
R = '/projects/EEG-foundation-model/yinghao/outputs_v9/r26_dist_20261027/runs'
R25 = json.load(open('/home/infres/yinwang/CS_QMI/handoff/r25_in2r_20261027/RESULTS_PER_RUN_CORE.json'))
which, out = sys.argv[1], sys.argv[2]
ARMS = ['d_vcs', 'd_bce', 'm_vcs', 'n_vcs', 'd_snce'] if which == 'flickr' else ['d_vcs', 'd_bce', 'm_vcs']
CONDS = ['f40', 'f60'] if which == 'flickr' else ['c40', 'cc']
SEEDS = [1001, 1002, 1003]
KEYS = ['i2t_r1', 'i2t_r5', 'i2t_r10', 't2i_r1', 't2i_r5', 't2i_r10']


def run(cell, arm):
    d = os.path.join(R, cell, arm)
    done = json.load(open(os.path.join(d, 'DONE.json')))
    ep = json.load(open(os.path.join(d, 'EPOCHS.json')))['cotrain']
    rs = [e['dev']['rsum'] for e in ep]
    assert done['status'] == 'COMPLETE' and len(ep) == 40 and [e['epoch'] for e in ep] == list(range(40)), (cell, arm)
    assert abs(max(rs) - done['best_rsum']) < 1e-6, (cell, arm, max(rs), done['best_rsum'])
    b = int(np.argmax(rs))
    skips = {}
    for e in ep:
        for p in ('A', 'B'):
            for k, v in e.get('train', {}).get(p, {}).items():
                if k.startswith('skip') or k.endswith('no_proto'):
                    skips[k] = skips.get(k, 0) + v
    peak = sorted(glob.glob(os.path.join(d, 'PEAK_*.json')))
    pk = json.load(open(peak[-1])) if peak else {}
    runj = sorted(glob.glob(os.path.join(d, 'RUN_*.json')))
    env = json.load(open(runj[-1])).get('env', {}) if runj else {}
    # cost metadata: PEAK files exist only for R26 arm runs; native H0 (r25_run.py) records GPU/host in RUN_*.json and
    # per-epoch seconds in EPOCHS.json. epoch_seconds_sum excludes data staging / model setup.
    pk.setdefault('gpu', env.get('gpu'))
    ep_sec = sum(e.get('seconds', 0) for e in ep) / 3600
    return {'best': max(rs), 'epoch': b, 'last': rs[-1], 'last5': float(np.mean(rs[-5:])),
            **{k: ep[b]['dev'][k] for k in KEYS}, 'skips': skips,
            'wall_h_last_attempt': (pk['wall_seconds'] / 3600) if 'wall_seconds' in pk else None, 'cotrain_epoch_seconds_h': ep_sec, 'gpu': pk.get('gpu'), 'host': env.get('host'), 'attempts': open(os.path.join(d, 'ATTEMPTS')).read().strip()}


res = {}
for c in CONDS:
    for s in SEEDS:
        cell = f'{c}_s{s}'
        res[cell] = {a: run(cell, a) for a in ARMS}
        if which == 'flickr':
            h = R25[cell]['H0']
            res[cell]['H0'] = {'best': h['best'], 'epoch': h['epoch'], 'last': h['last'], 'last5': h['last5'], 'source': 'R25 RESULTS_PER_RUN_CORE.json'}
            for k in ('V1', 'B1'):
                res[cell]['R25_' + k] = {'best': R25[cell][k]['best'], 'last5': R25[cell][k]['last5']}
        else:
            res[cell]['H0'] = run(cell, 'H0')
pairs = [('d_vcs', 'H0')] + [('d_vcs', a) for a in ARMS if a != 'd_vcs'] + [(a, 'H0') for a in ARMS if a != 'd_vcs']
comp = {}
for c in CONDS:
    for m in ('best', 'last5'):
        for x, y in pairs:
            d = [res[f'{c}_s{s}'][x][m] - res[f'{c}_s{s}'][y][m] for s in SEEDS]
            comp.setdefault(c, {}).setdefault(m, {})[f'{x}-{y}'] = {'per_seed': d, 'mean': float(np.mean(d)), 'n_pos': int(sum(v > 1e-9 for v in d)), 'n_tie': int(sum(abs(v) <= 1e-9 for v in d))}
    comp[c]['arm_means_best'] = {a: float(np.mean([res[f'{c}_s{s}'][a]['best'] for s in SEEDS])) for a in ARMS + ['H0']}
    comp[c]['arm_means_last5'] = {a: float(np.mean([res[f'{c}_s{s}'][a]['last5'] for s in SEEDS])) for a in ARMS + ['H0']}
json.dump({'scope': f'R26 {which}: full DEV best/last5 rSum, seed-paired; no TEST; no per-query ranks saved -> no image-grouped bootstrap',
           'runs': res, 'comparisons': comp}, open(out, 'w'), indent=1)
for c in CONDS:
    print(c, 'best means', {k: round(v, 2) for k, v in comp[c]['arm_means_best'].items()})
    print(c, 'last5 means', {k: round(v, 2) for k, v in comp[c]['arm_means_last5'].items()})
    for m in ('best', 'last5'):
        for k, v in comp[c][m].items():
            print(f'  {m:5s} {k:14s} mean {v["mean"]:+.2f}  pos {v["n_pos"]}/3  ', [round(x, 2) for x in v['per_seed']])
