"""R24 stage 1 analysis (PROTOCOL_AMENDMENT_1_STAGE1.json): selection rules over the fixed Neon candidate pool."""
import argparse, json, statistics
from collections import defaultdict
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ARMS = ('vcs_exact', 'bce', 'nce'); SEEDS = (1001, 1002, 1003)
OWN = {'VCS_J': 'vcs_exact', 'BCE': 'bce', 'NCE': 'nce'}
STRICT, LOOSE = 9.0, 42.0


def native_pick(c):
    best_r, best = 0, None
    for r in sorted({x['round'] for x in c}):
        b, bf = 0, None
        for x in c:
            if x['round'] == r and x['iter'] >= 0 and x['mi_native'] > b: b, bf = x['mi_native'], x
        if bf is None: bf = next(x for x in c if x['round'] == r and x['iter'] == -1)
        if b > best_r: best_r, best = b, bf
    return best          # None = the host wrote no field (no iterate with MI > 0): scored as the identity, see row['native']


def picks(row):
    c = row['candidates']; el = [x for x in c if x.get('n_valid', 0) >= 64]
    out = {'NATIVE': native_pick(c), 'ORACLE': min(c, key=lambda x: x['grid']), 'AFFINE': next(x for x in c if x['round'] == 0 and x['iter'] == -1)}
    if el:
        out['SOFTMI'] = max(el, key=lambda x: x['soft_mi'])
        for s in SEEDS:
            for rule, arm in OWN.items():
                out[f'{rule}_{s}'] = max(el, key=lambda x: x[f'{arm}_{s}']['own'])
            for arm in ARMS:
                out[f'COS_{arm}_{s}'] = max(el, key=lambda x: x[f'{arm}_{s}']['cos'])
    return out


def summarize(errs_by_core):
    cores = sorted(errs_by_core); allv = [v for c in cores for v in errs_by_core[c]]
    return dict(strict=100 * statistics.mean(np.mean([v < STRICT for v in errs_by_core[c]]) for c in cores),
                loose=100 * statistics.mean(np.mean([v < LOOSE for v in errs_by_core[c]]) for c in cores),
                median=float(np.median(allv)), p95=float(np.percentile(allv, 95)), n=len(allv), cores=len(cores))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--role', default='DEV'); ap.add_argument('--runs', default=str(HERE / 'runs/main'))
    ap.add_argument('--partial', action='store_true'); a = ap.parse_args()
    rows = [json.loads(l) for f in sorted(Path(a.runs).glob(f'{a.role}_s*.jsonl')) for l in open(f)]
    expected = 160 if a.role == 'DEV' else 16
    if len(rows) != expected and not a.partial:
        print('WAITING', len(rows), '/', expected); return
    aud = json.loads((HERE / 'AUDIT_SUMMARY.json').read_text()); unseen = set(aud['dev_cores_not_seen_by_neon_training'])
    mism = 0; per = []
    for r in rows:
        p = picks(r); nat = p['NATIVE']
        if nat is None:
            assert r['native']['output'] == 'none'; p['NATIVE'] = nat = dict(grid=r['native']['grid'], corner=r['native']['corner'])
        if abs(nat['grid'] - r['native']['grid']) > 1e-3: mism += 1
        per.append(dict(core=r['core'], k=r['k'], seen=r['core'] not in unseen, err={k: v['grid'] for k, v in p.items()},
                        corner={k: v['corner'] for k, v in p.items()}, seconds=r['seconds']))
    rules = sorted({k for x in per for k in x['err']})
    res = {}
    for subset, f in (('all', lambda x: True), ('seen_by_neon', lambda x: x['seen']), ('not_seen_by_neon', lambda x: not x['seen'])):
        sub = [x for x in per if f(x)]
        if not sub: continue
        tab = {}
        for rule in rules:
            ebc = defaultdict(list); cbc = defaultdict(list)
            for x in sub:
                if rule in x['err']: ebc[x['core']].append(x['err'][rule]); cbc[x['core']].append(x['corner'][rule])
            if ebc: tab[rule] = dict(grid=summarize(ebc), corner=summarize(cbc))
        def d(r1, r2, key='strict'):
            return round(tab[r1]['grid'][key] - tab[r2]['grid'][key], 3) if r1 in tab and r2 in tab else None
        comp = {}
        for s in SEEDS:
            comp[s] = {f'VCS_J-NATIVE_{k}': d(f'VCS_J_{s}', 'NATIVE', k) for k in ('strict', 'loose')}
            comp[s].update({f'VCS_J-{o}_strict': d(f'VCS_J_{s}', f'{o}_{s}') for o in ('BCE', 'NCE')})
            comp[s].update({f'{o}-NATIVE_strict': d(f'{o}_{s}', 'NATIVE') for o in ('BCE', 'NCE')})
            comp[s]['VCS_J-NATIVE_p95'] = round(tab[f'VCS_J_{s}']['grid']['p95'] - tab['NATIVE']['grid']['p95'], 3) if f'VCS_J_{s}' in tab else None
        mean = lambda k: statistics.mean(comp[s][k] for s in SEEDS) if all(comp[s][k] is not None for s in SEEDS) else None
        pos = lambda k: sum((comp[s][k] or 0) > 0 for s in SEEDS)
        keys = list(comp[SEEDS[0]])
        agg = {k: dict(mean=mean(k), positive_seeds=pos(k)) for k in keys}
        dec = dict(pool_miss=sum(x['err']['ORACLE'] >= STRICT for x in sub), native_selection_error=sum(x['err']['ORACLE'] < STRICT <= x['err']['NATIVE'] for x in sub),
                   native_success=sum(x['err']['NATIVE'] < STRICT for x in sub), n=len(sub))
        for rule in rules:
            if rule in ('NATIVE', 'ORACLE'): continue
            have = [x for x in sub if rule in x['err']]
            dec[f'{rule}_rescued'] = sum(x['err']['NATIVE'] >= STRICT > x['err'][rule] for x in have)
            dec[f'{rule}_broken'] = sum(x['err']['NATIVE'] < STRICT <= x['err'][rule] for x in have)
        res[subset] = dict(table=tab, per_seed=comp, aggregate=agg, decomposition=dec)
    vg = res['all']['aggregate']
    sig = dict(vcs_selection_gain=bool(vg['VCS_J-NATIVE_strict']['mean'] is not None and vg['VCS_J-NATIVE_strict']['mean'] >= 3 and vg['VCS_J-NATIVE_strict']['positive_seeds'] >= 2
                                       and vg['VCS_J-NATIVE_p95']['mean'] <= 0),
               vcs_specific=None)
    spec = [min(res['all']['per_seed'][s]['VCS_J-BCE_strict'], res['all']['per_seed'][s]['VCS_J-NCE_strict']) for s in SEEDS]
    sig['vcs_specific'] = bool(sig['vcs_selection_gain'] and statistics.mean(spec) >= 2 and sum(v > 0 for v in spec) >= 2)
    out = dict(role=a.role, n_rows=len(rows), native_recompute_mismatches=mism, native_no_output=sum(r['native'].get('output') == 'none' for r in rows), signal=sig, results=res,
               seconds_total=sum(x['seconds'] for x in per), partial=len(rows) != expected)
    name = f'STAGE1_RESULTS{"_" + a.role if a.role != "DEV" else ""}{"_PARTIAL" if out["partial"] else ""}.json'
    (HERE / name).write_text(json.dumps(out, indent=1, default=lambda o: o.item()) + '\n'); print(name)
    t = res['all']['table']
    for rule in ['ORACLE', 'NATIVE', 'AFFINE', 'SOFTMI'] + [f'{r}_{s}' for r in ('VCS_J', 'BCE', 'NCE') for s in SEEDS] + [f'COS_{a_}_{s}' for a_ in ARMS for s in SEEDS]:
        if rule in t: g = t[rule]['grid']; print(f'{rule:20s} strict {g["strict"]:6.2f} loose {g["loose"]:6.2f} med {g["median"]:7.2f} p95 {g["p95"]:7.1f}')
    print(json.dumps(res['all']['aggregate'], indent=0, default=lambda o: o.item())); print(res['all']['decomposition']); print('signal', sig, 'native mismatches', mism)


if __name__ == '__main__':
    main()
