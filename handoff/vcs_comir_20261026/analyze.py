"""R24 analysis: DEV results of the locked finals (BF primary, EPFL secondary), paired contrasts, the pre-registered signal rule,
the V10 CoMIR_native reproduction and cost. Prints WAITING until every locked final of a dataset is COMPLETE."""
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1]
OUT = Path('/projects/EEG-foundation-model/yinghao/outputs_v10/vcs_comir_20261026')
SEEDS = (1001, 1002, 1003)
V10 = {'bf': BASE / 'handoff/v10_native_geometry_recovery_20261019/families/4/results/FINAL_DEV/CoMIR_native',
       'epfl': BASE / 'handoff/v10_epfl_production_20261019/results/FINAL_DEV/CoMIR_native'}


def rows(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def per_scene(recs, scene_of):
    scenes = defaultdict(list)
    for r in recs:
        scenes[scene_of(r)].append(r['err_full'])
    return scenes


def metrics(recs, scene_of):
    scenes = per_scene(recs, scene_of); errs = [e for v in scenes.values() for e in v]
    return dict(success42=100 * statistics.fmean(sum(e < 42 for e in v) / len(v) for v in scenes.values()),
                success9=100 * statistics.fmean(sum(e < 9 for e in v) / len(v) for v in scenes.values()),
                median_err=float(np.median(errs)), p95_err=float(np.percentile(errs, 95)), n=len(errs), n_scenes=len(scenes))


def load(ds, name, seed):
    d = OUT / 'runs' / f'{ds}_final_{name}_s{seed}'
    c = json.loads((d / 'COMPLETE.json').read_text())
    assert c['status'] == 'COMPLETE' and c['identity']['seed'] == seed and c['identity']['phase'] == 'final'
    recs = rows(d / 'DEV.jsonl')
    return c, recs


def scene_fn(ds):
    if ds == 'bf':
        return lambda r: r['core']
    group = None
    return lambda r: r['scene']


def contrast(a_by_seed, b_by_seed, metric):
    vals = [a_by_seed[s][metric] - b_by_seed[s][metric] for s in SEEDS]
    return dict(per_seed=vals, mean=statistics.fmean(vals), positive=sum(v > 0 for v in vals))


def core_bootstrap(ds, a_recs, b_recs, threshold=9, B=10000):
    """Paired scene-level difference of success<threshold averaged over seeds; bootstrap over scenes."""
    fn = scene_fn(ds); diff = defaultdict(list)
    for s in SEEDS:
        pa, pb = per_scene(a_recs[s], fn), per_scene(b_recs[s], fn)
        assert set(pa) == set(pb)
        for k in pa:
            diff[k].append(sum(e < threshold for e in pa[k]) / len(pa[k]) - sum(e < threshold for e in pb[k]) / len(pb[k]))
    d = np.array([statistics.fmean(v) for v in diff.values()]) * 100
    g = np.random.default_rng(20261026); boots = d[g.integers(0, len(d), (B, len(d)))].mean(1)
    return dict(mean=float(d.mean()), ci95=[float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))], n_scenes=len(d))


def analyse(ds, lock):
    names = lock['finals']; missing = [f'{n}_s{s}' for n in names for s in SEEDS
                                       if not (OUT / 'runs' / f'{ds}_final_{n}_s{s}' / 'COMPLETE.json').exists()]
    if missing:
        return dict(status='WAITING', missing=missing)
    fn = scene_fn(ds); res, recs, cost = {}, {}, 0.0
    for n in names:
        res[n], recs[n] = {}, {}
        for s in SEEDS:
            c, r = load(ds, n, s); recs[n][s] = r; res[n][s] = metrics(r, fn); cost += c['seconds']
    v10, v10_recs = {}, {}
    for s in SEEDS:
        r = rows(V10[ds] / f'seed_{s}' / 'DEV.jsonl')
        if ds == 'epfl':
            ours = {(x['core'], x['perturbation']): x['scene'] for x in recs[names[0]][s]}
            r = [dict(x, scene=ours[(x.get('core') or x.get('window'), x['perturbation'])], core=x.get('core') or x.get('window')) for x in r]
        v10_recs[s] = r; v10[s] = metrics(r, fn)
    sel = lock['selected']; best = lock['best_critic']
    means = {n: {k: statistics.fmean(res[n][s][k] for s in SEEDS) for k in ('success42', 'success9', 'median_err', 'p95_err')} for n in names}
    means['V10_CoMIR_native(reused)'] = {k: statistics.fmean(v10[s][k] for s in SEEDS) for k in ('success42', 'success9', 'median_err', 'p95_err')}
    pairs = {}
    def add(label, a, b):
        A = res[a] if a in res else v10; Bm = res[b] if b in res else v10
        ar = recs[a] if a in recs else v10_recs; br = recs[b] if b in recs else v10_recs
        pairs[label] = dict(a=a, b=b, strict=contrast(A, Bm, 'success9'), loose=contrast(A, Bm, 'success42'),
                            median=contrast(A, Bm, 'median_err'), strict_scene_bootstrap=core_bootstrap(ds, ar, br, 9),
                            loose_scene_bootstrap=core_bootstrap(ds, ar, br, 42))
    add('best VCS - best NCE (primary)', best['vcs'], best['nce'])
    add('best VCS - best BCE', best['vcs'], best['bce'])
    add('vcs_exp - nce_mse (same critic family)', sel['vcs_exp']['name'], sel['nce_mse']['name'])
    add('vcs_cos - nce_cos', sel['vcs_cos']['name'], sel['nce_cos']['name'])
    add('vcs_exp - bce_exp', sel['vcs_exp']['name'], sel['bce_exp']['name'])
    add('vcs_cos - bce_cos', sel['vcs_cos']['name'], sel['bce_cos']['name'])
    add('best VCS - V10 CoMIR_native (reused)', best['vcs'], 'V10')
    add('nce_mse t0.5 (reproduction) - V10 CoMIR_native', 'nce_mse_t0.5_lr0.01', 'V10')
    p = pairs['best VCS - best NCE (primary)']
    signal = dict(rule='best VCS - best NCE >= +3 pp strict, >= 2/3 seeds positive, loose not worse by > 1 pp',
                  met=bool(p['strict']['mean'] >= 3 and p['strict']['positive'] >= 2 and p['loose']['mean'] >= -1),
                  vcs_specific_also_vs_bce=bool(pairs['best VCS - best BCE']['strict']['mean'] >= 3
                                                and pairs['best VCS - best BCE']['strict']['positive'] >= 2))
    return dict(status='COMPLETE', unit='BF: core (10 perturbations each, pooled = core macro); EPFL: scene unit macro',
                per_seed=res, v10_per_seed=v10, means=means, pairs=pairs, signal=signal if ds == 'bf' else None,
                final_unit_gpu_hours=cost / 3600)


def main():
    lock = json.loads((HERE / 'LOCK.json').read_text())
    out = dict(lock=dict(selected={k: v['name'] for k, v in lock['selected'].items()}, best_critic=lock['best_critic']))
    for ds in ('bf', 'epfl'):
        out[ds] = analyse(ds, lock)
    cal_cost = sum(json.loads((d / 'COMPLETE.json').read_text())['seconds'] for d in (OUT / 'runs').glob('bf_cal_*') if (d / 'COMPLETE.json').exists())
    smoke_cost = sum(json.loads((d / 'COMPLETE.json').read_text())['seconds'] for d in (OUT / 'smoke').glob('*') if (d / 'COMPLETE.json').exists())
    out['cost_gpu_hours'] = dict(cal=cal_cost / 3600, smoke=smoke_cost / 3600,
                                 finals={ds: out[ds].get('final_unit_gpu_hours') for ds in ('bf', 'epfl')})
    (HERE / 'ANALYSIS.json').write_text(json.dumps(out, indent=1) + '\n')
    for ds in ('bf', 'epfl'):
        a = out[ds]
        if a['status'] != 'COMPLETE':
            print(ds, 'WAITING', len(a['missing'])); continue
        print('==', ds)
        for n, m in a['means'].items():
            print(f"  {n:34s} strict {m['success9']:6.2f} loose {m['success42']:6.2f} median {m['median_err']:6.2f} p95 {m['p95_err']:7.2f}")
        for k, v in a['pairs'].items():
            print(f"  {k:48s} strict {v['strict']['mean']:+6.2f} ({v['strict']['positive']}/3) loose {v['loose']['mean']:+6.2f} "
                  f"scene-CI strict {v['strict_scene_bootstrap']['ci95']}")
        if a['signal']:
            print('  signal', a['signal'])
    print('cost', out['cost_gpu_hours'])


if __name__ == '__main__':
    main()
