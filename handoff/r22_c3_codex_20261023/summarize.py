#!/usr/bin/env python3
"""Aggregate all completed C3 seed triplets, preserving negative results."""
import fcntl
import json
import os
from pathlib import Path
import statistics
import sys

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1]
sys.path.insert(0, str(BASE / 'ssl_pilot_v10_r22c/v10/r22_pmi'))
import analyze as A

VARIANTS = ('J', 'PMI', 'PMI_C2', 'EPT', 'POS')
PAIRS = (('J', 'PMI'), ('PMI', 'PMI_C2'), ('J', 'PMI_C2'), ('J', 'POS'), ('POS', 'EPT'), ('EPT', 'PMI'))
A.VARIANTS = VARIANTS
A.RUNS = HERE / 'runs'
A.unit_record = lambda ds, key, seed: json.loads((HERE / 'units' / f'r22_pmi3_{ds}_{key}_s{seed}.json').read_text())


def main():
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Run scientific aggregation through Slurm')
    with (HERE / 'summary.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        result = {'status': 'INTERIM', 'label': 'PILOT/DEVELOPMENT', 'freeze_sha256': A.sha(HERE / 'FREEZE.json'),
                  'complete_units': [], 'pending_units': [], 'datasets': {}, 'cost': {},
                  'caveats': ['Clipped PMI is an application heuristic, not the exact variational bound.',
                              'Intervals resample scenes/cores conditional on three trained seeds; no training/selection uncertainty.',
                              'Same-job pairing does not remove nondeterministic registration noise.',
                              'Existing VCS and InfoNCE checkpoints are recipe-level controls, not a new matched retraining experiment.',
                              'CAL/DEV are reused development data; no independent confirmation or TEST access.']}
        for ds in ('bf', 'epfl'):
            datas, families = {}, {}
            for key in ('MLPp1', 'InfoNCE'):
                complete = True
                for seed in A.SEEDS:
                    name = f'r22_pmi3_{ds}_{key}_s{seed}'
                    folder = A.RUNS / ds / key / f'seed_{seed}'
                    p = folder / 'COMPLETE.json'
                    if not p.exists():
                        complete = False; result['pending_units'].append(name); continue
                    c = json.loads(p.read_text())
                    ident = json.loads((folder / 'IDENTITY.json').read_text())
                    if c['status'] != 'COMPLETE' or c['identity_sha256'] != A.sha(folder / 'IDENTITY.json'):
                        raise RuntimeError('Invalid completion: ' + name)
                    if ident['unit'] != A.unit_record(ds, key, seed) or ident['variants'] != list(VARIANTS):
                        raise RuntimeError('Identity mismatch: ' + name)
                    if not set(c['roles_read']) <= {'CAL', 'DEV'}:
                        raise RuntimeError('Unexpected role read: ' + name)
                    datas[key, seed] = A.load(ds, key, seed)
                    result['complete_units'].append(name)
                if not complete:
                    continue
                per = {s: A.analyse_unit(ds, datas[key, s]) for s in A.SEEDS}
                block = {'per_seed': {}, 'means': {}, 'pairs': {}}
                for role in ('CAL', 'DEV'):
                    block['per_seed'][role] = {str(s): {v: A.headline(ds, role, per[s][role], v) for v in VARIANTS} for s in A.SEEDS}
                    block['means'][role] = {v: {m: statistics.fmean(block['per_seed'][role][str(s)][v][m] for s in A.SEEDS)
                                               for m in ('s42', 's9', 'median')} for v in VARIANTS}
                for a, b in PAIRS:
                    pair = {}
                    for m in ('s42', 's9', 'median'):
                        d = [block['per_seed']['DEV'][str(s)][b][m] - block['per_seed']['DEV'][str(s)][a][m] for s in A.SEEDS]
                        pair[m] = {'per_seed': d, 'mean': statistics.fmean(d), 'positive_seeds': sum(x > 0 for x in d)}
                    pair['bootstrap'] = {m: A.bootstrap(ds, datas, key, a, b, threshold) for m, threshold in (('s42', 42), ('s9', 9))}
                    pair['flips'] = {str(s): A.flips(datas[key, s]['DEV'], a, b) for s in A.SEEDS}
                    block['pairs'][b + '_minus_' + a] = pair
                block['failure_attribution'] = {str(s): {v: A.failure_attribution(datas[key, s]['DEV'], v) for v in VARIANTS} for s in A.SEEDS}
                families[key] = block
            ds_result = {'families': families}
            if set(families) == {'MLPp1', 'InfoNCE'}:
                ds_result['VCS_minus_InfoNCE'] = {}
                for v in VARIANTS:
                    ds_result['VCS_minus_InfoNCE'][v] = {}
                    for m in ('s42', 's9', 'median'):
                        d = [families['MLPp1']['per_seed']['DEV'][str(s)][v][m] - families['InfoNCE']['per_seed']['DEV'][str(s)][v][m] for s in A.SEEDS]
                        ds_result['VCS_minus_InfoNCE'][v][m] = {'per_seed': d, 'mean': statistics.fmean(d)}
                ds_result['clip_gain_VCS_minus_InfoNCE'] = {m: statistics.fmean(
                    families['MLPp1']['pairs']['PMI_C2_minus_PMI'][m]['per_seed'][i] -
                    families['InfoNCE']['pairs']['PMI_C2_minus_PMI'][m]['per_seed'][i] for i in range(3)) for m in ('s42', 's9', 'median')}
            result['datasets'][ds] = ds_result
        receipts = [json.loads(p.read_text()) for p in sorted((HERE / 'receipts').glob('*.json'))]
        result['cost'] = {'attempts': receipts, 'known_gpu_hours': sum(r.get('allocated_seconds', 0) for r in receipts) / 3600,
                          'unfinished_receipts': sum(r.get('exit_code') is None for r in receipts)}
        if not result['pending_units']:
            result['status'] = 'COMPLETE'
        tmp = HERE / '.SUMMARY.tmp'
        tmp.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
        tmp.replace(HERE / 'SUMMARY.json')
        print('C3 SUMMARY', result['status'], len(result['complete_units']), '/12', flush=True)


if __name__ == '__main__':
    main()
