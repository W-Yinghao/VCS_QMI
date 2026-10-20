"""The separate BF1 capacity-extension table with the pre-registered statistics (never merged into the BF1 main table).

Run after the units finish (CPU only, no Torch):  python capacity_table.py  ->  bundle CAPACITY_TABLE.json / CAPACITY_TABLE.md.
Every K8/K32 unit is audited with capacity_common.verify_result; the K16 references are the pinned BF1 COMPLETE.json files.
"""
import argparse
import os
import statistics

import numpy as np

import capacity_common as C

B_RESAMPLES = 10000
BOOT_SEED = 20261019
RATES = ('success42', 'success9')


def k16_reference(prereg, family, seed):
    ref = prereg['K_reference']['units'][family][str(seed)]
    r = C.checked(ref)
    d = C.Path(ref['path']).parent
    for name, digest in r['artifacts'].items():
        if name.endswith(('.json', '.jsonl')):
            C.require(C.sha(d / name) == digest, 'K16 reference artifact changed: ' + name)
    return r, d


def m1_diagnostics(directory):
    rows = C.lines(C.Path(directory) / 'TRAIN_STEPS.jsonl')
    its = [r['m1_iterations'] for r in rows]
    return {'cap_hit_fraction': sum(r['m1_stop_reason'] == 'iteration_budget_exhausted' for r in rows) / len(rows),
            'median_iterations': statistics.median(its), 'mean_iterations': statistics.mean(its)}


def unit_cost(key):
    exits = [C.read(p) for p in (C.BUNDLE / 'exits').glob('*.json')] if (C.BUNDLE / 'exits').exists() else []
    mine = [e for e in exits if e['unit'] == key]
    return {'jobs': sorted(e['job'] for e in mine), 'allocated_gpu_hours': sum(e['allocated_seconds'] for e in mine) / 3600,
            'nodes': sorted({e['node'] for e in mine if e.get('node')})}


def interval(diff, rng):
    """diff: [seeds, cores] per-core paired differences; percentile 95% interval of the mean over cores of the seed mean."""
    d = diff.mean(0)
    idx = rng.integers(0, d.shape[0], size=(B_RESAMPLES, d.shape[0]))
    boot = d[idx].mean(1)
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def grid(lo, hi, per_seed):
    if lo <= 0 <= hi:
        return 'B_no_detectable_response'
    sign = 1 if lo > 0 else -1
    return 'A_consistent_response' if all(x * sign > 0 for x in per_seed) else 'C_mixed'


def cell(prereg, family, K, freeze_sha, rng):
    units = [u for u in prereg['units'] if u['family'] == family and u['K'] == K]
    rows, missing = [], []
    for u in units:
        if not (C.OUT / u['key'] / 'COMPLETE.json').exists() or (C.BUNDLE / 'budget' / f"{u['key']}.json").exists():
            missing.append(u['key'])
            continue
        r = C.verify_result(u, prereg, freeze_sha)
        ref, ref_dir = k16_reference(prereg, family, u['seed'])
        row = {'seed': u['seed'], 'unit': u['key'], 'selected_step': r['selected']['step'], 'k16_selected_step': ref['selected']['step'],
               'cost': unit_cost(u['key'])}
        for subset in ('DEV32', 'DEV24'):
            row[subset] = r[subset]
            row[subset + '_k16'] = ref[subset]
            row[subset + '_delta'] = {m: r[subset][m] - ref[subset][m] for m in ('success42', 'success9', 'median_err')}
        if family == 'MF_VCS_M1':
            row['m1'] = m1_diagnostics(C.OUT / u['key'])
            row['m1_k16'] = m1_diagnostics(ref_dir)
        row['_per_core'], row['_per_core_k16'], row['_dev24'] = r['per_core'], ref['per_core'], r['DEV24_cores']
        rows.append(row)
    out = {'family': family, 'K': K, 'seeds_complete': [x['seed'] for x in rows], 'missing_or_incomplete': missing}
    if missing:
        out['status'] = 'INCOMPLETE'
    else:
        out['status'] = 'COMPLETE'
        for subset in ('DEV32', 'DEV24'):
            cores = sorted(rows[0]['_per_core']) if subset == 'DEV32' else rows[0]['_dev24']
            out[subset] = {}
            for m in RATES:
                diff = np.array([[x['_per_core'][c][m] - x['_per_core_k16'][c][m] for c in cores] for x in rows])
                per_seed = [x[subset + '_delta'][m] for x in rows]
                C.require(np.allclose(diff.mean(1), per_seed, atol=1e-12), 'Per-core and pooled rates disagree')
                lo, hi = interval(diff, rng)
                out[subset][m] = {'delta_per_seed': per_seed, 'delta_mean': float(np.mean(per_seed)), 'ci95': [lo, hi],
                                  'grid': grid(lo, hi, per_seed)}
            out[subset]['median_err'] = {'delta_per_seed': [x[subset + '_delta']['median_err'] for x in rows]}
    for x in rows:
        for k in ('_per_core', '_per_core_k16', '_dev24'):
            x.pop(k)
    out['units'] = rows
    return out


def markdown(table):
    lines = ['# BF1 capacity extension (fixed training rule) — separate table', '',
             'Capacity response under a fixed training rule (固定训练规则下的容量响应): K8/K32 inherit the K16 CAL selection; '
             'includes optimisation effects; not an optimal-capacity curve. DEV is a development benchmark. Not part of the BF1 main table.', '',
             '| family | K | seed | ckpt | DEV32 <42 | <9 | median | Δ<42 vs K16 | Δ<9 | DEV24 <42 | <9 | GPU-h | node |',
             '|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for c in table['cells']:
        for u in c['units']:
            lines.append(f"| {c['family']} | {c['K']} | {u['seed']} | {u['selected_step']} | {u['DEV32']['success42']:.4f} | "
                         f"{u['DEV32']['success9']:.4f} | {u['DEV32']['median_err']:.3f} | {u['DEV32_delta']['success42']:+.4f} | "
                         f"{u['DEV32_delta']['success9']:+.4f} | {u['DEV24']['success42']:.4f} | {u['DEV24']['success9']:.4f} | "
                         f"{u['cost']['allocated_gpu_hours']:.3f} | {','.join(u['cost']['nodes'])} |")
    lines += ['', '| family | K | status | DEV32 Δ<42 mean [95% core bootstrap] | grid | DEV32 Δ<9 mean [95%] | grid | DEV24 Δ<42 [95%] | grid |',
              '|---|---|---|---|---|---|---|---|---|']
    for c in table['cells']:
        if c['status'] != 'COMPLETE':
            lines.append(f"| {c['family']} | {c['K']} | INCOMPLETE ({', '.join(c['missing_or_incomplete'])}) | | | | | | |")
            continue
        f = lambda s, m: f"{c[s][m]['delta_mean']:+.4f} [{c[s][m]['ci95'][0]:+.4f}, {c[s][m]['ci95'][1]:+.4f}]"
        lines.append(f"| {c['family']} | {c['K']} | COMPLETE | {f('DEV32', 'success42')} | {c['DEV32']['success42']['grid']} | "
                     f"{f('DEV32', 'success9')} | {c['DEV32']['success9']['grid']} | {f('DEV24', 'success42')} | {c['DEV24']['success42']['grid']} |")
    lines += ['', f"Total allocated L40S GPU-h (all jobs of the extension): {table['total_allocated_gpu_hours']:.3f} "
                  f"(ceiling {table['ceiling_gpu_hours']:.0f}).", '']
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--freeze', default=os.environ.get('V10_CAPACITY_FREEZE'))
    a = ap.parse_args()
    prereg = C.validate_prereg(C.frozen(a.freeze))
    rng = np.random.default_rng(BOOT_SEED)
    cells = [cell(prereg, f, K, a.freeze, rng) for f in C.FAMILIES for K in C.KS]
    exits = [C.read(p) for p in (C.BUNDLE / 'exits').glob('*.json')] if (C.BUNDLE / 'exits').exists() else []
    table = {'schema': 'v10.capacity_ext.table.v1', 'utc': C.utc(), 'name': C.NAME, 'freeze_sha256': a.freeze,
             'preregistration_sha256': C.sha(C.BUNDLE / 'PREREGISTRATION.json'), 'bootstrap': {'B': B_RESAMPLES, 'seed': BOOT_SEED},
             'scope': prereg['interpretation_scope'], 'cells': cells,
             'complete': all(c['status'] == 'COMPLETE' for c in cells),
             'total_allocated_gpu_hours': sum(e['allocated_seconds'] for e in exits) / 3600,
             'ceiling_gpu_hours': C.CEILING_SECONDS / 3600}
    C.write(C.BUNDLE / 'CAPACITY_TABLE.json', table)
    (C.BUNDLE / 'CAPACITY_TABLE.md').write_text(markdown(table))
    print(markdown(table))


if __name__ == '__main__':
    main()
