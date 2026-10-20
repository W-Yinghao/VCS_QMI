"""Capacity table under the 2026-10-19 hardware amendment (separate table, never merged into the BF1 main table).

Units: the 12 base units (base freeze, base verifier, all L40S) and the 6 seed-1003 units (amended verifier; RTX6000PRO, L40S or
H100).  The K16 references are all L40S.  Reported:
  * pooled: the pre-registered statistics of capacity_table.py (same bootstrap seed and consumption order), every unit with its GPU
  * by hardware stratum: the same statistics within each GPU type of the K8/K32 unit (default_rng(20261019 + 1)); a stratum with one
    seed is descriptive only; the L40S stratum is the hardware-matched sensitivity (K and K16 on the same GPU type)
No claim of cross-hardware equivalence: a non-L40S seed's K-vs-K16 difference includes the hardware difference.
CPU only (no Torch):  V10_CAPACITY_FREEZE=<base> V10_CAPACITY_HW_FREEZE=<hw> python capacity_table_hw.py
"""
import argparse
import os

import numpy as np

import capacity_common as C
import capacity_table as T
import hw_amendment as H

SUBSETS = ('DEV32', 'DEV24')


def unit_row(u, prereg, base_freeze, hw_freeze):
    base_dir, hw_dir = C.OUT / u['key'], H.OUT_HW / u['key']
    if (base_dir / 'COMPLETE.json').exists():
        r, d, source = C.verify_result(u, prereg, base_freeze), base_dir, 'base_freeze'
    elif u['key'] in H.UNIT_KEYS and (hw_dir / 'COMPLETE.json').exists():
        r, d, source = H.verify_result(u, prereg, base_freeze, hw_freeze_sha=hw_freeze), hw_dir, 'hw_amendment'
    else:
        return None
    hardware = H.hardware_of(d, r['artifacts'])
    gpus = sorted({g for g, _, _ in hardware})
    ref, ref_dir = T.k16_reference(prereg, u['family'], u['seed'])
    ref_gpus = sorted({g for g, _, _ in H.hardware_of(ref_dir, ref['artifacts'])})
    row = {'seed': u['seed'], 'unit': u['key'], 'source': source, 'hardware': hardware, 'k16_reference_hardware': ref_gpus,
           'stratum': gpus[0] if len(gpus) == 1 else 'mixed:' + '+'.join(gpus),
           'hardware_matched_to_k16': gpus == ref_gpus, 'selected_step': r['selected']['step'],
           'k16_selected_step': ref['selected']['step'], 'cost': T.unit_cost(u['key']),
           'budget_refusal_history': (C.BUNDLE / 'budget' / f"{u['key']}.json").exists()}
    for subset in SUBSETS:
        row[subset] = r[subset]
        row[subset + '_k16'] = ref[subset]
        row[subset + '_delta'] = {m: r[subset][m] - ref[subset][m] for m in ('success42', 'success9', 'median_err')}
    if u['family'] == 'MF_VCS_M1':
        row['m1'], row['m1_k16'] = T.m1_diagnostics(d), T.m1_diagnostics(ref_dir)
    row['_per_core'], row['_per_core_k16'], row['_dev24'] = r['per_core'], ref['per_core'], r['DEV24_cores']
    return row


def summarize(rows, rng):
    out = {'seeds': [x['seed'] for x in rows]}
    for subset in SUBSETS:
        cores = sorted(rows[0]['_per_core']) if subset == 'DEV32' else rows[0]['_dev24']
        out[subset] = {}
        for m in T.RATES:
            diff = np.array([[x['_per_core'][c][m] - x['_per_core_k16'][c][m] for c in cores] for x in rows])
            per_seed = [x[subset + '_delta'][m] for x in rows]
            C.require(np.allclose(diff.mean(1), per_seed, atol=1e-12), 'Per-core and pooled rates disagree')
            lo, hi = T.interval(diff, rng)
            out[subset][m] = {'delta_per_seed': per_seed, 'delta_mean': float(np.mean(per_seed)), 'ci95': [lo, hi],
                              'grid': T.grid(lo, hi, per_seed) if len(rows) >= 2 else 'single_seed_descriptive'}
        out[subset]['median_err'] = {'delta_per_seed': [x[subset + '_delta']['median_err'] for x in rows]}
    return out


def build(prereg, base_freeze, hw_freeze):
    pooled_rng, strata_rng = np.random.default_rng(T.BOOT_SEED), np.random.default_rng(T.BOOT_SEED + 1)
    cells = []
    for family in C.FAMILIES:
        for K in C.KS:
            units = [u for u in prereg['units'] if u['family'] == family and u['K'] == K]
            rows = [unit_row(u, prereg, base_freeze, hw_freeze) for u in units]
            missing = [u['key'] for u, x in zip(units, rows) if x is None]
            rows = [x for x in rows if x is not None]
            cell = {'family': family, 'K': K, 'status': 'INCOMPLETE' if missing else 'COMPLETE', 'missing': missing}
            if not missing:
                cell['pooled'] = summarize(rows, pooled_rng)
                cell['strata'] = {s: summarize([x for x in rows if x['stratum'] == s], strata_rng)
                                  for s in sorted({x['stratum'] for x in rows})}
            for x in rows:
                for k in ('_per_core', '_per_core_k16', '_dev24'):
                    x.pop(k)
            cell['units'] = rows
            cells.append(cell)
    return cells


def markdown(table):
    f = lambda s: f"{s['delta_mean']:+.4f} [{s['ci95'][0]:+.4f}, {s['ci95'][1]:+.4f}] {s['grid']}"
    out = ['# BF1 capacity extension (fixed training rule), hardware-amended table', '',
           'Capacity response under a fixed training rule (固定训练规则下的容量响应); separate from the BF1 main table. K16 references and '
           'the seed-1001/1002 units ran on NVIDIA L40S; a seed-1003 unit on another GPU type compares K across hardware as well. '
           'No claim of cross-hardware equivalence.', '',
           '| family | K | seed | GPU | node | ckpt | DEV32 <42 | <9 | median | Δ<42 vs K16 | Δ<9 | GPU-h |', '|---|---|---|---|---|---|---|---|---|---|---|---|']
    for c in table['cells']:
        for u in c['units']:
            out.append(f"| {c['family']} | {c['K']} | {u['seed']} | {u['stratum']} | {','.join(sorted({h for _, _, h in u['hardware']}))} | "
                       f"{u['selected_step']} | {u['DEV32']['success42']:.4f} | {u['DEV32']['success9']:.4f} | {u['DEV32']['median_err']:.3f} | "
                       f"{u['DEV32_delta']['success42']:+.4f} | {u['DEV32_delta']['success9']:+.4f} | {u['cost']['allocated_gpu_hours']:.3f} |")
    out += ['', '| family | K | view | seeds | DEV32 Δ<42 [95%] grid | DEV32 Δ<9 [95%] grid | DEV24 Δ<42 [95%] grid |', '|---|---|---|---|---|---|---|']
    for c in table['cells']:
        if c['status'] != 'COMPLETE':
            out.append(f"| {c['family']} | {c['K']} | INCOMPLETE ({', '.join(c['missing'])}) | | | | |")
            continue
        views = [('pooled (all hardware)', c['pooled'])] + [(f'stratum {s}', v) for s, v in c['strata'].items()]
        for name, v in views:
            out.append(f"| {c['family']} | {c['K']} | {name} | {','.join(map(str, v['seeds']))} | {f(v['DEV32']['success42'])} | "
                       f"{f(v['DEV32']['success9'])} | {f(v['DEV24']['success42'])} |")
    out += ['', f"Total allocated GPU-h of the extension (native, all types): {table['total_allocated_gpu_hours']:.3f} "
                f"(ceiling {table['ceiling_gpu_hours']:.0f}); by GPU type: {table['gpu_hours_by_type']}.", '']
    return '\n'.join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--freeze', default=os.environ.get('V10_CAPACITY_FREEZE'))
    ap.add_argument('--hw-freeze', default=os.environ.get('V10_CAPACITY_HW_FREEZE'))
    a = ap.parse_args()
    prereg = C.validate_prereg(C.frozen(a.freeze))
    H.validate_record(H.frozen_hw(a.hw_freeze), prereg)
    cells = build(prereg, a.freeze, a.hw_freeze)
    exits = [C.read(p) for p in (C.BUNDLE / 'exits').glob('*.json')] if (C.BUNDLE / 'exits').exists() else []
    runtime = {p.stem: C.read(p) for p in (C.BUNDLE / 'runtime').glob('*.json')} if (C.BUNDLE / 'runtime').exists() else {}
    by_type = {}
    for e in exits:
        gpu = runtime.get(e['job'], {}).get('environment', {}).get('gpu_name', 'unknown (no runtime record)')
        by_type[gpu] = by_type.get(gpu, 0.0) + e['allocated_seconds'] / 3600
    table = {'schema': 'v10.capacity_ext.table_hw.v1', 'utc': C.utc(), 'base_freeze_sha256': a.freeze, 'hw_freeze_sha256': a.hw_freeze,
             'hardware_amendment_sha256': C.sha(H.DIR / 'HARDWARE_AMENDMENT.json'), 'bootstrap': {'B': T.B_RESAMPLES,
             'pooled_seed': T.BOOT_SEED, 'strata_seed': T.BOOT_SEED + 1}, 'cells': cells,
             'complete': all(c['status'] == 'COMPLETE' for c in cells),
             'total_allocated_gpu_hours': sum(e['allocated_seconds'] for e in exits) / 3600,
             'gpu_hours_by_type': {k: round(v, 3) for k, v in sorted(by_type.items())}, 'ceiling_gpu_hours': C.CEILING_SECONDS / 3600}
    C.write(H.DIR / 'CAPACITY_TABLE_HW.json', table)
    (H.DIR / 'CAPACITY_TABLE_HW.md').write_text(markdown(table))
    print(markdown(table))


if __name__ == '__main__':
    main()
