"""R44 summary (numbers only). A: Flickr60 R30-H s1001/2/3 N-JOINT 16k, training seed 2002 (R44) vs 1001 (R41-A), paired per parent at
8000 / 12000 / 16000 (R41-A has no 12000 readout -> NA), nested means, J-BCE / J-NCE per seed. B/C: per condition x objective: ALL, PRESENT_ALL,
ROUTE_PRESENT, ROUTE_RANDOM and the package contrasts; per-arm J-BCE / J-NCE; routing exposure (mask rows / present rows / empty batches).
Writes RESULTS_R44.md, R44_RESULTS.csv, R44_CONTRASTS.csv, R44_CURVES.csv, R44_EXPOSURE.csv, R44_COST.csv, R44_FAILURES.csv. Missing -> NA.
Usage: python3 -I -B tools/summarize_r44.py"""
import csv, json, math
from pathlib import Path
import numpy as np
W = Path(__file__).resolve().parents[1]
OUT = Path('/projects/EEG-foundation-model/yinghao/outputs_v9/r40_scope_present_20261103')
R39R = Path('/projects/EEG-foundation-model/yinghao/outputs_v9/r39_complete_joint_20261102')
OBJ = ('J', 'BCE', 'NCE'); MET = ('i2t_r1', 'i2t_r5', 'i2t_r10', 't2i_r1', 't2i_r5', 't2i_r10', 'rsum')
ALL = {'F00_R30_H1001': lambda o: R39R / 'runs_r39x' / f'R39X_Flickr00_R30H_s1001_N_joint_global_{o}', 'F60_R30_H1001': lambda o: R39R / 'runs' / f'R39_Flickr60_N_joint_global_{o}',
       'F80_IN2R_H1001': lambda o: OUT / 'runs_r41' / f'R41_C_F80_IN2R_H1001_N_joint_{o}', 'CC_R31_H1001': lambda o: R39R / 'runs' / f'R39_CC152K_N_joint_global_{o}'}
PALL = {'F00_R30_H1001': lambda o: OUT / 'runs_r44' / f'R44_C_F00_R30_H1001_PRESENT_ALL_{o}_train1001', 'F60_R30_H1001': lambda o: OUT / 'runs' / f'R40_B_F60_R30_H1001_PRESENT_both_joint_{o}',
        'F80_IN2R_H1001': lambda o: OUT / 'runs_r44' / f'R44_C_F80_IN2R_H1001_PRESENT_ALL_{o}_train1001', 'CC_R31_H1001': lambda o: OUT / 'runs' / f'R40_B_CC31_H1001_PRESENT_both_joint_{o}'}


def rj(p):
    p = Path(p); return json.loads(p.read_text()) if p.exists() else None


def res(d):
    return rj(Path(d) / 'RESULT.json')


def rs(r):
    return r['final']['dev_retrieval']['rsum'] if r else None


def at(r, step):
    if not r:
        return None
    for ph in r['phases']:
        for x in ph['curves']:
            if x['step'] == step:
                return x['dev_retrieval']['rsum']
    return None


def f(x, nd=2):
    return 'NA' if x is None or (isinstance(x, float) and math.isnan(x)) else (f'{x:.{nd}f}' if isinstance(x, float) else str(x))


def sub(a, b):
    return None if a is None or b is None else a - b


def mean(v):
    v = [x for x in v if x is not None]; return float(np.mean(v)) if v else None


def wcsv(name, rows):
    if rows:
        keys = list(dict.fromkeys(k for r in rows for k in r))
        with open(W / name, 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rows)


def main():
    units = json.loads((W / 'configs' / 'UNITS_R44.json').read_text())
    rows, curves, cost, cx, expo, fails = [], [], [], [], [], []
    for u in units:
        d = OUT / 'runs_r44' / u; r = res(d); c = json.loads((W / 'configs' / f'{u}.json').read_text())
        base = dict(unit=u, block=c['block'], cond=c['cond'], arm=c.get('routing') or ('PRESENT_ALL' if c['supervision'] == 'N_PRESENT_ONLY' else 'ALL'), objective=c['objective'],
                    training_seed=c['init_seed'], status=r['status'] if r else 'NA')
        for fl in sorted(d.glob('FAILED_*.json')):
            fails.append(dict(unit=u, file=fl.name, last_line=json.loads(fl.read_text())['traceback'].strip().splitlines()[-1][:300]))
        if r is None:
            rows.append(base); continue
        r40 = r.get('r40') or {}; r44 = r.get('r44') or r['phases'][-1]['train_history'].get('r44') or {}
        rows.append(dict(base, **{k: r['final']['dev_retrieval'][k] for k in MET}, H_rsum=r['native_H_dev']['rsum'], one_step_r40=(r40.get('one_step_check') or {}).get('passed'),
                         routing_check=(r44.get('routing_check') or {}).get('passed'), gpu=r['env']['gpu'], job=r['env']['job']))
        for ph in r['phases']:
            for x in ph['curves']:
                curves.append(dict(base, step=x['step'], **{k: x['dev_retrieval'][k] for k in MET}))
        cost.append(dict(base, train_seconds=r['cost']['train_seconds'], eval_seconds=r['cost']['eval_seconds'], gpu=r['env']['gpu']))
        if r44:
            expo.append(dict(base, **{k: r44.get(k) for k in ('updates', 'opt_steps', 'mask_rows', 'query_rows', 'masked_rows', 'masked_rows_enc_grad_nonzero', 'masked_rows_head_grad_nonzero', 'empty_mask_batches', 'mask_hash')},
                             mask_fraction=(r44['mask_rows'] / r44['query_rows']) if r44.get('query_rows') else None))
    n_done = sum(1 for r in rows if r['status'] == 'COMPLETE')
    L = ['# R44 results (numbers only)', '', '> **R44 audit concluded (2026-11-03): the B/C ROUTE_PRESENT numbers come from a layout leak through the JOINT student centring (R44_AUDIT.md); do not use them.**', '', '> **Readout note (2026-11-03, R44 audit):** numbers in this file use the native IN²R per-batch centring over the sequential native layout (true image-caption pairs share a batch mean); JOINT students since R39 were also trained with that layout. Layout-free re-evaluations of the same checkpoints: CORRECTED_READOUTS.md; audit: R44_AUDIT.md; layout-free retraining: RESULTS_R47.md.', '', f'{n_done}/{len(units)} R44 units complete. DEV two-stage (frozen-H top-100 reranked). TEST not read.', '',
         '## A: Flickr60 R30-H N-JOINT 16000 updates, training seed 1001 (R41-A) vs 2002 (R44)', '',
         '| parent | obj | s1001 @8000 | s2002 @8000 | s1001 @12000 | s2002 @12000 | s1001 @16000 | s2002 @16000 | 2002-1001 @16000 |', '|---|---|---|---|---|---|---|---|---|']
    nest = {}
    for s in ('1001', '1002', '1003'):
        pk = f'F60_R30_H{s}'
        for o in OBJ:
            a1 = res(OUT / 'runs_r41' / f'R41_A_{pk}_N_joint_16k_{o}'); a2 = res(OUT / 'runs_r44' / f'R44_A_{pk}_ALL_{o}_train2002')
            v = {(sd, st): at(r_, st) for sd, r_ in (('1001', a1), ('2002', a2)) for st in (8000, 12000, 16000)}
            dd = sub(v[('2002', 16000)], v[('1001', 16000)]); nest.setdefault(o, []).append((v[('1001', 16000)], v[('2002', 16000)]))
            L.append(f"| {pk} | {o} | {f(v[('1001', 8000)])} | {f(v[('2002', 8000)])} | {f(v[('1001', 12000)])} | {f(v[('2002', 12000)])} | {f(v[('1001', 16000)])} | {f(v[('2002', 16000)])} | {f(dd)} |")
            cx.append(dict(block='A', parent=pk, objective=o, **{f's{sd}_at{st}': v[(sd, st)] for sd, st in v}, seed2002_minus_1001_16000=dd))
    L += ['', '### A: nested mean at 16000 (mean over parents of the per-parent seed mean) and objective gaps per seed', '', '| obj | nested mean | mean s1001 | mean s2002 |', '|---|---|---|---|']
    for o in OBJ:
        pr = nest.get(o, []); nm = mean([mean([a, b]) if None not in (a, b) else None for a, b in pr])
        L.append(f'| {o} | {f(nm)} | {f(mean([a for a, _ in pr]))} | {f(mean([b for _, b in pr]))} |')
    L += ['', '| parent | seed | J-BCE @16000 | J-NCE @16000 |', '|---|---|---|---|']
    for s in ('1001', '1002', '1003'):
        pk = f'F60_R30_H{s}'
        for sd in ('1001', '2002'):
            g = {o: at(res(OUT / 'runs_r41' / f'R41_A_{pk}_N_joint_16k_{o}') if sd == '1001' else res(OUT / 'runs_r44' / f'R44_A_{pk}_ALL_{o}_train2002'), 16000) for o in OBJ}
            L.append(f"| {pk} | {sd} | {f(sub(g['J'], g['BCE']))} | {f(sub(g['J'], g['NCE']))} |")
    L += ['', '## B/C: gradient routing (8000 updates, training seed 1001)', '',
          '| condition | obj | ALL | PRESENT_ALL | ROUTE_PRESENT | ROUTE_RANDOM | RP-ALL | RP-PA | RP-RR | RR-ALL |', '|---|---|---|---|---|---|---|---|---|---|']
    for ck in ALL:
        for o in OBJ:
            va, vp = rs(res(ALL[ck](o))), rs(res(PALL[ck](o)))
            vr, vq = rs(res(OUT / 'runs_r44' / f'R44_B_{ck}_ROUTE_PRESENT_{o}_train1001')), rs(res(OUT / 'runs_r44' / f'R44_B_{ck}_ROUTE_RANDOM_{o}_train1001'))
            L.append(f'| {ck} | {o} | {f(va)} | {f(vp)} | {f(vr)} | {f(vq)} | {f(sub(vr, va))} | {f(sub(vr, vp))} | {f(sub(vr, vq))} | {f(sub(vq, va))} |')
            cx.append(dict(block='BC', condition=ck, objective=o, ALL=va, PRESENT_ALL=vp, ROUTE_PRESENT=vr, ROUTE_RANDOM=vq, RP_minus_ALL=sub(vr, va), RP_minus_PA=sub(vr, vp), RP_minus_RR=sub(vr, vq), RR_minus_ALL=sub(vq, va)))
    L += ['', '### B/C: objective gaps per arm (J-BCE / J-NCE)', '', '| condition | arm | J-BCE | J-NCE |', '|---|---|---|---|']
    for ck in ALL:
        for arm, fn in (('ALL', ALL[ck]), ('PRESENT_ALL', PALL[ck]), ('ROUTE_PRESENT', lambda o, ck=ck: OUT / 'runs_r44' / f'R44_B_{ck}_ROUTE_PRESENT_{o}_train1001'),
                        ('ROUTE_RANDOM', lambda o, ck=ck: OUT / 'runs_r44' / f'R44_B_{ck}_ROUTE_RANDOM_{o}_train1001')):
            g = {o: rs(res(fn(o))) for o in OBJ}
            L.append(f"| {ck} | {arm} | {f(sub(g['J'], g['BCE']))} | {f(sub(g['J'], g['NCE']))} |")
    L += ['', 'Files: R44_RESULTS.csv, R44_CONTRASTS.csv, R44_CURVES.csv, R44_EXPOSURE.csv (routing mask / present rows, empty batches, masked-row gradient counters), R44_COST.csv, R44_FAILURES.csv.']
    (W / 'RESULTS_R44.md').write_text('\n'.join(L) + '\n')
    wcsv('R44_RESULTS.csv', rows); wcsv('R44_CONTRASTS.csv', cx); wcsv('R44_CURVES.csv', curves); wcsv('R44_EXPOSURE.csv', expo); wcsv('R44_COST.csv', cost)
    wcsv('R44_FAILURES.csv', fails or [dict(unit=None, file=None, last_line=None)])
    print('R44 summary', n_done, '/', len(units))


if __name__ == '__main__':
    main()
