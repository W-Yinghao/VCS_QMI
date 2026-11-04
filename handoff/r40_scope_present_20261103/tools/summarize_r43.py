"""R43 summary (numbers only). A: MS-COCO IN2R-native R33 H0 parents (20 / 60 / 80%) x {N-frozen, N-JOINT}. B: TRUST+N (R37 TRUST_NATIVE_NULL)
x {frozen, JOINT} on the 80% Flickr parents and CC152K CC31, against the same parent's N-frozen / N-JOINT (R41 C / R42 / R38 / R39).
Writes RESULTS_R43.md, R43_RESULTS.csv, R43_CONTRASTS.csv, R43_CURVES.csv, R43_COST.csv, R43_FAILURES.csv. Missing -> NA.
Usage: python3 -I -B tools/summarize_r43.py"""
import csv, glob, json, math
from pathlib import Path
import numpy as np
W = Path(__file__).resolve().parents[1]
OUT = Path('/projects/EEG-foundation-model/yinghao/outputs_v9/r40_scope_present_20261103')
R38R = Path('/projects/EEG-foundation-model/yinghao/outputs_v9/r38_direction_20261102/runs')
R39R = Path('/projects/EEG-foundation-model/yinghao/outputs_v9/r39_complete_joint_20261102/runs')
OBJ = ('J', 'BCE', 'NCE'); MET = ('i2t_r1', 'i2t_r5', 'i2t_r10', 't2i_r1', 't2i_r5', 't2i_r10', 'rsum')
COCO = {20: ['C20_IN2R_H1001'], 60: ['C60_IN2R_H1001', 'C60_IN2R_H1002', 'C60_IN2R_H1003'], 80: ['C80_IN2R_H1001', 'C80_IN2R_H1002', 'C80_IN2R_H1003']}
NREF = {'F80_IN2R_H1001': lambda k, o: OUT / 'runs_r41' / f'R41_C_F80_IN2R_H1001_N_{k}_{o}', 'F80_IN2R_H1002': lambda k, o: OUT / 'runs_r42' / f'R42_N_F80_IN2R_H1002_N_{k}_{o}',
        'F80_IN2R_H1003': lambda k, o: OUT / 'runs_r42' / f'R42_N_F80_IN2R_H1003_N_{k}_{o}',
        'CC31_H1001': lambda k, o: (R38R / f'r38_B_cc_R31_s1001_N_shared_{o}_cosine') if k == 'frozen' else (R39R / f'R39_CC152K_N_joint_global_{o}')}


def rj(p):
    p = Path(p); return json.loads(p.read_text()) if p.exists() else None


def rs(r, k='rsum'):
    return r['final']['dev_retrieval'][k] if r else None


def f(x, nd=2):
    return 'NA' if x is None or (isinstance(x, float) and math.isnan(x)) else (f'{x:.{nd}f}' if isinstance(x, float) else str(x))


def sub(a, b):
    return None if a is None or b is None else a - b


def wcsv(name, rows):
    if rows:
        keys = list(dict.fromkeys(k for r in rows for k in r))
        with open(W / name, 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rows)


def main():
    units = json.loads((W / 'configs' / 'UNITS_R43.json').read_text())
    rows, curves, cost, cx, fails = [], [], [], [], []
    for u in units:
        d = OUT / 'runs_r43' / u; r = rj(d / 'RESULT.json'); c = json.loads((W / 'configs' / f'{u}.json').read_text())
        base = dict(unit=u, block=c['block'], cond=c['cond'], supervision=c['supervision'], adaptation=c['adaptation'], objective=c['objective'], status=r['status'] if r else 'NA')
        for fl in sorted(d.glob('FAILED_*.json')):
            fails.append(dict(unit=u, file=fl.name, last_line=json.loads(fl.read_text())['traceback'].strip().splitlines()[-1][:300]))
        if r is None:
            rows.append(base); continue
        r40 = r.get('r40') or {}
        rows.append(dict(base, **{k: r['final']['dev_retrieval'][k] for k in MET}, H_rsum=r['native_H_dev']['rsum'], one_step_r40=(r40.get('one_step_check') or {}).get('passed'),
                         gpu=r['env']['gpu'], job=r['env']['job']))
        for ph in r['phases']:
            for x in ph['curves']:
                curves.append(dict(base, step=x['step'], **{k: x['dev_retrieval'][k] for k in MET}))
        cost.append(dict(base, train_seconds=r['cost']['train_seconds'], eval_seconds=r['cost']['eval_seconds'], gpu=r['env']['gpu']))
    n_done = sum(1 for r in rows if r['status'] == 'COMPLETE')
    L = ['# R43 results (numbers only)', '', f'{n_done}/{len(units)} R43 units complete. 8000 updates, shared G + cosine, DEV two-stage (frozen-H top-100 reranked). TEST not read.', '',
         '## A: MS-COCO IN2R-native parents (DEV = native 1K-image split, 1000 images x 5000 captions)', '',
         '| noise | parent | H | obj | N-frozen | N-JOINT | JOINT-FROZEN | frozen-H | JOINT-H |', '|---|---|---|---|---|---|---|---|---|']
    agg = {}
    for nz, pks in COCO.items():
        for pk in pks:
            for o in OBJ:
                rf = rj(OUT / 'runs_r43' / f'R43_A_{pk}_N_frozen_{o}' / 'RESULT.json'); rjt = rj(OUT / 'runs_r43' / f'R43_A_{pk}_N_joint_{o}' / 'RESULT.json')
                H = (rf or rjt or {}).get('native_H_dev', {}).get('rsum'); dj = sub(rs(rjt), rs(rf)); agg.setdefault((nz, o), []).append(dj)
                L.append(f'| {nz}% | {pk} | {f(H)} | {o} | {f(rs(rf))} | {f(rs(rjt))} | {f(dj)} | {f(sub(rs(rf), H))} | {f(sub(rs(rjt), H))} |')
                cx.append(dict(block='A', noise=nz, parent=pk, objective=o, H=H, N_frozen=rs(rf), N_joint=rs(rjt), JOINT_minus_FROZEN=dj, frozen_minus_H=sub(rs(rf), H), joint_minus_H=sub(rs(rjt), H)))
    L += ['', '### A: JOINT-FROZEN by noise level (mean / min / max over parents; n available)', '', '| noise | obj | mean | min | max | n |', '|---|---|---|---|---|---|']
    for (nz, o), v in sorted(agg.items()):
        v2 = [x for x in v if x is not None]
        L.append(f"| {nz}% | {o} | {f(float(np.mean(v2)) if v2 else None)} | {f(min(v2) if v2 else None)} | {f(max(v2) if v2 else None)} | {len(v2)} |")
    L += ['', '## B: TRUST+N (R37 TRUST_NATIVE_NULL) vs N on the same parent', '',
          '| parent | obj | N-frozen | TRUST-frozen | N-JOINT | TRUST-JOINT | TF-NF | TJ-NJ | TJ-TF | [TJ-TF]-[NJ-NF] |', '|---|---|---|---|---|---|---|---|---|---|']
    for pk, nref in NREF.items():
        for o in OBJ:
            nf, nj = rs(rj(nref('frozen', o) / 'RESULT.json')), rs(rj(nref('joint', o) / 'RESULT.json'))
            tf = rs(rj(OUT / 'runs_r43' / f'R43_B_{pk}_TRUST_frozen_{o}' / 'RESULT.json')); tj = rs(rj(OUT / 'runs_r43' / f'R43_B_{pk}_TRUST_joint_{o}' / 'RESULT.json'))
            it = None if None in (nf, nj, tf, tj) else (tj - tf) - (nj - nf)
            L.append(f'| {pk} | {o} | {f(nf)} | {f(tf)} | {f(nj)} | {f(tj)} | {f(sub(tf, nf))} | {f(sub(tj, nj))} | {f(sub(tj, tf))} | {f(it)} |')
            cx.append(dict(block='B', parent=pk, objective=o, N_frozen=nf, TRUST_frozen=tf, N_joint=nj, TRUST_joint=tj, TF_minus_NF=sub(tf, nf), TJ_minus_NJ=sub(tj, nj), TJ_minus_TF=sub(tj, tf), interaction=it))
    L += ['', 'Files: R43_RESULTS.csv (six recalls, in-unit check flags), R43_CONTRASTS.csv, R43_CURVES.csv, R43_COST.csv, R43_FAILURES.csv.']
    (W / 'RESULTS_R43.md').write_text('\n'.join(L) + '\n')
    wcsv('R43_RESULTS.csv', rows); wcsv('R43_CONTRASTS.csv', cx); wcsv('R43_CURVES.csv', curves); wcsv('R43_COST.csv', cost)
    wcsv('R43_FAILURES.csv', fails or [dict(unit=None, file=None, last_line=None)])
    print('R43 summary', n_done, '/', len(units))


if __name__ == '__main__':
    main()
