"""R42 summary (numbers only): noise dose-response of JOINT on IN2R-native Flickr parents (20/40/80%, R42 + the R41 C s1001 cells) and the
ORACLE_MISSING x {frozen, JOINT} cells on f80 s1001. Writes RESULTS_R42.md, R42_RESULTS.csv, R42_CURVE.csv, R42_CONTRASTS.csv, R42_CURVES.csv,
R42_COST.csv, R42_FAILURES.csv. Missing -> NA. Usage: python3 -I -B tools/summarize_r42.py"""
import csv, glob, json, math
from pathlib import Path
import numpy as np
W = Path(__file__).resolve().parents[1]
OUT = Path('/projects/EEG-foundation-model/yinghao/outputs_v9/r40_scope_present_20261103')
OBJ = ('J', 'BCE', 'NCE'); MET = ('i2t_r1', 'i2t_r5', 'i2t_r10', 't2i_r1', 't2i_r5', 't2i_r10', 'rsum')
PARENTS = {20: ['F20_IN2R_H1001', 'F20_IN2R_H1002', 'F20_IN2R_H1003'], 40: ['F40_IN2R_H1001', 'F40_IN2R_H1002', 'F40_IN2R_H1003'], 80: ['F80_IN2R_H1001', 'F80_IN2R_H1002', 'F80_IN2R_H1003']}


def rj(p):
    p = Path(p); return json.loads(p.read_text()) if p.exists() else None


def unit(pk, kind, o):
    if pk.endswith('H1001') and not pk.startswith('F20'):
        return OUT / 'runs_r41' / f'R41_C_{pk}_N_{kind}_{o}'
    return OUT / 'runs_r42' / f'R42_N_{pk}_N_{kind}_{o}'


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
    rows, curve, curves, cost, cx = [], [], [], [], []
    L = ['# R42 noise dose-response of JOINT on IN2R-native Flickr parents (numbers only)', '',
         'N route, 8000 updates, DEV two-stage (frozen-H top-100 reranked). s1001 cells at 40/80% are R41 C; all other parents are R42. TEST not read.', '',
         '| noise | parent | H | obj | N-frozen | N-JOINT | JOINT-FROZEN |', '|---|---|---|---|---|---|---|']
    agg = {}
    for nz, pks in PARENTS.items():
        for pk in pks:
            for o in OBJ:
                rf, rjt = rj(unit(pk, 'frozen', o) / 'RESULT.json'), rj(unit(pk, 'joint', o) / 'RESULT.json')
                H = (rf or rjt or {}).get('native_H_dev', {}).get('rsum')
                d = sub(rs(rjt), rs(rf)); agg.setdefault((nz, o), []).append(d)
                L.append(f'| {nz}% | {pk} | {f(H)} | {o} | {f(rs(rf))} | {f(rs(rjt))} | {f(d)} |')
                curve.append(dict(noise=nz, parent=pk, objective=o, H=H, N_frozen=rs(rf), N_joint=rs(rjt), JOINT_minus_FROZEN=d,
                                  frozen_minus_H=sub(rs(rf), H), joint_minus_H=sub(rs(rjt), H)))
                for kind, r in (('frozen', rf), ('joint', rjt)):
                    base = dict(noise=nz, parent=pk, cell=f'N_{kind}', objective=o, unit=unit(pk, kind, o).name, status=r['status'] if r else 'NA')
                    if r is None:
                        rows.append(base); continue
                    r40 = r.get('r40') or {}
                    rows.append(dict(base, **{k: r['final']['dev_retrieval'][k] for k in MET}, H_rsum=r['native_H_dev']['rsum'], one_step_r40=(r40.get('one_step_check') or {}).get('passed'),
                                     gpu=r['env']['gpu'], job=r['env']['job']))
                    for ph in r['phases']:
                        for x in ph['curves']:
                            curves.append(dict(base, step=x['step'], **{k: x['dev_retrieval'][k] for k in MET}))
                    cost.append(dict(base, train_seconds=r['cost']['train_seconds'], eval_seconds=r['cost']['eval_seconds'], gpu=r['env']['gpu']))
    L += ['', '## JOINT-FROZEN by noise level (mean over the 3 parents; min / max; n available)', '', '| noise | obj | mean | min | max | n |', '|---|---|---|---|---|---|']
    for (nz, o), v in sorted(agg.items()):
        v2 = [x for x in v if x is not None]
        L.append(f"| {nz}% | {o} | {f(float(np.mean(v2)) if v2 else None)} | {f(min(v2) if v2 else None)} | {f(max(v2) if v2 else None)} | {len(v2)} |")
        cx.append(dict(block='curve', noise=nz, objective=o, mean_JOINT_minus_FROZEN=float(np.mean(v2)) if v2 else None, min=min(v2) if v2 else None, max=max(v2) if v2 else None, n=len(v2)))
    L += ['', '## ORACLE_MISSING on the 80% parent f80 s1001 (8000)', '', '| obj | N-frozen (R41 C) | ORACLE-frozen | N-JOINT (R41 C) | ORACLE-JOINT | OJ-NJ | OF-NF | [OJ-OF]-[NJ-NF] |', '|---|---|---|---|---|---|---|---|']
    for o in OBJ:
        nf = rs(rj(OUT / 'runs_r41' / f'R41_C_F80_IN2R_H1001_N_frozen_{o}' / 'RESULT.json')); nj = rs(rj(OUT / 'runs_r41' / f'R41_C_F80_IN2R_H1001_N_joint_{o}' / 'RESULT.json'))
        rof = rj(OUT / 'oracle_runs_r42' / f'R42_O_F80_IN2R_H1001_ORACLE_frozen_{o}' / 'RESULT.json'); roj = rj(OUT / 'oracle_runs_r42' / f'R42_O_F80_IN2R_H1001_ORACLE_joint_{o}' / 'RESULT.json')
        of, oj = rs(rof), rs(roj)
        it = None if None in (nf, nj, of, oj) else (oj - of) - (nj - nf)
        L.append(f'| {o} | {f(nf)} | {f(of)} | {f(nj)} | {f(oj)} | {f(sub(oj, nj))} | {f(sub(of, nf))} | {f(it)} |')
        cx.append(dict(block='oracle_f80', objective=o, N_frozen=nf, ORACLE_frozen=of, N_joint=nj, ORACLE_joint=oj, OJ_minus_NJ=sub(oj, nj), OF_minus_NF=sub(of, nf), interaction=it))
        for kind, r in (('ORACLE_frozen', rof), ('ORACLE_joint', roj)):
            base = dict(noise=80, parent='F80_IN2R_H1001', cell=kind, objective=o, unit=f'R42_O_F80_IN2R_H1001_{kind}_{o}', status=r['status'] if r else 'NA')
            if r:
                rows.append(dict(base, **{k: r['final']['dev_retrieval'][k] for k in MET}, H_rsum=r['native_H_dev']['rsum'], one_step_r40=((r.get('r40') or {}).get('one_step_check') or {}).get('passed'),
                                 gpu=r['env']['gpu'], job=r['env']['job']))
                for ph in r['phases']:
                    for x in ph['curves']:
                        curves.append(dict(base, step=x['step'], **{k: x['dev_retrieval'][k] for k in MET}))
                cost.append(dict(base, train_seconds=r['cost']['train_seconds'], eval_seconds=r['cost']['eval_seconds'], gpu=r['env']['gpu']))
            else:
                rows.append(base)
    fails = []
    for p in sorted(glob.glob(str(OUT / '*runs_r42' / '*' / 'FAILED_*.json'))):
        x = rj(p); fails.append(dict(unit=x['unit'], file=p, utc=x['utc'], error=x['traceback'].strip().splitlines()[-1][:300]))
    for p in sorted(glob.glob(str(OUT / 'r42_parents' / 'cache' / '*' / 'FAILED_REPRO_*.json'))):
        x = rj(p); fails.append(dict(unit=f'parent build {Path(p).parent.name}', file=p, utc=x['utc'], error=f"native DEV {x['native']['rsum']} vs checkpoint {x['native']['checkpoint_best_rsum']} on {x['gpu']} (not exact; rebuilt on the training GPU type)"))
    wcsv('R42_RESULTS.csv', rows); wcsv('R42_CURVE.csv', curve); wcsv('R42_CONTRASTS.csv', cx); wcsv('R42_CURVES.csv', curves); wcsv('R42_COST.csv', cost)
    wcsv('R42_FAILURES.csv', fails or [dict(unit='none', file='', utc='', error='')])
    n42 = sum(1 for r in rows if r.get('status') == 'COMPLETE' and r['unit'].startswith('R42_'))
    L.insert(2, f'{n42}/48 R42 units complete.')
    L += ['', 'Files: R42_RESULTS.csv, R42_CURVE.csv, R42_CONTRASTS.csv, R42_CURVES.csv, R42_COST.csv, R42_FAILURES.csv (incl. the f40 s1003 RTX reproduction failure, rebuilt on V100S).']
    (W / 'RESULTS_R42.md').write_text('\n'.join(L) + '\n')
    print('R42 summary', n42)


if __name__ == '__main__':
    main()
