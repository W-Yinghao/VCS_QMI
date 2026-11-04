"""R46 summary (numbers only). A: COCO 80% c80 s1001 FIT pool full (R43-A) / 58k / 29k x {N-frozen, N-JOINT}, JOINT-FROZEN per pool. B: round 2
(REBUILT / CONTINUED) vs round 1 (8000) per condition x objective; secondary student-top-100 readout if R46_STUDENT_TOP100.json exists; coverage
table from R46_COVERAGE.json. Writes RESULTS_R46.md, R46_RESULTS.csv, R46_CONTRASTS.csv, R46_CURVES.csv, R46_COST.csv, R46_FAILURES.csv.
Missing -> NA. Usage: python3 -I -B tools/summarize_r46.py"""
import csv, json, math
from pathlib import Path
W = Path(__file__).resolve().parents[1]
OUT = Path('/projects/EEG-foundation-model/yinghao/outputs_v9/r40_scope_present_20261103')
R39R = Path('/projects/EEG-foundation-model/yinghao/outputs_v9/r39_complete_joint_20261102/runs')
OBJ = ('J', 'BCE', 'NCE'); MET = ('i2t_r1', 'i2t_r5', 'i2t_r10', 't2i_r1', 't2i_r5', 't2i_r10', 'rsum')
R1 = {'F60_R30_H1001': lambda o: R39R / f'R39_Flickr60_N_joint_global_{o}', 'F80_IN2R_H1001': lambda o: OUT / 'runs_r41' / f'R41_C_F80_IN2R_H1001_N_joint_{o}',
      'C80_IN2R_H1001': lambda o: OUT / 'runs_r43' / f'R43_A_C80_IN2R_H1001_N_joint_{o}'}


def rj(p):
    p = Path(p); return json.loads(p.read_text()) if p.exists() else None


def rs(d):
    r = rj(Path(d) / 'RESULT.json'); return r['final']['dev_retrieval']['rsum'] if r else None


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
    units = json.loads((W / 'configs' / 'UNITS_R46.json').read_text()); rows, curves, cost, cx, fails = [], [], [], [], []
    for u in units:
        d = OUT / 'runs_r46' / u; r = rj(d / 'RESULT.json'); c = json.loads((W / 'configs' / f'{u}.json').read_text())
        base = dict(unit=u, block=c['block'], cond=c['cond'], adaptation=c['adaptation'], objective=c['objective'], round2=c.get('round2'), status=r['status'] if r else 'NA')
        for fl in sorted(d.glob('FAILED_*.json')):
            fails.append(dict(unit=u, file=fl.name, last_line=json.loads(fl.read_text())['traceback'].strip().splitlines()[-1][:300]))
        if r is None:
            rows.append(base); continue
        rows.append(dict(base, **{k: r['final']['dev_retrieval'][k] for k in MET}, H_rsum=r['native_H_dev']['rsum'], gpu=r['env']['gpu'], job=r['env']['job']))
        for ph in r['phases']:
            for x in ph['curves']:
                curves.append(dict(base, step=x['step'], **{k: x['dev_retrieval'][k] for k in MET}))
        cost.append(dict(base, train_seconds=r['cost']['train_seconds'], eval_seconds=r['cost']['eval_seconds'], gpu=r['env']['gpu']))
    n_done = sum(1 for r in rows if r['status'] == 'COMPLETE')
    L = ['# R46 results (numbers only)', '', '> **Status note (2026-11-03):** R46 JOINT jobs (A JOINT arms, all of B) were cancelled by the user after the R44 audit (JOINT student layout leak); only units with DONE are tabulated, the rest are NA. Numbers use the native per-batch centring; layout-free re-evaluations: CORRECTED_READOUTS.md.', '', f'{n_done}/{len(units)} R46 units complete. DEV two-stage (frozen-H top-100 reranked), native 1K COCO / Flickr DEV. TEST not read.', '',
         '## A: COCO 80% c80 s1001, FIT pool restriction (8000 updates)', '', '| pool | obj | N-frozen | N-JOINT | JOINT-FROZEN |', '|---|---|---|---|---|']
    for pool, fz, jt in (('full (R43-A)', lambda o: OUT / 'runs_r43' / f'R43_A_C80_IN2R_H1001_N_frozen_{o}', lambda o: OUT / 'runs_r43' / f'R43_A_C80_IN2R_H1001_N_joint_{o}'),
                         ('58k', lambda o: OUT / 'runs_r46' / f'R46_A_C80_IN2R_H1001_P58K_N_frozen_{o}', lambda o: OUT / 'runs_r46' / f'R46_A_C80_IN2R_H1001_P58K_N_joint_{o}'),
                         ('29k', lambda o: OUT / 'runs_r46' / f'R46_A_C80_IN2R_H1001_P29K_N_frozen_{o}', lambda o: OUT / 'runs_r46' / f'R46_A_C80_IN2R_H1001_P29K_N_joint_{o}')):
        for o in OBJ:
            a, b = rs(fz(o)), rs(jt(o)); L.append(f'| {pool} | {o} | {f(a)} | {f(b)} | {f(sub(b, a))} |')
            cx.append(dict(block='A', pool=pool, objective=o, N_frozen=a, N_joint=b, JOINT_minus_FROZEN=sub(b, a)))
    L += ['', '## B: iterated N supervision, round 2 (8000 more updates from the round-1 step-8000 state)', '',
          '| condition | obj | round 1 (8000) | R2 CONTINUED | R2 REBUILT | REBUILT-CONTINUED | CONTINUED-R1 | REBUILT-R1 |', '|---|---|---|---|---|---|---|---|']
    for pk, f1 in R1.items():
        for o in OBJ:
            a = rs(f1(o)); c_ = rs(OUT / 'runs_r46' / f'R46_B_{pk}_R2_CONTINUED_{o}'); r_ = rs(OUT / 'runs_r46' / f'R46_B_{pk}_R2_REBUILT_{o}')
            L.append(f'| {pk} | {o} | {f(a)} | {f(c_)} | {f(r_)} | {f(sub(r_, c_))} | {f(sub(c_, a))} | {f(sub(r_, a))} |')
            cx.append(dict(block='B', condition=pk, objective=o, round1=a, R2_CONTINUED=c_, R2_REBUILT=r_, REBUILT_minus_CONTINUED=sub(r_, c_), CONTINUED_minus_R1=sub(c_, a), REBUILT_minus_R1=sub(r_, a)))
    st = rj(W / 'R46_STUDENT_TOP100.json')
    if st:
        L += ['', '### B secondary: round-2 student own top-100 two-stage DEV (post-hoc, final checkpoints)', '', '| unit | rsum (student top-100) | rsum (frozen-H top-100, primary) |', '|---|---|---|']
        for u, v in sorted(st.get('units', {}).items()):
            L.append(f"| {u} | {f(v.get('rsum_student_top100'))} | {f(v.get('rsum_primary'))} |")
    cov = rj(W / 'R46_COVERAGE.json')
    if cov:
        L += ['', '## Zero-training coverage (training record pool; truth columns Flickr only, evaluator-only)', '',
              '| condition | table | records | i2t partner in top32 | t2i partner in top32 | i2t observed pos slots | truth: t2i true image in top32 | truth: obs pair correct |', '|---|---|---|---|---|---|---|---|']
        for cn, tabs in cov['conditions'].items():
            for tn, s in tabs.items():
                L.append(f"| {cn} | {tn} | {s['n_records']} | {f(s['i2t_partner_in_top32'], 3)} | {f(s['t2i_partner_in_top32'], 3)} | {f(s['i2t_mean_observed_positive_slots'], 3)} | "
                         f"{f(s.get('truth_t2i_true_image_in_top32'), 3)} | {f(s.get('truth_observed_pair_correct'), 3)} |")
    L += ['', 'Files: R46_RESULTS.csv, R46_CONTRASTS.csv, R46_CURVES.csv, R46_COST.csv, R46_FAILURES.csv, R46_COVERAGE.json/csv.']
    (W / 'RESULTS_R46.md').write_text('\n'.join(L) + '\n')
    wcsv('R46_RESULTS.csv', rows); wcsv('R46_CONTRASTS.csv', cx); wcsv('R46_CURVES.csv', curves); wcsv('R46_COST.csv', cost)
    wcsv('R46_FAILURES.csv', fails or [dict(unit=None, file=None, last_line=None)])
    print('R46 summary', n_done, '/', len(units))


if __name__ == '__main__':
    main()
