"""R41 summary (numbers only). Reads the 39 R41 RESULT.json (runs_r41/, oracle_runs_r41/) and the read-only references (R38 N-FROZEN,
R39/R39X N-JOINT). Writes RESULTS_R41.md and R41_RESULTS.csv, R41_CONTRASTS.csv, R41_CURVES.csv, R41_COST.csv, R41_FAILURES.csv. Missing -> NA.
Usage: python3 -I -B tools/summarize_r41.py"""
import csv, glob, json, math
from pathlib import Path
W = Path(__file__).resolve().parents[1]
OUT = Path('/projects/EEG-foundation-model/yinghao/outputs_v9/r40_scope_present_20261103')
R39 = Path('/projects/EEG-foundation-model/yinghao/outputs_v9/r39_complete_joint_20261102')
R38 = Path('/projects/EEG-foundation-model/yinghao/outputs_v9/r38_direction_20261102/runs')
OBJ = ('J', 'BCE', 'NCE'); MET = ('i2t_r1', 'i2t_r5', 'i2t_r10', 't2i_r1', 't2i_r5', 't2i_r10', 'rsum')
JREF = {'F60_R30_H1001': R39 / 'runs' / 'R39_Flickr60_N_joint_global_{o}', 'F60_R30_H1002': R39 / 'runs_r39x' / 'R39X_Flickr60_R30H_s1002_N_joint_global_{o}',
        'F60_R30_H1003': R39 / 'runs_r39x' / 'R39X_Flickr60_R30H_s1003_N_joint_global_{o}', 'F00_R30_H1001': R39 / 'runs_r39x' / 'R39X_Flickr00_R30H_s1001_N_joint_global_{o}'}
FREF = {'F60_R30_H1001': R38 / 'r38_B_f60_R30_s1001_N_shared_{o}_cosine', 'F60_R30_H1002': R38 / 'r38_B_f60_R30_s1002_N_shared_{o}_cosine',
        'F60_R30_H1003': R38 / 'r38_B_f60_R30_s1003_N_shared_{o}_cosine', 'F00_R30_H1001': R39 / 'runs_r39x' / 'R39X_Flickr00_R30H_s1001_N_frozen_shared_{o}'}


def rj(p):
    p = Path(p); return json.loads(p.read_text()) if p.exists() else None


def res(path):
    return rj(Path(path) / 'RESULT.json')


def r41(uid):
    return res(OUT / ('oracle_runs_r41' if '_ORACLE_' in uid else 'runs_r41') / uid)


def rs(r, k='rsum', step=None):
    if r is None:
        return None
    if step is None:
        return r['final']['dev_retrieval'][k]
    c = [x for ph in r['phases'] for x in ph['curves'] if x['step'] == step]
    return c[0]['dev_retrieval'][k] if c else None


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
    units = sum((json.loads((W / 'configs' / f).read_text()) for f in ('UNITS_R41.json', 'UNITS_R41C.json', 'UNITS_R41F.json', 'UNITS_R41G.json') if (W / 'configs' / f).exists()), [])
    R = {u: r41(u) for u in units}
    rows, curves, cost, cx = [], [], [], []
    for u, r in R.items():
        c = json.loads((W / 'configs' / f'{u}.json').read_text())
        base = dict(unit=u, block=c['block'], cond=c['cond'], adaptation=c['adaptation'], supervision=c['supervision'], objective=c['objective'], steps=c['steps'], status=r['status'] if r else 'NA')
        if r is None:
            rows.append(base); continue
        r40 = r.get('r40') or {}
        rows.append(dict(base, **{k: r['final']['dev_retrieval'][k] for k in MET}, H_rsum=r['native_H_dev']['rsum'], d_vs_H=r['final']['dev_retrieval']['rsum'] - r['native_H_dev']['rsum'],
                         rsum_step8000=rs(r, step=8000), one_step_r40=(r40.get('one_step_check') or {}).get('passed'),
                         refs_ok=all(x['init_equal'] and x['student_init_equal'] in (None, True) for x in (r40.get('references') or {}).values()),
                         ref_chain_steps=';'.join(f"{k}:{len(x['chain_equal_at'])}" for k, x in (r40.get('references') or {}).items()), gpu=r['env']['gpu'], job=r['env']['job']))
        for ph in r['phases']:
            for x in ph['curves']:
                curves.append(dict(base, step=x['step'], **{k: x['dev_retrieval'][k] for k in MET}))
        cost.append(dict(base, train_seconds=r['cost']['train_seconds'], eval_seconds=r['cost']['eval_seconds'], params_trainable=r['cost']['params_trainable'],
                         peak_gpu_memory_GB=r['cost']['peak_gpu_memory_bytes'] / 2 ** 30, gpu=r['env']['gpu']))
    wcsv('R41_RESULTS.csv', rows); wcsv('R41_CURVES.csv', curves); wcsv('R41_COST.csv', cost)
    L = ['# R41 extension results (numbers only)', '', f"{sum(r is not None for r in R.values())}/{len(units)} R41 units complete (A-E 39, amendments F 3 and G 9). DEV two-stage (frozen-H top-100 reranked). TEST not read.", '']
    # A
    L += ['## A: budget 16000 vs 8000 (N x both_joint)', '', '| parent | obj | N-JOINT 8000 (ref) | 16k @8000 | 16k @16000 | 16k@16000 - ref8000 | N-frozen (ref) |', '|---|---|---|---|---|---|---|']
    for pk in JREF:
        for o in OBJ:
            r = R.get(f'R41_A_{pk}_N_joint_16k_{o}'); ref8 = rs(res(str(JREF[pk]).format(o=o))); fr = rs(res(str(FREF[pk]).format(o=o)))
            a8, a16 = rs(r, step=8000), rs(r)
            L.append(f'| {pk} | {o} | {f(ref8)} | {f(a8)} | {f(a16)} | {f(sub(a16, ref8))} | {f(fr)} |')
            cx.append(dict(block='A', parent=pk, objective=o, N_joint_8000_ref=ref8, A16k_at_8000=a8, A16k_at_16000=a16, A16k_minus_ref8000=sub(a16, ref8), A16k_16000_minus_8000=sub(a16, a8), N_frozen_ref=fr))
    # B
    L += ['', '## B: training seed 2002 vs seed 1001 (Flickr60 s1001 N-JOINT, 8000)', '', '| obj | seed1001 (R39) | seed2002 | diff |', '|---|---|---|---|']
    b = {}
    for o in OBJ:
        s1 = rs(res(str(JREF['F60_R30_H1001']).format(o=o))); s2 = rs(R.get(f'R41_B_F60_R30_H1001_seed2002_N_joint_{o}')); b[o] = (s1, s2)
        L.append(f'| {o} | {f(s1)} | {f(s2)} | {f(sub(s2, s1))} |'); cx.append(dict(block='B', parent='F60_R30_H1001', objective=o, seed1001=s1, seed2002=s2, diff=sub(s2, s1)))
    L += ['', f"J-BCE: seed1001 {f(sub(b['J'][0], b['BCE'][0]))}, seed2002 {f(sub(b['J'][1], b['BCE'][1]))}; J-NCE: seed1001 {f(sub(b['J'][0], b['NCE'][0]))}, seed2002 {f(sub(b['J'][1], b['NCE'][1]))}"]
    # C
    L += ['', '## C: IN2R-native Flickr parents (8000)', '', '| parent | H | obj | N-frozen | N-JOINT | JOINT-FROZEN |', '|---|---|---|---|---|---|']
    for pk in ('F40_IN2R_H1001', 'F80_IN2R_H1001'):
        for o in OBJ:
            fr, jt = R.get(f'R41_C_{pk}_N_frozen_{o}'), R.get(f'R41_C_{pk}_N_joint_{o}')
            H = (fr or jt or {}).get('native_H_dev', {}).get('rsum')
            L.append(f'| {pk} | {f(H)} | {o} | {f(rs(fr))} | {f(rs(jt))} | {f(sub(rs(jt), rs(fr)))} |')
            cx.append(dict(block='C', parent=pk, objective=o, H=H, N_frozen=rs(fr), N_joint=rs(jt), JOINT_minus_FROZEN=sub(rs(jt), rs(fr))))
    # D
    L += ['', '## D: ORACLE_MISSING on Flickr60 s1002 (8000)', '', '| obj | N-frozen (R38) | ORACLE-frozen | N-JOINT (R39X) | ORACLE-JOINT | OJ-NJ | OF-NF | [OJ-OF]-[NJ-NF] |', '|---|---|---|---|---|---|---|---|']
    for o in OBJ:
        nf = rs(res(str(FREF['F60_R30_H1002']).format(o=o))); nj = rs(res(str(JREF['F60_R30_H1002']).format(o=o)))
        of, oj = rs(R.get(f'R41_D_F60_R30_H1002_ORACLE_frozen_{o}')), rs(R.get(f'R41_D_F60_R30_H1002_ORACLE_joint_{o}'))
        it = None if None in (nf, nj, of, oj) else (oj - of) - (nj - nf)
        L.append(f'| {o} | {f(nf)} | {f(of)} | {f(nj)} | {f(oj)} | {f(sub(oj, nj))} | {f(sub(of, nf))} | {f(it)} |')
        cx.append(dict(block='D', parent='F60_R30_H1002', objective=o, N_frozen=nf, ORACLE_frozen=of, N_joint=nj, ORACLE_joint=oj, OJ_minus_NJ=sub(oj, nj), OF_minus_NF=sub(of, nf), interaction=it))
    # E
    L += ['', '## E: image_only (8000)', '', '| parent | obj | N-frozen | IMAGE_ONLY | N-JOINT | IMAGE-FROZEN | BOTH-IMAGE |', '|---|---|---|---|---|---|---|']
    for pk in ('F60_R30_H1002', 'F60_R30_H1003'):
        for o in OBJ:
            nf = rs(res(str(FREF[pk]).format(o=o))); nj = rs(res(str(JREF[pk]).format(o=o))); im = rs(R.get(f'R41_E_{pk}_image_only_{o}'))
            L.append(f'| {pk} | {o} | {f(nf)} | {f(im)} | {f(nj)} | {f(sub(im, nf))} | {f(sub(nj, im))} |')
            cx.append(dict(block='E', parent=pk, objective=o, N_frozen=nf, IMAGE_ONLY=im, N_joint=nj, IMAGE_minus_FROZEN=sub(im, nf), BOTH_minus_IMAGE=sub(nj, im)))
    # F: CC152K encoder lr x0.1
    L += ['', '## F (amendment): CC152K CC31 N-JOINT with student lr 1e-6 -> 1e-7 (10x lower), 8000', '', '| obj | N-frozen (R38) | N-JOINT (R39, enc lr x1) | N-JOINT enc lr x0.1 | F-NF | F-NJ |', '|---|---|---|---|---|---|']
    for o in OBJ:
        nf = rs(res(R38 / f'r38_B_cc_R31_s1001_N_shared_{o}_cosine')); nj = rs(res(R39 / 'runs' / f'R39_CC152K_N_joint_global_{o}')); fv = rs(R.get(f'R41_F_CC31_H1001_N_joint_enclr0p1_{o}'))
        L.append(f'| {o} | {f(nf)} | {f(nj)} | {f(fv)} | {f(sub(fv, nf))} | {f(sub(fv, nj))} |')
        cx.append(dict(block='F', parent='CC31_H1001', objective=o, N_frozen=nf, N_joint=nj, N_joint_enclr0p1=fv, F_minus_NF=sub(fv, nf), F_minus_NJ=sub(fv, nj)))
    # G1: 0% seed 2002
    L += ['', '## G1 (amendment): Flickr 0% R30-H f00_s1001 N-JOINT, training seed 2002 vs 1001 (8000)', '', '| obj | seed1001 (R39X) | seed2002 | diff |', '|---|---|---|---|']
    g1 = {}
    for o in OBJ:
        s1 = rs(res(str(JREF['F00_R30_H1001']).format(o=o))); s2 = rs(R.get(f'R41_G1_F00_R30_H1001_seed2002_N_joint_{o}')); g1[o] = (s1, s2)
        L.append(f'| {o} | {f(s1)} | {f(s2)} | {f(sub(s2, s1))} |'); cx.append(dict(block='G1', parent='F00_R30_H1001', objective=o, seed1001=s1, seed2002=s2, diff=sub(s2, s1)))
    L += ['', f"J-BCE: seed1001 {f(sub(g1['J'][0], g1['BCE'][0]))}, seed2002 {f(sub(g1['J'][1], g1['BCE'][1]))}; J-NCE: seed1001 {f(sub(g1['J'][0], g1['NCE'][0]))}, seed2002 {f(sub(g1['J'][1], g1['NCE'][1]))}"]
    # G2: oracle on s1003
    L += ['', '## G2 (amendment): ORACLE_MISSING on Flickr60 s1003 (8000)', '', '| obj | N-frozen (R38) | ORACLE-frozen | N-JOINT (R39X) | ORACLE-JOINT | OJ-NJ | OF-NF | [OJ-OF]-[NJ-NF] |', '|---|---|---|---|---|---|---|---|']
    for o in OBJ:
        nf = rs(res(str(FREF['F60_R30_H1003']).format(o=o))); nj = rs(res(str(JREF['F60_R30_H1003']).format(o=o)))
        of, oj = rs(R.get(f'R41_G2_F60_R30_H1003_ORACLE_frozen_{o}')), rs(R.get(f'R41_G2_F60_R30_H1003_ORACLE_joint_{o}'))
        it = None if None in (nf, nj, of, oj) else (oj - of) - (nj - nf)
        L.append(f'| {o} | {f(nf)} | {f(of)} | {f(nj)} | {f(oj)} | {f(sub(oj, nj))} | {f(sub(of, nf))} | {f(it)} |')
        cx.append(dict(block='G2', parent='F60_R30_H1003', objective=o, N_frozen=nf, ORACLE_frozen=of, N_joint=nj, ORACLE_joint=oj, OJ_minus_NJ=sub(oj, nj), OF_minus_NF=sub(of, nf), interaction=it))
    wcsv('R41_CONTRASTS.csv', cx)
    fails = []
    for p in sorted(glob.glob(str(OUT / '*runs_r41' / '*' / 'FAILED_*.json'))):
        x = rj(p); fails.append(dict(unit=x['unit'], file=p, utc=x['utc'], error=x['traceback'].strip().splitlines()[-1][:300]))
    for p in sorted(glob.glob(str(W / 'CPU_TEST_R41*.json'))):
        x = rj(p)
        for k, c in x['checks'].items():
            if not c['passed']:
                fails.append(dict(unit=f'{Path(p).name}:{k}', file=p, utc=x['utc'], error='see file'))
    wcsv('R41_FAILURES.csv', fails or [dict(unit='none', file='', utc='', error='')])
    L += ['', 'Files: R41_RESULTS.csv (six recalls, step-8000 rSum for A, in-unit check flags), R41_CONTRASTS.csv, R41_CURVES.csv, R41_COST.csv, R41_FAILURES.csv.']
    (W / 'RESULTS_R41.md').write_text('\n'.join(L) + '\n')
    print('R41 summary', sum(r is not None for r in R.values()))


if __name__ == '__main__':
    main()
