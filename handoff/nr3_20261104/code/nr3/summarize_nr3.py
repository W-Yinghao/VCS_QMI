#!/usr/bin/env python3
"""NR3 summarizer: numbers only. Writes RESULTS_NR3.md + CSVs in the NR3 handoff dir."""
import csv, json, sys
from pathlib import Path
import numpy as np
N3 = Path('/home/infres/yinwang/CS_QMI/handoff/nr3_20261104'); sys.path.insert(0, str(N3/'code'))
from nr3.paths import OUT, CONDS, cond_paths
rj = lambda p: json.loads(Path(p).read_text()) if Path(p).exists() else None
K = ['i2t_R1', 'i2t_R5', 'i2t_R10', 't2i_R1', 't2i_R5', 't2i_R10', 'rSum']
f2 = lambda x, p=2: f'{x:.{p}f}' if isinstance(x, (int, float)) and x is not None else 'NA'
sg = lambda x: f'{x:+.2f}' if isinstance(x, (int, float)) else 'NA'
OBJ = ('J', 'BCE', 'NCE'); STA = [0, 250, 500, 1000, 2000, 4000, 8000]; STB = [8000, 8250, 8500, 9000, 10000, 12000]


def wcsv(p, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(p, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys, restval=''); w.writeheader(); w.writerows(rows)


def unit(u, steps):
    run = OUT/'runs'/u; ev = {s: rj(run/f'eval_{s:06d}/METRICS.json') for s in steps}; ro = rj(run/'NR3_READOUT.json'); cost = rj(run/'TRAIN_COST.json')
    st = 'COMPLETE' if (run/'UNIT_COMPLETE.json').exists() else ('TRAINED' if (run/'DONE.json').exists() else ('RUNNING_OR_FAILED' if run.exists() else 'NOT_STARTED'))
    x = dict(unit=u, status=st, attempts=len((run/'ATTEMPTS.txt').read_text().splitlines()) if (run/'ATTEMPTS.txt').exists() else 0,
             curve={s: ev[s]['metrics']['rSum'] for s in steps if ev[s]}, readB_J={r['step']: r['READ_B_HREF_J'] for r in ro['steps']} if ro else {})
    m = ev[steps[-1]]
    if m:
        x.update({k: m['metrics'][k] for k in K}); x['delta_base'] = m['delta_rSum']
        for d in ('i2t', 't2i'): x[f'{d}_resc'], x[f'{d}_lost'] = m[d]['rescued'], m[d]['lost']
        x['eval_s_per_1000q'] = m['cost']['seconds_per_1000_queries']
    if cost: x.update(train_s=cost['wrapper_seconds'], gpu=cost['gpu'], pair_forwards=cost.get('pair_forwards_this_attempt'))
    return x


def main():
    L = ['# NR3_20261104 results (numbers only; DEV, base SigLIP2 top-100 rerank by s; TEST not read)', '',
         'Protocol: PROTOCOL_FROZEN_NR3.json (NR3_REF_INIT_AND_TASK_PREFIX_V1). Calibration: CALIBRATION_RECEIPTS.json. Prefixes: PREFIX_LINEAGE.json.', '']
    # D0
    L += ['## D0. NR2 HREF seed-1001 step-8000 checkpoints: FIT witness (READ-B, same Q_h per condition) and DEV', '',
          '| condition | objective | READ-B HREF-J raw | READ-B HREF-J calibrated (a,b fitted on CAL-A) | a | b | product-Q J (FIT_READOUT) | DEV rSum |', '|---|---|---|---|---|---|---|---|']
    d0rows = []
    for c in CONDS:
        d = rj(OUT/'d0'/c/'D0.json')
        if not d: L.append(f'| {c} | NA | | | | | | |'); continue
        for o in OBJ:
            x = d['checkpoints'][o]; r = dict(condition=c, objective=o, readB_J_raw=x['READ_B_HREF_J_raw'], readB_J_cal=x['READ_B_HREF_J_calibrated'],
                                              a=x['calibration_fit']['a'], b=x['calibration_fit']['b'], productQ_J=x['productQ_J_FIT_READOUT'], DEV_rSum=x['DEV']['rSum'])
            d0rows.append(r)
            L.append(f"| {c} | {o} | {f2(r['readB_J_raw'],4)} | {f2(r['readB_J_cal'],4)} | {f2(r['a'],4)} | {f2(r['b'],4)} | {f2(r['productQ_J'],4)} | {f2(r['DEV_rSum'])} |")
    L += ['', 'READ-B strata of the J checkpoints (raw g = s − log w): per direction × P/Q × hard/non-hard: n, median g, |T|>.99, median 1−T², mean dL_J/dg', '',
          '| condition | objective | dir | slot | stratum | n | median g | sat | median 1−T² | mean dLJ/dg |', '|---|---|---|---|---|---|---|---|---|---|']
    strows = []
    for c in CONDS:
        d = rj(OUT/'d0'/c/'D0.json')
        if not d: continue
        for o in OBJ:
            for dd in ('i2t', 't2i'):
                for slot in ('P', 'Q'):
                    for sname in ('hard', 'nonhard'):
                        s = d['checkpoints'][o]['strata'][dd][slot][sname]
                        if not s.get('n'): continue
                        r = dict(condition=c, objective=o, dir=dd, slot=slot, stratum=sname, n=s['n'], median_g=s['g_q']['50'] if '50' in s['g_q'] else s['g_q'][50],
                                 sat=s['sat_T_gt_099'], median_1mT2=s['one_minus_T2_q']['50'] if '50' in s['one_minus_T2_q'] else s['one_minus_T2_q'][50], mean_dLJ=s['mean_dLJ_dg'])
                        strows.append(r)
                        if o == 'J': L.append(f"| {c} | {o} | {dd} | {slot} | {sname} | {r['n']} | {f2(r['median_g'],3)} | {f2(r['sat'],4)} | {f2(r['median_1mT2'],4)} | {f2(r['mean_dLJ'],4)} |")
    L += ['', 'Step 0 on the identical first training batch (seed 1001): NR2 init (NR1 a0,b0) vs NR3-A init (HCAL a,b); P slot hard/non-hard and Q slot; dJ/dg identity; parameter gradients', '',
          '| condition | init | a0 | b0 | dir | P hard median g | P nonhard median g | Q hard median g | Q nonhard median g | P sat | Q sat | identity max diff | grad norm J/BCE/NCE | cos(J,BCE) |', '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    s0rows = []
    for c in CONDS:
        d = rj(OUT/'d0'/c/'D0.json')
        if not d: continue
        for tag in ('NR2_init', 'NR3A_init'):
            r0 = d['step0'][tag]
            for dd in ('i2t', 't2i'):
                S = r0['strata'][dd]; med = lambda z: (z['g_q'].get('50', z['g_q'].get(50)) if z.get('n') else None)
                gn = '/'.join(f"{r0['param_grads'][o]['grad_norm']:.3g}" for o in OBJ)
                row = dict(condition=c, init=tag, a0=r0['a0'], b0=r0['b0'], dir=dd, P_hard=med(S['P']['hard']), P_nonhard=med(S['P']['nonhard']), Q_hard=med(S['Q']['hard']),
                           Q_nonhard=med(S['Q']['nonhard']), P_sat=S['P']['all']['sat_T_gt_099'], Q_sat=S['Q']['all']['sat_T_gt_099'], identity=r0['identity'][dd]['max_abs_diff'],
                           grad_norms=gn, cos_J_BCE=r0['param_grad_cosines']['J_BCE'])
                s0rows.append(row)
                L.append(f"| {c} | {tag} | {f2(row['a0'],3)} | {f2(row['b0'],3)} | {dd} | {f2(row['P_hard'],3)} | {f2(row['P_nonhard'],3)} | {f2(row['Q_hard'],3)} | {f2(row['Q_nonhard'],3)} | "
                         f"{f2(row['P_sat'],4)} | {f2(row['Q_sat'],4)} | {row['identity']:.1e} | {gn} | {f2(row['cos_J_BCE'],3)} |")
    L += ['', 'Evaluator-only: READ-B P pairs of the J checkpoint split by injected-noise identity (median g; |T|>.99); CC152K_RECOVERED NA', '',
          '| condition | dir | clean fraction | clean median g | mismatched median g | clean sat | mismatched sat |', '|---|---|---|---|---|---|---|']
    for c in CONDS:
        t = rj(OUT/'d0'/c/'D0_TRUTH_EVALONLY.json')
        if not t: continue
        if t.get('status') == 'NA': L.append(f'| {c} | NA | NA | NA | NA | NA | NA |'); continue
        for dd in ('i2t', 't2i'):
            x = t['readB']['J'][dd]; med = lambda z: z['all']['g_q'].get('50', z['all']['g_q'].get(50)) if z['all'].get('n') else None
            L.append(f"| {c} | {dd} | {f2(x['clean_fraction'],3)} | {f2(med(x['clean']),3)} | {f2(med(x['mismatched']),3)} | {f2(x['clean']['all'].get('sat_T_gt_099'),4)} | {f2(x['mismatched']['all'].get('sat_T_gt_099'),4)} |")
    # calibration receipts
    cal = rj(N3/'CALIBRATION_RECEIPTS.json') or {}; lin = rj(N3/'PREFIX_LINEAGE.json') or {}
    L += ['', '## Calibration receipts (CAL-A, L-BFGS-B, bounds a∈[0.1,100], b∈[−20,20], ≤500 iterations)', '',
          '| condition | block | start (a,b) | fitted a | fitted b | BCE initial | BCE final | converged | iterations | at bound | READ-B HREF-J before | READ-B HREF-J after |', '|---|---|---|---|---|---|---|---|---|---|---|---|']
    for c in CONDS:
        if c not in cal: continue
        A, B = cal[c]['A'], cal[c]['B']
        L.append(f"| {c} | A (base cosine) | ({f2(A['start'][0],3)}, {f2(A['start'][1],3)}) | {f2(A['fit']['a'],4)} | {f2(A['fit']['b'],4)} | {f2(A['fit']['initial_bce'],4)} | {f2(A['fit']['final_bce'],4)} | {A['fit']['converged']} | {A['fit']['iterations']} | {A['fit']['at_bound']} | {f2(A['READ_B_J']['nr1'],4)} | {f2(A['READ_B_J']['hcal'],4)} |")
        L.append(f"| {c} | B (parent NCE s) | (1, 0) | {f2(B['fit']['a'],4)} | {f2(B['fit']['b'],4)} | {f2(B['fit']['initial_bce'],4)} | {f2(B['fit']['final_bce'],4)} | {B['fit']['converged']} | {B['fit']['iterations']} | {B['fit']['at_bound']} | {f2(B['READ_B_J']['raw'],4)} | {f2(B['READ_B_J']['calibrated'],4)} |")
    L += ['', '## Prefix lineage', '', '| condition | parent unit | parent ckpt sha256 | prefix sha256 | fold max abs err (pools) | DEV max abs score err | DEV i2t/t2i rows differing |', '|---|---|---|---|---|---|---|']
    for c in CONDS:
        if c not in lin: continue
        x = lin[c]; oc = x['prefix_dev_order_check']
        L.append(f"| {c} | {x['parent_unit']} | {x['parent_checkpoint_sha256'][:16]}… | {x['prefix_checkpoint_sha256'][:16]}… | {x['fold']['max_abs_err_vs_a_s_plus_b']:.2e} | {oc['max_abs_score_err']:.2e} | {oc['i_order_rows_differing']}/{oc['t_order_rows_differing']} |")
    # A results
    A = {f'{c}_{o}': unit(f'NR3_A_HCAL_{c}_G_{o}_s1001', STA) for c in CONDS for o in OBJ}
    nr2 = {f'{c}_{o}': (rj(cond_paths(c)['parent'](o)/'eval_008000/METRICS.json') or {}).get('metrics', {}).get('rSum') for c in CONDS for o in OBJ}
    base = {c: rj(cond_paths(c)['baseline'])['metrics']['rSum'] for c in CONDS}
    L += ['', '## A. HREF with HREF-aware calibrated init (8000 steps), step 8000', '',
          '| unit | ' + ' | '.join(K) + ' | Δbase | i2t resc/lost | t2i resc/lost | NR2-HREF s1001 rSum | A−NR2HREF | READ-B HREF-J | status |', '|' + '---|'*(len(K)+8)]
    arows = []
    for c in CONDS:
        for o in OBJ:
            x = A[f'{c}_{o}']; n2 = nr2[f'{c}_{o}']
            r = dict(condition=c, objective=o, **{k: x.get(k) for k in K}, delta_base=x.get('delta_base'), NR2_HREF=n2, A_minus_NR2=(x['rSum'] - n2) if 'rSum' in x and n2 else None,
                     readB_J=x['readB_J'].get(8000), status=x['status'])
            arows.append(r)
            L.append(f"| {x['unit']} | " + ' | '.join(f2(x.get(k)) for k in K) + f" | {sg(x.get('delta_base'))} | {x.get('i2t_resc','NA')}/{x.get('i2t_lost','NA')} | {x.get('t2i_resc','NA')}/{x.get('t2i_lost','NA')} | "
                     f"{f2(n2)} | {sg(r['A_minus_NR2'])} | {f2(r['readB_J'],4)} | {x['status']} |")
    L += ['', 'A within-condition objective differences (step 8000):', '', '| condition | base | A-J | A-BCE | A-NCE | J−BCE | J−NCE |', '|---|---|---|---|---|---|---|']
    for c in CONDS:
        v = {o: A[f'{c}_{o}'].get('rSum') for o in OBJ}
        dd = lambda p, q: sg(v[p] - v[q]) if v[p] is not None and v[q] is not None else 'NA'
        L.append(f"| {c} | {f2(base[c])} | {f2(v['J'])} | {f2(v['BCE'])} | {f2(v['NCE'])} | {dd('J','BCE')} | {dd('J','NCE')} |")
    # B results
    B = {f'{c}_{o}': unit(f'NR3_B_NCE2{o}_{c}_G_s1001', STB) for c in CONDS for o in OBJ}
    L += ['', '## B. Shared folded HREF-NCE prefix (step 8000) → J/BCE/NCE, 4000 added steps (total 12000)', '',
          '| unit | ' + ' | '.join(K) + ' | Δbase | i2t resc/lost | t2i resc/lost | prefix rSum | Δprefix | READ-B HREF-J prefix→12000 | status |', '|' + '---|'*(len(K)+8)]
    brows = []
    for c in CONDS:
        for o in OBJ:
            x = B[f'{c}_{o}']; p0 = x['curve'].get(8000)
            r = dict(condition=c, branch=f'NCE→{o}', **{k: x.get(k) for k in K}, delta_base=x.get('delta_base'), prefix_rSum=p0,
                     delta_prefix=(x['rSum'] - p0) if 'rSum' in x and p0 is not None else None, readB_J_prefix=x['readB_J'].get(8000), readB_J_end=x['readB_J'].get(12000), status=x['status'])
            brows.append(r)
            L.append(f"| {x['unit']} | " + ' | '.join(f2(x.get(k)) for k in K) + f" | {sg(x.get('delta_base'))} | {x.get('i2t_resc','NA')}/{x.get('i2t_lost','NA')} | {x.get('t2i_resc','NA')}/{x.get('t2i_lost','NA')} | "
                     f"{f2(p0)} | {sg(r['delta_prefix'])} | {f2(r['readB_J_prefix'],4)}→{f2(r['readB_J_end'],4)} | {x['status']} |")
    L += ['', 'B matched branch differences at total step 12000:', '', '| condition | prefix | NCE→J | NCE→BCE | NCE→NCE | J−NCE | J−BCE | BCE−NCE |', '|---|---|---|---|---|---|---|---|']
    for c in CONDS:
        v = {o: B[f'{c}_{o}'].get('rSum') for o in OBJ}; p0 = B[f'{c}_J']['curve'].get(8000)
        dd = lambda p, q: sg(v[p] - v[q]) if v[p] is not None and v[q] is not None else 'NA'
        L.append(f"| {c} | {f2(p0)} | {f2(v['J'])} | {f2(v['BCE'])} | {f2(v['NCE'])} | {dd('J','NCE')} | {dd('J','BCE')} | {dd('BCE','NCE')} |")
    L += ['', '## Curves (rSum)', '', '| unit | ' + ' | '.join(map(str, STA)) + ' |', '|' + '---|'*(len(STA)+1)]
    for x in A.values(): L.append(f"| {x['unit']} | " + ' | '.join(f2(x['curve'].get(s)) for s in STA) + ' |')
    L += ['', '| unit | ' + ' | '.join(map(str, STB)) + ' |', '|' + '---|'*(len(STB)+1)]
    for x in B.values(): L.append(f"| {x['unit']} | " + ' | '.join(f2(x['curve'].get(s)) for s in STB) + ' |')
    L += ['', '## READ-B HREF-J by step (same Q_h per condition)', '', '| unit | steps… |', '|---|---|']
    for x in list(A.values()) + list(B.values()):
        L.append(f"| {x['unit']} | " + ', '.join(f"{s}:{v:.4f}" for s, v in sorted(x['readB_J'].items())) + ' |')
    L += ['', '## Status and cost', '', '| unit | status | attempts | train s | GPU | pair forwards | eval s/1000q |', '|---|---|---|---|---|---|---|']
    for x in list(A.values()) + list(B.values()):
        L.append(f"| {x['unit']} | {x['status']} | {x['attempts']} | {f2(x.get('train_s'),1)} | {x.get('gpu','')} | {x.get('pair_forwards','')} | {f2(x.get('eval_s_per_1000q'),3)} |")
    wcsv(N3/'RESULTS_NR3_D0.csv', d0rows); wcsv(N3/'RESULTS_NR3_D0_STRATA.csv', strows); wcsv(N3/'RESULTS_NR3_STEP0.csv', s0rows)
    wcsv(N3/'RESULTS_NR3_A.csv', arows); wcsv(N3/'RESULTS_NR3_B.csv', brows)
    wcsv(N3/'RESULTS_NR3_CURVES.csv', [dict(unit=x['unit'], step=s, rSum=v, readB_J=x['readB_J'].get(s)) for x in list(A.values()) + list(B.values()) for s, v in x['curve'].items()])
    (N3/'RESULTS_NR3.md').write_text('\n'.join(L) + '\n'); print('\n'.join(L[:6]))


if __name__ == '__main__':
    main()
