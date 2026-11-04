#!/usr/bin/env python3
"""NR2 summarizer: numbers only (no conclusions). Writes RESULTS_NR2.md + CSVs in the NR2 handoff dir.
Reads NR2 runs, the NR1 seed-1001 G runs (read-only, CONFIRM 3-seed table / references) and NR2 read-only FIT readouts of them."""
import csv, json
from pathlib import Path
import numpy as np
H = Path('/home/infres/yinwang/CS_QMI/handoff/nr2_20261104'); O = Path('/home/infres/yinwang/CS_QMI/outputs_nr2/NR2_20261104')
O1 = Path('/home/infres/yinwang/CS_QMI/outputs_nr1/NR1_20261104')
STEPS = [0, 250, 500, 1000, 2000, 4000, 8000]
K = ['i2t_R1', 'i2t_R5', 'i2t_R10', 't2i_R1', 't2i_R5', 't2i_R10', 'rSum']
rj = lambda p: json.loads(Path(p).read_text()) if Path(p).exists() else None
rows = list(csv.DictReader(open(H/'package/configs/MATRIX.csv')))
base = rj(O1/'baseline/float64/METRICS.json')['metrics']


def unit_dirs():
    d = {r['unit_id']: (O/'runs'/r['unit_id'], r) for r in rows}
    for c in ('Flickr00', 'Flickr60', 'Flickr80'):
        for o in ('J', 'BCE', 'NCE'):
            u = f'NR1_{c}_G_{o}_s1001'
            d[u] = (O1/'runs'/u, dict(unit_id=u, block='NR1_REF', condition=c, objective=o, seed='1001', reference='original_NR1'))
    for f in sorted((H/'locked_configs_nr2b').glob('NR2B_*.json')):   # NR2B seed replication (HREF, U x seeds 1002/1003)
        c = json.loads(f.read_text())
        d[f.stem] = (O/'runs_nr2b'/f.stem, dict(unit_id=f.stem, block='NR2B', condition=c['condition'], objective=c['objective'],
                                              seed=str(c['seed']), reference=c['reference']))
    return d


def read_unit(u, run, r):
    ev = {s: rj(run/f'eval_{s:06d}/METRICS.json') for s in STEPS}
    fr = rj(run/'FIT_READOUT.json') or rj(O/'nr1_readout'/u/'FIT_READOUT.json')
    st = rj(run/'EVALONLY_FIT_STRATA.json') or rj(O/'nr1_readout'/u/'EVALONLY_FIT_STRATA.json')
    log = [json.loads(l) for l in open(run/'TRAIN.jsonl')] if (run/'TRAIN.jsonl').exists() else []
    cost = rj(run/'TRAIN_COST.json'); done = rj(run/'DONE.json')
    status = 'COMPLETE' if (run/'UNIT_COMPLETE.json').exists() else ('TRAINED_EVAL_PARTIAL' if done else ('RUNNING_OR_FAILED' if run.exists() else 'NOT_STARTED'))
    out = dict(unit_id=u, block=r['block'], condition=r['condition'], objective=r['objective'], seed=int(r['seed']),
               reference=r.get('reference', ''), status=status,
               attempts=len((run/'ATTEMPTS.txt').read_text().splitlines()) if (run/'ATTEMPTS.txt').exists() else 0)
    m8 = ev[8000]
    if m8:
        out.update({k: round(m8['metrics'][k], 2) for k in K}); out['delta_rSum'] = round(m8['delta_rSum'], 2)
        for d in ('i2t', 't2i'): out[f'{d}_rescued'], out[f'{d}_lost'] = m8[d]['rescued'], m8[d]['lost']
        out['eval_s_per_1000q'] = round(m8['cost']['seconds_per_1000_queries'], 3)
    out['curve'] = {s: round(ev[s]['metrics']['rSum'], 2) for s in STEPS if ev[s]}
    if log:
        last = log[-1]
        out['train_J_own_step8000'] = round(last.get('J_own', last.get('J')), 5) if 'J_own' in last or 'J' in last else None
        out['train_loss_step8000'] = round(last['loss'], 5)
        for d in ('i2t', 't2i'):
            if f'{d}_hard_frac_ref' in last:
                out[f'{d}_train_hard_frac_ref'] = round(float(np.mean([x[f'{d}_hard_frac_ref'] for x in log])), 4)
                out[f'{d}_train_hard_frac_pos'] = round(float(np.mean([x[f'{d}_hard_frac_pos'] for x in log])), 4)
                out[f'{d}_train_collision_frac_ref'] = round(float(np.mean([x[f'{d}_collision_frac_ref'] for x in log])), 5)
        out['grad_norm_mean_logged'] = round(float(np.mean([x['grad_norm'] for x in log])), 4)
    if fr:
        f8 = {x['step']: x for x in fr['steps']}
        for s in (0, 8000):
            if s in f8:
                out[f'fit_J_productQ_s{s}'] = round(f8[s]['J_productQ'], 5)
                out[f'fit_J_IW_productQ_s{s}'] = round(.5*(f8[s]['J_IW_productQ_i2t']+f8[s]['J_IW_productQ_t2i']), 5)
                out[f'fit_J_proposalQ_g_s{s}'] = round(.5*(f8[s]['J_proposalQ_g_i2t']+f8[s]['J_proposalQ_g_t2i']), 5)
        out['fit_J_curve'] = {x['step']: round(x['J_productQ'], 5) for x in fr['steps']}
    if st and 'steps' in st:
        s8 = {x['step']: x for x in st['steps']}.get(8000)
        if s8:
            out['evalonly_auc_clean_vs_mismatched_s8000'] = round(s8['auc_clean_vs_mismatched'], 4) if 'auc_clean_vs_mismatched' in s8 else None
            out['evalonly_auc_clean_vs_Q0_s8000'] = round(s8['auc_clean_vs_Q0'], 4)
            out['evalonly_auc_mismatched_vs_Q0_s8000'] = round(s8['auc_mismatched_vs_Q0'], 4) if 'auc_mismatched_vs_Q0' in s8 else None
    if cost:
        out.update(train_seconds=round(cost['wrapper_seconds'], 1), gpu=cost['gpu'], peak_gpu_MiB=round(cost['peak_gpu_mem_bytes']/2**20))
        out['pair_forwards'] = done.get('pair_forwards_this_attempt', 4096*8000)
    return out


def wcsv(p, rs):
    keys = list(dict.fromkeys(k for r in rs for k in r if not isinstance(r[k], dict)))
    with open(p, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys, restval='', extrasaction='ignore'); w.writeheader(); w.writerows(rs)


def main():
    U = {u: read_unit(u, run, r) for u, (run, r) in unit_dirs().items()}
    nr2 = [U[r['unit_id']] for r in rows]
    wcsv(H/'RESULTS_NR2_STEP8000.csv', nr2 + [U[u] for u in U if u.startswith('NR1_')])
    curves = [dict(unit_id=u, step=s, rSum=v, fit_J_productQ=x.get('fit_J_curve', {}).get(s, ''))
              for u, x in U.items() for s, v in x['curve'].items()]
    wcsv(H/'RESULTS_NR2_CURVES.csv', curves)
    L = ['# NR2_20261104 results (numbers only; DEV, base SigLIP2 top-100 rerank by s; TEST not read)', '',
         f"Protocol: PROTOCOL_FROZEN_NR2.json. Base rSum {base['rSum']:.2f}. NR1 rows are the existing seed-1001 G units (read-only).", '']
    done = sum(x['status'] == 'COMPLETE' for x in nr2)
    L += [f'Complete NR2 units: {done}/36', '']
    # CONFIRM 3-seed table
    L += ['## A. CONFIRM (NR1 G recipe; s1001 = NR1, s1002/s1003 = NR2), step 8000 rSum', '',
          '| condition | objective | s1001 | s1002 | s1003 | mean | sd | mean ΔrSum |', '|---|---|---|---|---|---|---|---|']
    conf = []
    for c in ('Flickr60', 'Flickr80'):
        for o in ('J', 'BCE', 'NCE'):
            v = [U[f'NR1_{c}_G_{o}_s1001'].get('rSum'), U[f'NR2_CONFIRM_{c}_G_{o}_s1002'].get('rSum'), U[f'NR2_CONFIRM_{c}_G_{o}_s1003'].get('rSum')]
            ok = [x for x in v if x is not None]
            mean = np.mean(ok) if len(ok) == 3 else None; sd = np.std(ok, ddof=1) if len(ok) == 3 else None
            conf.append(dict(condition=c, objective=o, s1001=v[0], s1002=v[1], s1003=v[2], mean=mean, sd=sd))
            f = lambda x: f'{x:.2f}' if x is not None else 'NA'
            L.append(f"| {c} | {o} | {f(v[0])} | {f(v[1])} | {f(v[2])} | {f(mean)} | {f(sd)} | {f(mean-base['rSum']) if mean is not None else 'NA'} |")
    L += ['', 'Paired differences per seed:', '', '| condition | pair | s1001 | s1002 | s1003 | mean |', '|---|---|---|---|---|---|']
    for c in ('Flickr60', 'Flickr80'):
        g = {x['objective']: x for x in conf if x['condition'] == c}
        for a, b in (('J', 'BCE'), ('J', 'NCE'), ('BCE', 'NCE')):
            d = [g[a][s]-g[b][s] if g[a][s] is not None and g[b][s] is not None else None for s in ('s1001', 's1002', 's1003')]
            ok = [x for x in d if x is not None]
            L.append(f"| {c} | {a}−{b} | " + ' | '.join(f'{x:+.2f}' if x is not None else 'NA' for x in d) + f" | {np.mean(ok):+.2f} |" if len(ok) == 3 else
                     f"| {c} | {a}−{b} | " + ' | '.join(f'{x:+.2f}' if x is not None else 'NA' for x in d) + ' | NA |')
    wcsv(H/'RESULTS_NR2_CONFIRM.csv', conf)
    # B. REFERENCE endpoints
    L += ['', '## B. REFERENCE (seed 1001) step 8000, with NR1 G s1001 reference', '',
          '| unit | ' + ' | '.join(K) + ' | ΔrSum | i2t resc/lost | t2i resc/lost | train hard frac ref i2t/t2i | train J own | FIT J productQ | FIT J proposalQ(g) |',
          '|' + '---|'*(len(K)+8)]
    order = [r['unit_id'] for r in rows if r['block'] == 'REFERENCE'] + [u for u in U if u.startswith('NR1_')]
    fm = lambda x, p=2: (f'{x:.{p}f}' if isinstance(x, (float, int)) else 'NA')
    for u in order:
        x = U[u]
        if 'rSum' not in x:
            L.append(f'| {u} | ' + ' | '.join('NA' for _ in K) + f" | NA | NA | NA | NA | NA | NA | NA |  ({x['status']})"); continue
        L.append(f"| {u} | " + ' | '.join(f"{x[k]:.2f}" for k in K) + f" | {x['delta_rSum']:+.2f} | {x['i2t_rescued']}/{x['i2t_lost']} | {x['t2i_rescued']}/{x['t2i_lost']} | "
                 f"{fm(x.get('i2t_train_hard_frac_ref'),4)}/{fm(x.get('t2i_train_hard_frac_ref'),4)} | {fm(x.get('train_J_own_step8000'),4)} | "
                 f"{fm(x.get('fit_J_productQ_s8000'),4)} | {fm(x.get('fit_J_proposalQ_g_s8000'),4)} |")
    # matched comparisons (rSum differences, same condition / objective / seed 1001)
    L += ['', '## C. Matched comparisons, step 8000 ΔrSum between arms (same condition, objective, seed 1001)', '',
          '| condition | objective | HIW−U | HREF−HIW | HREF−U | U−NR1 | HIW−NR1 | HREF−NR1 |', '|---|---|---|---|---|---|---|---|']
    mrows = []
    for c in ('Flickr00', 'Flickr60', 'Flickr80'):
        for o in ('J', 'BCE', 'NCE'):
            g = lambda a: U.get(f'NR2_{a}_{c}_G_{o}_s1001', {}).get('rSum')
            n1 = U[f'NR1_{c}_G_{o}_s1001'].get('rSum'); u_, hw, hr = g('U'), g('HIW'), g('HREF')
            dd = lambda a, b: (a-b) if a is not None and b is not None else None
            r = dict(condition=c, objective=o, HIW_minus_U=dd(hw, u_), HREF_minus_HIW=dd(hr, hw), HREF_minus_U=dd(hr, u_),
                     U_minus_NR1=dd(u_, n1), HIW_minus_NR1=dd(hw, n1), HREF_minus_NR1=dd(hr, n1)); mrows.append(r)
            L.append(f'| {c} | {o} | ' + ' | '.join(f'{r[k]:+.2f}' if r[k] is not None else 'NA' for k in list(r)[2:]) + ' |')
    wcsv(H/'RESULTS_NR2_MATCHED.csv', mrows)
    L += ['', 'Within-condition objective differences (seed 1001):', '', '| condition | arm | J−BCE | J−NCE | BCE−NCE |', '|---|---|---|---|---|']
    for c in ('Flickr00', 'Flickr60', 'Flickr80'):
        for a in ('U', 'HIW', 'HREF'):
            v = {o: U.get(f'NR2_{a}_{c}_G_{o}_s1001', {}).get('rSum') for o in ('J', 'BCE', 'NCE')}
            dd = lambda p, q: f'{v[p]-v[q]:+.2f}' if v[p] is not None and v[q] is not None else 'NA'
            L.append(f'| {c} | {a} | {dd("J","BCE")} | {dd("J","NCE")} | {dd("BCE","NCE")} |')
    # curves
    L += ['', '## D. rSum curves (steps ' + '/'.join(map(str, STEPS)) + ')', '', '| unit | ' + ' | '.join(map(str, STEPS)) + ' |', '|' + '---|'*(len(STEPS)+1)]
    for u in [r['unit_id'] for r in rows] + [u for u in U if u.startswith('NR1_')]:
        c = U[u]['curve']; L.append(f'| {u} | ' + ' | '.join(f'{c[s]:.2f}' if s in c else 'NA' for s in STEPS) + ' |')
    L += ['', '## E. FIT common product-Q J (T=tanh(s/2); same pool per condition; s for every unit) by step', '',
          '| unit | ' + ' | '.join(map(str, STEPS)) + ' |', '|' + '---|'*(len(STEPS)+1)]
    for u in [r['unit_id'] for r in rows] + [u for u in U if u.startswith('NR1_')]:
        c = U[u].get('fit_J_curve', {}); L.append(f'| {u} | ' + ' | '.join(f'{c[s]:.4f}' if s in c else 'NA' for s in STEPS) + ' |')
    L += ['', '## F. Evaluator-only FIT stratification at step 8000 (AUC of s; injected-noise identity isolated in evaluator)', '',
          '| unit | clean vs mismatched | clean vs Q0 | mismatched vs Q0 |', '|---|---|---|---|']
    for u in [r['unit_id'] for r in rows] + [u for u in U if u.startswith('NR1_')]:
        x = U[u]
        if 'evalonly_auc_clean_vs_Q0_s8000' in x:
            L.append(f"| {u} | {fm(x.get('evalonly_auc_clean_vs_mismatched_s8000'),4)} | {fm(x['evalonly_auc_clean_vs_Q0_s8000'],4)} | {fm(x.get('evalonly_auc_mismatched_vs_Q0_s8000'),4)} |")
    # H. NR2B: 3-seed HREF and U (s1001 = NR2 units, s1002/s1003 = NR2B units) and paired HREF-U per seed
    L += ['', '## H. NR2B seed replication: HREF and U, step 8000 rSum (s1001 = NR2, s1002/s1003 = NR2B)', '',
          '| condition | objective | reference | s1001 | s1002 | s1003 | mean | sd | mean ΔrSum |', '|---|---|---|---|---|---|---|---|---|']
    seedrows = []
    f2 = lambda x: f'{x:.2f}' if x is not None else 'NA'
    for c in ('Flickr00', 'Flickr60', 'Flickr80'):
        for o in ('J', 'BCE', 'NCE'):
            for a in ('U', 'HREF'):
                v = [U.get(f'NR2_{a}_{c}_G_{o}_s1001', {}).get('rSum')] + [U.get(f'NR2B_{a}_{c}_G_{o}_s{s}', {}).get('rSum') for s in (1002, 1003)]
                full = all(x is not None for x in v)
                mean = float(np.mean(v)) if full else None; sd = float(np.std(v, ddof=1)) if full else None
                dlt = f"{mean - base['rSum']:+.2f}" if full else 'NA'
                seedrows.append(dict(condition=c, objective=o, reference=a, s1001=v[0], s1002=v[1], s1003=v[2], mean=mean, sd=sd))
                L.append(f"| {c} | {o} | {a} | {f2(v[0])} | {f2(v[1])} | {f2(v[2])} | {f2(mean)} | {f2(sd)} | {dlt} |")
    L += ['', 'Paired HREF−U per seed (same condition, objective, seed):', '', '| condition | objective | s1001 | s1002 | s1003 | mean | n positive |', '|---|---|---|---|---|---|---|']
    pairrows = []
    for c in ('Flickr00', 'Flickr60', 'Flickr80'):
        for o in ('J', 'BCE', 'NCE'):
            g = {r['reference']: r for r in seedrows if r['condition'] == c and r['objective'] == o}
            d = [g['HREF'][s]-g['U'][s] if g['HREF'][s] is not None and g['U'][s] is not None else None for s in ('s1001', 's1002', 's1003')]
            full = all(x is not None for x in d)
            pairrows.append(dict(condition=c, objective=o, s1001=d[0], s1002=d[1], s1003=d[2], mean=float(np.mean(d)) if full else None,
                                 n_positive=sum(x > 0 for x in d) if full else None))
            L.append(f"| {c} | {o} | " + ' | '.join(f'{x:+.2f}' if x is not None else 'NA' for x in d) +
                     (f" | {np.mean(d):+.2f} | {sum(x > 0 for x in d)}/3 |" if full else ' | NA | NA |'))
    wcsv(H/'RESULTS_NR2B_SEEDS.csv', seedrows); wcsv(H/'RESULTS_NR2B_PAIRED.csv', pairrows)
    nb = [x for u, x in U.items() if u.startswith('NR2B_')]
    L += ['', f"NR2B complete units: {sum(x['status'] == 'COMPLETE' for x in nb)}/{len(nb)}", '',
          '| unit | status | attempts | train s | GPU | eval s/1000q |', '|---|---|---|---|---|---|']
    for x in nb:
        L.append(f"| {x['unit_id']} | {x['status']} | {x['attempts']} | {x.get('train_seconds','')} | {x.get('gpu','')} | {x.get('eval_s_per_1000q','')} |")
    # I. NR2C: COCO (U vs HREF, seed 1001); base = COCO DEV 1K base cosine
    OC = Path('/projects/EEG-foundation-model/yinghao/outputs_nr2/NR2C_20261104')
    bc = rj(OC/'baseline/float64/METRICS.json')
    if bc:
        L += ['', '## I. NR2C COCO (DEV 1000 images / 5000 captions; base SigLIP2 top-100 rerank by s; seed 1001)', '',
              '| base | ' + ' | '.join(K) + ' | i2t top100 cov | t2i top100 cov |', '|' + '---|'*(len(K)+3),
              '| SigLIP2 cosine | ' + ' | '.join(f"{bc['metrics'][k]:.2f}" for k in K) + f" | {bc['i2t']['coverage_topk']:.4f} | {bc['t2i']['coverage_topk']:.4f} |", '',
              '| unit | ' + ' | '.join(K) + ' | ΔrSum | i2t resc/lost | t2i resc/lost | train hard frac ref i2t/t2i | FIT J productQ | FIT J proposalQ(g) | status |',
              '|' + '---|'*(len(K)+8)]
        C = {}
        for f in sorted((H/'locked_configs_nr2c').glob('NR2C_*.json')):
            c = json.loads(f.read_text())
            C[f.stem] = x = read_unit(f.stem, OC/'runs'/f.stem, dict(block='NR2C', condition=c['condition'], objective=c['objective'], seed='1001', reference=c['reference']))
            if 'rSum' in x:
                L.append(f"| {f.stem} | " + ' | '.join(f"{x[k]:.2f}" for k in K) + f" | {x['delta_rSum']:+.2f} | {x['i2t_rescued']}/{x['i2t_lost']} | {x['t2i_rescued']}/{x['t2i_lost']} | "
                         f"{fm(x.get('i2t_train_hard_frac_ref'),4)}/{fm(x.get('t2i_train_hard_frac_ref'),4)} | {fm(x.get('fit_J_productQ_s8000'),4)} | {fm(x.get('fit_J_proposalQ_g_s8000'),4)} | {x['status']} |")
            else:
                L.append(f"| {f.stem} | " + ' | '.join('NA' for _ in K) + f" | NA | NA | NA | NA | NA | NA | {x['status']} |")
        L += ['', 'HREF−U (step 8000 rSum, seed 1001):', '', '| condition | objective | U | HREF | HREF−U |', '|---|---|---|---|---|']
        crow = []
        for c in ('COCO00', 'COCO60', 'COCO80'):
            for o in ('J', 'BCE', 'NCE'):
                u_, h_ = C[f'NR2C_U_{c}_G_{o}_s1001'].get('rSum'), C[f'NR2C_HREF_{c}_G_{o}_s1001'].get('rSum')
                d = h_ - u_ if u_ is not None and h_ is not None else None
                crow.append(dict(condition=c, objective=o, U=u_, HREF=h_, HREF_minus_U=d))
                L.append(f"| {c} | {o} | {fm(u_)} | {fm(h_)} | {f'{d:+.2f}' if d is not None else 'NA'} |")
        L += ['', 'NR2C rSum curves (steps ' + '/'.join(map(str, STEPS)) + '):', '', '| unit | ' + ' | '.join(map(str, STEPS)) + ' |', '|' + '---|'*(len(STEPS)+1)]
        for u, x in C.items():
            L.append(f'| {u} | ' + ' | '.join(f"{x['curve'][s]:.2f}" if s in x['curve'] else 'NA' for s in STEPS) + ' |')
        L += ['', '| unit | train s | GPU | eval s/1000q | eval-only AUC clean vs mismatched |', '|---|---|---|---|---|']
        for u, x in C.items():
            L.append(f"| {u} | {x.get('train_seconds','')} | {x.get('gpu','')} | {x.get('eval_s_per_1000q','')} | {fm(x.get('evalonly_auc_clean_vs_mismatched_s8000'),4)} |")
        wcsv(H/'RESULTS_NR2C_STEP8000.csv', list(C.values())); wcsv(H/'RESULTS_NR2C_PAIRED.csv', crow)
        wcsv(H/'RESULTS_NR2C_CURVES.csv', [dict(unit_id=u, step=s, rSum=v, fit_J_productQ=x.get('fit_J_curve', {}).get(s, '')) for u, x in C.items() for s, v in x['curve'].items()])
    # J. NR2D-A: COCO 3-seed (s1001 = NR2C, s1002/s1003 = NR2D) and paired HREF-U per seed
    if bc:
        RD = OC/'runs_nr2d'; CJ = {}
        for f in sorted((H/'locked_configs_nr2d').glob('NR2D_*COCO*.json')):
            c = json.loads(f.read_text())
            CJ[f.stem] = read_unit(f.stem, RD/f.stem, dict(block='NR2D_A', condition=c['condition'], objective=c['objective'], seed=str(c['seed']), reference=c['reference']))
        g = lambda a, c, o, s: (C if s == 1001 else CJ).get(f"{'NR2C' if s == 1001 else 'NR2D'}_{a}_{c}_G_{o}_s{s}", {}).get('rSum')
        L += ['', '## J. NR2D-A COCO seed replication: HREF and U, step 8000 rSum (s1001 = NR2C, s1002/s1003 = NR2D)', '',
              f"COCO base rSum {bc['metrics']['rSum']:.2f}. NR2D-A complete units: {sum(x['status'] == 'COMPLETE' for x in CJ.values())}/{len(CJ)}", '',
              '| condition | objective | reference | s1001 | s1002 | s1003 | mean | sd | mean ΔrSum |', '|---|---|---|---|---|---|---|---|---|']
        jrows, jpair = [], []
        for c in ('COCO00', 'COCO60', 'COCO80'):
            for o in ('J', 'BCE', 'NCE'):
                for a in ('U', 'HREF'):
                    v = [g(a, c, o, s) for s in (1001, 1002, 1003)]; full = all(x is not None for x in v)
                    mean = float(np.mean(v)) if full else None; sd = float(np.std(v, ddof=1)) if full else None
                    jrows.append(dict(condition=c, objective=o, reference=a, s1001=v[0], s1002=v[1], s1003=v[2], mean=mean, sd=sd))
                    dl = f"{mean - bc['metrics']['rSum']:+.2f}" if full else 'NA'
                    L.append(f"| {c} | {o} | {a} | " + ' | '.join(fm(x) for x in v) + f" | {fm(mean)} | {fm(sd)} | {dl} |")
        L += ['', 'Paired HREF−U per seed (COCO):', '', '| condition | objective | s1001 | s1002 | s1003 | mean | n positive |', '|---|---|---|---|---|---|---|']
        for c in ('COCO00', 'COCO60', 'COCO80'):
            for o in ('J', 'BCE', 'NCE'):
                d = [g('HREF', c, o, s) - g('U', c, o, s) if g('HREF', c, o, s) is not None and g('U', c, o, s) is not None else None for s in (1001, 1002, 1003)]
                full = all(x is not None for x in d)
                jpair.append(dict(condition=c, objective=o, s1001=d[0], s1002=d[1], s1003=d[2], mean=float(np.mean(d)) if full else None, n_positive=sum(x > 0 for x in d) if full else None))
                L.append(f"| {c} | {o} | " + ' | '.join(f'{x:+.2f}' if x is not None else 'NA' for x in d) + (f" | {np.mean(d):+.2f} | {sum(x > 0 for x in d)}/3 |" if full else ' | NA | NA |'))
        L += ['', 'NR2D-A rSum curves:', '', '| unit | ' + ' | '.join(map(str, STEPS)) + ' |', '|' + '---|'*(len(STEPS)+1)]
        for u, x in CJ.items():
            L.append(f'| {u} | ' + ' | '.join(f"{x['curve'][s]:.2f}" if s in x['curve'] else 'NA' for s in STEPS) + ' |')
        wcsv(H/'RESULTS_NR2D_COCO_SEEDS.csv', jrows); wcsv(H/'RESULTS_NR2D_COCO_PAIRED.csv', jpair); wcsv(H/'RESULTS_NR2D_COCO_STEP8000.csv', list(CJ.values()))
    # K. NR2D-B: CC152K_RECOVERED_20261104 (separate condition; never pooled with Flickr/COCO/original CC152K)
    CB = Path('/projects/EEG-foundation-model/yinghao/data/cc152k_raw/nr1/CC152K_RECOVERED_20261104')
    bcc = rj(CB/'baseline/METRICS.json')
    if bcc and (H/'locked_configs_nr2d').exists():
        RC = Path('/projects/EEG-foundation-model/yinghao/outputs_nr2/NR2D_20261104/runs_cc'); CK = {}
        for f in sorted((H/'locked_configs_nr2d').glob('NR2D_*CC152K*.json')):
            c = json.loads(f.read_text())
            CK[f.stem] = read_unit(f.stem, RC/f.stem, dict(block='NR2D_B', condition=c['condition'], objective=c['objective'], seed='1001', reference=c['reference']))
        NR1C = Path('/home/infres/yinwang/CS_QMI/outputs_nr1/NR1_20261104/runs_cc152k_recovered')
        L += ['', '## K. NR2D-B CC152K_RECOVERED_20261104 (recovered web snapshot; DEV 693 images / 693 captions; seed 1001; separate table)', '',
              '| base | ' + ' | '.join(K) + ' | i2t top100 cov | t2i top100 cov |', '|' + '---|'*(len(K)+3),
              '| SigLIP2 cosine | ' + ' | '.join(f"{bcc['metrics'][k]:.2f}" for k in K) + f" | {bcc['i2t']['coverage_topk']:.4f} | {bcc['t2i']['coverage_topk']:.4f} |", '',
              '| unit | ' + ' | '.join(K) + ' | ΔrSum | i2t resc/lost | t2i resc/lost | train hard frac ref i2t/t2i | FIT J productQ | FIT J proposalQ(g) | status |', '|' + '---|'*(len(K)+8)]
        for u, x in CK.items():
            if 'rSum' in x:
                L.append(f"| {u} | " + ' | '.join(f"{x[k]:.2f}" for k in K) + f" | {x['delta_rSum']:+.2f} | {x['i2t_rescued']}/{x['i2t_lost']} | {x['t2i_rescued']}/{x['t2i_lost']} | "
                         f"{fm(x.get('i2t_train_hard_frac_ref'),4)}/{fm(x.get('t2i_train_hard_frac_ref'),4)} | {fm(x.get('fit_J_productQ_s8000'),4)} | {fm(x.get('fit_J_proposalQ_g_s8000'),4)} | {x['status']} |")
            else:
                L.append(f"| {u} | " + ' | '.join('NA' for _ in K) + f" | NA | NA | NA | NA | NA | NA | {x['status']} |")
        L += ['', 'Matched differences (CC152K_RECOVERED, seed 1001; NR1 G = existing NR1 recovered-condition G unit, original NR1 sampler):', '',
              '| objective | U | HIW | HREF | NR1 G | HIW−U | HREF−U | HREF−HIW |', '|---|---|---|---|---|---|---|---|']
        krow = []
        for o in ('J', 'BCE', 'NCE'):
            u_, hw, hr = (CK.get(f'NR2D_{a}_CC152K_RECOVERED_G_{o}_s1001', {}).get('rSum') for a in ('U', 'HIW', 'HREF'))
            n1 = rj(NR1C/f'NR1_CC152K_RECOVERED_20261104_G_{o}_s1001/eval_008000/METRICS.json'); n1 = n1['metrics']['rSum'] if n1 else None
            dd = lambda a, b: a - b if a is not None and b is not None else None
            r = dict(objective=o, U=u_, HIW=hw, HREF=hr, NR1_G=n1, HIW_minus_U=dd(hw, u_), HREF_minus_U=dd(hr, u_), HREF_minus_HIW=dd(hr, hw)); krow.append(r)
            L.append(f"| {o} | {fm(u_)} | {fm(hw)} | {fm(hr)} | {fm(n1)} | " + ' | '.join(f'{r[k]:+.2f}' if r[k] is not None else 'NA' for k in ('HIW_minus_U', 'HREF_minus_U', 'HREF_minus_HIW')) + ' |')
        L += ['', 'NR2D-B rSum curves:', '', '| unit | ' + ' | '.join(map(str, STEPS)) + ' |', '|' + '---|'*(len(STEPS)+1)]
        for u, x in CK.items():
            L.append(f'| {u} | ' + ' | '.join(f"{x['curve'][s]:.2f}" if s in x['curve'] else 'NA' for s in STEPS) + ' |')
        L += ['', 'Evaluator-only clean/mismatched strata: NA (real web noise, no injected-noise truth).']
        wcsv(H/'RESULTS_NR2D_CC_STEP8000.csv', list(CK.values())); wcsv(H/'RESULTS_NR2D_CC_MATCHED.csv', krow)
        # L. NR2E: CC152K_RECOVERED 3-seed (s1001 = NR2D-B, s1002/s1003 = NR2E) and per-seed HREF-U
        RE = Path('/projects/EEG-foundation-model/yinghao/outputs_nr2/NR2D_20261104/runs_cc_nr2e'); CL = {}
        for f in sorted((H/'locked_configs_nr2e').glob('NR2E_*.json')):
            c = json.loads(f.read_text())
            CL[f.stem] = read_unit(f.stem, RE/f.stem, dict(block='NR2E', condition=c['condition'], objective=c['objective'], seed=str(c['seed']), reference=c['reference']))
        gl = lambda a, o, s: (CK if s == 1001 else CL).get(f"{'NR2D' if s == 1001 else 'NR2E'}_{a}_CC152K_RECOVERED_G_{o}_s{s}", {}).get('rSum')
        L += ['', '## L. NR2E CC152K_RECOVERED seed replication: HREF and U, step 8000 rSum (s1001 = NR2D-B, s1002/s1003 = NR2E)', '',
              f"CC152K_RECOVERED base rSum {bcc['metrics']['rSum']:.2f}. NR2E complete units: {sum(x['status'] == 'COMPLETE' for x in CL.values())}/{len(CL)}", '',
              '| objective | reference | s1001 | s1002 | s1003 | mean | sd | mean ΔrSum |', '|---|---|---|---|---|---|---|---|']
        lrows, lpair = [], []
        for o in ('J', 'BCE', 'NCE'):
            for a in ('U', 'HREF'):
                v = [gl(a, o, s) for s in (1001, 1002, 1003)]; full = all(x is not None for x in v)
                mean = float(np.mean(v)) if full else None; sd = float(np.std(v, ddof=1)) if full else None
                lrows.append(dict(objective=o, reference=a, s1001=v[0], s1002=v[1], s1003=v[2], mean=mean, sd=sd))
                dl = f"{mean - bcc['metrics']['rSum']:+.2f}" if full else 'NA'
                L.append(f"| {o} | {a} | " + ' | '.join(fm(x) for x in v) + f" | {fm(mean)} | {fm(sd)} | {dl} |")
        L += ['', 'Paired HREF−U per seed (CC152K_RECOVERED):', '', '| objective | s1001 | s1002 | s1003 | mean | n positive |', '|---|---|---|---|---|---|']
        for o in ('J', 'BCE', 'NCE'):
            d = [gl('HREF', o, s) - gl('U', o, s) if gl('HREF', o, s) is not None and gl('U', o, s) is not None else None for s in (1001, 1002, 1003)]
            full = all(x is not None for x in d)
            lpair.append(dict(objective=o, s1001=d[0], s1002=d[1], s1003=d[2], mean=float(np.mean(d)) if full else None, n_positive=sum(x > 0 for x in d) if full else None))
            L.append(f"| {o} | " + ' | '.join(f'{x:+.2f}' if x is not None else 'NA' for x in d) + (f" | {np.mean(d):+.2f} | {sum(x > 0 for x in d)}/3 |" if full else ' | NA | NA |'))
        L += ['', 'NR2E rSum curves:', '', '| unit | ' + ' | '.join(map(str, STEPS)) + ' |', '|' + '---|'*(len(STEPS)+1)]
        for u, x in CL.items():
            L.append(f'| {u} | ' + ' | '.join(f"{x['curve'][s]:.2f}" if s in x['curve'] else 'NA' for s in STEPS) + ' |')
        wcsv(H/'RESULTS_NR2E_CC_SEEDS.csv', lrows); wcsv(H/'RESULTS_NR2E_CC_PAIRED.csv', lpair); wcsv(H/'RESULTS_NR2E_CC_STEP8000.csv', list(CL.values()))
    L += ['', '## G. Status and cost', '', '| unit | status | attempts | train s | GPU | peak MiB | pair forwards | eval s/1000q |', '|---|---|---|---|---|---|---|---|']
    for x in nr2:
        L.append(f"| {x['unit_id']} | {x['status']} | {x['attempts']} | {x.get('train_seconds','')} | {x.get('gpu','')} | {x.get('peak_gpu_MiB','')} | {x.get('pair_forwards','')} | {x.get('eval_s_per_1000q','')} |")
    wcsv(H/'RESULTS_NR2_STATUS.csv', [{k: x.get(k, '') for k in ('unit_id', 'status', 'attempts', 'train_seconds', 'gpu', 'peak_gpu_MiB', 'pair_forwards', 'eval_s_per_1000q')} for x in nr2])
    (H/'RESULTS_NR2.md').write_text('\n'.join(L) + '\n'); print('\n'.join(L[:8]))


if __name__ == '__main__':
    main()
