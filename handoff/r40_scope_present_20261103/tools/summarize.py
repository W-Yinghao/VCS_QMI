"""R40 summary (numbers only; package docs/SERVER_PLAN_CN.md sections 3-5, 8). Reads the 27 R40 RESULT.json (runs/, oracle_runs/) and the
reused read-only references (R38 N-FROZEN, R39 N-JOINT, R39 ORACLE-FROZEN), SUBMITTED.tsv, FAILED_*.json, CPU_TEST*.json. Writes RESULTS.md and
CSVs: RESULTS_FIXED, CONTRASTS_SCOPE_2x2 (section 3), CONTRASTS_PRESENT (section 4), CONTRASTS_ORACLE (section 5), OBJECTIVE_CONTRASTS, CURVES,
RESCUED_LOST (vs H and between cells), BOOTSTRAP (descriptive, image-grouped), GRAD_SPLIT, PRESENT_LOGS, STUDENT_CHANGE, COST_LEDGER, FAILURES.
Missing units -> NA. The oracle-JOINT cells are read here only as finished RESULT/rank files (summary process, not a trainer).
Usage: python3 -I -B tools/summarize.py"""
import csv, glob, json, math
from pathlib import Path
import numpy as np
W = Path(__file__).resolve().parents[1]
OUT = Path('/projects/EEG-foundation-model/yinghao/outputs_v9/r40_scope_present_20261103')
R39_OUT = Path('/projects/EEG-foundation-model/yinghao/outputs_v9/r39_complete_joint_20261102')
R38_RUNS = Path('/projects/EEG-foundation-model/yinghao/outputs_v9/r38_direction_20261102/runs')
DS = {'Flickr60': dict(tag='F60_R30_H1001', r39='Flickr60', per=5, n_img=1014, n_cap=5070, r38='r38_B_f60_R30_s1001_N_shared_{o}_cosine'),
      'CC152K': dict(tag='CC31_H1001', r39='CC152K', per=1, n_img=1000, n_cap=1000, r38='r38_B_cc_R31_s1001_N_shared_{o}_cosine')}
OBJ = ('J', 'BCE', 'NCE')
CELLS = ('N_frozen', 'image_only', 'text_only', 'N_both_joint', 'PRESENT_frozen', 'PRESENT_both_joint', 'ORACLE_frozen', 'ORACLE_both_joint')
NEW = {'image_only', 'text_only', 'PRESENT_frozen', 'PRESENT_both_joint', 'ORACLE_both_joint'}
MET = ('i2t_r1', 'i2t_r5', 'i2t_r10', 't2i_r1', 't2i_r5', 't2i_r10', 'rsum')


def rj(p):
    return json.loads(Path(p).read_text())


def unit_dir(ds, cell, o):
    P = DS[ds]
    return {'N_frozen': R38_RUNS / P['r38'].format(o=o),
            'N_both_joint': R39_OUT / 'runs' / f"R39_{P['r39']}_N_joint_global_{o}",
            'ORACLE_frozen': R39_OUT / 'oracle_runs' / f'R39_Flickr60_ORACLE_MISSING_frozen_{o}',
            'image_only': OUT / 'runs' / f"R40_A_{P['tag']}_image_only_{o}", 'text_only': OUT / 'runs' / f"R40_A_{P['tag']}_text_only_{o}",
            'PRESENT_frozen': OUT / 'runs' / f"R40_B_{P['tag']}_PRESENT_frozen_{o}", 'PRESENT_both_joint': OUT / 'runs' / f"R40_B_{P['tag']}_PRESENT_both_joint_{o}",
            'ORACLE_both_joint': OUT / 'oracle_runs' / f"R40_C_{P['tag']}_ORACLE_joint_{o}"}[cell]


def cells(ds):
    return [c for c in CELLS if ds == 'Flickr60' or not c.startswith('ORACLE')]


def f(x, nd=2):
    return 'NA' if x is None or (isinstance(x, float) and math.isnan(x)) else (f'{x:.{nd}f}' if isinstance(x, float) else str(x))


def wcsv(name, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(W / name, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rows)


def sub(a, b):
    return None if a is None or b is None else a - b


def main():
    R = {(ds, c, o): (rj(unit_dir(ds, c, o) / 'RESULT.json') if (unit_dir(ds, c, o) / 'RESULT.json').exists() else None) for ds in DS for c in cells(ds) for o in OBJ}
    v = lambda ds, c, o, k='rsum': R[(ds, c, o)]['final']['dev_retrieval'][k] if R.get((ds, c, o)) else None
    rows, curves, rl, gsr, pl, sc, cost = [], [], [], [], [], [], []
    for (ds, cell, o), r in R.items():
        base = dict(dataset=ds, cell=cell, objective=o, unit=unit_dir(ds, cell, o).name, source='R40 (new)' if cell in NEW else 'reused reference (read-only)', status=r['status'] if r else 'NA')
        if r is None:
            rows.append(base); continue
        m = r['final']['dev_retrieval']; nat = r['native_H_dev']; tr = r['final'].get('transitions_vs_H', {})
        r40 = r.get('r40') or {}
        rows.append(dict(base, **{k: m[k] for k in MET}, H_rsum=nat['rsum'], d_vs_H=m['rsum'] - nat['rsum'],
                         tie_pess=r['final']['dev_retrieval_tie_bounded']['rsum_pessimistic'], tie_opt=r['final']['dev_retrieval_tie_bounded']['rsum_optimistic'],
                         step0_native_passed=(rj(unit_dir(ds, cell, o) / 'eval_step0.json').get('step0_native_check') or {}).get('passed') if (unit_dir(ds, cell, o) / 'eval_step0.json').exists() else None,
                         one_step_r40_passed=(r40.get('one_step_check') or {}).get('passed'), refs_ok=all(x['init_equal'] and x['student_init_equal'] in (None, True) and len(x['chain_equal_at']) == 7 for x in (r40.get('references') or {}).values()) if r40 else None,
                         gpu=r['env']['gpu'], job=r['env']['job']))
        rl.append(dict(base, **{k: tr.get(k) for k in ('i2t_rescued_r1', 'i2t_lost_r1', 't2i_rescued_r1', 't2i_lost_r1', 'i2t_rescued_r5', 'i2t_lost_r5', 't2i_rescued_r5', 't2i_lost_r5')},
                       reference='frozen H'))
        for ph in r['phases']:
            for c in ph['curves']:
                curves.append(dict(base, step=c['step'], **{k: c['dev_retrieval'][k] for k in MET}))
        for w in r40.get('grad_split_windows', []):
            gsr.append(dict(base, **w))
        if r.get('r39', {}) and r['r39'].get('joint'):
            fc = r['final'].get('r39_student_feature_change', {}); pdist = r['final'].get('r39_param_distance', {})
            sc.append(dict(base, **{f'{k}_mean_cos': x['mean_cos'] for k, x in fc.items()}, **{f'param_rel_l2_{p}': x['rel_l2_from_parent'] for p, x in pdist.items()},
                           **{f'param_rel_l2_{p}_{mm}': y['rel_l2_from_parent'] for p, x in pdist.items() for mm, y in (x.get('per_modality') or {}).items()}))
        bs = r.get('batch_stats', {}).get('phase1', {})
        if cell.startswith('PRESENT'):
            n = r['config']['steps']
            pl.append(dict(base, updates=n, **{k: bs.get(k) for k in ('valid_rows_i2t', 'valid_rows_t2i', 'empty_valid_batch_i2t', 'empty_valid_batch_t2i', 'valid_unique_queries_i2t',
                                                                       'valid_unique_queries_t2i', 'present_weight_i2t', 'present_weight_t2i')},
                           p_i2t=r['present_probs']['p_i2t'], p_t2i=r['present_probs']['p_t2i'], masked_grad=json.dumps(r40.get('masked_grad'))))
        st = (r.get('r40') or {}).get('one_step_check') or {}
        cost.append(dict(base, train_seconds=r['cost']['train_seconds'], eval_seconds=r['cost']['eval_seconds'], params_trainable=r['cost']['params_trainable'],
                         peak_gpu_memory_GB=r['cost']['peak_gpu_memory_bytes'] / 2 ** 30, gpu=r['env']['gpu']))
    wcsv('RESULTS_FIXED.csv', rows); wcsv('CURVES.csv', curves); wcsv('GRAD_SPLIT.csv', gsr); wcsv('PRESENT_LOGS.csv', pl); wcsv('STUDENT_CHANGE.csv', sc); wcsv('COST_LEDGER.csv', cost)
    # ------------------------------------------------ contrasts
    sx, px, ox, oc = [], [], [], []
    for ds in DS:
        for o in OBJ:
            for k in MET:
                F, I, T, B = (v(ds, c, o, k) for c in ('N_frozen', 'image_only', 'text_only', 'N_both_joint'))
                sx.append(dict(dataset=ds, objective=o, metric=k, FROZEN=F, IMAGE_ONLY=I, TEXT_ONLY=T, BOTH=B, IMAGE_ONLY_minus_FROZEN=sub(I, F), TEXT_ONLY_minus_FROZEN=sub(T, F),
                               BOTH_minus_IMAGE_ONLY=sub(B, I), BOTH_minus_TEXT_ONLY=sub(B, T), BOTH_minus_FROZEN=sub(B, F),
                               interaction_BOTH_minus_I_minus_T_plus_F=None if None in (F, I, T, B) else B - I - T + F))
                PF, PJ = v(ds, 'PRESENT_frozen', o, k), v(ds, 'PRESENT_both_joint', o, k)
                px.append(dict(dataset=ds, objective=o, metric=k, N_FROZEN=F, N_JOINT=B, PRESENT_FROZEN=PF, PRESENT_JOINT=PJ, PRESENT_minus_N_frozen=sub(PF, F), PRESENT_minus_N_joint=sub(PJ, B),
                               JOINT_minus_FROZEN_PRESENT=sub(PJ, PF), JOINT_minus_FROZEN_N=sub(B, F), interaction=None if None in (F, B, PF, PJ) else (PJ - PF) - (B - F)))
                if ds == 'Flickr60':
                    OF, OJ = v(ds, 'ORACLE_frozen', o, k), v(ds, 'ORACLE_both_joint', o, k)
                    ox.append(dict(dataset=ds, objective=o, metric=k, N_FROZEN=F, N_JOINT=B, ORACLE_FROZEN=OF, ORACLE_JOINT=OJ, ORACLE_JOINT_minus_N_JOINT=sub(OJ, B),
                                   ORACLE_JOINT_minus_ORACLE_FROZEN=sub(OJ, OF), ORACLE_FROZEN_minus_N_FROZEN=sub(OF, F),
                                   interaction=None if None in (F, B, OF, OJ) else (OJ - OF) - (B - F)))
        for c in cells(ds):
            oc.append(dict(dataset=ds, cell=c, J=v(ds, c, 'J'), BCE=v(ds, c, 'BCE'), NCE=v(ds, c, 'NCE'), J_minus_BCE=sub(v(ds, c, 'J'), v(ds, c, 'BCE')), J_minus_NCE=sub(v(ds, c, 'J'), v(ds, c, 'NCE'))))
    wcsv('CONTRASTS_SCOPE_2x2.csv', sx); wcsv('CONTRASTS_PRESENT.csv', px); wcsv('CONTRASTS_ORACLE.csv', ox); wcsv('OBJECTIVE_CONTRASTS.csv', oc)
    # ------------------------------------------------ rescued / lost between cells, grouped bootstrap
    def ranks(ds, c, o):
        p = unit_dir(ds, c, o) / 'dev_ranks_step8000.npz'
        return np.load(p) if p.exists() else None
    PAIRS = [('image_only', 'N_frozen'), ('text_only', 'N_frozen'), ('N_both_joint', 'image_only'), ('N_both_joint', 'text_only'), ('PRESENT_frozen', 'N_frozen'),
             ('PRESENT_both_joint', 'N_both_joint'), ('PRESENT_both_joint', 'PRESENT_frozen'), ('ORACLE_both_joint', 'N_both_joint'), ('ORACLE_both_joint', 'ORACLE_frozen')]
    INTER = {'scope_interaction': [('N_both_joint', 1), ('image_only', -1), ('text_only', -1), ('N_frozen', 1)],
             'present_interaction': [('PRESENT_both_joint', 1), ('PRESENT_frozen', -1), ('N_both_joint', -1), ('N_frozen', 1)],
             'oracle_interaction': [('ORACLE_both_joint', 1), ('ORACLE_frozen', -1), ('N_both_joint', -1), ('N_frozen', 1)]}
    rng = np.random.default_rng(20261103); bsr = []
    for ds, P in DS.items():
        img_of_cap = np.arange(P['n_cap']) // P['per']; caps_by_img = [np.nonzero(img_of_cap == i)[0] for i in range(P['n_img'])]
        Bn = 1000; idx = rng.integers(0, P['n_img'], (Bn, P['n_img'])); cache = {}
        def boot(key):
            if key not in cache:
                rk = ranks(ds, *key)
                if rk is None:
                    cache[key] = None
                else:
                    hi = {k: (rk['ranks_i'] < k).astype(float) for k in (1, 5, 10)}; ht = {k: (rk['ranks_t'] < k).astype(float) for k in (1, 5, 10)}
                    out = np.zeros(Bn)
                    for b in range(Bn):
                        ii = idx[b]; cc = np.concatenate([caps_by_img[i] for i in ii])
                        out[b] = sum(100 * hi[k][ii].mean() + 100 * ht[k][cc].mean() for k in (1, 5, 10))
                    cache[key] = out
            return cache[key]
        for o in OBJ:
            for a, b in PAIRS:
                if a not in cells(ds) or b not in cells(ds):
                    continue
                ra, rb = ranks(ds, a, o), ranks(ds, b, o)
                if ra is not None and rb is not None:
                    rl.append(dict(dataset=ds, cell=a, objective=o, reference=b, **{f'{dn}_rescued_r{k}': int(((ra[f'ranks_{dd}'] < k) & ~(rb[f'ranks_{dd}'] < k)).sum()) for dn, dd in (('i2t', 'i'), ('t2i', 't')) for k in (1, 5)},
                                   **{f'{dn}_lost_r{k}': int((~(ra[f'ranks_{dd}'] < k) & (rb[f'ranks_{dd}'] < k)).sum()) for dn, dd in (('i2t', 'i'), ('t2i', 't')) for k in (1, 5)}))
                xa, xb = boot((a, o)), boot((b, o))
                if xa is None or xb is None:
                    bsr.append(dict(dataset=ds, objective=o, contrast=f'{a} - {b}', mean_diff=None)); continue
                dd = xa - xb
                bsr.append(dict(dataset=ds, objective=o, contrast=f'{a} - {b}', mean_diff=float(dd.mean()), q025=float(np.quantile(dd, .025)), q975=float(np.quantile(dd, .975))))
            for nm, terms in INTER.items():
                if any(c not in cells(ds) for c, _ in terms):
                    continue
                xs = [boot((c, o)) for c, _ in terms]
                if any(x is None for x in xs):
                    bsr.append(dict(dataset=ds, objective=o, contrast=nm, mean_diff=None)); continue
                dd = sum(s * x for (c, s), x in zip(terms, xs))
                bsr.append(dict(dataset=ds, objective=o, contrast=nm, mean_diff=float(dd.mean()), q025=float(np.quantile(dd, .025)), q975=float(np.quantile(dd, .975))))
    for r in bsr:
        r['note'] = 'descriptive image-grouped bootstrap of DEV queries (1000 draws, same draws for all cells of a dataset); one parent state, not a confirmation'
    wcsv('RESCUED_LOST.csv', rl); wcsv('BOOTSTRAP.csv', bsr)
    # ------------------------------------------------ failures
    fails = []
    for p in sorted(glob.glob(str(OUT / '*runs' / '*' / 'FAILED_*.json'))):
        x = rj(p); fails.append(dict(unit=x['unit'], file=p, utc=x['utc'], error=x['traceback'].strip().splitlines()[-1][:300]))
    for p in sorted(glob.glob(str(W / 'CPU_TEST*.json'))):
        x = rj(p)
        for k, c in x['checks'].items():
            if not c['passed']:
                fails.append(dict(unit=f'{Path(p).name}:{k}', file=p, utc=x['utc'], error=(c.get('error') or json.dumps(c.get('detail'), default=str)).strip().splitlines()[-1][-300:]))
    wcsv('FAILURES.csv', fails or [dict(unit='none', file='', utc='', error='')])
    # ------------------------------------------------ RESULTS.md
    n_new = sum(1 for (ds, c, o), r in R.items() if r is not None and c in NEW)
    L = ['# R40 update scope x incomplete-correspondence supervision (numbers only)', '',
         f'{n_new}/27 R40 units complete. Reused read-only references: R38 N-FROZEN, R39 N-JOINT, R39 ORACLE-FROZEN. Fixed step 8000; DEV two-stage (frozen-H top-100 reranked, rest in H order). TEST not read.', '']
    for ds in DS:
        nat = next((r['native_H_dev']['rsum'] for (d_, c_, o_), r in R.items() if d_ == ds and r), None)
        L += [f'## {ds} (frozen H DEV rSum {f(nat)}): rSum at step 8000', '', '| cell | J | BCE | NCE | J-BCE | J-NCE |', '|---|---|---|---|---|---|']
        for c in cells(ds):
            a, b, cc = (v(ds, c, o) for o in OBJ)
            L.append(f"| {c}{'' if c in NEW else ' (ref)'} | {f(a)} | {f(b)} | {f(cc)} | {f(sub(a, b))} | {f(sub(a, cc))} |")
        L += ['', 'Section 3 (update scope, rSum):', '', '| objective | IMAGE-FROZEN | TEXT-FROZEN | BOTH-IMAGE | BOTH-TEXT | BOTH-IMAGE-TEXT+FROZEN |', '|---|---|---|---|---|---|']
        for o in OBJ:
            x = next(r for r in sx if r['dataset'] == ds and r['objective'] == o and r['metric'] == 'rsum')
            L.append(f"| {o} | {f(x['IMAGE_ONLY_minus_FROZEN'])} | {f(x['TEXT_ONLY_minus_FROZEN'])} | {f(x['BOTH_minus_IMAGE_ONLY'])} | {f(x['BOTH_minus_TEXT_ONLY'])} | {f(x['interaction_BOTH_minus_I_minus_T_plus_F'])} |")
        L += ['', 'Section 4 (PRESENT, rSum):', '', '| objective | PRESENT-N (frozen) | PRESENT-N (joint) | [PJ-PF]-[NJ-NF] |', '|---|---|---|---|']
        for o in OBJ:
            x = next(r for r in px if r['dataset'] == ds and r['objective'] == o and r['metric'] == 'rsum')
            L.append(f"| {o} | {f(x['PRESENT_minus_N_frozen'])} | {f(x['PRESENT_minus_N_joint'])} | {f(x['interaction'])} |")
        if ds == 'Flickr60':
            L += ['', 'Section 5 (ORACLE_MISSING, rSum):', '', '| objective | ORACLE_JOINT-N_JOINT | ORACLE_JOINT-ORACLE_FROZEN | [OJ-OF]-[NJ-NF] |', '|---|---|---|---|']
            for o in OBJ:
                x = next(r for r in ox if r['objective'] == o and r['metric'] == 'rsum')
                L.append(f"| {o} | {f(x['ORACLE_JOINT_minus_N_JOINT'])} | {f(x['ORACLE_JOINT_minus_ORACLE_FROZEN'])} | {f(x['interaction'])} |")
        L.append('')
    L += ['Files: RESULTS_FIXED.csv, CONTRASTS_SCOPE_2x2.csv / CONTRASTS_PRESENT.csv / CONTRASTS_ORACLE.csv (all six recalls + rSum), OBJECTIVE_CONTRASTS.csv, CURVES.csv,',
          'RESCUED_LOST.csv (vs frozen H and between cells), BOOTSTRAP.csv (descriptive), GRAD_SPLIT.csv, PRESENT_LOGS.csv, STUDENT_CHANGE.csv, COST_LEDGER.csv, FAILURES.csv; diag7/ (section 7 readouts).']
    (W / 'RESULTS.md').write_text('\n'.join(L) + '\n')
    print('R40 summary written', n_new)


if __name__ == '__main__':
    main()
