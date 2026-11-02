"""R38 summary (numbers only; evaluator process, no TEST file is opened). Reads the 30 R38 units (OUT/runs/<unit>) and the 12 read-only
reference units (R36 Flickr n60 N/T shared constant, R37C CC s1001 N/T shared constant), parent caches/preps. Writes RESULTS_FIXED.csv,
CURVES.csv, DIRECTIONAL_RECALL.csv, RANK_TRANSITIONS.csv, ARCH_X_SCHEDULE_CONTRASTS.csv, OBJECTIVE_CONTRASTS.csv, T_SCHEDULE.csv,
A_REAL_REPLICATION.csv, BOOTSTRAP.csv, CC_SUPPORT_CROSSTAB.csv, GRAD_DIRECTION_DIAG.csv, STEP0_CHECK.csv, NULL_READOUTS.csv,
COST_LEDGER.csv, RESULTS.md. Missing units are NA, never filled. Usage: python -I -B tools/summarize.py"""
import csv, json, subprocess, sys
from pathlib import Path
import numpy as np
W = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(W / 'r38c'))
from common import read_json, utc, install_guards, OUT, CONDS as CC

PY = sys.executable
R6 = ['i2t_r1', 'i2t_r5', 'i2t_r10', 't2i_r1', 't2i_r5', 't2i_r10']
RUNS = OUT / 'runs'
PAR = {'f60_R30_s1001': dict(tree='r38f', cond='n60', per=5, n_img=1014, cache='/projects/EEG-foundation-model/yinghao/outputs_v9/r34_direct_critic_20261102/cache/n60',
                             prep='/projects/EEG-foundation-model/yinghao/outputs_v9/r35_expanded_20261102/prep', key='n60', trust='trust_n60.npz'),
       'f60_R30_s1002': dict(tree='r38f', cond='n60s1002', per=5, n_img=1014, cache='/projects/EEG-foundation-model/yinghao/outputs_v9/r37_confirm_20261102/cache/n60s1002',
                             prep='/projects/EEG-foundation-model/yinghao/outputs_v9/r37_confirm_20261102/prep', key='n60s1002', trust='trust_n60s1002.npz'),
       'f60_R30_s1003': dict(tree='r38f', cond='n60s1003', per=5, n_img=1014, cache='/projects/EEG-foundation-model/yinghao/outputs_v9/r37_confirm_20261102/cache/n60s1003',
                             prep='/projects/EEG-foundation-model/yinghao/outputs_v9/r37_confirm_20261102/prep', key='n60s1003', trust='trust_n60s1003.npz'),
       'cc_R31_s1001': dict(tree='r38c', cond='cc', per=1, n_img=1000), 'cc_R31_s1002': dict(tree='r38c', cond='cc_s1002', per=1, n_img=1000),
       'cc_R31_s1003': dict(tree='r38c', cond='cc_s1003', per=1, n_img=1000)}
for k in ('cc_R31_s1001', 'cc_R31_s1002', 'cc_R31_s1003'):
    c = CC[PAR[k]['cond']]; PAR[k].update(cache=c['cache_dir'], prep=c['prep_dir'], key=c['prep_key'], trust=c['trust_file'])
CKS = [0, 250, 500, 1000, 2000, 4000, 8000]
FLICKR = ('f60_R30_s1001', 'f60_R30_s1002', 'f60_R30_s1003'); CCP = ('cc_R31_s1001', 'cc_R31_s1002', 'cc_R31_s1003')


def wcsv(name, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r)) if rows else ['empty']
    with open(W / name, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)


def tbl(rows, cols, fmt='{:.2f}'):
    out = ['|' + '|'.join(cols) + '|', '|' + '|'.join('---' for _ in cols) + '|']
    for r in rows:
        out.append('|' + '|'.join((fmt.format(r[k]) if isinstance(r.get(k), float) else ('NA' if r.get(k) is None else str(r.get(k)))) for k in cols) + '|')
    return '\n'.join(out)


def units():
    """all arms: R38 units (configs) + reused references, keyed by (parent, route, arch, schedule, objective)."""
    U = {}
    for u in read_json(W / 'configs' / 'UNITS.json'):
        c = read_json(W / 'configs' / f'{u}.json')
        U[(c['parent_key'], c['route'], c['architecture'], c['optimizer']['schedule'], c['objective'])] = dict(unit=u, dir=RUNS / u, block=c['block'], source='R38 new', cfg=c)
    for r in read_json(W / 'REUSE_REFERENCES.json')['references']:
        U[(r['parent_key'], r['route'], 'shared', 'constant', r['objective'])] = dict(unit=r['unit_id'], dir=Path(r['dir']), block='ref', source='reused read-only (R36/R37C)', cfg=None)
    return U


def boot_pair(rka, rkb, per, n_img, B=2000, seed=20261102):
    """image-grouped paired bootstrap of rSum(a) - rSum(b): resample DEV images; i2t = image query, t2i = its `per` captions."""
    rng = np.random.default_rng(seed)
    def parts(rk):
        return rk['ranks_i'], rk['ranks_t'].reshape(n_img, per)
    ia, ta = parts(rka); ib, tb = parts(rkb)
    def rs(sel):
        f = lambda i, t: sum(100 * (i[sel] < k).mean() for k in (1, 5, 10)) + sum(100 * (t[sel] < k).mean() for k in (1, 5, 10))
        return f(ia, ta) - f(ib, tb)
    full = rs(np.arange(n_img)); bs = np.array([rs(rng.integers(0, n_img, n_img)) for _ in range(B)])
    return dict(diff=float(full), ci2_5=float(np.quantile(bs, .025)), ci97_5=float(np.quantile(bs, .975)), frac_boot_gt0=float((bs > 0).mean()), B=B)


def trans(rka, rkb, k=1):
    out = {}
    for d, key in (('i2t', 'ranks_i'), ('t2i', 'ranks_t')):
        a, b = rka[key] < k, rkb[key] < k
        out[f'{d}_a_only_r{k}'] = int((a & ~b).sum()); out[f'{d}_b_only_r{k}'] = int((~a & b).sum()); out[f'{d}_both_r{k}'] = int((a & b).sum())
        out[f'{d}_neither_r{k}'] = int((~a & ~b).sum()); out[f'{d}_net_r{k}'] = out[f'{d}_a_only_r{k}'] - out[f'{d}_b_only_r{k}']
        out[f'{d}_top1_changed'] = int((rka[key.replace('ranks', 'top1')] != rkb[key.replace('ranks', 'top1')]).sum())
    return out


def main():
    install_guards(role='audit')
    U = units(); tmp = OUT / 'summary'; tmp.mkdir(parents=True, exist_ok=True)
    natrk, natm = {}, {}
    for pk, P in PAR.items():
        f = tmp / f'native_ranks_{P["cond"]}.npz'
        if not f.exists() and (Path(P['cache']) / 'CACHE_MANIFEST.json').exists():
            subprocess.run([PY, '-I', '-B', str(W / 'tools' / 'native_ranks.py'), str(W / P['tree']), P['cond'], str(f)], check=True)
        if f.exists():
            natrk[pk] = dict(np.load(f)); natm[pk] = json.loads(Path(str(f) + '.json').read_text())['metrics']
    R, RK = {}, {}
    fixed, curves, dirrec, step0, nulls, gdiag, cost, missing = [], [], [], [], [], [], [], []
    for key, u in sorted(U.items()):
        pk, route, arch, sch, obj = key
        base = dict(unit=u['unit'], block=u['block'], source=u['source'], parent=pk, route=route, architecture=arch, schedule=sch, objective=obj)
        rf = u['dir'] / 'RESULT.json'
        if not rf.exists():
            missing.append(u['unit']); fixed.append(dict(base, status='NA (not complete)')); continue
        r = read_json(rf); R[key] = r; fin = r['final']; dr = fin['dev_retrieval']; H = r['native_H_dev']
        RK[key] = dict(np.load(u['dir'] / 'dev_ranks_step8000.npz'))
        cv = {e['step']: e for e in r['phases'][0]['curves']}
        s0 = cv[0].get('step0_native_check') or read_json(u['dir'] / 'eval_step0.json').get('step0_native_check')
        h38 = r.get('r38') or {}
        gc = h38.get('step1_graph_check') or {}
        rs_curve = {s: cv[s]['dev_retrieval']['rsum'] for s in CKS if s in cv}
        row = dict(base, status='COMPLETE', **{k: dr[k] for k in R6}, rsum=dr['rsum'], H_rsum=H['rsum'], **{f'd_{k}': dr[k] - H[k] for k in R6}, d_rsum=dr['rsum'] - H['rsum'],
                   change_2000_to_8000=rs_curve[8000] - rs_curve[2000], curve_max_rsum_secondary=max(rs_curve.values()),
                   curve_argmax_step=max(rs_curve, key=rs_curve.get), rsum_curve=' / '.join(f'{rs_curve[s]:.2f}' for s in CKS),
                   step0_passed=s0['passed'] if s0 else None, graph_check_passed=gc.get('passed'), reference_unit=h38.get('reference_unit'),
                   reference_init_equal=h38.get('reference_init_equal'), reference_chain_equal_at=' '.join(map(str, h38.get('reference_chain_equal_at', []))) or None,
                   lr_last_update=(h38.get('lr_at_readout') or {}).get('8000'), gpu=r['env']['gpu'], job=r['env']['job'],
                   init_state_sha256=r['phases'][0]['identity']['init_state_sha256'][:16], batch_index_sha256=r['phases'][0]['identity']['batch_index_sha256'][:16])
        fixed.append(row)
        for s, e in sorted(cv.items()):
            curves.append(dict(base, step=s, rsum=e['dev_retrieval']['rsum'], d_rsum_vs_H=e['dev_retrieval']['rsum'] - H['rsum'], **{k: e['dev_retrieval'][k] for k in R6},
                               fit_list_observed_J_raw=e['fit_list_observed_J_raw']['J'], dev_list_clean_J_raw=e['dev_list_clean_J_raw']['J'],
                               train_loss_window=e.get('train_window', {}).get('loss_mean'), batch_J_window=e.get('train_window', {}).get('batch_J_mean'),
                               grad_norm_window=e.get('train_window', {}).get('grad_norm_mean'), lr_last_update=(h38.get('lr_at_readout') or {}).get(str(s)),
                               T_sat_frac=e.get('T_saturation_dev_top100', {}).get('frac_absT_ge_0p999'),
                               t2i_pairacc_Hhard10=e['pair_win_rates']['t2i_pairacc_Hhard10'], i2t_pairacc_Hhard10=e['pair_win_rates']['i2t_pairacc_Hhard10']))
        for d in ('i2t', 't2i'):
            dirrec.append(dict(base, direction=d, **{f'r{k}': dr[f'{d}_r{k}'] for k in (1, 5, 10)}, **{f'H_r{k}': H[f'{d}_r{k}'] for k in (1, 5, 10)},
                               **{f'd_r{k}': dr[f'{d}_r{k}'] - H[f'{d}_r{k}'] for k in (1, 5, 10)}, medr=dr[f'{d}_medr'], meanr=dr[f'{d}_meanr'],
                               **{f'{w}_r{k}_vs_H': fin['transitions_vs_H'][f'{d}_{w}_r{k}'] for w in ('rescued', 'lost') for k in (1, 5, 10)},
                               coverage=fin['candidate_coverage'][f'coverage_{d}'], cond_r1_covered=fin['candidate_coverage'][f'cond_r1_{d}_covered']))
        step0.append(dict(base, **({k: s0[k] for k in s0} if s0 else {}), graph_check=json.dumps(gc.get('routing')) if gc else None,
                          graph_scores=json.dumps(gc.get('step0_scores')) if gc else None, grad_sum=json.dumps(gc.get('sum_of_direction_grads_equals_total')) if gc else None,
                          second_update_passed=(gc.get('second_update') or {}).get('passed') if gc else None, graph_passed=gc.get('passed') if gc else None))
        nr = fin.get('fit_null_lists_observed_J_raw')
        if nr:
            nulls.append(dict(base, **{f'{d}_{k}': nr[d][k] for d in ('i2t', 't2i') for k in ('J_total', 'J_real_part', 'J_null_part', 'null_label_fraction')},
                              null_params=json.dumps(nr['null_params'])))
        for s, dg in sorted((h38.get('dir_grad_diag') or {}).items(), key=lambda x: int(x[0])):
            for p, rec in dg.items():
                gdiag.append(dict(base, step=int(s), peer=p, cos_trunk=rec.get('trunk', {}).get('cos'), cos_all=rec['all']['cos'],
                                  norm_i2t_all=rec['all']['norm_i2t'], norm_t2i_all=rec['all']['norm_t2i'],
                                  norm_i2t_trunk=rec.get('trunk', {}).get('norm_i2t'), norm_t2i_trunk=rec.get('trunk', {}).get('norm_t2i'),
                                  cos_out_i2t=rec.get('out_i2t', {}).get('cos')))
        cost.append(dict(base, item='unit', gpu=r['env']['gpu'], host=r['env']['host'], job=r['env']['job'], train_seconds=r['cost']['train_seconds'],
                         eval_seconds=r['cost']['eval_seconds'], unit_wall_seconds=r['cost']['unit_wall_seconds_this_process'], params_trainable=r['cost']['params_trainable'],
                         peak_gpu_gb=r['cost']['peak_gpu_memory_bytes'] / 2 ** 30, cost_scope='R38 new' if u['block'] != 'ref' else 'reused (R36/R37C cost, not R38)'))

    # ---- B: architecture x schedule (N), per parent x objective
    axs, objc, tsch, arep, boots, rtr = [], [], [], [], [], []
    def g(key, m):
        return (R[key]['final']['dev_retrieval'][m] if key in R else None)
    def df(a, b):
        return None if a is None or b is None else a - b
    for pk in FLICKR + CCP:
        for obj in ('J', 'BCE', 'NCE'):
            k = {(a, s): (pk, 'N', a, s, obj) for a in ('shared', 'directional') for s in ('constant', 'cosine')}
            row = dict(parent=pk, objective=obj)
            for m in ('rsum', 'i2t_r1', 't2i_r1', 'i2t_r5', 't2i_r5'):
                v = {ks: g(kk, m) for ks, kk in k.items()}
                for (a, s), x in v.items():
                    row[f'{m}_{a}_{s}'] = x
                row[f'{m}_dir_minus_shared_constant'] = df(v['directional', 'constant'], v['shared', 'constant'])
                row[f'{m}_dir_minus_shared_cosine'] = df(v['directional', 'cosine'], v['shared', 'cosine'])
                row[f'{m}_cos_minus_const_shared'] = df(v['shared', 'cosine'], v['shared', 'constant'])
                row[f'{m}_cos_minus_const_dir'] = df(v['directional', 'cosine'], v['directional', 'constant'])
                row[f'{m}_interaction'] = df(row[f'{m}_cos_minus_const_dir'], row[f'{m}_cos_minus_const_shared'])
            if natm.get(pk):
                row['H_rsum'] = natm[pk]['rsum']; row['H_i2t_r1'] = natm[pk]['i2t_r1']; row['H_t2i_r1'] = natm[pk]['t2i_r1']
            axs.append(row)
            for (a1, s1), (a2, s2), lab in ((('directional', 'constant'), ('shared', 'constant'), 'dir-shared|constant'), (('directional', 'cosine'), ('shared', 'cosine'), 'dir-shared|cosine'),
                                             (('shared', 'cosine'), ('shared', 'constant'), 'cos-const|shared'), (('directional', 'cosine'), ('directional', 'constant'), 'cos-const|dir')):
                ka, kb = k[a1, s1], k[a2, s2]
                if ka in RK and kb in RK:
                    P = PAR[pk]
                    boots.append(dict(parent=pk, objective=obj, contrast=lab, a=R[ka]['unit_id'], b=R[kb]['unit_id'], **boot_pair(RK[ka], RK[kb], P['per'], P['n_img'])))
                    rtr.append(dict(parent=pk, objective=obj, contrast=lab, a=R[ka]['unit_id'], b=R[kb]['unit_id'], **trans(RK[ka], RK[kb], 1), **{k_ + '_r5': v_ for k_, v_ in trans(RK[ka], RK[kb], 5).items() if 'top1' not in k_ and not k_.endswith('_r5')}))
    # ---- objective contrasts within every (parent, route, arch, schedule) cell
    cells = sorted({k[:4] for k in U})
    for cell in cells:
        v = {o: (cell + (o,)) for o in ('J', 'BCE', 'NCE')}
        row = dict(parent=cell[0], route=cell[1], architecture=cell[2], schedule=cell[3])
        for m in ('rsum', 'i2t_r1', 't2i_r1'):
            for o in ('J', 'BCE', 'NCE'):
                row[f'{m}_{o}'] = g(v[o], m)
            row[f'{m}_J_minus_BCE'] = df(row[f'{m}_J'], row[f'{m}_BCE']); row[f'{m}_J_minus_NCE'] = df(row[f'{m}_J'], row[f'{m}_NCE'])
        for o in ('BCE', 'NCE'):
            if v['J'] in RK and v[o] in RK:
                P = PAR[cell[0]]
                boots.append(dict(parent=cell[0], objective=f'J-{o}', contrast=f'J-{o}|{cell[1]},{cell[2]},{cell[3]}', a=R[v['J']]['unit_id'], b=R[v[o]]['unit_id'], **boot_pair(RK[v['J']], RK[v[o]], P['per'], P['n_img'])))
                rtr.append(dict(parent=cell[0], objective=f'J-{o}', contrast=f'J-{o}|{cell[1]},{cell[2]},{cell[3]}', a=R[v['J']]['unit_id'], b=R[v[o]]['unit_id'], **trans(RK[v['J']], RK[v[o]], 1)))
        objc.append(row)
    # ---- C: T cosine vs T constant (+ N at the same schedule)
    for pk in FLICKR + CCP:
        for obj in ('J', 'BCE', 'NCE'):
            kc, kk, nc = (pk, 'T', 'shared', 'cosine', obj), (pk, 'T', 'shared', 'constant', obj), (pk, 'N', 'shared', 'cosine', obj)
            row = dict(parent=pk, objective=obj, T_constant_rsum=g(kk, 'rsum'), T_cosine_rsum=g(kc, 'rsum'), T_cos_minus_const=df(g(kc, 'rsum'), g(kk, 'rsum')),
                       N_cosine_rsum=g(nc, 'rsum'), T_minus_N_cosine=df(g(kc, 'rsum'), g(nc, 'rsum')), H_rsum=natm.get(pk, {}).get('rsum'))
            for kx, lab in ((kk, 'T_constant'), (kc, 'T_cosine')):
                if kx in R:
                    cv = {e['step']: e['dev_retrieval']['rsum'] for e in R[kx]['phases'][0]['curves']}
                    row[f'{lab}_curve'] = ' / '.join(f'{cv[s]:.1f}' for s in CKS); row[f'{lab}_max_secondary'] = max(cv.values())
            tsch.append(row)
            if kc in RK and kk in RK:
                P = PAR[pk]; boots.append(dict(parent=pk, objective=obj, contrast='T cos-const', a=R[kc]['unit_id'], b=R[kk]['unit_id'], **boot_pair(RK[kc], RK[kk], P['per'], P['n_img'])))
    # ---- A: three CC152K R31 parents, N shared constant
    for obj in ('J', 'BCE', 'NCE'):
        row = dict(objective=obj)
        ds = []
        for pk in ('cc_R31_s1001', 'cc_R31_s1002', 'cc_R31_s1003'):
            kx = (pk, 'N', 'shared', 'constant', obj)
            row[f'{pk}_H_rsum'] = natm.get(pk, {}).get('rsum'); row[f'{pk}_rsum'] = g(kx, 'rsum')
            row[f'{pk}_d_rsum'] = df(g(kx, 'rsum'), natm.get(pk, {}).get('rsum')); ds.append(row[f'{pk}_d_rsum'])
            row[f'{pk}_d_i2t_r1'] = df(g(kx, 'i2t_r1'), natm.get(pk, {}).get('i2t_r1')); row[f'{pk}_d_t2i_r1'] = df(g(kx, 't2i_r1'), natm.get(pk, {}).get('t2i_r1'))
            if kx in RK and pk in natrk:
                boots.append(dict(parent=pk, objective=obj, contrast='unit-H (A block)', a=R[kx]['unit_id'], b='native H', **boot_pair(RK[kx], natrk[pk], 1, 1000)))
        row['mean_d_rsum_3parents'] = float(np.mean(ds)) if all(x is not None for x in ds) else None
        row['n_parents_positive'] = sum(1 for x in ds if x is not None and x > 0) if all(x is not None for x in ds) else None
        arep.append(row)
    for pk in ('cc_R31_s1001', 'cc_R31_s1002', 'cc_R31_s1003'):
        row = dict(objective='J-BCE / J-NCE', parent=pk)
        j, b, n = (g((pk, 'N', 'shared', 'constant', o), 'rsum') for o in ('J', 'BCE', 'NCE'))
        row.update(J_minus_BCE=df(j, b), J_minus_NCE=df(j, n)); arep.append(row)
    # ---- three-parent pooled contrasts (per dataset; never across datasets)
    pooled = []
    for ds, pks in (('Flickr60 (R30-H s1001/1002/1003)', FLICKR), ('CC152K (R31 H0_default s1001/1002/1003)', CCP)):
        for obj in ('J', 'BCE', 'NCE'):
            rows_ = [a for a in axs if a['parent'] in pks and a['objective'] == obj]
            ts = [x for x in tsch if x['parent'] in pks and x['objective'] == obj]
            row = dict(dataset=ds, objective=obj)
            for m in ('rsum', 'i2t_r1', 't2i_r1'):
                for c_ in ('dir_minus_shared_constant', 'dir_minus_shared_cosine', 'cos_minus_const_shared', 'cos_minus_const_dir', 'interaction'):
                    v = [a.get(f'{m}_{c_}') for a in rows_]
                    ok_ = len(v) == 3 and all(x is not None for x in v)
                    row[f'{m}_{c_}_mean3'] = float(np.mean(v)) if ok_ else None; row[f'{m}_{c_}_npos'] = sum(x > 0 for x in v) if ok_ else None
            v = [x['T_cos_minus_const'] for x in ts]; ok_ = len(v) == 3 and all(x is not None for x in v)
            row['T_cos_minus_const_mean3'] = float(np.mean(v)) if ok_ else None; row['T_cos_minus_const_npos'] = sum(x > 0 for x in v) if ok_ else None
            pooled.append(row)
        for cellname, pred in (('N shared constant', lambda o: o['route'] == 'N' and o['architecture'] == 'shared' and o['schedule'] == 'constant'),
                               ('N shared cosine', lambda o: o['route'] == 'N' and o['architecture'] == 'shared' and o['schedule'] == 'cosine'),
                               ('N directional constant', lambda o: o['route'] == 'N' and o['architecture'] == 'directional' and o['schedule'] == 'constant'),
                               ('N directional cosine', lambda o: o['route'] == 'N' and o['architecture'] == 'directional' and o['schedule'] == 'cosine'),
                               ('T shared constant', lambda o: o['route'] == 'T' and o['schedule'] == 'constant'), ('T shared cosine', lambda o: o['route'] == 'T' and o['schedule'] == 'cosine')):
            cs = [o for o in objc if o['parent'] in pks and pred(o)]
            row = dict(dataset=ds, objective=f'J-BCE / J-NCE | {cellname}')
            for c_ in ('rsum_J_minus_BCE', 'rsum_J_minus_NCE', 'i2t_r1_J_minus_BCE', 't2i_r1_J_minus_BCE'):
                v = [o.get(c_) for o in cs]; ok_ = len(v) == 3 and all(x is not None for x in v)
                row[f'{c_}_mean3'] = float(np.mean(v)) if ok_ else None; row[f'{c_}_npos'] = sum(x > 0 for x in v) if ok_ else None
            pooled.append(row)
    ident = []
    for pk in PAR:
        for route in ('N', 'T'):
            us = [(k, R[k]) for k in R if k[0] == pk and k[1] == route]
            if not us:
                continue
            sk = {(r_.get('r38') or {}).get('init_state_sha256_shared_keys') or r_['phases'][0]['identity']['init_state_sha256'] for _, r_ in us}
            ch = {r_['phases'][0]['identity']['batch_index_sha256'] for _, r_ in us}
            ident.append(dict(parent=pk, route=route, n_units=len(us), n_distinct_init_shared_keys=len(sk), n_distinct_batch_chains=len(ch),
                              identical=len(sk) == 1 and len(ch) == 1))
    # ---- support stats per parent (observed relations only)
    sup = []
    for pk, P in PAR.items():
        try:
            pm = read_json(Path(P['prep']) / 'PREP_MANIFEST.json')
        except FileNotFoundError:
            sup.append(dict(parent=pk, status='NA (prep missing)')); continue
        ca = pm['candidates'][P['key']]; tr = pm['trust'][P['key']] if 'trust' in pm else pm[f"trust_{P['key']}"]
        obs = np.load(Path(P['cache']) / ('observed_map.npy' if (Path(P['cache']) / 'observed_map.npy').exists() else 'observed_map.npy')) if (Path(P['cache']) / 'observed_map.npy').exists() else None
        row = dict(parent=pk, trust_n=tr['n_subset'], trust_retention=tr['retention'], **{k: v for k, v in ca.items() if k.startswith(('i2t_frac', 't2i_frac'))},
                   a0=pm['calibration_init'][P['key']]['a0'], b0=pm['calibration_init'][P['key']]['b0'])
        cand = np.load(Path(P['prep']) / f"candidates_{P['key']}.npz"); T = np.load(Path(P['prep']) / P['trust'])
        if obs is None:   # Flickr: R34 cache stores the observed map under the cache manifest
            man = read_json(Path(P['cache']) / 'CACHE_MANIFEST.json'); om = man.get('observed_map', {}).get('path') or man.get('noise_file')
            obs = np.load(om).astype(np.int64) if om else None
        if obs is not None:
            ct = cand['t2i_top32'].astype(np.int64); ci = cand['i2t_top32'].astype(np.int64)
            t_in = (ct == obs[:, None]).any(1)            # record's caption query: observed image in native top-32
            i_in = (obs[ci[obs]] == obs[:, None]).any(1)   # record's image query: some top-32 caption with that observed image
            trm = np.zeros(len(obs), bool); trm[T['subset']] = True
            pos_t = np.where(ct == obs[:, None], np.arange(32)[None], 99).min(1)
            row.update(n_records=int(len(obs)), trust_and_t2i_in_top32=int((trm & t_in).sum()), trust_and_t2i_not_in_top32=int((trm & ~t_in).sum()),
                       notrust_and_t2i_in_top32=int((~trm & t_in).sum()), notrust_and_t2i_not_in_top32=int((~trm & ~t_in).sum()),
                       trust_and_i2t_in_top32=int((trm & i_in).sum()), trust_and_i2t_not_in_top32=int((trm & ~i_in).sum()),
                       notrust_and_i2t_in_top32=int((~trm & i_in).sum()), notrust_and_i2t_not_in_top32=int((~trm & ~i_in).sum()),
                       t2i_observed_rank_hist_1_2to5_6to10_11to32_absent=' '.join(str(int(x)) for x in (
                           (pos_t == 0).sum(), ((pos_t >= 1) & (pos_t < 5)).sum(), ((pos_t >= 5) & (pos_t < 10)).sum(), ((pos_t >= 10) & (pos_t < 32)).sum(), (pos_t == 99).sum())),
                       note='observed relation only (Flickr: injected-noise observed map, truth not read); trust = native GMM subset of the parent')
        sup.append(row)
    for fr in fixed:   # DEV coverage of the frozen H top-100
        pass
    other = []
    for k in ('cc_s1002', 'cc_s1003'):
        try:
            m = read_json(Path(CC[k]['cache_dir']) / 'CACHE_MANIFEST.json'); p = read_json(Path(CC[k]['prep_dir']) / 'PREP_MANIFEST.json')
            other += [dict(item=f'cache {k}', seconds=m['elapsed_seconds'], gpu=m['env']['gpu'], job=m['env']['job'], cost_scope='R38 new'),
                      dict(item=f'prep {k}', seconds=p['elapsed_seconds'], gpu=p['env']['gpu'], job=p['env']['job'], cost_scope='R38 new')]
        except FileNotFoundError:
            other.append(dict(item=f'cache/prep {k}', seconds=None, cost_scope='missing'))
    for f in ('CPU_TEST.json', 'PARENT_ARTIFACT_CHECK.json'):
        if (W / f).exists():
            x = read_json(W / f); other.append(dict(item=f, seconds=x.get('seconds'), gpu='cpu', job=x.get('job'), cost_scope='R38 new'))
    other.append(dict(item='staging of CC152K train/dev to node-local disk inside each prep job', seconds='included in job wall time (logs/r38_prep_*)', cost_scope='R38 new'))
    other.append(dict(item='parent training (R30 f60 s1001; R31 cc s1001/s1002/s1003)', seconds='existing runs, not R38 cost', cost_scope='parent (separate)'))
    wcsv('RESULTS_FIXED.csv', fixed); wcsv('CURVES.csv', curves); wcsv('DIRECTIONAL_RECALL.csv', dirrec); wcsv('RANK_TRANSITIONS.csv', rtr)
    wcsv('ARCH_X_SCHEDULE_CONTRASTS.csv', axs); wcsv('OBJECTIVE_CONTRASTS.csv', objc); wcsv('T_SCHEDULE.csv', tsch); wcsv('A_REAL_REPLICATION.csv', arep)
    wcsv('BOOTSTRAP.csv', boots); wcsv('CC_SUPPORT_CROSSTAB.csv', sup); wcsv('GRAD_DIRECTION_DIAG.csv', gdiag); wcsv('STEP0_CHECK.csv', step0)
    wcsv('NULL_READOUTS.csv', nulls); wcsv('COST_LEDGER.csv', cost + other); wcsv('POOLED_THREE_PARENT.csv', pooled); wcsv('IDENTITY_CROSSCHECK.csv', ident)
    comp = [f for f in fixed if f.get('status') == 'COMPLETE']
    new = [f for f in comp if f['source'] == 'R38 new']
    gsum = {}
    for rr in gdiag:
        gsum.setdefault((rr['parent'], rr['architecture'], rr['schedule'], rr['objective']), []).append(rr)
    gtab = [dict(parent=k[0], architecture=k[1], schedule=k[2], objective=k[3], **{f'cos_trunk_step{s}': float(np.nanmean([x['cos_trunk'] for x in v if x['step'] == s and x['cos_trunk'] is not None] or [np.nan])) for s in (1, 250, 1000, 4000)})
            for k, v in sorted(gsum.items())]
    tot_new = sum(c_['unit_wall_seconds'] for c_ in cost if c_['cost_scope'] == 'R38 new')
    md = ['# R38 结果（仅数字；开发实验，本线未读 TEST）', '',
          f'生成 {utc()}。冻结 encoder 的 critic fits；固定主终点 8000 步（曲线最大值只作次要读数）。Flickr60 R30-H1001 与 CC152K R31 父模型不同系列，只报同父模型内的差。'
          f' 复用（只读）的 shared/constant 参照：R36（Flickr s1001 N/T）、R37（Flickr s1002/s1003 N/T）、R37C（CC s1001 N/T）；CC s1002/s1003 的 N shared/constant 为 R38 A 块。54 项为用户补充（包内 B/C 格应用到全部父模型，冻结前登记、非结果驱动）。', '',
          f'未完成单元：{missing if missing else "无"}。新单元完成 {len(new)}/84（包内 30 + 用户补充 54）。', '',
          '## 父模型 H（native DEV，按各自 cache 复算）',
          tbl([dict(parent=pk, **{k: natm[pk][k] for k in R6}, rsum=natm[pk]['rsum']) for pk in PAR if pk in natm], ['parent'] + R6 + ['rsum']), '',
          '## B：结构 × 日程（N 路线；rSum 与双向 R@1；shared/constant = 复用参照）',
          tbl(axs, ['parent', 'objective', 'H_rsum', 'rsum_shared_constant', 'rsum_shared_cosine', 'rsum_directional_constant', 'rsum_directional_cosine',
                    'rsum_dir_minus_shared_constant', 'rsum_dir_minus_shared_cosine', 'rsum_cos_minus_const_shared', 'rsum_cos_minus_const_dir', 'rsum_interaction']), '',
          tbl(axs, ['parent', 'objective', 'H_i2t_r1', 'i2t_r1_shared_constant', 'i2t_r1_shared_cosine', 'i2t_r1_directional_constant', 'i2t_r1_directional_cosine',
                    'i2t_r1_dir_minus_shared_constant', 'i2t_r1_dir_minus_shared_cosine']), '',
          tbl(axs, ['parent', 'objective', 'H_t2i_r1', 't2i_r1_shared_constant', 't2i_r1_shared_cosine', 't2i_r1_directional_constant', 't2i_r1_directional_cosine',
                    't2i_r1_dir_minus_shared_constant', 't2i_r1_dir_minus_shared_cosine']), '',
          '## 三父状态合并（每数据集内；均值与正号数，不跨数据集）',
          tbl([x for x in pooled if 'rsum_dir_minus_shared_constant_mean3' in x], ['dataset', 'objective', 'rsum_dir_minus_shared_constant_mean3', 'rsum_dir_minus_shared_constant_npos',
              'rsum_dir_minus_shared_cosine_mean3', 'rsum_dir_minus_shared_cosine_npos', 'rsum_cos_minus_const_shared_mean3', 'rsum_cos_minus_const_dir_mean3',
              'i2t_r1_dir_minus_shared_constant_mean3', 't2i_r1_dir_minus_shared_constant_mean3', 'T_cos_minus_const_mean3', 'T_cos_minus_const_npos']), '',
          tbl([x for x in pooled if 'rsum_J_minus_BCE_mean3' in x], ['dataset', 'objective', 'rsum_J_minus_BCE_mean3', 'rsum_J_minus_BCE_npos', 'rsum_J_minus_NCE_mean3', 'rsum_J_minus_NCE_npos',
              'i2t_r1_J_minus_BCE_mean3', 't2i_r1_J_minus_BCE_mean3']), '',
          '## 同父模型同路线单元的初始化 / 抽样一致性（共享键 init 哈希与批索引哈希链）', tbl(ident, ['parent', 'route', 'n_units', 'n_distinct_init_shared_keys', 'n_distinct_batch_chains', 'identical']), '',
          '## 目标对比（同父模型、同路线、同结构、同日程）',
          tbl(objc, ['parent', 'route', 'architecture', 'schedule', 'rsum_J', 'rsum_BCE', 'rsum_NCE', 'rsum_J_minus_BCE', 'rsum_J_minus_NCE', 'i2t_r1_J_minus_BCE', 't2i_r1_J_minus_BCE']), '',
          '## C：T 日程压力（shared；constant = 复用参照）',
          tbl(tsch, ['parent', 'objective', 'H_rsum', 'T_constant_rsum', 'T_cosine_rsum', 'T_cos_minus_const', 'N_cosine_rsum', 'T_minus_N_cosine', 'T_constant_curve', 'T_cosine_curve']), '',
          '## A：CC152K 三个 R31 父状态（N shared constant；s1001 = R37C 复用）',
          tbl([a for a in arep if 'mean_d_rsum_3parents' in a], ['objective'] + [f'{pk}_{m}' for pk in ('cc_R31_s1001', 'cc_R31_s1002', 'cc_R31_s1003') for m in ('H_rsum', 'rsum', 'd_rsum')] + ['mean_d_rsum_3parents', 'n_parents_positive']), '',
          tbl([a for a in arep if 'J_minus_BCE' in a], ['parent', 'J_minus_BCE', 'J_minus_NCE']), '',
          '## 全部单元固定终点（8000 步）',
          tbl(comp, ['unit', 'source', 'objective'] + R6 + ['rsum', 'd_rsum', 'change_2000_to_8000', 'curve_max_rsum_secondary', 'step0_passed', 'graph_check_passed', 'reference_init_equal']), '',
          '## R@1 转移（a 对 b：仅 a 对 / 仅 b 对 / 净；top1 改变数）',
          tbl(rtr, ['parent', 'objective', 'contrast', 'i2t_a_only_r1', 'i2t_b_only_r1', 'i2t_net_r1', 'i2t_top1_changed', 't2i_a_only_r1', 't2i_b_only_r1', 't2i_net_r1', 't2i_top1_changed'], '{:.0f}'), '',
          '## 按图像分组配对 bootstrap（rSum 差，B=2000，描述性）', tbl(boots, ['parent', 'objective', 'contrast', 'diff', 'ci2_5', 'ci97_5', 'frac_boot_gt0']), '',
          '## 候选支持（训练池，观测关系；trust = 父模型原生 GMM 子集）',
          tbl(sup, ['parent', 'n_records', 'trust_retention', 'i2t_frac_images_with_an_observed_caption_in_top32', 't2i_frac_captions_observed_image_in_top32',
                    'i2t_frac_observed_partner_at_rank1', 't2i_frac_observed_partner_at_rank1', 'trust_and_t2i_in_top32', 'trust_and_t2i_not_in_top32', 'notrust_and_t2i_in_top32',
                    'notrust_and_t2i_not_in_top32', 't2i_observed_rank_hist_1_2to5_6to10_11to32_absent'], '{:.4f}'), '',
          '## 方向梯度夹角（训练批，trunk 上 i2t 与 t2i 梯度 cos，两 peer 平均；step 1 起 trunk 有梯度）', tbl(gtab, ['parent', 'architecture', 'schedule', 'objective', 'cos_trunk_step1', 'cos_trunk_step250', 'cos_trunk_step1000', 'cos_trunk_step4000'], '{:.3f}'), '',
          '## 成本', f'R38 新单元 GPU 墙钟合计 {tot_new:.0f} s（unit_wall_seconds）；cache/prep 与 CPU 检查见下。',
          tbl([c_ for c_ in cost if c_['cost_scope'] == 'R38 new'] + other, ['unit', 'item', 'gpu', 'job', 'train_seconds', 'eval_seconds', 'unit_wall_seconds', 'seconds', 'peak_gpu_gb', 'cost_scope'], '{:.1f}'), '',
          '文件：RESULTS_FIXED.csv、CURVES.csv、DIRECTIONAL_RECALL.csv、RANK_TRANSITIONS.csv、ARCH_X_SCHEDULE_CONTRASTS.csv、OBJECTIVE_CONTRASTS.csv、T_SCHEDULE.csv、A_REAL_REPLICATION.csv、BOOTSTRAP.csv、CC_SUPPORT_CROSSTAB.csv、GRAD_DIRECTION_DIAG.csv、STEP0_CHECK.csv、NULL_READOUTS.csv、COST_LEDGER.csv。']
    (W / 'RESULTS.md').write_text('\n'.join(md) + '\n', encoding='utf-8')
    print('R38_SUMMARY', len(new), 'new complete; missing', missing)


if __name__ == '__main__':
    main()
