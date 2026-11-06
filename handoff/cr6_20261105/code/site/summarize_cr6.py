"""CR6 summarizer (CPU, no training). Numbers only; DEV only; no TEST.
Tables: endpoints/curves of every CR6 unit and in-group reference; A (source) and B (student scope) contrasts; scope x source
interactions; grouped resampling of fixed-ranking paired differences (explicit image-caption groups from the evaluator-only DEV
positives; Flickr: 1 image + its captions, CC: 1 pair); source-vs-source diagnostics (calibration, candidate-Top1 agreement, row mass,
image load, exposure); costs. Plan §6 extras for CR4/CR5/CR5P PC runs: six recalls, net Top1 rescue, grouped intervals of J-X.
Intervals describe the fixed DEV set and fixed predictions only (not new-source / new-data generalisation); never used for selection.
"""
from __future__ import annotations
import csv, hashlib, json, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import cr6_common as K                                                       # noqa: E402

B_RS = 2000; SEED_RS = 20261106
R6 = K.GRAPHS6; RECALLS = ['i2t_R1', 'i2t_R5', 'i2t_R10', 't2i_R1', 't2i_R5', 't2i_R10']
CR4 = Path('/projects/EEG-foundation-model/yinghao/outputs_cr4/CR4_20261104')


def rj(p): return json.loads(Path(p).read_text())


def metrics(d, s=8000):
    p = Path(d)/f'eval_{s:06d}'/'METRICS.json'
    return rj(p) if p.exists() else None


def groups_for(cond):
    from nr1.io import FeatureBank
    dev, truth = K.dev_paths(cond); bank = FeatureBank(dev, 'dev'); P = rj(truth)['positives']
    imap = {x: i for i, x in enumerate(bank.image_ids)}; tmap = {x: i for i, x in enumerate(bank.text_ids)}
    tg = np.full(len(bank.text_ids), -1)
    for im, cap in P: tg[tmap[cap]] = imap[im]          # explicit pairs: every DEV caption has exactly one positive image in both sets
    assert (tg >= 0).all()
    return len(bank.image_ids), tg


def rsum_from_ranks(ri, rt, gi, gt_idx):
    v = [100*np.mean(ri[gi] <= k) for k in (1, 5, 10)] + [100*np.mean(rt[gt_idx] <= k) for k in (1, 5, 10)]
    return float(sum(v))


_G = {}


def grouped_ci(cond, da, db, s=8000):
    """paired rSum(a)-rSum(b), resampling image groups (i2t query = the image; t2i queries = captions of the image)."""
    pa, pb = Path(da)/f'eval_{s:06d}'/'RANKS_EVAL_ONLY.npz', Path(db)/f'eval_{s:06d}'/'RANKS_EVAL_ONLY.npz'
    if not (pa.exists() and pb.exists()): return None
    if cond not in _G:
        ni, tg = groups_for(cond); caps = [np.flatnonzero(tg == g) for g in range(ni)]; _G[cond] = (ni, caps)
    ni, caps = _G[cond]; A, B = np.load(pa), np.load(pb)
    point = rsum_from_ranks(A['i_ranks'], A['t_ranks'], np.arange(ni), np.arange(len(A['t_ranks']))) - rsum_from_ranks(B['i_ranks'], B['t_ranks'], np.arange(ni), np.arange(len(B['t_ranks'])))
    rng = np.random.default_rng(SEED_RS); out = np.empty(B_RS)
    for b in range(B_RS):
        g = rng.integers(0, ni, ni); t = np.concatenate([caps[x] for x in g])
        out[b] = rsum_from_ranks(A['i_ranks'], A['t_ranks'], g, t) - rsum_from_ranks(B['i_ranks'], B['t_ranks'], g, t)
    return dict(diff=point, lo95=float(np.quantile(out, .025)), hi95=float(np.quantile(out, .975)), resamples=B_RS, groups=ni)


def unit_rows():
    fz = rj(K.FROZEN); rows, curves, dirs = [], [], {}
    for u, i in fz['units'].items():
        if i['stage'] == 'A1_ESTIMATOR': continue
        d = K.OUT/'students'/i['condition']/u; dirs[u] = d
        for s in K.SAVES:
            m = metrics(d, s)
            if m: curves.append(dict(unit=u, step=s, rSum=m['metrics']['rSum'], student_cos_full=m['student_cosine_full']['rSum'], g_full=m['g_full_gallery']['rSum']))
        m = metrics(d); rows.append(dict(unit=u, kind=i['stage'], group=i['group'], condition=i['condition'], relation=i['relation'], scope=i['scope'], source_set=i['source_set'],
                                        student_seed=i['head_seed'], rSum=m['metrics']['rSum'] if m else None, **({k: m['metrics'][k] for k in RECALLS} if m else {}),
                                        net_top1_rescue_i2t=(m['i2t']['rescued_top1']-m['i2t']['lost_top1']) if m else None,
                                        net_top1_rescue_t2i=(m['t2i']['rescued_top1']-m['t2i']['lost_top1']) if m else None,
                                        student_cos_full=m['student_cosine_full']['rSum'] if m else None, g_full=m['g_full_gallery']['rSum'] if m else None))
    for rid, i in fz['references'].items():
        for gname in i['groups']:
            g = K.OUT/'groups'/gname/'refs'/rid
            if not (g/'UNIT_COMPLETE.json').exists(): continue
            uc = rj(g/'UNIT_COMPLETE.json'); d = g/'eval'; dirs[(rid, gname)] = d
            for s in K.SAVES:
                m = metrics(d, s)
                if m: curves.append(dict(unit=f'{rid}@{gname}', step=s, rSum=m['metrics']['rSum'], student_cos_full=m['student_cosine_full']['rSum'], g_full=m['g_full_gallery']['rSum']))
            m = metrics(d); rows.append(dict(unit=rid, kind='REFERENCE', group=gname, condition=i['condition'], relation=i['relation'], scope='FROZEN', source_set='ORIGINAL_CR4',
                                            student_seed=i['head_seed'], reference_status=uc['status'], rSum=m['metrics']['rSum'] if m else None,
                                            **({k: m['metrics'][k] for k in RECALLS} if m else {}),
                                            net_top1_rescue_i2t=(m['i2t']['rescued_top1']-m['i2t']['lost_top1']) if m else None,
                                            net_top1_rescue_t2i=(m['t2i']['rescued_top1']-m['t2i']['lost_top1']) if m else None,
                                            student_cos_full=m['student_cosine_full']['rSum'] if m else None, g_full=m['g_full_gallery']['rSum'] if m else None))
    return fz, rows, curves, dirs


def write_csv(name, rows):
    if not rows: return
    k = []; [k.append(x) for r in rows for x in r if x not in k]
    with open(K.TREE/name, 'w', newline='') as f: w = csv.DictWriter(f, k); w.writeheader(); w.writerows(rows)


def source_diagnostics():
    """old (CR1 s1001) vs new (src2002) Flickr60 sources: label-free."""
    out = {}
    try:
        for S in ('J', 'BCE'):
            cal = {e: dict(old=rj(K.CR1/'Flickr60'/'experts'/S/e/'DONE.json')['calibration'], new=rj(K.OUT/'A1'/'experts'/S/e/'DONE.json')['calibration']) for e in K.EXPERTS}
            zo, zn = np.load(K.CR1/'Flickr60'/'scores'/f'{S}.npz'), np.load(K.OUT/'A1'/'scores'/f'{S}.npz')
            assert np.array_equal(zo['candidates'], zn['candidates'])
            so, sn = zo['scores'], zn['scores']
            ro, rn = np.load(K.CR1/'Flickr60'/'relations'/f'{S}_CAP.npz'), np.load(K.OUT/'A1'/'relations'/f'{S}_CAP.npz')
            mo, mn = ro['weights'].sum(1), rn['weights'].sum(1)
            lo = np.bincount(ro['candidates'].ravel(), ro['weights'].ravel(), minlength=29000); ln = np.bincount(rn['candidates'].ravel(), rn['weights'].ravel(), minlength=29000)
            out[S] = dict(calibration={e: dict(a_old=v['old']['a'], b_old=v['old']['b'], a_new=v['new']['a'], b_new=v['new']['b']) for e, v in cal.items()},
                          candidate_top1_agreement=float((so.argmax(1) == sn.argmax(1)).mean()),
                          score_pearson=float(np.corrcoef(so.ravel(), sn.ravel())[0, 1]),
                          within_row_spearman_mean=float(np.mean([np.corrcoef(np.argsort(np.argsort(a)), np.argsort(np.argsort(b)))[0, 1] for a, b in zip(so[::50], sn[::50])])),
                          cap_top1_agreement=float((ro['weights'].argmax(1) == rn['weights'].argmax(1)).mean()),
                          row_mass=dict(old_mean=float(mo.mean()), new_mean=float(mn.mean()), mean_abs_diff=float(np.abs(mo-mn).mean()), corr=float(np.corrcoef(mo, mn)[0, 1])),
                          image_load=dict(old_q=np.quantile(lo, [.5, .9, .99, 1]).tolist(), new_q=np.quantile(ln, [.5, .9, .99, 1]).tolist(), corr=float(np.corrcoef(lo, ln)[0, 1])))
        zj, zb = np.load(K.OUT/'A1'/'scores'/'J.npz'), np.load(K.OUT/'A1'/'scores'/'BCE.npz')
        zjo, zbo = np.load(K.CR1/'Flickr60'/'scores'/'J.npz'), np.load(K.CR1/'Flickr60'/'scores'/'BCE.npz')
        out['J_vs_BCE_candidate_top1_agreement'] = dict(new=float((zj['scores'].argmax(1) == zb['scores'].argmax(1)).mean()), old=float((zjo['scores'].argmax(1) == zbo['scores'].argmax(1)).mean()))
        from scipy.sparse import load_npz
        fz = rj(K.FROZEN); cr5 = rj(K.CR5_TREE/'PROTOCOL_FROZEN_CR5.json')['units']
        for r in ('J_KEEP_TOP1', 'BCE_KEEP_TOP1', 'J_MASS_OBS'):
            go = load_npz(cr5[f'CR5_Flickr60_{r}_COMPLETE_PC_s1002']['graph_path']).tocsr(); gn = load_npz(K.OUT/'A1'/'graphs'/f'{r}_src2002.npz').tocsr()
            mo, mn = np.asarray(go.sum(1)).ravel(), np.asarray(gn.sum(1)).ravel(); co, cn = np.asarray(go.sum(0)).ravel(), np.asarray(gn.sum(0)).ravel()
            same_edges = (go != 0).multiply(gn != 0).nnz
            out[f'graph_{r}'] = dict(nnz_old=int(go.nnz), nnz_new=int(gn.nnz), shared_edges=int(same_edges), row_mass_mean_old=float(mo.mean()), row_mass_mean_new=float(mn.mean()),
                                     image_load_max_old=float(co.max()), image_load_max_new=float(cn.max()), image_load_corr=float(np.corrcoef(co, cn)[0, 1]),
                                     images_with_mass_old=int((co > 0).sum()), images_with_mass_new=int((cn > 0).sum()),
                                     sha_new=K.sha256(K.OUT/'A1'/'graphs'/f'{r}_src2002.npz'), differs_from_old=K.sha256(K.OUT/'A1'/'graphs'/f'{r}_src2002.npz') != K.sha256(cr5[f'CR5_Flickr60_{r}_COMPLETE_PC_s1002']['graph_path']))
    except FileNotFoundError as e:
        out['incomplete'] = str(e)
    return out


def five_run_extras():
    """CR4 s1001 / CR5 s1002,s1003 / CR5P t2002,t2003 PC runs: six recalls, net top1 rescue, grouped interval of J - X."""
    fz4 = rj(K.CR4_TREE/'PROTOCOL_FROZEN_CR4.json')
    def cdir(c, r, s):
        d = {1001: CR4/'students'/c/f'CR4_{c}_{r}_COMPLETE_PC_s1001', 1002: K.CR5/'students'/c/f'CR5_{c}_{r}_COMPLETE_PC_s1002', 1003: K.CR5/'students'/c/f'CR5_{c}_{r}_COMPLETE_PC_s1003',
             2002: K.CR5P/'students'/c/f'CR5P1105_{c}_{r}_COMPLETE_PC_t2002', 2003: K.CR5P/'students'/c/f'CR5P1105_{c}_{r}_COMPLETE_PC_t2003'}[s]
        uc = d/'UNIT_COMPLETE.json'
        return Path(rj(uc)['reused_dir']) if uc.exists() and rj(uc).get('reused_dir') else d
    rec, diffs = [], []
    for c in K.CONDS:
        for s in (1001, 1002, 1003, 2002, 2003):
            for r in R6:
                m = metrics(cdir(c, r, s))
                if not m: continue
                rec.append(dict(condition=c, student_seed=s, relation=r, rSum=m['metrics']['rSum'], **{k: m['metrics'][k] for k in RECALLS},
                                net_top1_rescue_i2t=m['i2t']['rescued']-m['i2t']['lost'], net_top1_rescue_t2i=m['t2i']['rescued']-m['t2i']['lost']))
            for x in ('OBS', 'J_MASS_OBS', 'BASE_KEEP_TOP1', 'BCE_KEEP_TOP1', 'NCE_KEEP_TOP1'):
                ci = grouped_ci(c, cdir(c, 'J_KEEP_TOP1', s), cdir(c, x, s))
                if ci: diffs.append(dict(condition=c, student_seed=s, contrast=f'J_KEEP_TOP1-{x}', **ci))
    return rec, diffs


def main():
    fz, rows, curves, dirs = unit_rows()
    write_csv('TASK_ENDPOINTS_CR6.csv', rows); write_csv('TASK_CURVES_CR6.csv', curves)
    get = lambda **kw: next((r for r in rows if all(r.get(k) == v for k, v in kw.items())), None)
    A, Bt, inter, cis = [], [], [], []
    for s in (1002, 1003):   # A: source effect inside the A2 group (same GPU), same student seed
        gname = f'A2_Flickr60_t{s}'
        new = {r: get(kind='A2_SOURCE_STUDENT', relation=r, student_seed=s) for r in ('J_KEEP_TOP1', 'BCE_KEEP_TOP1', 'J_MASS_OBS')}
        old = {r: get(kind='REFERENCE', relation=r, group=gname) for r in R6}
        val = lambda d, r: (d.get(r) or {}).get('rSum')
        for lab, a, b in [('J-BCE', 'J_KEEP_TOP1', 'BCE_KEEP_TOP1'), ('J-J_MASS_OBS', 'J_KEEP_TOP1', 'J_MASS_OBS')]:
            vn = (val(new, a)-val(new, b)) if val(new, a) is not None and val(new, b) is not None else None
            vo = (val(old, a)-val(old, b)) if val(old, a) is not None and val(old, b) is not None else None
            A.append(dict(student_seed=s, contrast=lab, new_source=vn, old_source=vo, new_minus_old=(vn-vo) if vn is not None and vo is not None else None))
        for x in ('OBS', 'BASE_KEEP_TOP1', 'NCE_KEEP_TOP1'):   # J_new - X_old (X has no new source); both in the same group
            vn = (val(new, 'J_KEEP_TOP1')-val(old, x)) if val(new, 'J_KEEP_TOP1') is not None and val(old, x) is not None else None
            vo = (val(old, 'J_KEEP_TOP1')-val(old, x)) if val(old, 'J_KEEP_TOP1') is not None and val(old, x) is not None else None
            A.append(dict(student_seed=s, contrast=f'J-{x}', new_source=vn, old_source=vo, new_minus_old=(vn-vo) if vn is not None and vo is not None else None))
        for r in ('J_KEEP_TOP1', 'BCE_KEEP_TOP1', 'J_MASS_OBS'):
            A.append(dict(student_seed=s, contrast=f'{r} new-old source', new_source=val(new, r), old_source=val(old, r),
                          new_minus_old=(val(new, r)-val(old, r)) if val(new, r) is not None and val(old, r) is not None else None))
            u = f'CR6_A2_Flickr60_{r}_src2002_t{s}'; rid = f'CR6_REF_Flickr60_{r}_PC_t{s}'
            if u in dirs and (rid, gname) in dirs:
                ci = grouped_ci('Flickr60', dirs[u], dirs[(rid, gname)])
                if ci: cis.append(dict(contrast=f'{r} src2002 - src1001', student_seed=s, **ci))
    for c in K.CONDS:   # B: scope effect and scope x source interaction (same B group)
        gname = f'B_{c}_t1002'
        tail = {r: get(kind='B_STUDENT_SCOPE', condition=c, relation=r) for r in R6}; froz = {r: get(kind='REFERENCE', group=gname, relation=r) for r in R6}
        v = lambda d, r: (d.get(r) or {}).get('rSum')
        for r in R6:
            Bt.append(dict(condition=c, relation=r, TAIL=v(tail, r), FROZEN=v(froz, r), D_adapt=(v(tail, r)-v(froz, r)) if v(tail, r) is not None and v(froz, r) is not None else None,
                           TAIL_student_cos_full=(tail.get(r) or {}).get('student_cos_full'), FROZEN_student_cos_full=(froz.get(r) or {}).get('student_cos_full')))
            u = f'CR6_B_{c}_{r}_TAIL_t1002'; rid = f'CR6_REF_{c}_{r}_PC_t1002'
            if u in dirs and (rid, gname) in dirs:
                ci = grouped_ci(c, dirs[u], dirs[(rid, gname)])
                if ci: cis.append(dict(contrast=f'{c} {r} TAIL-FROZEN', student_seed=1002, **ci))
        for x in ('OBS', 'J_MASS_OBS', 'BASE_KEEP_TOP1', 'BCE_KEEP_TOP1', 'NCE_KEEP_TOP1'):
            dt = (v(tail, 'J_KEEP_TOP1')-v(tail, x)) if v(tail, 'J_KEEP_TOP1') is not None and v(tail, x) is not None else None
            df = (v(froz, 'J_KEEP_TOP1')-v(froz, x)) if v(froz, 'J_KEEP_TOP1') is not None and v(froz, x) is not None else None
            inter.append(dict(condition=c, contrast=f'J-{x}', D_source_TAIL=dt, D_source_FROZEN=df, interaction_I=(dt-df) if dt is not None and df is not None else None))
    write_csv('A_SOURCE_CONTRASTS.csv', A); write_csv('B_SCOPE_EFFECTS.csv', Bt); write_csv('B_SCOPE_X_SOURCE_INTERACTION.csv', inter); write_csv('GROUPED_RESAMPLING_CR6.csv', cis)
    sd = source_diagnostics(); K.write_json(K.TREE/'SOURCE_VS_SOURCE_DIAGNOSTICS.json', sd)
    rec5, diff5 = five_run_extras(); write_csv('CR4_CR5_CR5P_PC_SIX_RECALLS.csv', rec5); write_csv('CR4_CR5_CR5P_PC_GROUPED_J_DIFFS.csv', diff5)
    # costs
    cost = []
    for p in sorted((K.OUT/'A1'/'experts').glob('*/*/DONE.json')):
        d = rj(p); cost.append(dict(item=d['unit'], kind='source_train', gpu=d.get('gpu'), train_seconds=d['synchronized_train_seconds_this_attempt'], calibration_seconds=d['calibration_seconds']))
    for p in sorted((K.OUT/'A1'/'scores').glob('*_COST.json')): cost.append(dict(item=p.name, kind='score_export', gpu=rj(p)['env']['gpu_name'], wall_seconds=rj(p)['wall_seconds']))
    if (K.OUT/'A1'/'GRAPH_REGISTRY_SRC2002.json').exists(): cost.append(dict(item='CAP+KEEP_TOP1+J_MASS_OBS', kind='graph_build_cpu', **rj(K.OUT/'A1'/'GRAPH_REGISTRY_SRC2002.json')['cpu_seconds_this_run']))
    for p in sorted((K.OUT/'students').glob('*/*/DONE.json')):
        d = rj(p); ev = [rj(q)['cost'] for q in sorted(p.parent.glob('eval_*/METRICS.json'))]
        cost.append(dict(item=d['unit'], kind='encoder_adapt+head' if d.get('scope') == 'TAIL_LORA8' else 'student_head', gpu=d.get('gpu'), train_seconds=d['cost']['synchronized_train_seconds'],
                         eval_seconds=sum(e['feature_seconds']+e['score_seconds'] for e in ev), peak_gpu_memory_bytes=d['cost'].get('peak_gpu_memory_bytes'),
                         mean_unique_images=d['cost'].get('mean_unique_images'), mean_unique_texts=d['cost'].get('mean_unique_texts')))
    for p in sorted((K.OUT/'groups').glob('*/refs/*/train/DONE.json')):
        d = rj(p); cost.append(dict(item=d['unit']+'@'+p.parts[-4], kind='reference_rerun', gpu=d.get('gpu'), train_seconds=d['cost']['synchronized_train_seconds']))
    for p in sorted((K.OUT/'groups').glob('B_*/TAIL_ACCEPTANCE_IN_JOB.json')): cost.append(dict(item=p.parts[-2], kind='hidden_cache_load', seconds=rj(p).get('hidden_load_seconds')))
    write_csv('COSTS_CR6.csv', cost)
    genv = {p.parent.name: {k: rj(p)[k] for k in ('gpu_name', 'driver')} for p in sorted((K.OUT/'groups').glob('*/GROUP_ENV.json'))}
    if (K.OUT/'A1'/'GROUP_ENV.json').exists(): genv['A1'] = {k: rj(K.OUT/'A1'/'GROUP_ENV.json')[k] for k in ('gpu_name', 'driver')}
    f2 = lambda x: '' if x is None else f'{x:.2f}'; s2 = lambda x: '' if x is None else f'{x:+.2f}'
    L = ['# CR6 results (numbers only; DEV, fixed step 8000; no TEST)', '', f"Protocol {K.PROTOCOL_ID}; frozen sha {K.sha256(K.FROZEN)}", '',
         '## Group GPUs', '', *[f'- {k}: {v}' for k, v in genv.items()], '',
         '## A: source replicate (Flickr60; new = src2002 graphs, old = CR4 graphs re-verified/re-run in the same group)', '', '| student | contrast | new source | old source | new−old |', '|---|---|---|---|---|']
    L += [f"| {a['student_seed']} | {a['contrast']} | {s2(a['new_source']) if 'source' not in a['contrast'] else f2(a['new_source'])} | {s2(a['old_source']) if 'source' not in a['contrast'] else f2(a['old_source'])} | {s2(a['new_minus_old'])} |" for a in A]
    L += ['', '## B: scope effect D_adapt = TAIL − FROZEN (same graph, same group)', '', '| condition | graph | TAIL | FROZEN | D_adapt | TAIL student-cos full | FROZEN student-cos full |', '|---|---|---|---|---|---|---|']
    L += [f"| {b['condition']} | {b['relation']} | {f2(b['TAIL'])} | {f2(b['FROZEN'])} | {s2(b['D_adapt'])} | {f2(b['TAIL_student_cos_full'])} | {f2(b['FROZEN_student_cos_full'])} |" for b in Bt]
    L += ['', '## B: scope × source interaction I = D_source(TAIL) − D_source(FROZEN)', '', '| condition | contrast | D_source TAIL | D_source FROZEN | I |', '|---|---|---|---|---|']
    L += [f"| {i['condition']} | {i['contrast']} | {s2(i['D_source_TAIL'])} | {s2(i['D_source_FROZEN'])} | {s2(i['interaction_I'])} |" for i in inter]
    L += ['', '## Grouped resampling (image groups; fixed predictions; descriptive)', '', '| contrast | student | diff | 95% interval |', '|---|---|---|---|']
    L += [f"| {c['contrast']} | {c['student_seed']} | {s2(c['diff'])} | [{c['lo95']:+.2f}, {c['hi95']:+.2f}] |" for c in cis]
    L += ['', 'Files: TASK_ENDPOINTS_CR6.csv, TASK_CURVES_CR6.csv, A_SOURCE_CONTRASTS.csv, B_SCOPE_EFFECTS.csv, B_SCOPE_X_SOURCE_INTERACTION.csv, GROUPED_RESAMPLING_CR6.csv,',
          'SOURCE_VS_SOURCE_DIAGNOSTICS.json, CR4_CR5_CR5P_PC_SIX_RECALLS.csv, CR4_CR5_CR5P_PC_GROUPED_J_DIFFS.csv, COSTS_CR6.csv.',
          'Source and student layers are reported separately; 2 sources x 2 students are not four independent replicates. Intervals: fixed DEV/predictions only.']
    (K.TREE/'RESULTS_CR6.md').write_text('\n'.join(L)+'\n'); print('\n'.join(L))


if __name__ == '__main__': main()
