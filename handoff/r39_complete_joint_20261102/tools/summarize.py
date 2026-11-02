"""R39 summary (numbers only; package docs/SERVER_PLAN_CN.md sections 6-7). Reads the 21 R39 RESULT.json (runs/ and oracle_runs/),
the 6 reused R38 N-FROZEN references (read-only), the R39 prep readouts (coverage, teacher-only rerank), the isolated Flickr audit,
SUBMITTED.tsv / FAILED_*.json. Writes RESULTS.md and CSVs: RESULTS_FIXED, CONTRASTS_2x2, OBJECTIVE_CONTRASTS, VS_H_TEACHER,
RESCUED_LOST, CURVES, COVERAGE, LABEL_SOURCES, STUDENT_CHANGE, BOOTSTRAP, COST_LEDGER, FAILURES. Missing units -> NA.
Usage: python3 -I -B tools/summarize.py"""
import csv, glob, json, math, sys
from pathlib import Path
import numpy as np
W = Path(__file__).resolve().parents[1]
OUT = Path('/projects/EEG-foundation-model/yinghao/outputs_v9/r39_complete_joint_20261102')
R38_RUNS = Path('/projects/EEG-foundation-model/yinghao/outputs_v9/r38_direction_20261102/runs')
DS = {'Flickr60': dict(key='f60', per=5, n_img=1014, n_cap=5070, r38='r38_B_f60_R30_s1001_N_shared_{o}_cosine'),
      'CC152K': dict(key='cc', per=1, n_img=1000, n_cap=1000, r38='r38_B_cc_R31_s1001_N_shared_{o}_cosine')}
OBJ = ('J', 'BCE', 'NCE')
CELLS = ('N_frozen', 'COMPLETE_frozen', 'N_joint', 'COMPLETE_joint', 'ORACLE_frozen')
MET = ('i2t_r1', 'i2t_r5', 'i2t_r10', 't2i_r1', 't2i_r5', 't2i_r10', 'rsum')


def rj(p):
    return json.loads(Path(p).read_text())


def unit_name(ds, cell, o):
    if cell == 'N_frozen':
        return DS[ds]['r38'].format(o=o)
    sup, enc = cell.split('_')
    return f"R39_{ds}_{'ORACLE_MISSING' if sup == 'ORACLE' else sup}_{'joint_global' if enc == 'joint' else 'frozen'}_{o}"


def unit_dir(ds, cell, o):
    u = unit_name(ds, cell, o)
    if cell == 'N_frozen':
        return R38_RUNS / u
    return (OUT / ('oracle_runs' if cell.startswith('ORACLE') else 'runs')) / u


def load():
    R = {}
    for ds in DS:
        for cell in CELLS:
            if ds == 'CC152K' and cell.startswith('ORACLE'):
                continue
            for o in OBJ:
                d = unit_dir(ds, cell, o)
                R[(ds, cell, o)] = rj(d / 'RESULT.json') if (d / 'RESULT.json').exists() else None
    return R


def f(x, nd=2):
    return 'NA' if x is None or (isinstance(x, float) and math.isnan(x)) else (f'{x:.{nd}f}' if isinstance(x, float) else str(x))


def wcsv(name, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(W / name, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rows)


def recall_vectors(rk, per):
    """per-query hit indicators for R@1/5/10 from saved two-stage ranks (ranks are 0-based)."""
    return {k: (rk['ranks_i'] < k).astype(float) for k in (1, 5, 10)}, {k: (rk['ranks_t'] < k).astype(float) for k in (1, 5, 10)}


def main():
    R = load()
    rows, curves, rl, sc, ls, cost = [], [], [], [], [], []
    teach = {ds: rj(OUT / 'prep' / DS[ds]['key'] / f"TEACHER_ONLY_RERANK_{DS[ds]['key']}.json") if (OUT / 'prep' / DS[ds]['key'] / f"TEACHER_ONLY_RERANK_{DS[ds]['key']}.json").exists() else None for ds in DS}
    for (ds, cell, o), r in R.items():
        base = dict(dataset=ds, cell=cell, objective=o, unit=unit_name(ds, cell, o), status=r['status'] if r else 'NA')
        if r is None:
            rows.append(base); continue
        m = r['final']['dev_retrieval']; nat = r['native_H_dev']
        tr = r['final'].get('transitions_vs_H', {})
        row = dict(base, **{k: m[k] for k in MET}, H_rsum=nat['rsum'], d_vs_H=m['rsum'] - nat['rsum'],
                   teacher_only_rsum=teach[ds]['dev_retrieval']['rsum'] if teach[ds] else None,
                   d_vs_teacher_only=m['rsum'] - teach[ds]['dev_retrieval']['rsum'] if teach[ds] else None,
                   tie_pess=r['final']['dev_retrieval_tie_bounded']['rsum_pessimistic'], tie_opt=r['final']['dev_retrieval_tie_bounded']['rsum_optimistic'],
                   cond_r1_i2t=r['final']['candidate_coverage']['cond_r1_i2t_covered'], cond_r1_t2i=r['final']['candidate_coverage']['cond_r1_t2i_covered'],
                   dev_list_clean_J=r['final']['dev_list_clean_J_raw'].get('J') if isinstance(r['final'].get('dev_list_clean_J_raw'), dict) else None,
                   fit_obs_label_J=(r['final'].get('fit_null_lists_observed_J_raw') or {}).get('J_total_bidirectional'),
                   fit_completed_label_J=(r['final'].get('fit_null_lists_completed_J_raw') or {}).get('J_total_bidirectional'),
                   gpu=r['env']['gpu'], job=r['env']['job'])
        rows.append(row)
        rl.append(dict(base, **{k: tr.get(k) for k in ('i2t_rescued_r1', 'i2t_lost_r1', 't2i_rescued_r1', 't2i_lost_r1', 'i2t_rescued_r5', 'i2t_lost_r5', 't2i_rescued_r5', 't2i_lost_r5',
                                                        'i2t_rescued_r10', 'i2t_lost_r10', 't2i_rescued_r10', 't2i_lost_r10')},
                       net_r1=(tr.get('i2t_rescued_r1', 0) - tr.get('i2t_lost_r1', 0) + tr.get('t2i_rescued_r1', 0) - tr.get('t2i_lost_r1', 0))))
        for ph in r['phases']:
            for c in ph['curves']:
                curves.append(dict(base, step=c['step'], rsum=c['dev_retrieval']['rsum'], **{k: c['dev_retrieval'][k] for k in MET[:-1]}))
        r39 = r.get('r39') or {}
        if cell.endswith('joint'):
            fc = r['final'].get('r39_student_feature_change', {}); pdist = r['final'].get('r39_param_distance', {})
            gsw = r39.get('grad_split_windows', [])
            sc.append(dict(base, **{f'{k}_mean_cos': v['mean_cos'] for k, v in fc.items()}, **{f'param_rel_l2_{p}': v['rel_l2_from_parent'] for p, v in pdist.items()},
                           encoder_grad_norm_mean=float(np.mean([w['encoder_norm_mean'] for w in gsw])) if gsw else None,
                           head_grad_norm_mean=float(np.mean([w['head_norm_mean'] for w in gsw])) if gsw else None,
                           frac_updates_clipped=float(np.mean([w['frac_clipped'] for w in gsw])) if gsw else None,
                           step0_equivalence_max_abs=max(v['max_abs'] for part in ('train_batch_grad', 'dev_all_nograd') for v in r39.get('step0_cache_equivalence', {}).get(part, {}).values()) if r39.get('step0_cache_equivalence') else None,
                           one_step_check=(r39.get('one_step_check') or {}).get('passed')))
        bs = r.get('batch_stats', {}).get('phase1', {})
        ls.append(dict(base, **{k: v for k, v in bs.items() if k.startswith(('complete_', 'oracle_', 'null_label', 'effective_pos'))}))
        cost.append(dict(base, train_seconds=r['cost']['train_seconds'], eval_seconds=r['cost']['eval_seconds'], params_trainable=r['cost']['params_trainable'],
                         peak_gpu_memory_GB=r['cost']['peak_gpu_memory_bytes'] / 2 ** 30, gpu=r['env']['gpu'], kind='unit (R38 reference, reused)' if cell == 'N_frozen' else 'unit'))
    wcsv('RESULTS_FIXED.csv', rows); wcsv('RESCUED_LOST.csv', rl); wcsv('CURVES.csv', curves); wcsv('STUDENT_CHANGE.csv', sc); wcsv('LABEL_SOURCES.csv', ls)
    # ---------------------------------------------------------------- 2x2 contrasts, objective contrasts
    def v(ds, cell, o, k='rsum'):
        r = R.get((ds, cell, o)); return r['final']['dev_retrieval'][k] if r else None
    def sub(a, b):
        return None if a is None or b is None else a - b
    cx, oc = [], []
    for ds in DS:
        for o in OBJ:
            for k in MET:
                nf, cf, nj, cj = (v(ds, c, o, k) for c in ('N_frozen', 'COMPLETE_frozen', 'N_joint', 'COMPLETE_joint'))
                cx.append(dict(dataset=ds, objective=o, metric=k, N_frozen=nf, COMPLETE_frozen=cf, N_joint=nj, COMPLETE_joint=cj,
                               COMPLETE_minus_N_frozen=sub(cf, nf), JOINT_minus_FROZEN_N=sub(nj, nf), JOINT_minus_FROZEN_COMPLETE=sub(cj, cf),
                               interaction=None if None in (cf, nf, nj, cj) else cj - cf - nj + nf,
                               ORACLE_frozen=v(ds, 'ORACLE_frozen', o, k) if ds == 'Flickr60' else None,
                               ORACLE_minus_N_frozen=sub(v(ds, 'ORACLE_frozen', o, k), nf) if ds == 'Flickr60' else None,
                               ORACLE_minus_COMPLETE_frozen=sub(v(ds, 'ORACLE_frozen', o, k), cf) if ds == 'Flickr60' else None))
        for cell in CELLS:
            if ds == 'CC152K' and cell.startswith('ORACLE'):
                continue
            for k in MET:
                oc.append(dict(dataset=ds, cell=cell, metric=k, J=v(ds, cell, 'J', k), BCE=v(ds, cell, 'BCE', k), NCE=v(ds, cell, 'NCE', k),
                               J_minus_BCE=sub(v(ds, cell, 'J', k), v(ds, cell, 'BCE', k)), J_minus_NCE=sub(v(ds, cell, 'J', k), v(ds, cell, 'NCE', k))))
    wcsv('CONTRASTS_2x2.csv', cx); wcsv('OBJECTIVE_CONTRASTS.csv', oc)
    vt = [dict(dataset=r['dataset'], cell=r['cell'], objective=r['objective'], rsum=r.get('rsum'), H_rsum=r.get('H_rsum'), d_vs_H=r.get('d_vs_H'),
               teacher_only_rsum=r.get('teacher_only_rsum'), d_vs_teacher_only=r.get('d_vs_teacher_only')) for r in rows]
    for ds in DS:
        if teach[ds]:
            vt.append(dict(dataset=ds, cell='TEACHER_ONLY_RERANK', objective='-', rsum=teach[ds]['dev_retrieval']['rsum'], H_rsum=teach[ds]['native_H']['rsum'],
                           d_vs_H=teach[ds]['diff_vs_H']['rsum'], teacher_only_rsum=teach[ds]['dev_retrieval']['rsum'], d_vs_teacher_only=0.0))
    wcsv('VS_H_TEACHER.csv', vt)
    # ---------------------------------------------------------------- grouped bootstrap (descriptive): image-grouped resampling of DEV queries
    rng = np.random.default_rng(20261102); bsr = []
    def ranks(ds, cell, o):
        d = unit_dir(ds, cell, o) / 'dev_ranks_step8000.npz'
        return np.load(d) if d.exists() else None
    for ds, P in DS.items():
        img_of_cap = np.arange(P['n_cap']) // P['per']
        B = 1000; idx = rng.integers(0, P['n_img'], (B, P['n_img']))
        def rs(rk):
            hi, ht = recall_vectors(rk, P['per'])
            out = np.zeros(B)
            caps_by_img = [np.nonzero(img_of_cap == i)[0] for i in range(P['n_img'])]
            for b in range(B):
                ii = idx[b]; cc = np.concatenate([caps_by_img[i] for i in ii])
                out[b] = sum(100 * hi[k][ii].mean() + 100 * ht[k][cc].mean() for k in (1, 5, 10))
            return out
        pairs = []
        for o in OBJ:
            pairs += [(('COMPLETE_frozen', o), ('N_frozen', o)), (('N_joint', o), ('N_frozen', o)), (('COMPLETE_joint', o), ('COMPLETE_frozen', o))]
            if ds == 'Flickr60':
                pairs += [(('ORACLE_frozen', o), ('N_frozen', o))]
        for cell in ('N_frozen', 'COMPLETE_frozen', 'N_joint', 'COMPLETE_joint'):
            pairs += [((cell, 'J'), (cell, 'BCE')), ((cell, 'J'), (cell, 'NCE'))]
        cache = {}
        for a, b in pairs:
            ra, rb = ranks(ds, *a), ranks(ds, *b)
            if ra is None or rb is None:
                bsr.append(dict(dataset=ds, a='_'.join(a), b='_'.join(b), mean_diff=None)); continue
            for key, rr in ((a, ra), (b, rb)):
                if key not in cache:
                    cache[key] = rs(rr)
            dd = cache[a] - cache[b]
            bsr.append(dict(dataset=ds, a='_'.join(a), b='_'.join(b), mean_diff=float(dd.mean()), q025=float(np.quantile(dd, .025)), q975=float(np.quantile(dd, .975)),
                            note='descriptive image-grouped bootstrap of DEV queries (1000 draws); one target parent, not a confirmation'))
    wcsv('BOOTSTRAP.csv', bsr)
    # ---------------------------------------------------------------- coverage / label sources from prep
    cov_rows = []
    for ds, P in DS.items():
        p = OUT / 'prep' / P['key'] / f"COVERAGE_{P['key']}.json"
        if p.exists():
            c = rj(p)
            for dn in ('i2t', 't2i'):
                for wt in ('per_query_id', 'training_record_weighted'):
                    cov_rows.append(dict(dataset=ds, direction=dn, weighting=wt, n_queries=c[dn]['n_queries'], n_missing=c[dn]['n_missing'], n_completed=c[dn]['n_completed'], **c[dn][wt]))
                cov_rows.append(dict(dataset=ds, direction=dn, weighting='sampler_stream_8000', **c[dn]['sampler_stream_8000']))
    aud = OUT / 'audit' / 'COMPLETION_AUDIT_F60.json'
    if aud.exists():
        a = rj(aud)['results']
        for dn in ('i2t', 't2i'):
            for wt in ('per_query_id', 'training_record_weighted'):
                cov_rows.append(dict(dataset='Flickr60', direction=dn, weighting=f'AUDIT_{wt} (isolated evaluator, truth = k//5)', **a[dn][wt]))
    wcsv('COVERAGE.csv', cov_rows)
    # ---------------------------------------------------------------- costs (prep / teachers), failures
    for ds, P in DS.items():
        for nm in ('COMPLETE_PREP_MANIFEST.json', 'JOINT_PREP_MANIFEST.json', 'JOINT_SMOKE.json'):
            p = OUT / 'prep' / P['key'] / nm
            if p.exists():
                m = rj(p)
                cost.append(dict(dataset=ds, cell='PREP', objective='-', unit=nm, kind='prep (one-off, shared by the cell units)',
                                 train_seconds=m.get('elapsed_seconds', m.get('wall')), gpu=(m.get('env') or {}).get('gpu', m.get('gpu'))))
                if nm == 'COMPLETE_PREP_MANIFEST.json':
                    for k, t in m['teachers'].items():
                        cost.append(dict(dataset=ds, cell='TEACHER', objective='-', unit=t['unit'], kind='auxiliary teacher: offline full-gallery scoring in prep (s); its parent training + cache encoding are existing runs, counted at their own records',
                                         train_seconds=t['seconds'], gpu=t['gpu']))
    wcsv('COST_LEDGER.csv', cost)
    fails = []
    for p in sorted(glob.glob(str(OUT / '*runs' / '*' / 'FAILED_*.json'))):
        x = rj(p); fails.append(dict(unit=x['unit'], file=p, utc=x['utc'], error=x['traceback'].strip().splitlines()[-1][:300]))
    for p in sorted(glob.glob(str(W / 'CPU_TEST_*.json'))):
        x = rj(p)
        for k, c in x['checks'].items():
            if not c['passed']:
                fails.append(dict(unit=f'CPU_TEST_{x["part"]}:{k}', file=p, utc=x['utc'], error=(c.get('error') or json.dumps(c.get('detail')))[-300:]))
    wcsv('FAILURES.csv', fails or [dict(unit='none', file='', utc='', error='')])
    # ---------------------------------------------------------------- RESULTS.md
    n_done = sum(1 for (ds, c, o), r in R.items() if r is not None and c != 'N_frozen')
    L = ['# R39 COMPLETE x JOINT results (numbers only)', '',
         f'{n_done}/21 R39 units complete; 6 R38 N-shared-cosine references reused (not resubmitted). Fixed step 8000. DEV two-stage (frozen-H top-100 reranked, rest in H order). TEST not read.', '']
    for ds in DS:
        L += [f'## {ds}: DEV rSum at step 8000', '', '| cell | J | BCE | NCE | J-BCE | J-NCE |', '|---|---|---|---|---|---|']
        for cell in CELLS:
            if ds == 'CC152K' and cell.startswith('ORACLE'):
                continue
            a, b, c = (v(ds, cell, o) for o in OBJ)
            L.append(f'| {cell} | {f(a)} | {f(b)} | {f(c)} | {f(sub(a, b))} | {f(sub(a, c))} |')
        r0 = next((r for (d_, c_, o_), r in R.items() if d_ == ds and r), None)
        L += ['', f"frozen H native DEV rSum {f(r0['native_H_dev']['rsum']) if r0 else 'NA'}; teacher-only rerank rSum {f(teach[ds]['dev_retrieval']['rsum']) if teach[ds] else 'NA'}", '',
              '| objective | COMPLETE-N (frozen) | JOINT-FROZEN (N) | JOINT-FROZEN (COMPLETE) | interaction |', '|---|---|---|---|---|']
        for o in OBJ:
            x = next(r for r in cx if r['dataset'] == ds and r['objective'] == o and r['metric'] == 'rsum')
            L.append(f"| {o} | {f(x['COMPLETE_minus_N_frozen'])} | {f(x['JOINT_minus_FROZEN_N'])} | {f(x['JOINT_minus_FROZEN_COMPLETE'])} | {f(x['interaction'])} |")
        L.append('')
    L += ['Files: RESULTS_FIXED.csv, CONTRASTS_2x2.csv (all recalls), OBJECTIVE_CONTRASTS.csv, VS_H_TEACHER.csv, RESCUED_LOST.csv, CURVES.csv, COVERAGE.csv (incl. isolated Flickr audit rows),',
          'LABEL_SOURCES.csv (training-batch label counts), STUDENT_CHANGE.csv, BOOTSTRAP.csv (descriptive), COST_LEDGER.csv, FAILURES.csv.',
          'J readouts: DEV list-J uses the clean DEV annotation; FIT observed-label J and completed-label J are different label objects (P changed) and are not ranked against each other.']
    (W / 'RESULTS.md').write_text('\n'.join(L) + '\n')
    print('summary written', n_done)


if __name__ == '__main__':
    main()
