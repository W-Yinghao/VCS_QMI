"""R50 summary (numbers only; every row carries protocol_id). PRIMARY = FIT_FIXED_CENTER_V1 (R50 cache / units / re-reads); SECONDARY_TRANSDUCTIVE = the R47 G /
R48 original split-global readout (reported in a separate column, never mixed or picked per cell). H = the same-protocol parent H (unit native_H_dev = R50
cache native_dev). N reference = the R48 N-frozen / R47 G N-JOINT weights re-read under PRIMARY (r50_rereads; reuse receipt REUSE_RECEIPT_R50.json).
Contrasts per parent x objective (step 8000): PRESENT-N | frozen, PRESENT-N | joint, interaction (PJ-PF)-(NJ-NF); ORACLE-N | frozen, | joint, interaction;
per cell J-BCE, J-NCE. Curves (0/250/.../8000), DEV top-100 coverage + covered-query conditional R@1, FIT natural top-32 coverage, p_d, oracle exposure
(full-pool diagnostics of the oracle process + in-training relabel counters), cost. No TEST read. Missing values = NA.
Writes RESULTS_R50.md, R50_RESULTS.csv, R50_CONTRASTS.csv, R50_CURVES.csv, R50_REFERENCES.csv, R50_H.csv, R50_ORACLE_EXPOSURE.csv.
Usage: python3 -I -B tools/summarize_r50.py"""
import csv, glob, json, os
from pathlib import Path
W = Path(__file__).resolve().parents[1]
OUT = Path('/projects/EEG-foundation-model/yinghao/outputs_v9/r40_scope_present_20261103'); R50OUT = Path('/home/infres/yinwang/CS_QMI/outputs_v9_r50')
PID = 'FIT_FIXED_CENTER_V1'; SEC = 'SECONDARY_TRANSDUCTIVE'
OBJ = ('J', 'BCE', 'NCE'); STEPS = (0, 250, 500, 1000, 2000, 4000, 8000)
REC = ('i2t_r1', 'i2t_r5', 'i2t_r10', 't2i_r1', 't2i_r5', 't2i_r10', 'rsum')
PAR = {'F60_R30H1001': ('F60_R30_H1001', 'n60', 'Flickr60 R30-H s1001'), 'CC_R31H1001': ('CC31_H1001', 'cc', 'CC152K R31-H0 s1001'),
       'F80_IN2RH1001': ('F80_IN2R_H1001', 'f80in', 'Flickr80 IN2R H s1001')}
rj = lambda p: json.loads(Path(p).read_text()) if Path(p).exists() else None
f2 = lambda x: 'NA' if x is None else f'{x:.2f}'
sub = lambda a, b: None if a is None or b is None else a - b
mat = list(csv.DictReader(open('/home/infres/yinwang/CS_QMI/r50_layout_audit/VCS_Layout_Audit_R50_20261104/MATRIX.csv')))


def unit_dir(u, oracle):
    return (R50OUT / ('oracle_runs_r50' if oracle else 'runs_r50') / u)


def ev(d, s):
    e = rj(Path(d) / f'eval_step{s}.json'); return e


rows, curves, refrows = [], [], []
ref_cache = {}


def ref(pk, enc, o):
    u = f'R48_{pk}_N_frozen_lfG_{o}' if enc == 'frozen' else f'R47_G_{pk}_N_joint_{o}'
    if u not in ref_cache:
        rr = rj(R50OUT / 'r50_rereads' / u / 'REREAD.json'); src = rj(OUT / ('runs_r48' if enc == 'frozen' else 'runs_r47') / u / 'RESULT.json')
        ref_cache[u] = dict(unit=u, reread=rr, src=src)
    return ref_cache[u]


for r in mat:
    pk, base, lab = PAR[r['parent']]; o = r['objective']; enc = r['encoder']; oracle = r['oracle'] == 'True'; u = r['unit']
    d = unit_dir(u, oracle); res = rj(d / 'RESULT.json'); done = (d / 'DONE.json').exists()
    fin = (res or {}).get('final', {}).get('dev_retrieval') if res else None
    H = (res or {}).get('native_H_dev')
    R = ref(pk, enc, o); nref = ((R['reread'] or {}).get('per_step') or {}).get('8000', {}).get('primary')
    if H is None and R['reread']:
        H = R['reread']['H_primary']
    row = dict(protocol_id=PID, unit=u, block=r['block'], parent=lab, supervision=r['supervision'], encoder=enc, objective=o, oracle=oracle, status='DONE' if done else 'NA')
    for k in REC:
        row[k] = (fin or {}).get(k)
    row['H_rsum'] = (H or {}).get('rsum'); row['N_ref_unit'] = R['unit']; row['N_ref_rsum'] = (nref or {}).get('rsum')
    row['minus_H'] = sub(row['rsum'], row['H_rsum']); row['minus_N'] = sub(row['rsum'], row['N_ref_rsum'])
    e8 = ev(d, 8000) if done else None
    cov = (e8 or {}).get('candidate_coverage') or {}
    row.update(dev_cov_i2t=cov.get('coverage_i2t'), dev_cov_t2i=cov.get('coverage_t2i'), cond_r1_i2t_covered=cov.get('cond_r1_i2t_covered'), cond_r1_t2i_covered=cov.get('cond_r1_t2i_covered'))
    tr = (e8 or {}).get('transitions_vs_H') or {}
    row.update(i2t_rescued_r1=tr.get('i2t_rescued_r1'), i2t_lost_r1=tr.get('i2t_lost_r1'), t2i_rescued_r1=tr.get('t2i_rescued_r1'), t2i_lost_r1=tr.get('t2i_lost_r1'))
    pp = (res or {}).get('present_probs') or {}
    row.update(p_i2t=pp.get('p_i2t'), p_t2i=pp.get('p_t2i'))
    st = ((res or {}).get('batch_stats') or {}).get('phase1', {})
    if oracle:
        row.update(oracle_rows_i2t=st.get('oracle_i2t_rows'), oracle_relabelled_i2t=st.get('oracle_i2t_relabelled_rows'), oracle_rows_t2i=st.get('oracle_t2i_rows'),
                   oracle_relabelled_t2i=st.get('oracle_t2i_relabelled_rows'), oracle_true_slots_used=(st.get('oracle_i2t_true_slots_used') or 0) + (st.get('oracle_t2i_true_slots_used') or 0) if st else None)
    if present := (r['supervision'] == 'PRESENT_ONLY'):
        row.update(valid_rows_i2t=st.get('valid_rows_i2t'), valid_rows_t2i=st.get('valid_rows_t2i'))
    c = (res or {}).get('cost') or {}
    row.update(train_seconds=c.get('train_seconds'), eval_seconds=c.get('eval_seconds'), peak_gpu_gb=(c.get('peak_gpu_memory_bytes') or 0) / 2 ** 30 if c else None, gpu=(res or {}).get('env', {}).get('gpu'))
    rows.append(row)
    for s in STEPS:
        e = ev(d, s)
        curves.append(dict(protocol_id=PID, unit=u, step=s, rsum=(e or {}).get('dev_retrieval', {}).get('rsum')))
for u, R in ref_cache.items():
    rr = R['reread'] or {}; ps = rr.get('per_step') or {}
    row = dict(protocol_id=PID, unit=u, status='REREAD' if rr else 'NA', H_rsum=(rr.get('H_primary') or {}).get('rsum'), H_secondary_rsum=(rr.get('H_secondary_transductive') or {}).get('rsum'))
    p8, s8 = (ps.get('8000') or {}).get('primary') or {}, (ps.get('8000') or {}).get('secondary_transductive') or {}
    for k in REC:
        row[k] = p8.get(k); row[f'secondary_{k}'] = s8.get(k)
    row['minus_H'] = sub(row['rsum'], row['H_rsum']); row['secondary_minus_H'] = sub(row['secondary_rsum'], row['H_secondary_rsum'])
    refrows.append(row)
    for s in STEPS:
        x = ps.get(str(s)) or {}
        curves.append(dict(protocol_id=PID, unit=u, step=s, rsum=(x.get('primary') or {}).get('rsum'), secondary_rsum=(x.get('secondary_transductive') or {}).get('rsum')))
# H table (5 parents)
hrows = []
import subprocess, sys
HCODE = r'''
import json, sys, numpy as np
sys.path.insert(0, sys.argv[1]); import evals as EV
try:
    from numpy._core._multiarray_umath import __cpu_features__ as F
except ImportError:
    from numpy.core._multiarray_umath import __cpu_features__ as F
assert not F.get('AVX512F') and not F.get('AVX512_SKX'), 'run with the R50 amendment-A2 tie policy (NPY_DISABLE_CPU_FEATURES)'
out = {}
for k, p in (('primary', sys.argv[2]), ('secondary', sys.argv[3])):
    m = EV.full_metrics(np.load(p))[0]; out[k] = {x: float(m[x]) for x in ('i2t_r1', 'i2t_r5', 'i2t_r10', 't2i_r1', 't2i_r5', 't2i_r10', 'rsum')}
print(json.dumps(out))
'''
A2ENV = dict(os.environ, NPY_DISABLE_CPU_FEATURES='AVX512F AVX512CD AVX512_SKX AVX512_CLX AVX512_CNL AVX512_ICL AVX512_SPR')
for b, lab, tree in (('n60', 'Flickr60 R30-H s1001', 'r50f'), ('cc', 'CC152K R31-H0 s1001', 'r50c'), ('f80in', 'Flickr80 IN2R H s1001', 'r50f'), ('n00', 'Flickr0 R30-H s1001', 'r50f'),
                     ('c60in_s1001', 'COCO60 IN2R H s1001', 'r47o')):
    cm = rj(OUT / 'r50_parents/cache' / b / 'CACHE_MANIFEST.json'); nd = (cm or {}).get('native_dev') or {}
    pr = subprocess.run([sys.executable, '-I', '-B', '-c', HCODE, str(W / tree), str(OUT / 'r50_parents/cache' / b / 'native_dev_sims.npy'), str(OUT / 'r47_parents/cache' / b / 'native_dev_sims.npy')],
                        capture_output=True, text=True, env=A2ENV, cwd='/tmp')
    hm = json.loads(pr.stdout.strip().splitlines()[-1]) if pr.returncode == 0 else dict(primary={}, secondary={})
    hrows.append(dict(protocol_id=PID, base=b, parent=lab, **{k: hm['primary'].get(k) for k in REC}, secondary_rsum=hm['secondary'].get('rsum'), native_layout_rsum=nd.get('native_layout_rsum'),
                      build_record_rsum_build_node_dispatch=nd.get('rsum'), tie_policy='A2 (numpy without AVX-512)'))
# contrasts
by = {(r['parent'], r['supervision'], r['encoder'], r['objective']): r['rsum'] for r in rows}
nref = {}
for r in rows:
    nref[(r['parent'], r['encoder'], r['objective'])] = r['N_ref_rsum']
con = []
for lab in sorted({r['parent'] for r in rows}):
    for sup in sorted({r['supervision'] for r in rows if r['parent'] == lab}):
        for o in OBJ:
            xf, xj = by.get((lab, sup, 'frozen', o)), by.get((lab, sup, 'both_joint', o))
            nf, nj = nref.get((lab, 'frozen', o)), nref.get((lab, 'both_joint', o))
            con.append(dict(protocol_id=PID, parent=lab, supervision=sup, objective=o, minus_N_frozen=sub(xf, nf), minus_N_joint=sub(xj, nj),
                            interaction=sub(sub(xj, xf), sub(nj, nf)), joint_minus_frozen=sub(xj, xf), N_joint_minus_frozen=sub(nj, nf)))
obj_con = []
for lab in sorted({r['parent'] for r in rows}):
    for sup in sorted({r['supervision'] for r in rows if r['parent'] == lab}) + ['N_reference']:
        for enc in ('frozen', 'both_joint'):
            g = (lambda o: nref.get((lab, enc, o))) if sup == 'N_reference' else (lambda o: by.get((lab, sup, enc, o)))
            obj_con.append(dict(protocol_id=PID, parent=lab, supervision=sup, encoder=enc, J_minus_BCE=sub(g('J'), g('BCE')), J_minus_NCE=sub(g('J'), g('NCE'))))
# oracle exposure (full pool, oracle process diagnostics)
ox = []
for f in sorted(glob.glob(str(R50OUT / 'oracle_runs_r50' / 'ORACLE_DIAGNOSTICS_R50_*.json'))):
    x = rj(f)
    for cond, v in x['conds'].items():
        ox.append(dict(protocol_id=PID, file=Path(f).name, cond=cond, observed_mismatch=v['observed_mismatch_fraction_full_pool'], true_in_top32_i2t=v['true_partner_in_top32']['i2t'],
                       true_in_top32_t2i=v['true_partner_in_top32']['t2i'], observed_in_top32_i2t=v['observed_partner_in_top32']['i2t'], observed_in_top32_t2i=v['observed_partner_in_top32']['t2i'],
                       relabelled_rows_i2t=v['relabelled_rows']['i2t'], relabelled_rows_t2i=v['relabelled_rows']['t2i'],
                       relabelled_of_null_i2t=v['relabelled_fraction_of_null_rows']['i2t'], relabelled_of_null_t2i=v['relabelled_fraction_of_null_rows']['t2i'],
                       added_true_mass_per_row_i2t=v['expected_added_true_mass_per_row']['i2t'], added_true_mass_per_row_t2i=v['expected_added_true_mass_per_row']['t2i']))
seen = set(); ox = [x for x in ox if not (x['cond'] in seen or seen.add(x['cond']))]


def wcsv(name, rs):
    if not rs:
        return
    keys = list(dict.fromkeys(k for r in rs for k in r))
    with open(W / name, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rs)


wcsv('R50_RESULTS.csv', rows); wcsv('R50_CONTRASTS.csv', con + obj_con); wcsv('R50_CURVES.csv', curves); wcsv('R50_REFERENCES.csv', refrows); wcsv('R50_H.csv', hrows); wcsv('R50_ORACLE_EXPOSURE.csv', ox)
L = ['# R50 results (numbers only)', '', f'protocol_id = {PID} on every row (PRIMARY). SECONDARY_TRANSDUCTIVE = R47 G / R48 original split-global readout, separate columns only. '
     'DEV only; no TEST read. Primary endpoint step 8000. NA = not available.', '',
     f"Formal units DONE: {sum(r['status'] == 'DONE' for r in rows)}/24. References re-read: {sum(r['status'] == 'REREAD' for r in refrows)}/18.", '',
     '## Parent H (same protocol)', '', '| parent | PRIMARY rSum | i2t R@1/5/10 | t2i R@1/5/10 | SECONDARY (R47 G) rSum | native-layout rSum |', '|---|---:|---|---|---:|---:|']
for h in hrows:
    L.append(f"| {h['parent']} | {f2(h['rsum'])} | {f2(h['i2t_r1'])}/{f2(h['i2t_r5'])}/{f2(h['i2t_r10'])} | {f2(h['t2i_r1'])}/{f2(h['t2i_r5'])}/{f2(h['t2i_r10'])} | {f2(h['secondary_rsum'])} | {f2(h['native_layout_rsum'])} |")
L += ['', '## N references re-read (R48 N-frozen / R47 G N-JOINT weights, step 8000)', '', '| unit | PRIMARY rSum | -H | i2t R@1/5/10 | t2i R@1/5/10 | SECONDARY rSum | SECONDARY -H(sec) |', '|---|---:|---:|---|---|---:|---:|']
for r in refrows:
    L.append(f"| {r['unit']} | {f2(r['rsum'])} | {f2(r['minus_H'])} | {f2(r['i2t_r1'])}/{f2(r['i2t_r5'])}/{f2(r['i2t_r10'])} | {f2(r['t2i_r1'])}/{f2(r['t2i_r5'])}/{f2(r['t2i_r10'])} | {f2(r['secondary_rsum'])} | {f2(r['secondary_minus_H'])} |")
L += ['', '## R50 units (step 8000)', '', '| unit | rSum | -H | -N | i2t R@1/5/10 | t2i R@1/5/10 | DEV cov i2t/t2i | train s | GPU |', '|---|---:|---:|---:|---|---|---|---:|---|']
for r in rows:
    L.append(f"| {r['unit']} | {f2(r['rsum'])} | {f2(r['minus_H'])} | {f2(r['minus_N'])} | {f2(r['i2t_r1'])}/{f2(r['i2t_r5'])}/{f2(r['i2t_r10'])} | {f2(r['t2i_r1'])}/{f2(r['t2i_r5'])}/{f2(r['t2i_r10'])} | "
             f"{f2(r['dev_cov_i2t'] and 100 * r['dev_cov_i2t'])}/{f2(r['dev_cov_t2i'] and 100 * r['dev_cov_t2i'])} | {f2(r['train_seconds'])} | {r['gpu'] or 'NA'} |")
L += ['', '## 2x2 contrasts (rSum, step 8000)', '', '| parent | supervision | obj | X-N frozen | X-N joint | interaction (XJ-XF)-(NJ-NF) |', '|---|---|---|---:|---:|---:|']
for c in con:
    L.append(f"| {c['parent']} | {c['supervision']} | {c['objective']} | {f2(c['minus_N_frozen'])} | {f2(c['minus_N_joint'])} | {f2(c['interaction'])} |")
L += ['', '## Objective contrasts', '', '| parent | supervision | encoder | J-BCE | J-NCE |', '|---|---|---|---:|---:|']
for c in obj_con:
    L.append(f"| {c['parent']} | {c['supervision']} | {c['encoder']} | {f2(c['J_minus_BCE'])} | {f2(c['J_minus_NCE'])} |")
L += ['', '## Oracle exposure (full FIT pool, oracle process)', '', '| cond | obs mismatch | true in top32 i2t/t2i | observed in top32 i2t/t2i | relabelled rows i2t/t2i | relabelled of null rows i2t/t2i | added true mass/row i2t/t2i |', '|---|---:|---|---|---|---|---|']
for x in ox:
    L.append(f"| {x['cond']} | {x['observed_mismatch']:.4f} | {x['true_in_top32_i2t']:.4f}/{x['true_in_top32_t2i']:.4f} | {x['observed_in_top32_i2t']:.4f}/{x['observed_in_top32_t2i']:.4f} | "
             f"{x['relabelled_rows_i2t']:.4f}/{x['relabelled_rows_t2i']:.4f} | {x['relabelled_of_null_i2t']:.4f}/{x['relabelled_of_null_t2i']:.4f} | {x['added_true_mass_per_row_i2t']:.4f}/{x['added_true_mass_per_row_t2i']:.4f} |")
L += ['', 'In-training oracle counters (batch_stats sums of the per-update row counts over all updates): R50_RESULTS.csv columns oracle_*.',
      'Curves: R50_CURVES.csv. Reuse receipt: REUSE_RECEIPT_R50.json. Normalisation: NORMALIZATION_MANIFEST.json.']
(W / 'RESULTS_R50.md').write_text('\n'.join(L) + '\n')
print('R50 summary', sum(r['status'] == 'DONE' for r in rows), 'units', sum(r['status'] == 'REREAD' for r in refrows), 'refs')
