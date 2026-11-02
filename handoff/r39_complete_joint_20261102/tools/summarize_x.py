"""R39X summary (numbers only): per additional parent x objective: N-frozen reference (R38 / R37S reused, or the new R39X 0% frozen cell)
vs N-JOINT at step 8000; JOINT-FROZEN; J-BCE / J-NCE per cell; rescued/lost vs H; curves; student feature change; costs; identity of
each JOINT/frozen pair (init-state hash of the head, batch-index chain). Writes R39X_RESULTS.csv, R39X_CURVES.csv and appends an R39X
section to RESULTS.md (run tools/summarize.py first). Missing -> NA. Usage: python3 -I -B tools/summarize_x.py"""
import csv, json, math
from pathlib import Path
W = Path(__file__).resolve().parents[1]
OUT = Path('/projects/EEG-foundation-model/yinghao/outputs_v9/r39_complete_joint_20261102/runs_r39x')
OBJ = ('J', 'BCE', 'NCE')
MET = ('i2t_r1', 'i2t_r5', 'i2t_r10', 't2i_r1', 't2i_r5', 't2i_r10', 'rsum')
PARENTS = ['Flickr00_R30H_s1001', 'Flickr60_R30H_s1002', 'Flickr60_R30H_s1003', 'CC152K_R31H0_s1002', 'CC152K_R31H0_s1003', 'CC152K_R37S_s1001']


def rj(p):
    p = Path(p); return json.loads(p.read_text()) if p.exists() else None


def f(x):
    return 'NA' if x is None or (isinstance(x, float) and math.isnan(x)) else f'{x:.2f}'


def main():
    rows, curves = [], []
    for P in PARENTS:
        for o in OBJ:
            ju = f'R39X_{P}_N_joint_global_{o}'; jc = rj(W / 'configs' / f'{ju}.json')
            jr = rj(OUT / ju / 'RESULT.json')
            if jc['reference_unit']:
                fu = jc['reference_unit']['unit_id']; fr = rj(jc['reference_unit']['result_json']); src = 'reused reference'
            else:
                fu = f'R39X_{P}_N_frozen_shared_{o}'; fr = rj(OUT / fu / 'RESULT.json'); src = 'new R39X frozen cell'
            row = dict(parent=P, objective=o, frozen_unit=fu, frozen_source=src, joint_unit=ju)
            H = (jr or fr or {}).get('native_H_dev', {}).get('rsum')
            row['H_rsum'] = H
            for tag, r in (('frozen', fr), ('joint', jr)):
                for k in MET:
                    row[f'{tag}_{k}'] = r['final']['dev_retrieval'][k] if r else None
                if r:
                    t = r['final'].get('transitions_vs_H', {})
                    row[f'{tag}_net_r1_vs_H'] = t.get('i2t_rescued_r1', 0) - t.get('i2t_lost_r1', 0) + t.get('t2i_rescued_r1', 0) - t.get('t2i_lost_r1', 0)
                    for ph in r['phases']:
                        for c in ph['curves']:
                            curves.append(dict(parent=P, objective=o, cell=tag, unit=r['unit_id'], step=c['step'], rsum=c['dev_retrieval']['rsum']))
            row['joint_minus_frozen_rsum'] = None if not (fr and jr) else jr['final']['dev_retrieval']['rsum'] - fr['final']['dev_retrieval']['rsum']
            row['frozen_minus_H'] = None if not fr else fr['final']['dev_retrieval']['rsum'] - H
            row['joint_minus_H'] = None if not jr else jr['final']['dev_retrieval']['rsum'] - H
            if fr and jr:   # identity of the pair (head init + sampled batches)
                row['pair_init_equal'] = fr['phases'][0]['identity']['init_state_sha256'][:64] == jr['phases'][0]['identity']['init_state_sha256'][:64]
                row['pair_batch_chain_equal'] = fr['phases'][0]['identity']['batch_index_sha256'] == jr['phases'][0]['identity']['batch_index_sha256']
            if jr:
                fc = jr['final'].get('r39_student_feature_change', {}); pdist = jr['final'].get('r39_param_distance', {})
                row['student_dev_img_mean_cos_A'] = (fc.get('dev_img_A') or {}).get('mean_cos'); row['student_dev_cap_mean_cos_A'] = (fc.get('dev_cap_A') or {}).get('mean_cos')
                row['student_rel_l2_A'] = (pdist.get('A') or {}).get('rel_l2_from_parent')
                row['joint_train_s'] = jr['cost']['train_seconds']; row['joint_eval_s'] = jr['cost']['eval_seconds']; row['joint_gpu'] = jr['env']['gpu']
                row['step0_eq_passed'] = (jr.get('r39') or {}).get('step0_cache_equivalence', {}).get('passed'); row['one_step_passed'] = ((jr.get('r39') or {}).get('one_step_check') or {}).get('passed')
            rows.append(row)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(W / 'R39X_RESULTS.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rows)
    with open(W / 'R39X_CURVES.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=['parent', 'objective', 'cell', 'unit', 'step', 'rsum']); w.writeheader(); w.writerows(curves)
    n = sum(1 for r in rows if r['joint_rsum'] is not None) + sum(1 for r in rows if r['frozen_source'].startswith('new') and r['frozen_rsum'] is not None)
    L = ['', '## R39X JOINT confirmation extension (numbers only)', '',
         f'{n}/21 R39X units complete (15 JOINT on five additional parents + 3 JOINT and 3 new N-frozen cells on Flickr 0%). Fixed step 8000, DEV two-stage, TEST not read.', '',
         '| parent | H | obj | N-frozen | N-JOINT | JOINT-FROZEN | frozen-H | JOINT-H | net R@1 vs H (frozen / JOINT) |', '|---|---|---|---|---|---|---|---|---|']
    for r in rows:
        L.append(f"| {r['parent']} | {f(r['H_rsum'])} | {r['objective']} | {f(r['frozen_rsum'])} | {f(r['joint_rsum'])} | {f(r['joint_minus_frozen_rsum'])} | {f(r['frozen_minus_H'])} | "
                 f"{f(r['joint_minus_H'])} | {r.get('frozen_net_r1_vs_H', 'NA')} / {r.get('joint_net_r1_vs_H', 'NA')} |")
    L += ['', '| parent | cell | J-BCE | J-NCE |', '|---|---|---|---|']
    for P in PARENTS:
        for cell in ('frozen', 'joint'):
            v = {r['objective']: r[f'{cell}_rsum'] for r in rows if r['parent'] == P}
            jb = None if None in (v['J'], v['BCE']) else v['J'] - v['BCE']; jn = None if None in (v['J'], v['NCE']) else v['J'] - v['NCE']
            L.append(f'| {P} | {cell} | {f(jb)} | {f(jn)} |')
    L += ['', 'Files: R39X_RESULTS.csv (all recalls, pair identity, student change, costs), R39X_CURVES.csv. N-frozen cells: reused R38 / R37S N-shared-cosine units; '
          'Flickr 0% frozen cells are new R39X units (no R38 reference at 0%).']
    import glob
    fails = [dict(file=f, **{k: json.loads(Path(f).read_text())[k] for k in ('unit', 'utc')}, error=json.loads(Path(f).read_text())['traceback'].strip().splitlines()[-1][:300])
             for f in sorted(glob.glob(str(OUT / '**' / 'FAILED_*.json'), recursive=True))]
    with open(W / 'R39X_FAILURES.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=['unit', 'utc', 'file', 'error']); w.writeheader(); w.writerows(fails)
    L += [f'Failed attempts (kept, rerun after amendment A1 of PROTOCOL_FROZEN_R39X.json; technical in-unit checks only): {len(fails)} (R39X_FAILURES.csv).']
    res = (W / 'RESULTS.md').read_text()
    if '## R39X' in res:
        res = res[:res.index('\n## R39X')]
    (W / 'RESULTS.md').write_text(res.rstrip('\n') + '\n' + '\n'.join(L) + '\n')
    print('R39X summary', n)


if __name__ == '__main__':
    main()
