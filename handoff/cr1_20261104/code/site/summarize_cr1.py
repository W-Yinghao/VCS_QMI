"""CR1 summary: numbers only -> RESULTS_CR1.md + CSVs in the tree. Reads finished outputs; never trains."""
from __future__ import annotations
import csv, json, re, sys
from pathlib import Path
import numpy as np

TREE = Path('/home/infres/yinwang/CS_QMI/handoff/cr1_20261104')
OUT = Path('/projects/EEG-foundation-model/yinghao/outputs_cr1/CR1_20261104')
CONDS = ('Flickr00', 'Flickr60', 'CC152K_RECOVERED_20261104'); OBJS = ('J', 'BCE', 'NCE')
EXP = ('E01', 'E02', 'E03', 'E12', 'E13', 'E23')
ARMS = ('OBS', 'BASE_ROW', 'BASE_CAP', 'J_ROW', 'J_CAP', 'BCE_CAP', 'NCE_CAP')
GRAPHS = ('BASE_ROW', 'BASE_CAP', 'J_ROW', 'J_CAP', 'BCE_CAP', 'NCE_CAP')
STEPS = (0, 250, 500, 1000, 2000, 4000, 8000)
REC = ('i2t_R1', 'i2t_R5', 'i2t_R10', 't2i_R1', 't2i_R5', 't2i_R10', 'rSum')
rj = lambda p: json.loads(Path(p).read_text())
f2 = lambda x: '' if x is None else (f'{x:.2f}' if isinstance(x, float) else str(x))


def write_csv(name, rows):
    if not rows: return
    keys = list(rows[0]);  [keys.append(k) for r in rows for k in r if k not in keys]
    with open(TREE/name, 'w', newline='') as f:
        w = csv.DictWriter(f, keys); w.writeheader(); w.writerows(rows)


def costs():
    import datetime as dt
    rows = []
    with open(TREE/'SUBMITTED_CR1.tsv') as f: sub = list(csv.DictReader(f, delimiter='\t'))
    for r in sub:
        logs = list((TREE/'logs').glob(f"*_{r['job_id']}.out"))
        if not logs: rows.append(dict(job=r['job_id'], name=r['job_name'], stage=r['stage'], status='NO_LOG')); continue
        txt = logs[0].read_text(errors='replace').splitlines()
        st = next((l for l in txt if l.startswith('JOB ')), None); en = next((l for l in reversed(txt) if l.startswith('EXIT_OK')), None)
        t = lambda s_: dt.datetime.strptime(s_.split()[-1], '%Y-%m-%dT%H:%M:%SZ')
        parts = st.split() if st else []
        rows.append(dict(job=r['job_id'], name=r['job_name'], stage=r['stage'], node=parts[2] if len(parts) > 2 else '',
                         device=' '.join(parts[3:-1]) if len(parts) > 4 else 'CPU', status='EXIT_OK' if en else 'NOT_OK',
                         wall_seconds=(t(en)-t(st)).total_seconds() if (st and en) else None, note=r['note']))
    return rows


def main():
    task, curves, est, graphs, scores, cost = [], [], [], [], [], []
    for c in CONDS:
        # ---------------- estimators
        for o in OBJS:
            for e in EXP:
                p = OUT/c/'experts'/o/e/'DONE.json'
                r = dict(condition=c, objective=o, expert=e, status='DONE' if p.exists() else 'MISSING')
                if p.exists():
                    d = rj(p); last = [json.loads(l) for l in (p.parent/'TRAIN.jsonl').read_text().splitlines()][-1]
                    att = (p.parent/'ATTEMPTS.txt').read_text().split('\n')[0].split()
                    r.update(allowed_records=d['allowed_records'], J_raw_step0=d['J_raw_step0']['J'], J_raw_step8000=d['J_raw_step8000']['J'],
                             J_cal_step8000=d['J_calibrated_step8000']['J'], BCE_raw_step8000=d['J_raw_step8000']['BCE'],
                             BCE_cal_step8000=d['J_calibrated_step8000']['BCE'], sat_T_raw_step8000=d['J_raw_step8000']['saturated_T_fraction'],
                             mean_P_raw=d['J_raw_step8000']['mean_P'], mean_Q_raw=d['J_raw_step8000']['mean_Q'],
                             cal_a=d['calibration']['a'], cal_b=d['calibration']['b'], cal_at_bound=d['calibration']['at_bound'],
                             last_batch_loss=last['loss'], last_batch_J=last['J_batch'], train_seconds=d['synchronized_train_seconds_this_attempt'],
                             calibration_seconds=d['calibration_seconds'], resumed_from=d['resumed_from_step'], job=att[0] if att else '',
                             node=att[1] if len(att) > 1 else '', gpu=' '.join(att[2:-1]) if len(att) > 3 else '')
                est.append(r)
        # ---------------- score exports
        for s in ('BASE',) + OBJS:
            p = OUT/c/'scores'/f'{s}.npz.json'
            if not p.exists(): scores.append(dict(condition=c, source=s, status='MISSING')); continue
            d = rj(p)
            for e in EXP:
                L = d['lineage'][e]; q = L['calibrated_quantiles_0_1_10_25_50_75_90_99_100']
                scores.append(dict(condition=c, source=s, expert=e, edges=L['edges'], q01=q[1], q10=q[2], q50=q[4], q90=q[6], q99=q[7],
                                   max=q[8], cal_a=L['calibration']['a'], cal_b=L['calibration']['b'], export_seconds=d['seconds']))
        # ---------------- graphs
        diag = rj(OUT/c/'GRAPH_DIAG.json') if (OUT/c/'GRAPH_DIAG.json').exists() else None
        for g in GRAPHS:
            p = OUT/c/'relations'/f'{g}.npz.json'
            if not p.exists(): graphs.append(dict(condition=c, graph=g, status='MISSING')); continue
            d = rj(p); s = d['solver']; lf = d['label_free_stats']; rel = d['relation']
            r = dict(condition=c, graph=g, accepted_mass_per_text=s['accepted_mass_fraction'], mean_slack=lf['mean_slack'],
                     edges=rel['edges'], edge_ESS=rel['edge_ESS'], text_coverage=rel['text_coverage_positive'],
                     image_coverage=rel['image_coverage_positive'], max_image_load=rel['max_image_load'],
                     max_load_over_cap=lf['max_load_over_capacity'], active_capacity_fraction=lf['active_capacity_fraction'],
                     saturated_capacity_fraction=lf['saturated_capacity_fraction'], mass_on_observed_edge=lf['mass_on_observed_edge'],
                     new_edge_mass=lf['mass_off_observed_edge_new'], observed_mass_removed=lf['observed_mass_removed'],
                     top1_is_observed=lf['top1_weight_is_observed'], kkt_residual=s['kkt_residual'],
                     max_capacity_violation=s['max_capacity_violation'], solver_iterations=s['iterations'],
                     solve_cpu_seconds=d['seconds']['cpu'], solve_wall_seconds=d['seconds']['wall'])
            if diag and isinstance(diag['arms'].get(g), dict) and 'truth' in diag['arms'][g]:
                t = diag['arms'][g]['truth']
                r.update(cand_true_coverage=t['candidate_true_coverage'], true_mass_over_real_mass=t['true_mass_over_real_mass'],
                         true_mass_per_text=t['true_mass_per_text'], wrong_obs_true_mass_share=t['wrong_observation_true_mass_share'],
                         wrong_obs_rows_true_mass_per_row=t['wrong_obs_rows_true_mass_per_row'],
                         correct_obs_kept_mass_per_row=t['correct_obs_rows_kept_obs_mass_per_row'],
                         base_wrong_true_mass_share=t['base_wrong_true_mass_share'], new_edge_true_mass_share=t['new_edge_true_mass_share'],
                         top1_weight_correct=t['top1_weight_correct'])
            graphs.append(r)
        # ---------------- students
        ref = {}
        for arm in ARMS:
            st = OUT/c/'students'/arm; r = dict(condition=c, arm=arm)
            m8 = st/'eval_008000'/'METRICS.json'
            if not m8.exists(): r['status'] = 'MISSING'; task.append(r); continue
            M = rj(m8); d = rj(st/'DONE.json'); ex = d['exposure']; bj = rj(st/'eval_008000'/'BRIDGE.json')
            r.update({k: M['metrics'][k] for k in REC}); r['base_rSum'] = M['base']['rSum']; r['delta_base'] = M['delta_rSum']
            for dd in ('i2t', 't2i'): r[f'{dd}_rescued'] = M[dd]['rescued']; r[f'{dd}_lost'] = M[dd]['lost']; r[f'{dd}_base_top100_coverage'] = M[dd]['coverage_topk']
            att = (st/'ATTEMPTS.txt').read_text().split('\n')[0].split()
            r.update(train_seconds=d['synchronized_train_seconds_this_attempt'], eval_score_seconds_step8000=M['cost']['score_seconds'],
                     eval_ms_per_1000_queries=1000*M['cost']['seconds_per_1000_queries'], eval_pairs_scored=M['cost']['pairs_scored'],
                     graph_edges=ex['graph_edges'], distinct_edges_sampled=ex['distinct_edges_sampled'], distinct_edge_fraction=ex['distinct_edge_fraction'],
                     max_edge_repeats=ex['max_edge_repeats'], mean_unique_images=ex['mean_unique_images_per_batch'],
                     mean_unique_texts=ex['mean_unique_texts_per_batch'], mean_rect_pairs=ex['mean_rectangle_pairs'],
                     relation_edge_ESS=d['relation_summary']['edge_ESS'], relation_total_mass=d['relation_summary']['total_real_mass'],
                     bridge_cr1_vs_nr1_fp32=bj['equivalence']['cr1_vs_nr1_fp32_max_abs'], job=att[0] if att else '', gpu=' '.join(att[2:-1]))
            ref[arm] = r['rSum']; task.append(r)
            for s in STEPS:
                pm = st/f'eval_{s:06d}'/'METRICS.json'
                if pm.exists():
                    Ms = rj(pm); curves.append(dict(condition=c, arm=arm, step=s, **{k: Ms['metrics'][k] for k in REC}, delta_base=Ms['delta_rSum']))
        for r in task:
            if r['condition'] == c and 'rSum' in r:
                for k in ('OBS', 'BASE_CAP', 'BCE_CAP', 'NCE_CAP', 'J_ROW'):
                    r[f'minus_{k}'] = r['rSum'] - ref[k] if k in ref else None
    write_csv('TASK_ENDPOINTS.csv', task); write_csv('TASK_CURVES.csv', curves); write_csv('ESTIMATOR_ENDPOINTS.csv', est)
    write_csv('GRAPH_STATS.csv', graphs); write_csv('SCORE_EXPORTS.csv', scores)
    # precision-mass curves (Flickr, evaluator-only, read-only)
    pmc = []
    for c in CONDS:
        p = OUT/c/'GRAPH_DIAG.json'
        if not p.exists(): continue
        d = rj(p)
        for g in GRAPHS:
            a = d['arms'].get(g)
            if isinstance(a, dict) and 'curves' in a:
                for q, mpt, prec in a['curves']['precision_vs_accepted_mass']:
                    pmc.append(dict(condition=c, graph=g, fraction_of_graph_mass=q, mass_per_text=mpt, true_mass_share=prec))
    write_csv('GRAPH_PRECISION_MASS.csv', pmc)
    cr = costs(); write_csv('COSTS.csv', cr)
    # ---------------- markdown
    L = ['# CR1 results (numbers only)', '', 'protocol `CR1_PRODUCTQ_CROSSFIT_PARTIAL_RELATION_V1`; frozen `PROTOCOL_FROZEN_CR1.json` '
         f"sha256 `{__import__('hashlib').sha256((TREE/'PROTOCOL_FROZEN_CR1.json').read_bytes()).hexdigest()}`; DEV only, seed 1001, step 8000; TEST not read.", '']
    L += ['## Task endpoints (step 8000, full DEV, base Top-100, all queries)', '',
          '| condition | arm | i2t R1 | R5 | R10 | t2i R1 | R5 | R10 | rSum | Δbase | ΔOBS | −BASE_CAP | −BCE_CAP | −NCE_CAP | i2t resc/lost | t2i resc/lost |', '|' + '---|'*16]
    for r in task:
        if 'rSum' not in r: L.append(f"| {r['condition']} | {r['arm']} | MISSING |" + ' |'*14); continue
        L.append('| ' + ' | '.join([r['condition'], r['arm']] + [f2(r[k]) for k in REC] + [f2(r['delta_base']), f2(r['minus_OBS']), f2(r['minus_BASE_CAP']),
                 f2(r['minus_BCE_CAP']), f2(r['minus_NCE_CAP']), f"{r['i2t_rescued']}/{r['i2t_lost']}", f"{r['t2i_rescued']}/{r['t2i_lost']}"]) + ' |')
    L += ['', 'Base rSum (step 0 reproduces base order exactly): ' + ', '.join(f"{c} {f2(next((r['base_rSum'] for r in task if r['condition']==c and 'base_rSum' in r), None))}" for c in CONDS), '']
    L += ['## Relation graphs (FIT only; truth columns Flickr evaluator-only)', '',
          '| condition | graph | accepted mass/text | mean slack | edges | edge ESS | text cov | active cap | max load/cap | mass on obs | new mass | true-mass share | new-edge true share | wrong-obs true share | base-wrong true share | KKT |', '|' + '---|'*16]
    g3 = lambda x: '' if x is None else (f'{x:.4f}' if isinstance(x, float) else str(x))
    for r in graphs:
        if 'edges' not in r: L.append(f"| {r['condition']} | {r['graph']} | MISSING |" + ' |'*13); continue
        L.append('| ' + ' | '.join([r['condition'], r['graph']] + [g3(r.get(k)) for k in ('accepted_mass_per_text', 'mean_slack', 'edges', 'edge_ESS', 'text_coverage',
                 'active_capacity_fraction', 'max_load_over_cap', 'mass_on_observed_edge', 'new_edge_mass', 'true_mass_over_real_mass', 'new_edge_true_mass_share',
                 'wrong_obs_true_mass_share', 'base_wrong_true_mass_share')] + [f"{r['kkt_residual']:.1e}"]) + ' |')
    L += ['', '## Relation estimators (54; CAL pools inside each expert\'s allowed domain; step 8000)', '',
          '| condition | objective | experts done | allowed records (min–max) | raw J step0 (mean) | raw J step8000 (mean) | calibrated J (mean) | cal a (mean) | train s (mean) |', '|' + '---|'*9]
    for c in CONDS:
        for o in OBJS:
            rr = [r for r in est if r['condition'] == c and r['objective'] == o and r['status'] == 'DONE']
            if not rr: L.append(f'| {c} | {o} | 0/6 |' + ' |'*6); continue
            mean = lambda k: float(np.mean([r[k] for r in rr]))
            L.append(f"| {c} | {o} | {len(rr)}/6 | {min(r['allowed_records'] for r in rr)}–{max(r['allowed_records'] for r in rr)} | {mean('J_raw_step0'):.4f} | "
                     f"{mean('J_raw_step8000'):.4f} | {mean('J_cal_step8000'):.4f} | {mean('cal_a'):.3f} | {mean('train_seconds'):.1f} |")
    L += ['', '## Context (read-only)', '']
    for c in ('Flickr00', 'Flickr60'):
        p = OUT/c/'GRAPH_DIAG.json'
        if p.exists():
            d = rj(p); o = d['arms']['OBS_restricted_to_candidates']
            L.append(f"- {c} FIT (evaluator-only, annotation correspondence): observed noise rate {d['observed_noise_rate']:.4f}; "
                     f"true partner in label-blind Top-32 {d['candidate_true_coverage']:.4f}; base top-1 correct {d['base_top1_correct']:.4f}; "
                     f"OBS edges restricted to Top-32 true share {o['true_mass_over_real_mass']:.4f}.")
    for c in CONDS:
        p = OUT/c/'prepared'/'PREP.json'
        if p.exists():
            d = rj(p); L.append(f"- {c} PREP: observed partner in Top-32 {d['observed_partner_in_top32_fraction_label_stat']:.4f}; boundary-tie rows {d['candidate_ties']['rows_with_boundary_ties']}; "
                                f"student common init a={d['student_common_initialization']['a']:.3f}, b={d['student_common_initialization']['b']:.3f} (at_bound={d['student_common_initialization']['at_bound']}).")
    nr = {}
    with open('/home/infres/yinwang/CS_QMI/handoff/nr1_20261104/RESULTS_NR1_STEP8000.csv') as f:
        for r in csv.DictReader(f):
            if r['architecture'] == 'G' and r['condition'] in ('Flickr00', 'Flickr60'): nr[(r['condition'], r['objective'])] = float(r['rSum'])
    for o in OBJS:
        q = Path(f'/home/infres/yinwang/CS_QMI/outputs_nr1/NR1_20261104/runs_cc152k_recovered/NR1_CC152K_RECOVERED_20261104_G_{o}_s1001/eval_008000/METRICS.json')
        if q.exists(): nr[('CC152K_RECOVERED_20261104', o)] = rj(q)['metrics']['rSum']
    L.append('- Background (not CR1 arms; different trainer: NR1 full-position objectives directly on observed pairs, same SigLIP2/DEV): NR1 G step-8000 rSum ' +
             '; '.join(f"{c} J/BCE/NCE {nr.get((c,'J'),float('nan')):.2f}/{nr.get((c,'BCE'),float('nan')):.2f}/{nr.get((c,'NCE'),float('nan')):.2f}" for c in CONDS) + '.')
    gpu = sum(r['wall_seconds'] or 0 for r in cr if r.get('device', 'CPU') != 'CPU' and r.get('status') == 'EXIT_OK')
    cpu = sum(r['wall_seconds'] or 0 for r in cr if r.get('device') == 'CPU' and r.get('status') == 'EXIT_OK' and r['stage'] != 'FEEDER')
    L += ['', f'## Cost (job wall clock from logs, successful jobs; per-fit synchronized seconds in the CSVs)', '',
          f'GPU jobs: {gpu/3600:.3f} GPU·h total; CPU jobs (excl. feeder): {cpu/3600:.3f} h. Failed/cancelled attempts listed in COSTS.csv / SUBMITTED_CR1.tsv.']
    L += ['', 'Files: TASK_ENDPOINTS.csv, TASK_CURVES.csv, ESTIMATOR_ENDPOINTS.csv, GRAPH_STATS.csv, GRAPH_PRECISION_MASS.csv, SCORE_EXPORTS.csv, COSTS.csv, SUBMITTED_CR1.tsv.']
    (TREE/'RESULTS_CR1.md').write_text('\n'.join(L)+'\n'); print('\n'.join(L))


if __name__ == '__main__': main()
