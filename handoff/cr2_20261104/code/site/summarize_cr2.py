"""CR2 summary (numbers only) -> RESULTS_CR2.md + CSVs. Reuses CR1 TASK_ENDPOINTS.csv exactly (21 units)."""
from __future__ import annotations
import csv, datetime as dt, json
from pathlib import Path
import numpy as np

TREE = Path('/home/infres/yinwang/CS_QMI/handoff/cr2_20261104'); OUT = Path('/projects/EEG-foundation-model/yinghao/outputs_cr2/CR2_20261104')
CR1T = Path('/home/infres/yinwang/CS_QMI/handoff/cr1_20261104')
CONDS = ('Flickr00', 'Flickr60', 'CC152K_RECOVERED_20261104'); SRC = ('BASE', 'J', 'BCE', 'NCE')
REC = ('i2t_R1', 'i2t_R5', 'i2t_R10', 't2i_R1', 't2i_R5', 't2i_R10', 'rSum'); STEPS = (0, 250, 500, 1000, 2000, 4000, 8000)
rj = lambda p: json.loads(Path(p).read_text())
f2 = lambda x: '' if x is None else (f'{x:.2f}' if isinstance(x, float) else str(x))
f4 = lambda x: '' if x is None else (f'{x:.4f}' if isinstance(x, float) else str(x))


def wcsv(name, rows):
    if not rows: return
    keys = []; [keys.append(k) for r in rows for k in r if k not in keys]
    with open(TREE/name, 'w', newline='') as f:
        w = csv.DictWriter(f, keys); w.writeheader(); w.writerows(rows)


def main():
    cr1 = {(r['condition'], r['arm']): float(r['rSum']) for r in csv.DictReader(open(CR1T/'TASK_ENDPOINTS.csv'))}
    rows = list(csv.DictReader(open(TREE/'package'/'configs'/'MATRIX.csv')))
    task, curves = [], []
    for r in rows:
        u, c = r['unit_id'], r['condition']; st = OUT/'students'/c/u
        e = dict(unit_id=u, condition=c, source=r['source'], policy=r['policy'],
                 group='ORACLE' if r['oracle'] == 'True' else ('SELECTION' if r['source'] == 'OBS' else 'REPAIR'))
        m8 = st/'eval_008000'/'METRICS.json'
        if not m8.exists(): e['status'] = 'MISSING'; task.append(e); continue
        M = rj(m8); D = rj(st/'DONE.json'); ex = D['exposure']; att = (st/'ATTEMPTS.txt').read_text().split('\n')[0].split()
        e.update({k: M['metrics'][k] for k in REC}); e['base_rSum'] = M['base']['rSum']; e['delta_base'] = M['delta_rSum']
        e['minus_CR1_OBS'] = e['rSum'] - cr1[(c, 'OBS')]
        if r['parent_graph'] not in ('NONE',): e['minus_CR1_parent_CAP'] = e['rSum'] - cr1[(c, r['parent_graph'])]
        for d in ('i2t', 't2i'): e[f'{d}_rescued'] = M[d]['rescued']; e[f'{d}_lost'] = M[d]['lost']
        e.update(train_seconds=D['synchronized_train_seconds_this_attempt'], eval_ms_per_1000_queries=1000*M['cost']['seconds_per_1000_queries'],
                 graph_edges=ex['graph_edges'], distinct_edges_sampled=ex['distinct_edges_sampled'], edge_ESS=ex['edge_ESS'],
                 distinct_images_queried=ex['distinct_images_queried'], distinct_texts_queried=ex['distinct_texts_queried'],
                 mean_unique_images=ex['mean_unique_images_per_batch'], mean_unique_texts=ex['mean_unique_texts_per_batch'],
                 pair_scores_total=ex['pair_scores_total'], max_edge_repeats=ex['max_edge_repeats'],
                 relation_total_mass=D['relation_summary']['total_real_mass'], job=att[0] if att else '', gpu=' '.join(att[2:-1]))
        task.append(e)
        for s in STEPS:
            p = st/f'eval_{s:06d}'/'METRICS.json'
            if p.exists():
                Ms = rj(p); curves.append(dict(unit_id=u, condition=c, step=s, **{k: Ms['metrics'][k] for k in REC}, delta_base=Ms['delta_rSum']))
    by = {(e['condition'], e['source'], e['policy']): e.get('rSum') for e in task}
    for e in task:
        if e.get('rSum') is None: continue
        c, s, p = e['condition'], e['source'], e['policy']
        if p == 'KEEP_TOP1' and by.get((c, s, 'KEEP_SOFT')) is not None: e['TOP1_minus_SOFT'] = e['rSum'] - by[(c, s, 'KEEP_SOFT')]
        if e['group'] == 'REPAIR':
            for o in ('BASE', 'BCE', 'NCE'):
                if s != o and by.get((c, o, p)) is not None: e[f'minus_{o}_same_policy'] = e['rSum'] - by[(c, o, p)]
        if p == 'RANDOM_SUPPORT' and by.get((c, 'OBS', 'SUPPORT_OBS')) is not None: e['minus_SUPPORT_OBS'] = e['rSum'] - by[(c, 'OBS', 'SUPPORT_OBS')]
        if e['group'] == 'ORACLE' and by.get((c, 'J', 'KEEP_TOP1')) is not None:
            e['minus_J_KEEP_TOP1'] = e['rSum'] - by[(c, 'J', 'KEEP_TOP1')]; e['minus_J_KEEP_SOFT'] = e['rSum'] - by[(c, 'J', 'KEEP_SOFT')]
    wcsv('TASK_ENDPOINTS.csv', task); wcsv('TASK_CURVES.csv', curves)
    # graphs
    graphs = []
    for c in CONDS:
        lf = OUT/'audit'/f'AUDIT_LABELFREE_POSTBUILD_{c}.json'; tr = OUT/'evaluator_only'/f'AUDIT_TRUTH_POSTBUILD_{c}.json'
        L = rj(lf) if lf.exists() else {'graphs': {}}; T = rj(tr) if tr.exists() else {'graphs': {}}
        names = list(L['graphs']) + [n for n in T['graphs'] if n not in L['graphs']]
        for n in names:
            g = L['graphs'].get(n, {}); t = T['graphs'].get(n, {})
            graphs.append(dict(condition=c, graph=n, nnz=g.get('nnz'), row_mass_mean=g.get('row_mass_mean'), receipt_row_mass_max_change=g.get('receipt_row_mass_max_change'),
                               observed_mass_per_text=g.get('observed_mass_per_text'), H_rows_obs_frac_of_row_mass=g.get('supported_rows_observed_fraction_of_row_mass'),
                               H0_rows_obs_mass=g.get('unsupported_rows_observed_mass_mean'), cond_entropy_mean=g.get('conditional_entropy_mean'),
                               max_cond_mass_mean=g.get('max_conditional_mass_mean'), row_ESS_mean=g.get('row_ess_mean'), global_edge_ESS=g.get('global_edge_ess'),
                               max_load_over_CR1_cap=g.get('max_load_over_CR1_cap'), images_over_CR1_cap=g.get('images_over_CR1_cap'),
                               slack_mean=g.get('slack_mean'), true_mass_share=t.get('true_mass_share'), H0_true_mass_share=t.get('H0_true_mass_share'),
                               correct_obs_mass_lost=t.get('correct_obs_rows_obs_mass_lost_vs_1'), wrong_obs_true_mass=t.get('wrong_obs_rows_true_mass_mean'),
                               new_edge_mass=t.get('new_edge_mass_per_text'), new_edge_true_share=t.get('new_edge_true_share'),
                               base_wrong_true_mass=t.get('base_wrong_rows_true_mass_mean')))
    wcsv('GRAPH_STATS_CR2.csv', graphs)
    # costs
    cost = []
    for r in csv.DictReader(open(TREE/'SUBMITTED_CR2.tsv'), delimiter='\t'):
        lg = list((TREE/'logs').glob(f"*_{r['job_id']}.out"))
        if not lg: cost.append(dict(job=r['job_id'], stage=r['stage'], status='NO_LOG', note=r['note'])); continue
        tx = lg[0].read_text(errors='replace').splitlines(); s = next((l for l in tx if l.startswith('JOB ')), None)
        en = next((l for l in reversed(tx) if l.startswith('EXIT_OK')), None); tt = lambda l: dt.datetime.strptime(l.split()[-1], '%Y-%m-%dT%H:%M:%SZ')
        parts = s.split() if s else []
        cost.append(dict(job=r['job_id'], name=r['job_name'], stage=r['stage'], device=' '.join(parts[3:-1]) if len(parts) > 4 else 'CPU',
                         status='EXIT_OK' if en else 'NOT_OK', wall_seconds=(tt(en)-tt(s)).total_seconds() if s and en else None, note=r['note']))
    wcsv('COSTS_CR2.csv', cost)
    # markdown
    L = ['# CR2 results (numbers only)', '', f"protocol `CR2_ANCHOR_PRESERVATION_CONCENTRATION_V1`; frozen `PROTOCOL_FROZEN_CR2.json` sha256 "
         f"`{__import__('hashlib').sha256((TREE/'PROTOCOL_FROZEN_CR2.json').read_bytes()).hexdigest()}`; DEV only, seed 1001, step 8000; 0 new estimator fits; TEST not read.", '']
    hdr = '| condition | unit | i2t R1 | R5 | R10 | t2i R1 | R5 | R10 | rSum | Δbase | −CR1 OBS | −CR1 parent CAP | extra | i2t resc/lost | t2i resc/lost |'
    for grp, title in (('REPAIR', '24 anchor-preserving repairs'), ('SELECTION', '6 observation-selection controls'), ('ORACLE', '2 isolated oracle diagnostics (not deployable, not an upper bound)')):
        L += [f'## {title}', '', hdr, '|' + '---|'*15]
        for e in task:
            if e['group'] != grp: continue
            if e.get('rSum') is None: L.append(f"| {e['condition']} | {e['unit_id']} | MISSING |" + ' |'*12); continue
            extra = '; '.join(f"{k}={e[k]:+.2f}" for k in ('TOP1_minus_SOFT', 'minus_BASE_same_policy', 'minus_BCE_same_policy', 'minus_NCE_same_policy',
                                                         'minus_SUPPORT_OBS', 'minus_J_KEEP_TOP1', 'minus_J_KEEP_SOFT') if k in e)
            L.append('| ' + ' | '.join([e['condition'], f"{e['source']} {e['policy']}"] + [f2(e[k]) for k in REC] + [f2(e['delta_base']), f2(e['minus_CR1_OBS']),
                     f2(e.get('minus_CR1_parent_CAP')), extra, f"{e['i2t_rescued']}/{e['i2t_lost']}", f"{e['t2i_rescued']}/{e['t2i_lost']}"]) + ' |')
        L.append('')
    L += ['CR1 reference rSum (reused exactly, CR1 TASK_ENDPOINTS.csv): ' + '; '.join(f"{c} OBS {cr1[(c,'OBS')]:.2f}, BASE/J/BCE/NCE_CAP "
          + '/'.join(f"{cr1[(c, s+'_CAP')]:.2f}" for s in SRC) for c in CONDS), '']
    L += ['## Relation graphs (CR2 CSR; truth columns Flickr evaluator-only; CC label-free only)', '',
          '| condition | graph | nnz | row mass | obs mass/text | H0 obs mass | cond. entropy | max cond. mass | row ESS | max load/CR1 cap | images over cap | true-mass share | correct-obs mass lost | wrong-obs true mass | new-edge true share |', '|' + '---|'*15]
    for g in graphs:
        if not g['graph'].startswith('CR2_') and not g['graph'].startswith('CR1_'): continue
        L.append('| ' + ' | '.join([g['condition'], g['graph']] + [f4(g[k]) for k in ('nnz', 'row_mass_mean', 'observed_mass_per_text', 'H0_rows_obs_mass', 'cond_entropy_mean',
                 'max_cond_mass_mean', 'row_ESS_mean', 'max_load_over_CR1_cap', 'images_over_CR1_cap', 'true_mass_share', 'correct_obs_mass_lost', 'wrong_obs_true_mass', 'new_edge_true_share')]) + ' |')
    gpu = sum(x['wall_seconds'] or 0 for x in cost if x.get('device') not in ('CPU', None) and x.get('status') == 'EXIT_OK')
    cpu = sum(x['wall_seconds'] or 0 for x in cost if x.get('device') == 'CPU' and x.get('status') == 'EXIT_OK' and x['stage'] != 'FEEDER')
    L += ['', f'## Cost', '', f'GPU jobs {gpu/3600:.3f} GPU·h; CPU jobs (excl. feeder) {cpu/3600:.3f} h; per-job ledger COSTS_CR2.csv; failed/cancelled in SUBMITTED_CR2.tsv.',
          '', 'Files: TASK_ENDPOINTS.csv, TASK_CURVES.csv, GRAPH_STATS_CR2.csv, COSTS_CR2.csv, audit JSONs under outputs_cr2/.../audit and evaluator_only.']
    (TREE/'RESULTS_CR2.md').write_text('\n'.join(L)+'\n'); print('\n'.join(L))


if __name__ == '__main__': main()
