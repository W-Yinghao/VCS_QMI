"""CR3 summary (numbers only) -> RESULTS_CR3.md + CSVs. REUSE cells read the historical CR1/CR2 unit directory named in the receipt."""
import csv, datetime as dt, hashlib, json
from pathlib import Path

TREE = Path('/home/infres/yinwang/CS_QMI/handoff/cr3_20261104'); OUT = Path('/projects/EEG-foundation-model/yinghao/outputs_cr3/CR3_20261104')
REC = ('i2t_R1', 'i2t_R5', 'i2t_R10', 't2i_R1', 't2i_R5', 't2i_R10', 'rSum'); STEPS = (0, 250, 500, 1000, 2000, 4000, 8000)
rj = lambda p: json.loads(Path(p).read_text())
f2 = lambda x: '' if x is None else (f'{x:+.2f}' if isinstance(x, float) else str(x))


def wcsv(name, rows):
    if not rows: return
    keys = []; [keys.append(k) for r in rows for k in r if k not in keys]
    with open(TREE/name, 'w', newline='') as f: w = csv.DictWriter(f, keys); w.writeheader(); w.writerows(rows)


def main():
    fz = rj(TREE/'PROTOCOL_FROZEN_CR3.json'); plan = fz['unit_plan']; receipts = rj(TREE/'REUSE_OR_RERUN_RECEIPTS.json')
    rows = list(csv.DictReader(open(TREE/'package'/'configs'/'MATRIX.csv')))
    task, curves = [], []
    for r in rows:
        u, c, rel, cons = r['unit_id'], r['condition'], r['relation'], r['consumer']
        st = OUT/'students_A1'/c/u  # amendment CR3_A1: every cell runs inside its condition's group job
        ge = OUT/'students_A1'/c/'GROUP_ENV.json'; genv = rj(ge) if ge.exists() else {}
        e = dict(unit_id=u, condition=c, relation=rel, consumer=cons, oracle=r['oracle'], plan=plan[u], unit_dir=str(st), group_gpu=genv.get('gpu_name'), group_driver=genv.get('driver'))
        m8 = st/'eval_008000'/'METRICS.json'
        if not m8.exists(): e['status'] = 'MISSING'; task.append(e); continue
        M = rj(m8); D = rj(st/'DONE.json'); att = (st/'ATTEMPTS.txt').read_text().split('\n')[0]
        e.update({k: M['metrics'][k] for k in REC}); e['delta_base'] = M['delta_rSum']
        for d in ('i2t', 't2i'): e[f'{d}_rescued'] = M[d]['rescued']; e[f'{d}_lost'] = M[d]['lost']
        e['attempt'] = att
        cost = D.get('cost', {})
        e.update(train_seconds=D['synchronized_train_seconds_this_attempt'], mean_pairs_per_update=cost.get('mean_pair_scores_per_update'),
                 pairs_per_query_event=cost.get('pair_scores_per_query_event'), mean_cache_reads=cost.get('mean_cache_reads_per_update'),
                 max_pairs_update=cost.get('max_pair_scores_in_an_update'), peak_gpu_mem_MB=(cost['peak_gpu_memory_bytes']/2**20 if cost.get('peak_gpu_memory_bytes') else None),
                 complete_build_cpu_s=cost.get('complete_build_cpu_seconds'), edge_hits_sha256=D.get('exposure', {}).get('edge_hits_sha256'),
                 distinct_edges_sampled=D.get('exposure', {}).get('distinct_edges_sampled'))
        task.append(e)
        for s in STEPS:
            p = st/f'eval_{s:06d}'/'METRICS.json'
            if p.exists(): Ms = rj(p); curves.append(dict(unit_id=u, condition=c, relation=rel, consumer=cons, step=s, **{k: Ms['metrics'][k] for k in REC}))
    idx = {(e['condition'], e['relation'], e['consumer']): e.get('rSum') for e in task}
    for e in task:
        if e.get('rSum') is None: continue
        c, rel, cons = e['condition'], e['relation'], e['consumer']
        other = idx.get((c, rel, 'RECT' if cons == 'COMPLETE' else 'COMPLETE'))
        if cons == 'COMPLETE' and other is not None: e['COMPLETE_minus_RECT'] = e['rSum'] - other
        for ref in ('OBS', 'J_MASS_OBS', 'BASE_KEEP_TOP1', 'BCE_KEEP_TOP1', 'NCE_KEEP_TOP1'):
            if ref != rel and idx.get((c, ref, cons)) is not None: e[f'minus_{ref}_same_consumer'] = e['rSum'] - idx[(c, ref, cons)]
    # same-draw check: paired RECT/COMPLETE units share edge-exposure hashes
    for e in task:
        if e['consumer'] == 'COMPLETE' and e.get('edge_hits_sha256'):
            p = next((x for x in task if x['condition'] == e['condition'] and x['relation'] == e['relation'] and x['consumer'] == 'RECT'), None)
            if p is not None: e['same_edge_draws_as_RECT'] = (p.get('edge_hits_sha256') == e['edge_hits_sha256']) if p.get('edge_hits_sha256') else 'RECT_reused_historical_(no_hash)'
    wcsv('TASK_ENDPOINTS.csv', task); wcsv('TASK_CURVES.csv', curves)
    # transfer stats
    tr = []
    for p in sorted((OUT/'transfer_audit').glob('*/*.json')):
        d = rj(p)
        for mode in ('RECT', 'COMPLETE'):
            s = d['summary'][mode]; row = dict(condition=p.parent.name, relation=p.stem, mode=mode, mean_pair_scores=s['mean_pair_scores'], mean_extra_pair_scores=s['mean_extra_pair_scores'])
            for dr in ('i2t', 't2i'):
                for k, v in s[dr].items(): row[f'{dr}_{k}'] = v
            tr.append(row)
    wcsv('TRANSFER_STATS.csv', tr)
    cost = []
    for r in csv.DictReader(open(TREE/'SUBMITTED_CR3.tsv'), delimiter='\t'):
        lg = list((TREE/'logs').glob(f"*_{r['job_id']}.out"))
        if not lg: cost.append(dict(job=r['job_id'], stage=r['stage'], status='NO_LOG', note=r['note'])); continue
        tx = lg[0].read_text(errors='replace').splitlines(); s = next((l for l in tx if l.startswith('JOB ')), None); en = next((l for l in reversed(tx) if l.startswith('EXIT_OK')), None)
        tt = lambda l: dt.datetime.strptime(l.split()[-1], '%Y-%m-%dT%H:%M:%SZ'); parts = s.split() if s else []
        cost.append(dict(job=r['job_id'], name=r['job_name'], stage=r['stage'], device=' '.join(parts[3:-1]) if len(parts) > 4 else 'CPU',
                         status='EXIT_OK' if en else 'NOT_OK', wall_seconds=(tt(en)-tt(s)).total_seconds() if s and en else None, note=r['note']))
    wcsv('RESOURCE_LEDGER.csv', cost)
    L = ['# CR3 results (numbers only)', '', f"protocol `CR3_FIXED_GRAPH_SUPPORT_COMPLETE_V1`; frozen sha256 `{hashlib.sha256((TREE/'PROTOCOL_FROZEN_CR3.json').read_bytes()).hexdigest()}`; "
         f"amendment CR3_A1 (per-condition GPU lock; groups: " + '; '.join(f"{c}={(rj(OUT/'students_A1'/c/'GROUP_ENV.json')['gpu_name'] if (OUT/'students_A1'/c/'GROUP_ENV.json').exists() else 'n/a')}" for c in ('Flickr00','Flickr60','CC152K_RECOVERED_20261104')) + f"), torch {fz['gpu_env_lock']['torch']}, CUDA {fz['gpu_env_lock']['cuda']}; DEV only; seed 1001; step 8000; TEST not read.", '',
         '| condition | relation | consumer | plan | i2t R1 | R5 | R10 | t2i R1 | R5 | R10 | rSum | Δbase | COMPLETE−RECT | −OBS (same consumer) | −J_MASS_OBS | −BASE/BCE/NCE TOP1 | i2t resc/lost | t2i resc/lost |', '|' + '---|'*18]
    for e in task:
        if e.get('rSum') is None: L.append(f"| {e['condition']} | {e['relation']} | {e['consumer']} | {e['plan']} | MISSING |" + ' |'*13); continue
        srcd = '/'.join(f2(e.get(f'minus_{k}_same_consumer')) for k in ('BASE_KEEP_TOP1', 'BCE_KEEP_TOP1', 'NCE_KEEP_TOP1'))
        L.append('| ' + ' | '.join([e['condition'], e['relation'], e['consumer'], e['plan']] + [f'{e[k]:.2f}' for k in REC] +
                 [f2(e['delta_base']), f2(e.get('COMPLETE_minus_RECT')), f2(e.get('minus_OBS_same_consumer')), f2(e.get('minus_J_MASS_OBS_same_consumer')), srcd,
                  f"{e['i2t_rescued']}/{e['i2t_lost']}", f"{e['t2i_rescued']}/{e['t2i_lost']}"]) + ' |')
    L += ['', '## COMPLETE / RECT cost (per update, 64 query events)', '', '| condition | relation | consumer | pairs/update | pairs/query event | cache reads/update | max pairs | peak GPU MB | train s | same draws |', '|' + '---|'*10]
    for e in task:
        if e.get('rSum') is None: continue
        g = lambda k, f='{:.1f}': '' if e.get(k) is None else f.format(e[k])
        L.append(f"| {e['condition']} | {e['relation']} | {e['consumer']} | {g('mean_pairs_per_update')} | {g('pairs_per_query_event', '{:.2f}')} | {g('mean_cache_reads')} | {g('max_pairs_update', '{:.0f}')} | {g('peak_gpu_mem_MB')} | {g('train_seconds')} | {e.get('same_edge_draws_as_RECT', '')} |")
    gpu = sum(x['wall_seconds'] or 0 for x in cost if x.get('device') not in ('CPU', None) and x.get('status') == 'EXIT_OK')
    cpu = sum(x['wall_seconds'] or 0 for x in cost if x.get('device') == 'CPU' and x.get('status') == 'EXIT_OK' and x['stage'] != 'FEEDER')
    L += ['', f'Resources: GPU jobs {gpu/3600:.3f} GPU·h; CPU jobs (excl. feeder) {cpu/3600:.3f} h (RESOURCE_LEDGER.csv).',
          '', 'Files: TASK_ENDPOINTS.csv, TASK_CURVES.csv, TRANSFER_STATS.csv, REUSE_OR_RERUN_RECEIPTS.json, RESOURCE_LEDGER.csv, cr2_audit_export/, PROTOCOL_FROZEN_CR3.json.']
    (TREE/'RESULTS_CR3.md').write_text('\n'.join(L)+'\n'); print('\n'.join(L))


if __name__ == '__main__': main()
