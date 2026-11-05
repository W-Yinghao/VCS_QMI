"""CR4 summary (numbers only): RESULTS_CR4.md, TASK_ENDPOINTS.csv, TASK_CURVES.csv, CONSUMER_INCREMENT.csv, SOURCE_INCREMENT.csv,
REFERENCE_RECEIPTS.json, SAME_DRAW_PAIR_TARGET_RECEIPT.csv, GPU_ENV.json, RESOURCE_LEDGER.csv."""
import csv, datetime as dt, hashlib, json
from pathlib import Path

TREE = Path('/home/infres/yinwang/CS_QMI/handoff/cr4_20261104'); OUT = Path('/projects/EEG-foundation-model/yinghao/outputs_cr4/CR4_20261104')
CR3S = Path('/projects/EEG-foundation-model/yinghao/outputs_cr3/CR3_20261104/students_A1')
REC = ('i2t_R1', 'i2t_R5', 'i2t_R10', 't2i_R1', 't2i_R5', 't2i_R10', 'rSum'); STEPS = (0, 250, 500, 1000, 2000, 4000, 8000)
rj = lambda p: json.loads(Path(p).read_text())
f2 = lambda x: '' if x is None else (f'{x:+.2f}' if isinstance(x, float) else str(x))


def wcsv(n, rows):
    if not rows: return
    k = []; [k.append(x) for r in rows for x in r if x not in k]
    with open(TREE/n, 'w', newline='') as f: w = csv.DictWriter(f, k); w.writeheader(); w.writerows(rows)


def metrics(d):
    p = Path(d)/'eval_008000'/'METRICS.json'
    return rj(p) if p.exists() else None


def main():
    fz = rj(TREE/'PROTOCOL_FROZEN_CR4.json'); conds = sorted({i['condition'] for i in fz['units'].values()})
    rr = {c: rj(OUT/'groups'/c/'REFERENCE_RECEIPT.json') for c in conds if (OUT/'groups'/c/'REFERENCE_RECEIPT.json').exists()}
    genv = {c: rj(OUT/'groups'/c/'GROUP_ENV.json') for c in conds if (OUT/'groups'/c/'GROUP_ENV.json').exists()}
    def ref_dir(c, rel, cons):
        u = f'CR3_{c}_{rel}_{cons}_s1001'
        if c in rr and rr[c]['decision'].startswith('RERUN'): return OUT/'refs_rerun'/c/u
        return CR3S/c/u
    task, curves, cons_inc, same = [], [], [], []
    for u, i in fz['units'].items():
        c, rel = i['condition'], i['relation']; d = Path(i['result_dir']); uc = d/'UNIT_COMPLETE.json'; status = rj(uc)['status'] if uc.exists() else 'MISSING'
        src = Path(rj(uc)['reused_dir']) if uc.exists() and rj(uc).get('reused_dir') else d
        e = dict(unit_id=u, condition=c, relation=rel, oracle=i['oracle'], plan=i['plan'], status=status, result_dir=str(src), group_gpu=genv.get(c, {}).get('gpu_name'))
        M = metrics(src)
        if M:
            e.update({k: M['metrics'][k] for k in REC}); e['delta_base'] = M['delta_rSum']
            for dd in ('i2t', 't2i'): e[f'{dd}_rescued'] = M[dd]['rescued']; e[f'{dd}_lost'] = M[dd]['lost']
            D = rj(src/'DONE.json'); co = D.get('cost', {})
            e.update(train_seconds=D['synchronized_train_seconds_this_attempt'], pairs_per_update=co.get('mean_pair_scores_per_update'),
                     multi_positive_view_fraction=co.get('multi_positive_view_fraction'), no_negative_view_fraction=co.get('no_negative_view_fraction'),
                     peak_gpu_MB=co['peak_gpu_memory_bytes']/2**20 if co.get('peak_gpu_memory_bytes') else None)
            for s in STEPS:
                p = src/f'eval_{s:06d}'/'METRICS.json'
                if p.exists(): Ms = rj(p); curves.append(dict(unit_id=u, condition=c, relation=rel, step=s, **{k: Ms['metrics'][k] for k in REC}))
            mc, mr = metrics(ref_dir(c, rel, 'COMPLETE')), metrics(ref_dir(c, rel, 'RECT'))
            e['ref_COMPLETE_CE_rSum'] = mc['metrics']['rSum'] if mc else None; e['ref_RECT_rSum'] = mr['metrics']['rSum'] if mr else None
            e['PC_minus_COMPLETE_CE'] = e['rSum']-e['ref_COMPLETE_CE_rSum'] if mc else None; e['PC_minus_RECT_CE'] = e['rSum']-e['ref_RECT_rSum'] if mr else None
            cons_inc.append({k: e.get(k) for k in ('unit_id', 'condition', 'relation', 'rSum', 'ref_COMPLETE_CE_rSum', 'ref_RECT_rSum', 'PC_minus_COMPLETE_CE', 'PC_minus_RECT_CE')})
            refd = ref_dir(c, rel, 'COMPLETE'); rd = rj(refd/'DONE.json') if (refd/'DONE.json').exists() else {}
            pc_hash = D.get('same_draw_receipt', {}).get('edge_hits_sha256') or D.get('exposure', {}).get('edge_hits_sha256')
            ce_hash = rd.get('exposure', {}).get('edge_hits_sha256') or rd.get('same_draw_receipt', {}).get('edge_hits_sha256')
            same.append(dict(unit_id=u, pc_edge_hits_sha256=pc_hash, complete_ce_ref_edge_hits_sha256=ce_hash, same_draws=pc_hash == ce_hash if (pc_hash and ce_hash) else None,
                             pc_packed_sha256=D.get('same_draw_receipt', {}).get('packed_pairs_targets_weights_mask_sha256')))
        task.append(e)
    idx = {(e['condition'], e['relation']): e.get('rSum') for e in task}
    src_inc = []
    for e in task:
        if e.get('rSum') is None: continue
        for ref in ('OBS', 'J_MASS_OBS', 'BASE_KEEP_TOP1', 'BCE_KEEP_TOP1', 'NCE_KEEP_TOP1'):
            if ref != e['relation'] and idx.get((e['condition'], ref)) is not None: e[f'minus_{ref}_PC'] = e['rSum']-idx[(e['condition'], ref)]
        mo = metrics(ref_dir(e['condition'], 'OBS', 'RECT'))
        e['minus_RECT_OBS_system'] = e['rSum']-mo['metrics']['rSum'] if mo else None
        if e['relation'] == 'J_KEEP_TOP1': src_inc.append({k: e.get(k) for k in ('condition', 'rSum', 'minus_OBS_PC', 'minus_J_MASS_OBS_PC', 'minus_BASE_KEEP_TOP1_PC', 'minus_BCE_KEEP_TOP1_PC', 'minus_NCE_KEEP_TOP1_PC', 'minus_RECT_OBS_system')})
    wcsv('TASK_ENDPOINTS.csv', task); wcsv('TASK_CURVES.csv', curves); wcsv('CONSUMER_INCREMENT.csv', cons_inc); wcsv('SOURCE_INCREMENT.csv', src_inc); wcsv('SAME_DRAW_PAIR_TARGET_RECEIPT.csv', same)
    ident = {u: rj(Path(i['result_dir'])/'IDENTITY_RECEIPT.json') for u, i in fz['units'].items() if (Path(i['result_dir'])/'IDENTITY_RECEIPT.json').exists()}
    (TREE/'REFERENCE_RECEIPTS.json').write_text(json.dumps({'group_reference_receipts': rr, 'identity_receipts': ident}, indent=1))
    (TREE/'GPU_ENV.json').write_text(json.dumps(genv, indent=1))
    led = []
    for r in csv.DictReader(open(TREE/'SUBMITTED_CR4.tsv'), delimiter='\t'):
        lg = list((TREE/'logs').glob(f"*_{r['job_id']}.out"))
        if not lg: led.append(dict(job=r['job_id'], stage=r['stage'], status='NO_LOG', note=r['note'])); continue
        tx = lg[0].read_text(errors='replace').splitlines(); s = next((l for l in tx if l.startswith('JOB ')), None); en = next((l for l in reversed(tx) if l.startswith('EXIT_OK')), None)
        tt = lambda l: dt.datetime.strptime(l.split()[-1], '%Y-%m-%dT%H:%M:%SZ'); parts = s.split() if s else []
        led.append(dict(job=r['job_id'], name=r['job_name'], stage=r['stage'], device=' '.join(parts[3:-1]) if len(parts) > 4 else 'CPU', status='EXIT_OK' if en else 'NOT_OK',
                        wall_seconds=(tt(en)-tt(s)).total_seconds() if s and en else None, note=r['note']))
    wcsv('RESOURCE_LEDGER.csv', led)
    gtxt = ', '.join("%s=%s (%s)" % (c, v.get('gpu_name'), v.get('driver')) for c, v in genv.items())
    rtxt = ', '.join("%s=%s" % (c, v['decision']) for c, v in rr.items())
    fsha = hashlib.sha256((TREE/'PROTOCOL_FROZEN_CR4.json').read_bytes()).hexdigest()
    L = ['# CR4 results (numbers only)', '', f"protocol `CR4_FIXED_GRAPH_POSITIVE_EXCLUSIVE_V1`; frozen sha256 `{fsha}`; group GPUs: {gtxt}; reference decisions: {rtxt}; DEV only; seed 1001; step 8000; TEST not read.", '',
         '| condition | relation | plan / status | i2t R1 | R5 | R10 | t2i R1 | R5 | R10 | rSum | Δbase | PC−COMPLETE_CE | PC−RECT_CE | −OBS(PC) | −J_MASS_OBS(PC) | −BASE/BCE/NCE TOP1 (PC) | −RECT OBS | i2t resc/lost | t2i resc/lost |', '|' + '---|'*19]
    for e in task:
        if e.get('rSum') is None: L.append(f"| {e['condition']} | {e['relation']} | {e['status']} |" + ' |'*16); continue
        sd = '/'.join(f2(e.get(f'minus_{k}_PC')) for k in ('BASE_KEEP_TOP1', 'BCE_KEEP_TOP1', 'NCE_KEEP_TOP1'))
        L.append('| ' + ' | '.join([e['condition'], e['relation'], f"{e['plan']} / {e['status']}"] + [f'{e[k]:.2f}' for k in REC] + [f2(e['delta_base']), f2(e.get('PC_minus_COMPLETE_CE')),
                 f2(e.get('PC_minus_RECT_CE')), f2(e.get('minus_OBS_PC')), f2(e.get('minus_J_MASS_OBS_PC')), sd, f2(e.get('minus_RECT_OBS_system')),
                 f"{e['i2t_rescued']}/{e['i2t_lost']}", f"{e['t2i_rescued']}/{e['t2i_lost']}"]) + ' |')
    gpu = sum(x['wall_seconds'] or 0 for x in led if x.get('device') not in ('CPU', None) and x.get('status') == 'EXIT_OK')
    L += ['', f'Resources: GPU jobs {gpu/3600:.3f} GPU·h (RESOURCE_LEDGER.csv). Files: TASK_ENDPOINTS.csv, TASK_CURVES.csv, CONSUMER_INCREMENT.csv, SOURCE_INCREMENT.csv, '
          'REFERENCE_RECEIPTS.json, SAME_DRAW_PAIR_TARGET_RECEIPT.csv, GRAD_COMPONENT_AUDIT_*.json, SOURCE_GRAPH_HASHES.json, GPU_ENV.json, PROTOCOL_FROZEN_CR4.json.']
    (TREE/'RESULTS_CR4.md').write_text('\n'.join(L)+'\n'); print('\n'.join(L))


if __name__ == '__main__': main()
