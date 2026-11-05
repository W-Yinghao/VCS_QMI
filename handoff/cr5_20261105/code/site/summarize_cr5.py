"""CR5 summary: 3-seed (CR4 s1001 + CR5 s1002/s1003) mean +- sd per cell, per-seed PC-RECT; numbers only."""
import csv, datetime as dt, hashlib, json, statistics as stt
from pathlib import Path
TREE = Path('/home/infres/yinwang/CS_QMI/handoff/cr5_20261105'); OUT = Path('/projects/EEG-foundation-model/yinghao/outputs_cr5/CR5_20261105'); CR4 = Path('/projects/EEG-foundation-model/yinghao/outputs_cr4/CR4_20261104')
REC = ('i2t_R1', 'i2t_R5', 'i2t_R10', 't2i_R1', 't2i_R5', 't2i_R10', 'rSum'); rj = lambda p: json.loads(Path(p).read_text())
CONDS = ('Flickr60', 'CC152K_RECOVERED_20261104'); RELS = ('OBS', 'J_MASS_OBS', 'BASE_KEEP_TOP1', 'J_KEEP_TOP1', 'BCE_KEEP_TOP1', 'NCE_KEEP_TOP1')
def resolve(d):
    d = Path(d); uc = d/'UNIT_COMPLETE.json'
    if uc.exists() and rj(uc).get('reused_dir'): return Path(rj(uc)['reused_dir'])
    return d
def metr(d):
    p = resolve(d)/'eval_008000'/'METRICS.json'; return rj(p) if p.exists() else None
def cell_dir(c, rel, cons, s):
    if s == 1001: return CR4/'refs_rerun'/c/f'CR3_{c}_{rel}_RECT_s1001' if cons == 'RECT' else CR4/'students'/c/f'CR4_{c}_{rel}_COMPLETE_PC_s1001'
    return OUT/'students'/c/f'CR5_{c}_{rel}_{cons}_s{s}'
ms = lambda xs: (stt.mean(xs), stt.stdev(xs) if len(xs) > 1 else 0.0)
rows, per = [], []; V = {}
for c in CONDS:
    for rel in RELS:
        for cons in ('RECT', 'COMPLETE_PC'):
            r = dict(condition=c, relation=rel, consumer=cons)
            for s in (1001, 1002, 1003):
                M = metr(cell_dir(c, rel, cons, s)); r[f'rSum_s{s}'] = M['metrics']['rSum'] if M else None; V[(c, rel, cons, s)] = M
                if M:
                    for k in REC[:-1]: r[f'{k}_s{s}'] = M['metrics'][k]
            xs = [r[f'rSum_s{s}'] for s in (1001, 1002, 1003) if r[f'rSum_s{s}'] is not None]
            if xs: r['rSum_mean'], r['rSum_sd'] = ms(xs); r['n_seeds'] = len(xs)
            rows.append(r)
        d = dict(condition=c, relation=rel)
        diffs = []
        for s in (1001, 1002, 1003):
            a, b = V.get((c, rel, 'COMPLETE_PC', s)), V.get((c, rel, 'RECT', s))
            d[f'PC_minus_RECT_s{s}'] = (a['metrics']['rSum']-b['metrics']['rSum']) if (a and b) else None
            if a and b: diffs.append(d[f'PC_minus_RECT_s{s}'])
        if diffs: d['PC_minus_RECT_mean'], d['PC_minus_RECT_sd'] = ms(diffs); d['PC_gt_RECT_seeds'] = sum(x > 0 for x in diffs)
        per.append(d)
for name, rr in (('TASK_ENDPOINTS_3SEED.csv', rows), ('PC_MINUS_RECT.csv', per)):
    k = []; [k.append(x) for r in rr for x in r if x not in k]
    with open(TREE/name, 'w', newline='') as f: w = csv.DictWriter(f, k); w.writeheader(); w.writerows(rr)
# source / OBS increments within consumer, per seed then mean
inc = []
for c in CONDS:
    for cons in ('RECT', 'COMPLETE_PC'):
        for ref in ('OBS', 'J_MASS_OBS', 'BASE_KEEP_TOP1', 'BCE_KEEP_TOP1', 'NCE_KEEP_TOP1'):
            xs = [V[(c, 'J_KEEP_TOP1', cons, s)]['metrics']['rSum']-V[(c, ref, cons, s)]['metrics']['rSum'] for s in (1001, 1002, 1003) if V.get((c, 'J_KEEP_TOP1', cons, s)) and V.get((c, ref, cons, s))]
            if xs: m, sd = ms(xs); inc.append(dict(condition=c, consumer=cons, comparison=f'J_KEEP_TOP1-{ref}', mean=m, sd=sd, n=len(xs), positive_seeds=sum(x > 0 for x in xs), per_seed=';'.join(f'{x:+.2f}' for x in xs)))
        xs = [V[(c, 'J_KEEP_TOP1', cons, s)]['metrics']['rSum']-V[(c, 'OBS', 'RECT', s)]['metrics']['rSum'] for s in (1001, 1002, 1003) if V.get((c, 'J_KEEP_TOP1', cons, s)) and V.get((c, 'OBS', 'RECT', s))]
        if xs: m, sd = ms(xs); inc.append(dict(condition=c, consumer=cons, comparison='J_KEEP_TOP1-OBS_RECT(system)', mean=m, sd=sd, n=len(xs), positive_seeds=sum(x > 0 for x in xs), per_seed=';'.join(f'{x:+.2f}' for x in xs)))
k = []; [k.append(x) for r in inc for x in r if x not in k]
with open(TREE/'SOURCE_INCREMENT_3SEED.csv', 'w', newline='') as f: w = csv.DictWriter(f, k); w.writeheader(); w.writerows(inc)
genv = {p.parent.name: rj(p) for p in (OUT/'groups').glob('*/GROUP_ENV.json')}
ident = {p.parent.name: rj(p) for p in (OUT/'students').glob('*/*/IDENTITY_RECEIPT.json')}
(TREE/'GPU_ENV.json').write_text(json.dumps(genv, indent=1)); (TREE/'IDENTITY_RECEIPTS.json').write_text(json.dumps(ident, indent=1))
f2 = lambda x: '' if x is None else f'{x:.2f}'; s2 = lambda x: '' if x is None else f'{x:+.2f}'
L = ['# CR5 results (numbers only)', '', f"protocol `CR5_SEED_CONFIRM_RECT_VS_COMPLETE_PC_V1`; frozen sha256 `{hashlib.sha256((TREE/'PROTOCOL_FROZEN_CR5.json').read_bytes()).hexdigest()}`; seeds 1001 (CR4: PC + same-group RECT refs_rerun), 1002/1003 (CR5); groups: "
     + ', '.join(f"{k}={v['gpu_name']} ({v['driver']})" for k, v in sorted(genv.items())) + '; DEV only, step 8000, TEST not read.', '',
     '| condition | relation | RECT s1001 / s1002 / s1003 | RECT mean ± sd | PC s1001 / s1002 / s1003 | PC mean ± sd | PC−RECT per seed | PC−RECT mean ± sd | PC>RECT seeds |', '|' + '---|'*9]
for d in per:
    c, rel = d['condition'], d['relation']; R_ = next(r for r in rows if r['condition'] == c and r['relation'] == rel and r['consumer'] == 'RECT'); P_ = next(r for r in rows if r['condition'] == c and r['relation'] == rel and r['consumer'] == 'COMPLETE_PC')
    L.append(f"| {c} | {rel} | {' / '.join(f2(R_.get(f'rSum_s{s}')) for s in (1001, 1002, 1003))} | {f2(R_.get('rSum_mean'))} ± {f2(R_.get('rSum_sd'))} | {' / '.join(f2(P_.get(f'rSum_s{s}')) for s in (1001, 1002, 1003))} | "
             f"{f2(P_.get('rSum_mean'))} ± {f2(P_.get('rSum_sd'))} | {' / '.join(s2(d.get(f'PC_minus_RECT_s{s}')) for s in (1001, 1002, 1003))} | {s2(d.get('PC_minus_RECT_mean'))} ± {f2(d.get('PC_minus_RECT_sd'))} | {d.get('PC_gt_RECT_seeds', '')} |")
L += ['', '## J_KEEP_TOP1 increments within consumer (3 seeds)', '', '| condition | consumer | comparison | mean ± sd | positive seeds | per seed |', '|---|---|---|---|---|---|']
for r in inc: L.append(f"| {r['condition']} | {r['consumer']} | {r['comparison']} | {r['mean']:+.2f} ± {r['sd']:.2f} | {r['positive_seeds']}/{r['n']} | {r['per_seed']} |")
L += ['', 'Identity receipts (CC OBS/J_MASS_OBS PC): ' + '; '.join(f"{k}={v['decision']}" for k, v in sorted(ident.items())), '',
      'Files: TASK_ENDPOINTS_3SEED.csv, PC_MINUS_RECT.csv, SOURCE_INCREMENT_3SEED.csv, GPU_ENV.json, IDENTITY_RECEIPTS.json, SUBMITTED_CR5.tsv.']
(TREE/'RESULTS_CR5.md').write_text('\n'.join(L)+'\n'); print('\n'.join(L))
