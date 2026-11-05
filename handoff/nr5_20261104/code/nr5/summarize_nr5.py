#!/usr/bin/env python3
"""NR5 summary (numbers only): 30 endpoints, curves, A/B paired tables (+ NR4 seed-1001 pairs for A), D0 tables, ledgers."""
import csv, json
from pathlib import Path
H = Path('/home/infres/yinwang/CS_QMI/handoff/nr5_20261104'); OUT = Path('/projects/EEG-foundation-model/yinghao/outputs_nr5/NR5_20261104')
N4 = Path('/projects/EEG-foundation-model/yinghao/outputs_nr4/NR4_20261104/runs'); N3 = Path('/projects/EEG-foundation-model/yinghao/outputs_nr3/NR3_20261104/runs')
rj = lambda p: json.loads(Path(p).read_text()); OBJS = ['J', 'BCE', 'NCE']
rows = list(csv.DictReader(open(H/'package/MATRIX_NEW_30.csv', encoding='utf-8-sig')))


def main():
    ends, curves, hashes, ledger, missing = [], [], [], [], []
    for r in rows:
        u = r['unit_id']; cfg = rj(H/f'configs_production/{u}.json'); p0 = 0 if cfg['init'] == 'NATIVE' else 8000; d = OUT/'runs'/u
        for st in cfg['score_checks']:
            m = d/f'eval_{p0+st:06d}'/'METRICS.json'
            if not m.exists(): missing.append(f'{u}@{st}'); continue
            M = rj(m); row = dict(unit=u, group=r['group'], condition=r['condition'], objective=r['objective'], scope=r['scope'], readout=r['readout'], stage_step=st, **M['metrics'],
                                  base_rSum=M['base']['rSum'], delta_vs_base=M['delta_rSum'], student_cos_full_rSum=M['student_cosine_full']['rSum'],
                                  current_score_full_rSum=M.get('current_score_full_gallery', {}).get('metrics', {}).get('rSum'),
                                  i2t_rescued=M['i2t']['rescued'], i2t_lost=M['i2t']['lost'], t2i_rescued=M['t2i']['rescued'], t2i_lost=M['t2i']['lost'])
            curves.append(row)
            if st == max(cfg['score_checks']): ends.append(row)
            ps = rj(m.parent/'PREDICTIONS_SHA256.json'); hashes.append(dict(unit=u, stage_step=st, predictions_sha256=ps['predictions_sha256'], features_sha256=ps.get('features_sha256')))
        if (d/'TRAIN_COST.json').exists():
            tc = rj(d/'TRAIN_COST.json'); ex = rj(d/'EXPOSURE_LEDGER.json') if (d/'EXPOSURE_LEDGER.json').exists() else None
            ledger.append(dict(unit=u, gpu=tc.get('gpu'), attempts=(d/'ATTEMPTS.txt').read_text().strip().splitlines(), synchronized_update_loop_seconds=tc.get('synchronized_update_loop_seconds'),
                               wrapper_seconds=tc.get('wrapper_seconds'), pair_scores_train=tc.get('pair_forwards_this_attempt'), tail_item_forwards_train=tc.get('raw_encoder_item_forwards'),
                               peak_gpu_mem_bytes=tc.get('peak_gpu_mem_bytes'), exposure=ex))
    for fn, data in (('RESULTS_NR5_ENDPOINTS.csv', ends), ('RESULTS_NR5_CURVES.csv', curves), ('RESULTS_NR5_SCORE_HASHES.csv', hashes)):
        if data:
            with open(H/fn, 'w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=list(data[0])); w.writeheader(); w.writerows(data)
    (H/'RESOURCE_AND_EXPOSURE_LEDGER_NR5.json').write_text(json.dumps(dict(units=ledger, shared='NR4 hidden cache reused (Flickr 45.97 GB, CC 78.44 GB; COCO not used); no new large cache'), indent=1))
    E = {(x['condition'], x['objective'], x['scope'], x['readout']): x for x in ends}
    f2 = lambda v: 'NA' if v is None else f'{v:.2f}'
    df = lambda a, b: 'NA' if a is None or b is None else f'{a-b:+.2f}'
    L = ['# RESULTS_NR5 (numbers only; DEV all queries, frozen base Top100, float64; TEST not read)', '', f'endpoints {len(ends)}/30; missing evals {len(missing)}', '',
         '## Endpoints', '', '| group | cond | obj | scope | readout | i2t R1 | R5 | R10 | t2i R1 | R5 | R10 | rSum | Δbase | student-cos full | current-s full |', '|' + '---|'*15]
    for x in ends:
        L.append(f"| {x['group']} | {x['condition']} | {x['objective']} | {x['scope']} | {x['readout']} | " + ' | '.join(f"{x[k]:.2f}" for k in ('i2t_R1', 'i2t_R5', 'i2t_R10', 't2i_R1', 't2i_R5', 't2i_R10', 'rSum'))
                 + f" | {x['delta_vs_base']:+.2f} | {x['student_cos_full_rSum']:.2f} | {f2(x['current_score_full_rSum'])} |")
    g = lambda c, o, s, r: E[(c, o, s, r)]['rSum'] if (c, o, s, r) in E else None
    n4 = lambda c, o, s: rj(N4/f'NR4_TAIL_LORA8_{c}_G_{o}_s1001'/'eval_012000'/'METRICS.json')['metrics']['rSum'] if s == 'TAIL_LORA8' else rj(N3/f'NR3_B_NCE2{o}_{c}_G_s1001'/'eval_012000'/'METRICS.json')['metrics']['rSum']
    L += ['', '## A: structure and objective increments (seed 2002) beside NR4 seed 1001', '', '| cond | obj | FROZEN t2002 | TAIL t2002 | TAIL−FROZEN t2002 | FROZEN s1001 | TAIL s1001 | TAIL−FROZEN s1001 |', '|---|---|---|---|---|---|---|---|']
    for c in ('Flickr60', 'CC152K_RECOVERED'):
        for o in OBJS:
            a, b = g(c, o, 'FROZEN', 'G'), g(c, o, 'TAIL_LORA8', 'G'); a1, b1 = n4(c, o, 'FROZEN'), n4(c, o, 'TAIL_LORA8')
            L.append(f'| {c} | {o} | {f2(a)} | {f2(b)} | {df(b, a)} | {a1:.2f} | {b1:.2f} | {b1-a1:+.2f} |')
    L += ['', '| cond | scope | J−BCE t2002 | J−NCE t2002 | J−BCE s1001 | J−NCE s1001 |', '|---|---|---|---|---|---|']
    for c in ('Flickr60', 'CC152K_RECOVERED'):
        for s in ('FROZEN', 'TAIL_LORA8'):
            L.append(f"| {c} | {s} | {df(g(c,'J',s,'G'), g(c,'BCE',s,'G'))} | {df(g(c,'J',s,'G'), g(c,'NCE',s,'G'))} | {n4(c,'J',s)-n4(c,'BCE',s):+.2f} | {n4(c,'J',s)-n4(c,'NCE',s):+.2f} |")
    L += ['', '## B: structure increment (G_ZERO − COS) and objective increments', '', '| cond | obj | COS | G_ZERO | G_ZERO−COS |', '|---|---|---|---|---|']
    for c in ('Flickr00', 'Flickr60', 'CC152K_RECOVERED'):
        for o in OBJS:
            L.append(f"| {c} | {o} | {f2(g(c,o,'TAIL_LORA8','COS'))} | {f2(g(c,o,'TAIL_LORA8','G_ZERO'))} | {df(g(c,o,'TAIL_LORA8','G_ZERO'), g(c,o,'TAIL_LORA8','COS'))} |")
    L += ['', '| cond | readout | J−BCE | J−NCE | BCE−NCE |', '|---|---|---|---|---|']
    for c in ('Flickr00', 'Flickr60', 'CC152K_RECOVERED'):
        for rd in ('COS', 'G_ZERO'):
            q = lambda o: g(c, o, 'TAIL_LORA8', rd)
            L.append(f'| {c} | {rd} | {df(q("J"), q("BCE"))} | {df(q("J"), q("NCE"))} | {df(q("BCE"), q("NCE"))} |')
    L += ['', '## Curves (rSum by stage step)', '']
    C = {}
    for x in curves: C.setdefault(x['unit'], {})[x['stage_step']] = x['rSum']
    for u, v in C.items(): L.append(f"- {u}: " + ', '.join(f'{k}:{v[k]:.2f}' for k in sorted(v)))
    d0 = sorted(OUT.glob('d0/*/D0.json'))
    if d0:
        L += ['', '## D0 (NR4 TAIL endpoints; base Top100 rSum and full-gallery rSum)', '', '| unit | E0G0 | E1G1 | E1G0 | E0G1 | E1COS(top100) | E1COS full | E1G1 full |', '|---|---|---|---|---|---|---|---|']
        for f in d0:
            for u, U in rj(f)['units'].items():
                cm, fg = U['combos'], U['full_gallery']
                L.append(f"| {u} | " + ' | '.join(f"{cm[k]['metrics']['rSum']:.2f}" for k in ('E0G0', 'E1G1', 'E1G0', 'E0G1', 'E1COS')) + f" | {fg['E1COS_full']['rSum']:.2f} | {fg['E1G1_full']['rSum']:.2f} |")
    if missing: L += ['', 'missing: ' + ', '.join(missing)]
    (H/'RESULTS_NR5.md').write_text('\n'.join(L) + '\n'); print('\n'.join(L))


if __name__ == '__main__':
    main()
