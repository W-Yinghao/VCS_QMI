#!/usr/bin/env python3
"""NR4 summary: 36 rows (24 NR4 POST32/TAIL_LORA8 + 12 reused NR3-B FROZEN), numbers only.
Writes RESULTS_NR4.md, RESULTS_NR4_ENDPOINTS.csv (full precision, stage 4000), RESULTS_NR4_CURVES.csv (all stages),
RESULTS_NR4_SCORE_HASHES.csv (per-eval predictions/features sha256), RESOURCE_LEDGER_NR4.json."""
import csv, json
from pathlib import Path
H = Path('/home/infres/yinwang/CS_QMI/handoff/nr4_20261104'); OUT = Path('/projects/EEG-foundation-model/yinghao/outputs_nr4/NR4_20261104')
N3 = Path('/projects/EEG-foundation-model/yinghao/outputs_nr3/NR3_20261104/runs')
CONDS = ['Flickr00', 'Flickr60', 'COCO60', 'CC152K_RECOVERED']; OBJS = ['J', 'BCE', 'NCE']; SCOPES = ['FROZEN', 'POST32', 'TAIL_LORA8']; STAGES = [0, 250, 500, 1000, 2000, 4000]
rj = lambda p: json.loads(Path(p).read_text())
KEYS = ['i2t_R1', 'i2t_R5', 'i2t_R10', 't2i_R1', 't2i_R5', 't2i_R10', 'rSum']


def run_dir(c, o, s):
    return N3/f'NR3_B_NCE2{o}_{c}_G_s1001' if s == 'FROZEN' else OUT/'runs'/f'NR4_{s}_{c}_G_{o}_s1001'


def main():
    endpoints, curves, hashes, ledger, missing = [], [], [], [], []
    for c in CONDS:
        for o in OBJS:
            for s in SCOPES:
                d = run_dir(c, o, s)
                for st in STAGES:
                    m = d/f'eval_{8000+st:06d}'/'METRICS.json'
                    if not m.exists(): missing.append(f'{d.name}/eval_{8000+st:06d}'); continue
                    M = rj(m); row = dict(condition=c, objective=o, scope=s, unit=d.name, stage_step=st, total_step=8000+st, **M['metrics'],
                                          base_rSum=M['base']['rSum'], delta_rSum_vs_base=M['delta_rSum'],
                                          i2t_rescued=M['i2t']['rescued'], i2t_lost=M['i2t']['lost'], t2i_rescued=M['t2i']['rescued'], t2i_lost=M['t2i']['lost'],
                                          i2t_cov=M['i2t']['coverage_topk'], t2i_cov=M['t2i']['coverage_topk'],
                                          student_cos_rSum=M.get('student_cosine_full', {}).get('rSum') if s != 'FROZEN' else M['base']['rSum'],
                                          i2t_recoverable_outside_top100=M['i2t'].get('recoverable_outside_base_topk'), t2i_recoverable_outside_top100=M['t2i'].get('recoverable_outside_base_topk'))
                    curves.append(row)
                    if st == 4000: endpoints.append(row)
                    ps = rj(m.parent/'PREDICTIONS_SHA256.json')
                    hashes.append(dict(unit=d.name, stage_step=st, predictions_sha256=ps['predictions_sha256'], features_sha256=ps.get('features_sha256'), checkpoint_sha256=M.get('checkpoint_sha256')))
                if s != 'FROZEN' and (d/'TRAIN_COST.json').exists():
                    tc = rj(d/'TRAIN_COST.json'); ev = [rj(d/f'eval_{8000+st:06d}'/'METRICS.json') for st in STAGES if (d/f'eval_{8000+st:06d}'/'METRICS.json').exists()]
                    ledger.append(dict(unit=d.name, gpu=tc.get('gpu'), attempts=(d/'ATTEMPTS.txt').read_text().strip().splitlines(), training_wrapper_seconds=tc.get('wrapper_seconds'),
                                       synchronized_update_loop_seconds=tc.get('synchronized_update_loop_seconds'), pair_score_forwards_train=tc.get('pair_forwards_this_attempt'),
                                       raw_encoder_item_forwards_train=tc.get('raw_encoder_item_forwards'), peak_gpu_mem_bytes_train=tc.get('peak_gpu_mem_bytes'),
                                       eval_feature_seconds=sum(e['cost']['feature_seconds'] for e in ev), eval_score_seconds=sum(e['cost']['score_seconds'] for e in ev),
                                       eval_pairs_scored=sum(e['cost']['pairs_scored'] for e in ev),
                                       eval_raw_encoder_item_forwards=[e['features'].get('raw_encoder_item_forwards') for e in ev]))
    E = {(r['condition'], r['objective'], r['scope']): r for r in endpoints}
    with open(H/'RESULTS_NR4_ENDPOINTS.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(curves[0])); w.writeheader(); w.writerows(endpoints)
    with open(H/'RESULTS_NR4_CURVES.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(curves[0])); w.writeheader(); w.writerows(curves)
    with open(H/'RESULTS_NR4_SCORE_HASHES.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(hashes[0])); w.writeheader(); w.writerows(hashes)
    shared = dict(raw_prep={str(p.parent.relative_to(OUT)): rj(p)['seconds'] for p in sorted(OUT.glob('raw/*/*/RAW_MANIFEST.json'))},
                  hidden_prep={str(p.parent.relative_to(OUT)): dict(seconds=rj(p)['seconds'], bytes=rj(p)['bytes'], gpu=rj(p)['gpu']) for p in sorted(OUT.glob('raw/*/fit/HIDDEN_MANIFEST.json'))},
                  prefix='NR3 folded NCE prefix (reused; cost in NR3/NR2 ledgers)', frozen_receipts={p.parent.name: rj(p)['rerun_seconds'] for p in sorted(OUT.glob('reuse_check/*/RECEIPT.json'))})
    (H/'RESOURCE_LEDGER_NR4.json').write_text(json.dumps(dict(shared=shared, units=ledger), indent=1))
    L = ['# RESULTS_NR4 (numbers only; DEV, all queries, frozen base top100, float64; TEST not read)', '',
         f'endpoints {len(endpoints)}/36; missing evals {len(missing)}', '', '## Endpoints (stage 4000 = total step 12000)', '',
         '| cond | obj | scope | i2t R1 | R5 | R10 | t2i R1 | R5 | R10 | rSum | Δbase | rescued i/t | lost i/t | student-cos rSum |', '|' + '---|'*14]
    for c in CONDS:
        for o in OBJS:
            for s in SCOPES:
                r = E.get((c, o, s))
                if r is None: L.append(f'| {c} | {o} | {s} | NA |' + ' |'*10); continue
                L.append(f"| {c} | {o} | {s} | {r['i2t_R1']:.2f} | {r['i2t_R5']:.2f} | {r['i2t_R10']:.2f} | {r['t2i_R1']:.2f} | {r['t2i_R5']:.2f} | {r['t2i_R10']:.2f} | {r['rSum']:.2f} | {r['delta_rSum_vs_base']:+.2f} | {r['i2t_rescued']}/{r['t2i_rescued']} | {r['i2t_lost']}/{r['t2i_lost']} | {r['student_cos_rSum']:.2f} |")
    g = lambda c, o, s: E[(c, o, s)]['rSum'] if (c, o, s) in E else None
    df = lambda a, b: f'{a-b:+.2f}' if a is not None and b is not None else 'NA'
    L += ['', '## Structure increment (rSum, stage 4000)', '', '| cond | obj | POST−FROZEN | TAIL−FROZEN | TAIL−POST |', '|---|---|---|---|---|']
    for c in CONDS:
        for o in OBJS:
            L.append(f'| {c} | {o} | {df(g(c,o,"POST32"), g(c,o,"FROZEN"))} | {df(g(c,o,"TAIL_LORA8"), g(c,o,"FROZEN"))} | {df(g(c,o,"TAIL_LORA8"), g(c,o,"POST32"))} |')
    L += ['', '## Objective increment (rSum, stage 4000)', '', '| cond | scope | J−BCE | J−NCE | BCE−NCE |', '|---|---|---|---|---|']
    for c in CONDS:
        for s in SCOPES:
            L.append(f'| {c} | {s} | {df(g(c,"J",s), g(c,"BCE",s))} | {df(g(c,"J",s), g(c,"NCE",s))} | {df(g(c,"BCE",s), g(c,"NCE",s))} |')
    L += ['', '## Curves (rSum by stage step 0/250/500/1000/2000/4000)', '', '| cond | obj | scope | ' + ' | '.join(map(str, STAGES)) + ' |', '|' + '---|'*(3+len(STAGES))]
    C = {(r['condition'], r['objective'], r['scope'], r['stage_step']): r['rSum'] for r in curves}
    for c in CONDS:
        for o in OBJS:
            for s in SCOPES:
                L.append(f'| {c} | {o} | {s} | ' + ' | '.join(f"{C[(c,o,s,st)]:.2f}" if (c, o, s, st) in C else 'NA' for st in STAGES) + ' |')
    if missing: L += ['', 'missing: ' + ', '.join(missing)]
    (H/'RESULTS_NR4.md').write_text('\n'.join(L) + '\n'); print('\n'.join(L))


if __name__ == '__main__':
    main()
