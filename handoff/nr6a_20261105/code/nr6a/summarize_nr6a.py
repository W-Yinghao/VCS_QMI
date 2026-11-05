#!/usr/bin/env python3
"""NR6-A summary (numbers only): endpoints (stage 4000) for the 24 NR6-A units, and per condition x objective the TAIL-FROZEN and
objective increments across seeds 1001 (NR4 TAIL / NR3-B FROZEN), 2002 (NR5-A for Flickr60/CC; NR6-A for COCO60), 3003 (NR6-A)."""
import csv, json
from pathlib import Path
H = Path('/home/infres/yinwang/CS_QMI/handoff/nr6a_20261105'); O6 = Path('/projects/EEG-foundation-model/yinghao/outputs_nr6/NR6A_20261105/runs')
O5 = Path('/projects/EEG-foundation-model/yinghao/outputs_nr5/NR5_20261104/runs'); O4 = Path('/projects/EEG-foundation-model/yinghao/outputs_nr4/NR4_20261104/runs'); O3 = Path('/projects/EEG-foundation-model/yinghao/outputs_nr3/NR3_20261104/runs')
rj = lambda p: json.loads(Path(p).read_text()); OBJS = ('J', 'BCE', 'NCE'); CONDS = ('Flickr60', 'CC152K_RECOVERED', 'COCO60')


def ep(c, o, sc, seed):
    if seed == 1001:
        p = (O4/f'NR4_TAIL_LORA8_{c}_G_{o}_s1001' if sc == 'TAIL_LORA8' else O3/f'NR3_B_NCE2{o}_{c}_G_s1001')/'eval_012000'/'METRICS.json'
    elif seed == 2002 and c != 'COCO60':
        p = O5/f'NR5_A_{c}_{sc}_{o}_t2002'/'eval_012000'/'METRICS.json'
    else:
        p = O6/f'NR6A_{c}_{sc}_{o}_t{seed}'/'eval_012000'/'METRICS.json'
    return rj(p) if p.exists() else None


def main():
    rows, L = [], ['# RESULTS_NR6A (numbers only; DEV all queries, frozen base Top100, float64; TEST not read)', '']
    for c in CONDS:
        for seed in (2002, 3003) if c == 'COCO60' else (3003,):
            for sc in ('FROZEN', 'TAIL_LORA8'):
                for o in OBJS:
                    M = ep(c, o, sc, seed); u = f'NR6A_{c}_{sc}_{o}_t{seed}'
                    if M: rows.append(dict(unit=u, condition=c, objective=o, scope=sc, seed=seed, **M['metrics'], base_rSum=M['base']['rSum'], student_cos_full_rSum=M['student_cosine_full']['rSum'],
                                           current_score_full_rSum=M.get('current_score_full_gallery', {}).get('metrics', {}).get('rSum'), stage0=json.dumps((rj(O6/u/'eval_008000'/'PREDICTIONS_SHA256.json') or {}).get('stage0_vs_prefix'))))
    if rows:
        with open(H/'RESULTS_NR6A_ENDPOINTS.csv', 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    L += [f'NR6-A endpoints {len(rows)}/24', '', '| unit | i2t R1 | R5 | R10 | t2i R1 | R5 | R10 | rSum | student-cos full | current-s full |', '|---|---|---|---|---|---|---|---|---|---|']
    for r in rows:
        L.append(f"| {r['unit']} | " + ' | '.join(f"{r[k]:.2f}" for k in ('i2t_R1', 'i2t_R5', 'i2t_R10', 't2i_R1', 't2i_R5', 't2i_R10', 'rSum')) + f" | {r['student_cos_full_rSum']:.2f} | {r['current_score_full_rSum']:.2f} |")
    g = lambda c, o, sc, s: (ep(c, o, sc, s) or {}).get('metrics', {}).get('rSum')
    d = lambda a, b: None if a is None or b is None else a - b
    f = lambda v: 'NA' if v is None else f'{v:+.2f}'
    L += ['', '## TAIL_LORA8 − FROZEN (rSum) by seed', '', '| cond | obj | s1001 | s2002 | s3003 | mean | positive |', '|---|---|---|---|---|---|---|']
    for c in CONDS:
        for o in OBJS:
            v = [d(g(c, o, 'TAIL_LORA8', s), g(c, o, 'FROZEN', s)) for s in (1001, 2002, 3003)]; vv = [x for x in v if x is not None]
            L.append(f"| {c} | {o} | {' | '.join(f(x) for x in v)} | {f(sum(vv)/len(vv)) if vv else 'NA'} | {sum(x > 0 for x in vv)}/{len(vv)} |")
    L += ['', '## Objective increments (rSum) by seed', '', '| cond | scope | pair | s1001 | s2002 | s3003 | mean |', '|---|---|---|---|---|---|---|']
    for c in CONDS:
        for sc in ('FROZEN', 'TAIL_LORA8'):
            for a, b in (('J', 'BCE'), ('J', 'NCE'), ('BCE', 'NCE')):
                v = [d(g(c, a, sc, s), g(c, b, sc, s)) for s in (1001, 2002, 3003)]; vv = [x for x in v if x is not None]
                L.append(f"| {c} | {sc} | {a}−{b} | {' | '.join(f(x) for x in v)} | {f(sum(vv)/len(vv)) if vv else 'NA'} |")
    (H/'RESULTS_NR6A.md').write_text('\n'.join(L) + '\n'); print('\n'.join(L))


if __name__ == '__main__':
    main()
