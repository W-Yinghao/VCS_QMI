#!/usr/bin/env python3
"""NR1 summarizer: numbers only (no conclusions). Writes RESULTS_NR1.md + CSVs in the handoff dir."""
import csv, json
from pathlib import Path
H = Path('/home/infres/yinwang/CS_QMI/handoff/nr1_20261104'); O = Path('/home/infres/yinwang/CS_QMI/outputs_nr1/NR1_20261104')
STEPS = [0, 250, 500, 1000, 2000, 4000, 8000]
K = ['i2t_R1', 'i2t_R5', 'i2t_R10', 't2i_R1', 't2i_R5', 't2i_R10', 'rSum']
rows = list(csv.DictReader(open(H/'code/configs/MATRIX.csv')))
blocked = json.loads((H/'BLOCKED_DATA_CC152K.json').read_text())
rj = lambda p: json.loads(Path(p).read_text()) if Path(p).exists() else None
base = rj(O/'baseline/float64/METRICS.json')
curves, final, status = [], [], []
for r in rows:
    u, run = r['unit_id'], O/'runs'/r['unit_id']
    if r['condition'] == 'CC152K_OBS':
        status.append(dict(unit_id=u, status='BLOCKED_DATA', note='no raw CC152K images (BLOCKED_DATA_CC152K.json)')); continue
    done, cost, chk = rj(run/'DONE.json'), rj(run/'TRAIN_COST.json'), rj(run/'STEP0_GRAD_CHECK.json')
    ev = {s: rj(run/f'eval_{s:06d}/METRICS.json') for s in STEPS}
    st = 'COMPLETE' if (run/'UNIT_COMPLETE.json').exists() else ('TRAINED_EVAL_PARTIAL' if done else ('RUNNING_OR_FAILED' if run.exists() else 'NOT_STARTED'))
    s0 = ev[0]['order_vs_base'] if ev[0] else None
    status.append(dict(unit_id=u, status=st, attempts=len((run/'ATTEMPTS.txt').read_text().splitlines()) if (run/'ATTEMPTS.txt').exists() else 0,
                       step0_grad_check=chk['ok'] if chk else None, step0_order_equals_base=(s0 == dict(i2t_queries_order_differs=0, t2i_queries_order_differs=0)) if s0 else None,
                       train_seconds=round(cost['wrapper_seconds'], 1) if cost else None, gpu=cost['gpu'] if cost else None,
                       peak_gpu_mem_MiB=round(cost['peak_gpu_mem_bytes']/2**20) if cost else None,
                       eval_seconds_per_1000_queries_step8000=round(ev[8000]['cost']['seconds_per_1000_queries'], 3) if ev[8000] else None))
    for s, m in ev.items():
        if not m: continue
        row = dict(unit_id=u, condition=r['condition'], architecture=r['architecture'], objective=r['objective'], step=s,
                   **{k: round(m['metrics'][k], 2) for k in K}, delta_rSum=round(m['delta_rSum'], 2),
                   i2t_rescued=m['i2t']['rescued'], i2t_lost=m['i2t']['lost'], t2i_rescued=m['t2i']['rescued'], t2i_lost=m['t2i']['lost'])
        curves.append(row)
        if s == 8000: final.append(row)
def wcsv(p, rs):
    if not rs: Path(p).write_text(''); return
    with open(p, 'w', newline='') as f:
        keys = list(dict.fromkeys(k for r in rs for k in r)); w = csv.DictWriter(f, fieldnames=keys, restval=''); w.writeheader(); w.writerows(rs)
wcsv(H/'RESULTS_NR1_STEP8000.csv', final); wcsv(H/'RESULTS_NR1_CURVES.csv', curves); wcsv(H/'RESULTS_NR1_STATUS.csv', status)
L = ['# NR1_20261104 results (numbers only; DEV, top-100 rerank of base SigLIP2 candidates; TEST not read)', '']
L += ['Model google/siglip2-base-patch16-224 @75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2 (FixRes 224), encoder frozen, protocol NR1_SAMPLEWISE_L2_V1 / NR1_IID_PRODUCT_PQ_V1 / NR1_BASE_TOP100_ALL_QUERY_V1, score dtype float64.', '']
if base:
    L += ['## Baseline (untrained base cosine, DEV 1014 images / 5070 captions)', '', '| ' + ' | '.join(K) + ' | i2t top100 cov | t2i top100 cov |', '|' + '---|'*(len(K)+2)]
    L += ['| ' + ' | '.join(f"{base['metrics'][k]:.2f}" for k in K) + f" | {base['i2t']['coverage_topk']:.4f} | {base['t2i']['coverage_topk']:.4f} |", '']
L += ['## Fixed endpoint step 8000', '', '| unit | ' + ' | '.join(K) + ' | ΔrSum | i2t resc/lost | t2i resc/lost |', '|' + '---|'*(len(K)+4)]
for r in final:
    L.append(f"| {r['unit_id']} | " + ' | '.join(f"{r[k]:.2f}" for k in K) + f" | {r['delta_rSum']:+.2f} | {r['i2t_rescued']}/{r['i2t_lost']} | {r['t2i_rescued']}/{r['t2i_lost']} |")
L += ['', '## rSum curves (steps ' + '/'.join(map(str, STEPS)) + ')', '', '| unit | ' + ' | '.join(map(str, STEPS)) + ' |', '|' + '---|'*(len(STEPS)+1)]
for r in rows:
    c = {x['step']: x['rSum'] for x in curves if x['unit_id'] == r['unit_id']}
    if c: L.append(f"| {r['unit_id']} | " + ' | '.join(f"{c[s]:.2f}" if s in c else 'NA' for s in STEPS) + ' |')
L += ['', '## Unit status', '', '| unit | status | attempts | step0 grad check | step0 order==base | train s | GPU | peak MiB | eval s/1000q |', '|---|---|---|---|---|---|---|---|---|']
for s in status:
    L.append(f"| {s['unit_id']} | {s['status']} | {s.get('attempts','')} | {s.get('step0_grad_check','')} | {s.get('step0_order_equals_base','')} | {s.get('train_seconds','')} | {s.get('gpu','')} | {s.get('peak_gpu_mem_MiB','')} | {s.get('eval_seconds_per_1000_queries_step8000','')} |")
(H/'RESULTS_NR1.md').write_text('\n'.join(L) + '\n'); print('\n'.join(L[:12]))
