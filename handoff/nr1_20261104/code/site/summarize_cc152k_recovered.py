#!/usr/bin/env python3
"""Numbers only: CC152K_RECOVERED_20261104 base + 6 units (step 8000 and curves)."""
import json
from pathlib import Path
C = 'CC152K_RECOVERED_20261104'; D = Path(f'/projects/EEG-foundation-model/yinghao/data/cc152k_raw/nr1/{C}')
RUNS = Path('/home/infres/yinwang/CS_QMI/outputs_nr1/NR1_20261104/runs_cc152k_recovered'); H = Path('/home/infres/yinwang/CS_QMI/handoff/nr1_20261104/cc152k')
K = ['i2t_R1', 'i2t_R5', 'i2t_R10', 't2i_R1', 't2i_R5', 't2i_R10', 'rSum']; S = [0, 250, 500, 1000, 2000, 4000, 8000]
rj = lambda p: json.loads(Path(p).read_text()) if Path(p).exists() else None
sp = rj(H/'RECOVERED_SPLITS.json'); b = rj(D/'baseline/METRICS.json'); cal = rj(D/'calibration.json')
L = [f'# {C} (numbers only; recovered web snapshot, scenario C; DEV only; TEST not read)', '', f"protocol_id {sp['protocol_id']}; FIT {sp['counts']['fit']['kept']}/150000, DEV {sp['counts']['dev']['kept']}/1000 images and captions (paired filtering); calibration a0={cal['a0']:.3f} b0={cal['b0']:.3f}", '',
     '| row | ' + ' | '.join(K) + ' | ΔrSum | i2t resc/lost | t2i resc/lost |', '|' + '---|'*(len(K)+4),
     '| base | ' + ' | '.join(f"{b['metrics'][k]:.2f}" for k in K) + f" | 0.00 | – | – |  (top100 coverage i2t {b['i2t']['coverage_topk']:.4f}, t2i {b['t2i']['coverage_topk']:.4f})"]
cur = []
for a in ['G', 'GL']:
    for o in ['J', 'BCE', 'NCE']:
        u = f'NR1_{C}_{a}_{o}_s1001'; m = rj(RUNS/u/'eval_008000/METRICS.json'); m0 = rj(RUNS/u/'eval_000000/METRICS.json')
        if m: L.append(f"| {a} {o} | " + ' | '.join(f"{m['metrics'][k]:.2f}" for k in K) + f" | {m['delta_rSum']:+.2f} | {m['i2t']['rescued']}/{m['i2t']['lost']} | {m['t2i']['rescued']}/{m['t2i']['lost']} |")
        else: L.append(f'| {a} {o} | NA |')
        cur.append(f"| {a} {o} | " + ' | '.join(f"{rj(RUNS/u/f'eval_{s:06d}/METRICS.json')['metrics']['rSum']:.2f}" if rj(RUNS/u/f'eval_{s:06d}/METRICS.json') else 'NA' for s in S) +
                   f" | {m0['order_vs_base'] if m0 else 'NA'} |")
L += ['', '| unit | ' + ' | '.join(map(str, S)) + ' | step0 order vs base |', '|' + '---|'*(len(S)+2)] + cur
(H/'RESULTS_CC152K_RECOVERED.md').write_text('\n'.join(L) + '\n'); print('\n'.join(L))
