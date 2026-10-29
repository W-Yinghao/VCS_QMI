"""R27 result tables (full DEV only, no TEST). A0 = R25 native H0 (same prefix, same noise lock).
Reads each finished run's EPOCHS.json + DONE.json (checks 40 co-train epochs and DONE best == max rsum).
Writes R27_SUMMARY.json and R27_TABLES.md: per-run best/last/last5/six recalls, per-arm means, seed-paired
differences for the design comparisons. Units not finished are listed as missing; no interpretation.
Usage: summarize_r27.py"""
import json, os, glob
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = '/projects/EEG-foundation-model/yinghao/outputs_v9/r27_qtarget_20261029/runs'
R25 = '/projects/EEG-foundation-model/yinghao/outputs_v9/r25_in2r_20261027/groups'
ARMS = ['A0_native', 'a1_iface', 'a2_mean', 'a3_vcs', 'a4_bce', 'a5_snce', 'a6_delta']
CELLS = [f'f{n}_s{s}' for n in (40, 60) for s in (1001, 1002, 1003)]
KEYS = ['i2t_r1', 'i2t_r5', 'i2t_r10', 't2i_r1', 't2i_r5', 't2i_r10']
PAIRS = [('a3_vcs', 'A0_native'), ('a1_iface', 'A0_native'), ('a3_vcs', 'a1_iface'), ('a3_vcs', 'a2_mean'),
         ('a3_vcs', 'a6_delta'), ('a6_delta', 'a2_mean'), ('a3_vcs', 'a4_bce'), ('a3_vcs', 'a5_snce'), ('a2_mean', 'a1_iface')]


def run(d):
    done = os.path.join(d, 'DONE.json')
    if not os.path.exists(done):
        return None
    ep = json.load(open(os.path.join(d, 'EPOCHS.json')))['cotrain']
    rs = [e['dev']['rsum'] for e in ep]
    assert len(ep) == 40 and [e['epoch'] for e in ep] == list(range(40)), d
    assert abs(max(rs) - json.load(open(done))['best_rsum']) < 1e-6, d
    b = int(np.argmax(rs))
    peaks = sorted(glob.glob(os.path.join(d, 'PEAK_*.json')))
    att = open(os.path.join(d, 'ATTEMPTS')).read().strip() if os.path.exists(os.path.join(d, 'ATTEMPTS')) else None
    return {'best': max(rs), 'epoch': b, 'last': rs[-1], 'last5': float(np.mean(rs[-5:])), **{k: ep[b]['dev'][k] for k in KEYS},
            'attempts': att, 'gpu': [json.load(open(p)).get('gpu') for p in peaks],
            'wall_h_sum_recorded': sum(json.load(open(p)).get('wall_seconds', 0) for p in peaks) / 3600}


res, missing = {}, []
for c in CELLS:
    res[c] = {}
    for a in ARMS:
        r = run(os.path.join(R25, c, 'H0')) if a == 'A0_native' else run(os.path.join(R, c, a))
        if r is None:
            missing.append(f'{c}/{a}')
        else:
            res[c][a] = r
comp = {}
for n in (40, 60):
    cs = [f'f{n}_s{s}' for s in (1001, 1002, 1003)]
    comp[f'f{n}'] = {'arm_mean_best': {a: float(np.mean([res[c][a]['best'] for c in cs])) for a in ARMS if all(a in res[c] for c in cs)},
                     'arm_mean_last5': {a: float(np.mean([res[c][a]['last5'] for c in cs])) for a in ARMS if all(a in res[c] for c in cs)},
                     'pairs_best': {}}
    for x, y in PAIRS:
        d = [res[c][x]['best'] - res[c][y]['best'] for c in cs if x in res[c] and y in res[c]]
        comp[f'f{n}']['pairs_best'][f'{x}-{y}'] = {'per_seed': d, 'n': len(d), 'mean': float(np.mean(d)) if d else None,
                                                   'n_pos': int(sum(v > 1e-9 for v in d)), 'n_tie': int(sum(abs(v) <= 1e-9 for v in d))}
out = {'scope': 'R27 Flickr30K full DEV (1,014 img / 5,070 cap), best rSum over 40 co-train epochs; A0 = R25 H0; no TEST; no per-query ranks',
       'missing': missing, 'runs': res, 'comparisons': comp}
json.dump(out, open(os.path.join(HERE, 'R27_SUMMARY.json'), 'w'), indent=1)
L = ['# R27 结果表（自动生成，full DEV；未读 TEST；只列数字，不含解读）', '', f'缺失单元：{", ".join(missing) if missing else "无"}', '',
     '## 逐格最佳 full-DEV rSum', '', '| 格 | ' + ' | '.join(ARMS) + ' |', '|---|' + '---|' * len(ARMS)]
for c in CELLS:
    L.append(f'| {c} | ' + ' | '.join(f"{res[c][a]['best']:.2f}" if a in res[c] else '—' for a in ARMS) + ' |')
for n in (40, 60):
    m = comp[f'f{n}']
    L += ['', f'## Flickr {n}%：三 seed 均值（最佳 / 末5轮）', '', '| 臂 | best | last5 |', '|---|---|---|']
    L += [f'| {a} | {m["arm_mean_best"][a]:.2f} | {m["arm_mean_last5"][a]:.2f} |' for a in m['arm_mean_best']]
    L += ['', f'## Flickr {n}%：逐 seed 配对差（最佳 rSum）', '', '| 比较 | n | 均值 | >0 | 平 | 逐 seed |', '|---|---|---|---|---|---|']
    for k, v in m['pairs_best'].items():
        if v['n']:
            L.append(f'| {k} | {v["n"]} | {v["mean"]:+.2f} | {v["n_pos"]} | {v["n_tie"]} | ' + ', '.join(f'{x:+.2f}' for x in v['per_seed']) + ' |')
open(os.path.join(HERE, 'R27_TABLES.md'), 'w').write('\n'.join(L) + '\n')
print('\n'.join(L))
