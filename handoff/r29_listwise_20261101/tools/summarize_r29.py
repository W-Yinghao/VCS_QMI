"""R29 number table (Flickr30K full DEV; no TEST; numbers only, no interpretation). Arms H native / N InfoNCE / P pointwise VCS /
L listwise VCS / B listwise BCE in the main alignment term (warmup + reliable-pair alignment), common untrained init, noisy
branch = R27 A1, frozen config PROTOCOL_FROZEN.json. Cells f00_s1001 / f60_s1001 (pilots) and f60_s1002 / f60_s1003 (H/N/L/B).
Per run: best full-DEV rSum over 40 co-train epochs, its epoch and six Recalls, last, last5, best and last DEV of the 5 warmup epochs.
Per-run checks: 40 co-train epochs 0..39, DONE best == max rSum. Unfinished units = NA, never replaced.
Writes R29_SUMMARY.json and R29_TABLES.md. Usage: summarize_r29.py"""
import json, os
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = '/projects/EEG-foundation-model/yinghao/outputs_v9/r29_listwise_20261101/runs'
ARMS = ['h_native', 'n_infonce', 'p_pointwise_vcs', 'l_listwise_vcs', 'b_listwise_bce']
CELLS = {'f00_s1001': ARMS, 'f60_s1001': ARMS, 'f60_s1002': ['h_native', 'n_infonce', 'l_listwise_vcs', 'b_listwise_bce'],
         'f60_s1003': ['h_native', 'n_infonce', 'l_listwise_vcs', 'b_listwise_bce']}
KEYS = ['i2t_r1', 'i2t_r5', 'i2t_r10', 't2i_r1', 't2i_r5', 't2i_r10']
PAIRS = [('l_listwise_vcs', 'n_infonce'), ('l_listwise_vcs', 'p_pointwise_vcs'), ('l_listwise_vcs', 'b_listwise_bce'),
         ('l_listwise_vcs', 'h_native'), ('n_infonce', 'h_native'), ('b_listwise_bce', 'n_infonce'), ('p_pointwise_vcs', 'n_infonce')]


def run(d):
    if not os.path.exists(f'{d}/DONE.json'):
        return None
    e = json.load(open(f'{d}/EPOCHS.json'))
    ep = e['cotrain']; rs = [x['dev']['rsum'] for x in ep]
    assert len(ep) == 40 and [x['epoch'] for x in ep] == list(range(40)), d
    assert abs(max(rs) - json.load(open(f'{d}/DONE.json'))['best_rsum']) < 1e-6, d
    b = int(np.argmax(rs))
    wr = [x['dev']['rsum'] for x in e.get('warmup', []) if 'dev' in x]
    return {'best': max(rs), 'epoch': b, 'last': rs[-1], 'last5': float(np.mean(rs[-5:])), **{k: ep[b]['dev'][k] for k in KEYS},
            'warmup_best': max(wr) if wr else None, 'warmup_last': wr[-1] if wr else None, 'n_warmup_dev': len(wr)}


res, missing = {}, []
for c, arms in CELLS.items():
    res[c] = {}
    for a in arms:
        res[c][a] = run(f'{RUNS}/{c}/{a}')
        if res[c][a] is None:
            missing.append(f'{c}/{a}')
pairs = {c: {f'{x}-{y}': ({m: res[c][x][m] - res[c][y][m] for m in ['best', 'last5'] + KEYS}
                          if x in res[c] and y in res[c] and res[c][x] and res[c][y] else None)
             for x, y in PAIRS if x in CELLS[c] and y in CELLS[c]} for c in CELLS}
ms = {}
for a in ['h_native', 'n_infonce', 'l_listwise_vcs', 'b_listwise_bce']:
    v = [res[c][a] for c in ('f60_s1001', 'f60_s1002', 'f60_s1003')]
    have = [x for x in v if x]
    ms[a] = {'n_seeds': len(have), 'mean_best': float(np.mean([x['best'] for x in have])) if have else None,
             'mean_last5': float(np.mean([x['last5'] for x in have])) if have else None}
md = {}
for x, y in PAIRS:
    if x in ms and y in ms:
        d = [res[c][x]['best'] - res[c][y]['best'] for c in ('f60_s1001', 'f60_s1002', 'f60_s1003') if res[c].get(x) and res[c].get(y)]
        md[f'{x}-{y}'] = {'per_seed': d, 'mean': float(np.mean(d)) if d else None, 'n_pos': int(sum(v > 1e-9 for v in d))}
json.dump({'scope': 'Flickr30K full DEV best rSum over 40 co-train epochs; R29 main-alignment arms; no TEST', 'missing': missing,
           'runs': res, 'pairs': pairs, 'f60_multiseed': {'means': ms, 'paired': md}}, open(os.path.join(HERE, 'R29_SUMMARY.json'), 'w'), indent=1)
f = lambda v: 'NA' if v is None else f'{v:.2f}'
g = lambda v: 'NA' if v is None else f'{v:+.2f}'
L = ['# R29 数字表（主对齐项替换；自动生成；full DEV；未读 TEST；只列数字，不含解读）', '',
     'H 原生 A1 配方 / N InfoNCE / P 点对 VCS（T=tanh(s/τ+b)）/ L 候选竞争 VCS（T=(Kπ−1)/(Kπ+1)）/ B 同 π 参数化 balanced BCE；替换位置为 warmup 与可信配对对齐项，noisy 分支保持 A1；共同未训练初值，warmup 5 + co-train 40；τ=0.05，批内候选 K=128；配置见 `handoff/r29_listwise_20261101/PROTOCOL_FROZEN.json`。0% 为恒等映射（无人工错配）。未完成单元记 NA，不以其他臂替代。', '',
     f'未完成单元：{", ".join(missing) if missing else "无"}', '']
for c, arms in CELLS.items():
    L += [f'## {c}', '', '| 臂 | best | best epoch | last | last5 | i2t R@1 | i2t R@5 | i2t R@10 | t2i R@1 | t2i R@5 | t2i R@10 | warmup best | warmup last |',
          '|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for a in arms:
        r = res[c][a]
        L.append(f'| {a} | NA |' + ' |' * 11 + ' |' if r is None else
                 f"| {a} | {f(r['best'])} | {r['epoch']} | {f(r['last'])} | {f(r['last5'])} | " + ' | '.join(f(r[k]) for k in KEYS) +
                 f" | {f(r['warmup_best'])} | {f(r['warmup_last'])} |")
    L += ['', '| 比较 | best | last5 | i2t R@1 | t2i R@1 |', '|---|---|---|---|---|']
    for k, v in pairs[c].items():
        L.append(f'| {k} | ' + (' | '.join(g(v[m]) for m in ('best', 'last5', 'i2t_r1', 't2i_r1')) if v else 'NA | NA | NA | NA') + ' |')
    L.append('')
L += ['## 60% 多 seed（seed 1001–1003；均值只在已完成 seed 上计算并标注 n）', '', '| 臂 | n | 均值 best | 均值 last5 |', '|---|---|---|---|']
for a, m in ms.items():
    L.append(f"| {a} | {m['n_seeds']} | {f(m['mean_best'])} | {f(m['mean_last5'])} |")
L += ['', '| 比较 | 逐 seed best 差 | 均值 | >0 |', '|---|---|---|---|']
for k, m in md.items():
    L.append(f"| {k} | {', '.join(g(x) for x in m['per_seed']) or '—'} | {g(m['mean'])} | {m['n_pos']}/{len(m['per_seed'])} |")
open(os.path.join(HERE, 'R29_TABLES.md'), 'w').write('\n'.join(L) + '\n')
print('\n'.join(L))
