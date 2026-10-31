"""R27 A1 cross-dataset table (COCO 40% / CC152K full DEV; no TEST; numbers only, no interpretation).
A1 = R27 a1_iface run with code_a1x on the R26 prefixes / noise maps; the R26 arms (H0 native, D-VCS, D-BCE, M-VCS,
N-VCS, D-sNCE) are read from R26 analysis/CROSS_SUMMARY.json (same cells, same prefixes). Unfinished A1 cells = NA,
never replaced. Per-run checks as summarize_r27.py (40 co-train epochs, DONE best == max rsum).
Writes A1X_SUMMARY.json and A1X_TABLES.md. Usage: summarize_a1x.py"""
import json, os, glob
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = '/projects/EEG-foundation-model/yinghao/outputs_v9/r27_a1cross_20261030/runs'
R26 = '/home/infres/yinwang/CS_QMI/handoff/r26_dist_20261027/analysis/CROSS_SUMMARY.json'
CELLS = ['c40_s1001', 'c40_s1002', 'c40_s1003', 'cc_s1001', 'cc_s1002', 'cc_s1003']
R26_ARMS = ['H0', 'd_vcs', 'd_bce', 'm_vcs', 'n_vcs', 'd_snce']
KEYS = ['i2t_r1', 'i2t_r5', 'i2t_r10', 't2i_r1', 't2i_r5', 't2i_r10']


def run(d):
    if not os.path.exists(os.path.join(d, 'DONE.json')):
        return None
    ep = json.load(open(os.path.join(d, 'EPOCHS.json')))['cotrain']
    rs = [e['dev']['rsum'] for e in ep]
    assert len(ep) == 40 and [e['epoch'] for e in ep] == list(range(40)), d
    assert abs(max(rs) - json.load(open(os.path.join(d, 'DONE.json')))['best_rsum']) < 1e-6, d
    b = int(np.argmax(rs))
    peaks = sorted(glob.glob(os.path.join(d, 'PEAK_*.json')))
    pk = [json.load(open(p)) for p in peaks]
    return {'best': max(rs), 'epoch': b, 'last': rs[-1], 'last5': float(np.mean(rs[-5:])), **{k: ep[b]['dev'][k] for k in KEYS},
            'attempts': open(os.path.join(d, 'ATTEMPTS')).read().strip() if os.path.exists(os.path.join(d, 'ATTEMPTS')) else None,
            'gpu': [p.get('gpu') for p in pk], 'img_gather': [p.get('img_gather') for p in pk],
            'wall_h_sum_recorded': sum(p.get('wall_seconds', 0) for p in pk) / 3600}


r26 = json.load(open(R26))['runs']
res, missing = {}, []
for c in CELLS:
    res[c] = {a: {k: r26[c][a][k] for k in ['best', 'epoch', 'last', 'last5'] + KEYS} for a in R26_ARMS}
    r = run(os.path.join(RUNS, c, 'a1_iface'))
    if r is None:
        missing.append(f'{c}/a1_iface')
    else:
        res[c]['a1_iface'] = r
ARMS = ['a1_iface'] + R26_ARMS
pairs = {c: {f'a1_iface-{a}': ({m: res[c]['a1_iface'][m] - res[c][a][m] for m in ('best', 'last5', 'i2t_r1', 't2i_r1')}
                               if 'a1_iface' in res[c] else None) for a in R26_ARMS} for c in CELLS}
means = {}
for ds in ('c40', 'cc'):
    cs = [c for c in CELLS if c.startswith(ds) and 'a1_iface' in res[c]]
    means[ds] = {'cells_with_a1': cs, 'mean_best_over_these_cells': {a: float(np.mean([res[c][a]['best'] for c in cs])) for a in ARMS} if cs else None}
json.dump({'scope': 'COCO 40% (1K-image DEV) / CC152K full DEV, best rSum over 40 co-train epochs; R27 A1 (code_a1x) vs R26 arms on the same prefixes; no TEST',
           'missing': missing, 'runs': res, 'pairs': pairs, 'means_over_finished_a1_cells': means},
          open(os.path.join(HERE, 'A1X_SUMMARY.json'), 'w'), indent=1)
f = lambda v: 'NA' if v is None else f'{v:+.2f}'
L = ['# R27 A1 跨数据集数字表（自动生成；full DEV；未读 TEST；只列数字，不含解读）', '',
     'A1 = R27 a1_iface（另一 peer 选候选身份、当前 peer 重编码），用 code_a1x 在 R26 的共同前缀与噪声映射上运行；R26 各臂（H0 原生、D-VCS、D-BCE、M-VCS、N-VCS、D-sNCE）取自 `r26_dist_20261027/analysis/CROSS_SUMMARY.json`。COCO DEV 为 1K 图 / 5K caption，CC152K DEV 为 1K / 1K。', '',
     f'未完成 A1 单元（记 NA，不以其他臂替代）：{", ".join(missing) if missing else "无"}', '',
     '## 逐格最佳 full-DEV rSum', '', '| 格 | ' + ' | '.join(ARMS) + ' |', '|---|' + '---|' * len(ARMS)]
for c in CELLS:
    L.append(f'| {c} | ' + ' | '.join(f"{res[c][a]['best']:.2f}" if a in res[c] else 'NA' for a in ARMS) + ' |')
L += ['', '## A1 各格终点', '', '| 格 | best | best epoch | last | last5 | i2t R@1 | i2t R@5 | i2t R@10 | t2i R@1 | t2i R@5 | t2i R@10 | attempts | GPU / 取行方式 |', '|---|---|---|---|---|---|---|---|---|---|---|---|---|']
for c in CELLS:
    r = res[c].get('a1_iface')
    L.append(f'| {c} | NA |' + ' |' * 11 + ' |' if r is None else
             f"| {c} | {r['best']:.2f} | {r['epoch']} | {r['last']:.2f} | {r['last5']:.2f} | " + ' | '.join(f'{r[k]:.2f}' for k in KEYS) +
             f" | {r['attempts']} | {', '.join(f'{g} / {m}' for g, m in zip(r['gpu'], r['img_gather']))} |")
L += ['', '## 逐 seed 配对差 A1 − R26 各臂（best rSum / last5 / i2t R@1 / t2i R@1）', '', '| 格 | 比较 | best | last5 | i2t R@1 | t2i R@1 |', '|---|---|---|---|---|---|']
for c in CELLS:
    for k, v in pairs[c].items():
        L.append(f'| {c} | {k} | ' + (' | '.join(f(v[m]) for m in ('best', 'last5', 'i2t_r1', 't2i_r1')) if v else 'NA | NA | NA | NA') + ' |')
L += ['', '## 仅在 A1 已完成格上的均值（best rSum；不是三 seed 均值）', '', '| 数据集 | 已完成格 | ' + ' | '.join(ARMS) + ' |', '|---|---|' + '---|' * len(ARMS)]
for ds, m in means.items():
    L.append(f"| {ds} | {', '.join(m['cells_with_a1']) or '—'} | " + (' | '.join(f"{m['mean_best_over_these_cells'][a]:.2f}" for a in ARMS) if m['mean_best_over_these_cells'] else 'NA') + ' |')
open(os.path.join(HERE, 'A1X_TABLES.md'), 'w').write('\n'.join(L) + '\n')
print('\n'.join(L))
