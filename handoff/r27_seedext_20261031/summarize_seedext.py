"""Multi-seed number table for the Flickr seed1002/1003 extension of R27-D / R27-O / R27-O3 / R27-CS (full DEV only; no TEST;
numbers only, no interpretation). seed1001 = the original single-seed units; seeds 1002/1003 = the extension snapshots
(training code byte-identical apart from the seed gate). A1 reference = R27 a1_iface of the SAME cell/seed. Unfinished cells = NA,
never replaced; means are over the seeds that are finished and are labelled with their n. Per-run checks as summarize_r27.py.
Writes SEEDEXT_SUMMARY.json and SEEDEXT_TABLES.md. Usage: summarize_seedext.py"""
import json, os, glob
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
O = '/projects/EEG-foundation-model/yinghao/outputs_v9'
EXT = f'{O}/r27_seedext_20261031'
ORIG = {'r27d': f'{O}/r27_directional_20261030/runs', 'r27o': f'{O}/r27_oracle_20261030/runs',
        'r27o3': f'{O}/r27_oracle3_20261031/runs', 'r27cs': f'{O}/r27_candsrc_20261031/runs'}
ARMS = {'r27d': ['d1_vcs_i2t', 'd2_vcs_t2i', 'd3_snce_i2t', 'd4_snce_t2i'], 'r27o': ['o1_rescue_native_t2i', 'o2_rescue_vcs_t2i'],
        'r27o3': ['o3_true_target_t2i'], 'r27cs': ['bn_cross_t2i', 'cn_mix_t2i']}
A1 = f'{O}/r27_qtarget_20261029/runs'
SEEDS = (1001, 1002, 1003)
KEYS = ['i2t_r1', 'i2t_r5', 'i2t_r10', 't2i_r1', 't2i_r5', 't2i_r10']


def run(d):
    if not os.path.exists(os.path.join(d, 'DONE.json')):
        return None
    ep = json.load(open(os.path.join(d, 'EPOCHS.json')))['cotrain']
    rs = [e['dev']['rsum'] for e in ep]
    assert len(ep) == 40 and [e['epoch'] for e in ep] == list(range(40)), d
    assert abs(max(rs) - json.load(open(os.path.join(d, 'DONE.json')))['best_rsum']) < 1e-6, d
    b = int(np.argmax(rs))
    return {'best': max(rs), 'epoch': b, 'last': rs[-1], 'last5': float(np.mean(rs[-5:])), **{k: ep[b]['dev'][k] for k in KEYS}}


res, missing = {}, []
for n in (40, 60):
    for s in SEEDS:
        cell = f'f{n}_s{s}'
        res[cell] = {'a1_iface': run(f'{A1}/{cell}/a1_iface')}
        for line, arms in ARMS.items():
            for a in arms:
                d = f'{ORIG[line]}/{cell}/{a}' if s == 1001 else f'{EXT}/{line}/runs/{cell}/{a}'
                r = run(d)
                res[cell][a] = r
                if r is None:
                    missing.append(f'{cell}/{a}')
all_arms = ['a1_iface'] + [a for arms in ARMS.values() for a in arms]
summ = {}
for n in (40, 60):
    summ[f'f{n}'] = {}
    for a in all_arms:
        v = [res[f'f{n}_s{s}'][a] for s in SEEDS]
        have = [(s, x) for s, x in zip(SEEDS, v) if x is not None]
        d = [res[f'f{n}_s{s}'][a]['best'] - res[f'f{n}_s{s}']['a1_iface']['best'] for s in SEEDS
             if res[f'f{n}_s{s}'][a] is not None and res[f'f{n}_s{s}']['a1_iface'] is not None] if a != 'a1_iface' else []
        summ[f'f{n}'][a] = {'n_seeds': len(have), 'seeds': [s for s, _ in have],
                            'mean_best': float(np.mean([x['best'] for _, x in have])) if have else None,
                            'mean_last5': float(np.mean([x['last5'] for _, x in have])) if have else None,
                            'minus_a1_per_seed': d, 'minus_a1_mean': float(np.mean(d)) if d else None,
                            'minus_a1_n_pos': int(sum(x > 1e-9 for x in d))}
json.dump({'scope': 'Flickr30K 40%/60% x seeds 1001-1003 full DEV best rSum; seed1001 original units, 1002/1003 extension; A1 same seed; no TEST',
           'missing': missing, 'runs': res, 'summary': summ}, open(os.path.join(HERE, 'SEEDEXT_SUMMARY.json'), 'w'), indent=1)
f = lambda v: 'NA' if v is None else f'{v:.2f}'
L = ['# Flickr 多 seed 数字表（R27-D / O / O3 / CS；自动生成；full DEV；未读 TEST；只列数字，不含解读）', '',
     'seed1001 为原单 seed 单元；seed1002/1003 为种子扩展（训练代码除种子门槛外逐字节相同）。A1 参照取同格同 seed 的 R27 a1_iface。未完成单元记 NA，不以其他臂替代；均值只在已完成的 seed 上计算并标注 n。R27-O / O3 为显式使用真对应的诊断，不进入主结果表。', '',
     f'未完成单元：{", ".join(missing) if missing else "无"}', '']
for n in (40, 60):
    L += [f'## Flickr {n}%：逐 seed 最佳 full-DEV rSum', '', '| 臂 | s1001 | s1002 | s1003 | n | 均值 best | 均值 last5 | −A1 逐 seed | −A1 均值 | >0 |', '|---|---|---|---|---|---|---|---|---|---|']
    for a in all_arms:
        v = [res[f'f{n}_s{s}'][a] for s in SEEDS]; m = summ[f'f{n}'][a]
        dm = m['minus_a1_mean']
        dm_s = '—' if a == 'a1_iface' else ('NA' if dm is None else '%+.2f' % dm)
        pos_s = '—' if a == 'a1_iface' else '%d/%d' % (m['minus_a1_n_pos'], len(m['minus_a1_per_seed']))
        per_s = ', '.join('%+.2f' % x for x in m['minus_a1_per_seed']) or '—'
        L.append('| %s | ' % a + ' | '.join(f(x['best'] if x else None) for x in v) +
                 ' | %d | %s | %s | %s | %s | %s |' % (m['n_seeds'], f(m['mean_best']), f(m['mean_last5']), per_s, dm_s, pos_s))
    L.append('')
open(os.path.join(HERE, 'SEEDEXT_TABLES.md'), 'w').write('\n'.join(L) + '\n')
print('\n'.join(L))
