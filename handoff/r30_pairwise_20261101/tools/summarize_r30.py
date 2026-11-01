"""R30 v1.1 + A1 number tables (Flickr30K full DEV; no TEST; numbers only, no interpretation).
Reads MATRIX_A1.csv and each unit's METRICS.jsonl (completed_cotrain_epochs 0 = parent boundary, 1..N) and DONE.json.
Stage table: fixed completed_cotrain_epochs=10 for every stage_view=yes path (not the best of the first 10 epochs), with
E0/E1/E2/E5/E10, E10-E0 and cumulative updates; checked against E10_RECORD.json. Stage comparisons per noise/seed:
HL-HN, HL-HB, LN-LL, HN-LN, HL-HH, I=(HL-HN)-(LL-LN); a formal mean over seeds only when all expected seeds are complete
(else NA), plus a separately labelled exploratory_partial_mean. Full table: every 40-epoch unit, best full-DEV rSum over
40 epochs with the six Recalls of that same epoch, final, last5; paired differences within the same noise/seed.
Unfinished = NA, never replaced. Writes R30_SUMMARY.json, R30_STAGE_E10.md, R30_FULL40.md. Usage: summarize_r30.py"""
import csv, json, os
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = '/projects/EEG-foundation-model/yinghao/outputs_v9/r30_pairwise_20261101/runs'
KEYS = ['i2t_r1', 'i2t_r5', 'i2t_r10', 't2i_r1', 't2i_r5', 't2i_r10']
rows = list(csv.DictReader(open(os.path.join(HERE, 'MATRIX_A1.csv'))))


def metrics(u):
    p = f'{RUNS}/{u}/METRICS.jsonl'
    if not os.path.exists(p):
        return {}
    out = {}
    for l in open(p):
        r = json.loads(l)
        out[int(r['completed_cotrain_epochs'])] = r
    return out


def path_label(r):
    if r['stage_view'] != 'yes':
        return None
    return r['warmup_source'] + {'H': 'H', 'L': 'L', 'N': 'N', 'B': 'B'}[r['continuation_objective']]


res = {}
for r in rows:
    u = r['unit_id']; m = metrics(u); n_ep = int(r['co_train_epochs'])
    done = os.path.exists(f'{RUNS}/{u}/DONE.json')
    e = {'unit_id': u, 'noise': int(r['noise_percent']), 'seed': int(r['seed']), 'source': r['warmup_source'],
         'objective': r['continuation_objective'], 'scale_rule': r['scale_rule'], 'epochs_planned': n_ep,
         'stage_path': path_label(r), 'done': done, 'epochs_logged': max([k for k in m if k > 0], default=0)}
    for k in (0, 1, 2, 5, 10):
        e[f'E{k}'] = m[k]['full_dev_rsum'] if k in m else None
    if 10 in m:
        e['E10_recalls'] = {k: m[10][k] for k in KEYS}; e['E10_updates'] = m[10]['updates_cumulative']
        rec = f'{RUNS}/{u}/E10_RECORD.json'
        if os.path.exists(rec):
            er = json.load(open(rec)); e['E10_record_matches'] = abs(er['dev']['rsum'] - m[10]['full_dev_rsum']) < 1e-9 and er['recomputed_equal']
    e['E10_minus_E0'] = e['E10'] - e['E0'] if e['E10'] is not None and e['E0'] is not None else None
    if n_ep == 40 and done:
        ep = [m[k] for k in range(1, 41)]
        rs = [x['full_dev_rsum'] for x in ep]; b = int(np.argmax(rs))
        dj = json.load(open(f'{RUNS}/{u}/DONE.json'))
        assert dj['epochs_done'] == 40 and abs(dj['best_rsum'] - max(rs)) < 1e-6, u
        e['full'] = {'best': max(rs), 'best_completed_epoch': b + 1, **{k: ep[b][k] for k in KEYS}, 'final': rs[-1],
                     'last5': float(np.mean(rs[-5:]))}
    res[u] = e

# stage comparisons
stage = {}
for (nz, sd) in sorted({(e['noise'], e['seed']) for e in res.values() if e['stage_path']}):
    P = {e['stage_path']: e['E10'] for e in res.values() if e['stage_path'] and e['noise'] == nz and e['seed'] == sd}
    g = lambda a, b: P[a] - P[b] if P.get(a) is not None and P.get(b) is not None else None
    c = {'HL-HN': g('HL', 'HN'), 'HL-HB': g('HL', 'HB'), 'LN-LL': g('LN', 'LL'), 'HN-LN': g('HN', 'LN'), 'HL-HH': g('HL', 'HH')}
    c['I=(HL-HN)-(LL-LN)'] = c['HL-HN'] + c['LN-LL'] if c['HL-HN'] is not None and c['LN-LL'] is not None else None
    stage[f'f{nz:02d}_s{sd}'] = {'paths_E10': P, 'comparisons': c}
seeds60 = [k for k in stage if k.startswith('f60')]
stage_means = {}
for comp in ['HL-HN', 'HL-HB', 'LN-LL', 'HN-LN', 'HL-HH', 'I=(HL-HN)-(LL-LN)']:
    v = [stage[k]['comparisons'][comp] for k in seeds60]
    have = [x for x in v if x is not None]
    stage_means[comp] = {'expected_n': len(seeds60), 'n_complete': len(have),
                         'formal_mean': float(np.mean(have)) if len(have) == len(seeds60) and have else None,
                         'exploratory_partial_mean': float(np.mean(have)) if have and len(have) < len(seeds60) else None}

# full-method comparisons within a noise/seed
full_pairs = {}
FP = [('Ls', 'L0'), ('PV', 'Ls'), ('PV', 'PN'), ('PV', 'PB'), ('PN', 'H0'), ('PV', 'H0'), ('PB', 'H0'), ('L0', 'H0'), ('Ls', 'H0'),
      ('L0', 'HN'), ('L0', 'HB'), ('HN', 'H0'), ('HB', 'H0'), ('PV', 'HH'), ('PN', 'HH'), ('PB', 'HH')]
for (nz, sd) in sorted({(e['noise'], e['seed']) for e in res.values() if e['epochs_planned'] == 40}):
    def arm(e):
        if e['objective'] == 'H':
            return 'H0' if e['unit_id'].startswith('r30_') else 'HH'
        if e['objective'] == 'L':
            return 'Ls' if e['scale_rule'] == 'fit_static' else 'L0'
        if e['objective'] in ('N', 'B'):
            return 'H' + e['objective']
        return e['objective']
    A = {arm(e): e.get('full') for e in res.values() if e['epochs_planned'] == 40 and e['noise'] == nz and e['seed'] == sd}
    full_pairs[f'f{nz:02d}_s{sd}'] = {f'{a}-{b}': ({m: A[a][m] - A[b][m] for m in ['best', 'last5', 'final', 'i2t_r1', 't2i_r1']}
                                                   if A.get(a) and A.get(b) else None) for a, b in FP if a in A and b in A}
missing = [u for u, e in res.items() if not e['done']]
json.dump({'scope': 'R30 v1.1 + A1, Flickr30K full DEV; stage table at fixed completed_cotrain_epochs=10; full table = best of 40 epochs; no TEST',
           'not_finished': missing, 'units': res, 'stage': stage, 'stage_60_means': stage_means, 'full_pairs': full_pairs},
          open(os.path.join(HERE, 'R30_SUMMARY.json'), 'w'), indent=1)
f = lambda v: 'NA' if v is None else f'{v:.2f}'
g = lambda v: 'NA' if v is None else f'{v:+.2f}'
L = ['# R30 阶段表：固定 completed_cotrain_epochs=10（自动生成；full DEV；未读 TEST；只列数字，不含解读）', '',
     '每条阶段路径读第 10 个完成的 co-train 轮（raw_epoch 9）的 full-DEV rSum，不是前 10 轮 best；E0 为父状态边界。H→L 与 H→H 在 seed1001 取 40 轮单元 L0/H0 的 E10，在 seed1002/1003 的 H→H 取 40 轮对照的 E10。未完成记 NA。60% 跨 seed 正式均值仅在三 seed 齐全时给出，否则为 NA；部分均值单列为 exploratory_partial_mean。', '',
     f'未完成单元：{", ".join(missing) if missing else "无"}', '',
     '| unit | 噪声 | seed | 路径 | E0 | E1 | E2 | E5 | E10 | E10−E0 | E10 累计更新 | E10 i2t R@1 | E10 t2i R@1 |', '|---|---|---|---|---|---|---|---|---|---|---|---|---|']
for u, e in sorted(res.items(), key=lambda x: (x[1]['noise'], x[1]['seed'], x[1]['stage_path'] or 'z')):
    if not e['stage_path']:
        continue
    rr = e.get('E10_recalls') or {}
    L.append(f"| {u} | {e['noise']}% | {e['seed']} | {e['source']}→{e['objective']} | {f(e['E0'])} | {f(e['E1'])} | {f(e['E2'])} | {f(e['E5'])} | {f(e['E10'])} | {g(e['E10_minus_E0'])} | {e.get('E10_updates', 'NA')} | {f(rr.get('i2t_r1'))} | {f(rr.get('t2i_r1'))} |")
L += ['', '## 阶段比较（同噪声、同 seed 配对，E10 rSum 差）', '', '| 格 | HL−HN | HL−HB | LN−LL | HN−LN | HL−HH | I=(HL−HN)−(LL−LN) |', '|---|---|---|---|---|---|---|']
for k, v in stage.items():
    c = v['comparisons']
    L.append(f"| {k} | " + ' | '.join(g(c[x]) for x in ['HL-HN', 'HL-HB', 'LN-LL', 'HN-LN', 'HL-HH', 'I=(HL-HN)-(LL-LN)']) + ' |')
L += ['', '## 60% 跨 seed（正式均值三 seed 齐全才给出）', '', '| 比较 | 已完成/应有 | 正式均值 | exploratory_partial_mean |', '|---|---|---|---|']
for k, v in stage_means.items():
    L.append(f"| {k} | {v['n_complete']}/{v['expected_n']} | {g(v['formal_mean'])} | {g(v['exploratory_partial_mean'])} |")
open(os.path.join(HERE, 'R30_STAGE_E10.md'), 'w').write('\n'.join(L) + '\n')
L = ['# R30 完整表：40 轮（自动生成；full DEV；未读 TEST；只列数字，不含解读）', '',
     '40 轮内 best full-DEV rSum 及同一 epoch 的六项 Recall、final、last5（epoch 记为 completed_cotrain_epochs）。H0/L0/Ls/PN/PV/PB 从共同 H 父状态出发；HN/HB 为 seed1001 延长到 40 轮的阶段路径（修订 A1）；60% seed1002/1003 的 PN/PV/PB 与 HH 为修订 A1 的配对确认。Ls/PN/PV/PB 系数见 COEFFICIENTS.json。未完成记 NA。', '',
     '| unit | 噪声 | seed | 臂 | best | best epoch | i2t R@1 | i2t R@5 | i2t R@10 | t2i R@1 | t2i R@5 | t2i R@10 | final | last5 |', '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
for u, e in sorted(res.items(), key=lambda x: (x[1]['noise'], x[1]['seed'], x[0])):
    if e['epochs_planned'] != 40:
        continue
    F = e.get('full')
    lab = e['source'] + '→' + e['objective'] + ('(s)' if e['scale_rule'] == 'fit_static' else '')
    L.append(f"| {u} | {e['noise']}% | {e['seed']} | {lab} | NA |" + ' |' * 9 if not F else
             f"| {u} | {e['noise']}% | {e['seed']} | {lab} | {f(F['best'])} | {F['best_completed_epoch']} | " + ' | '.join(f(F[k]) for k in KEYS) + f" | {f(F['final'])} | {f(F['last5'])} |")
L += ['', '## 同格配对差（best / last5 / final / i2t R@1 / t2i R@1）', '', '| 格 | 比较 | best | last5 | final | i2t R@1 | t2i R@1 |', '|---|---|---|---|---|---|---|']
for k, v in full_pairs.items():
    for c, d in v.items():
        L.append(f'| {k} | {c} | ' + (' | '.join(g(d[m]) for m in ['best', 'last5', 'final', 'i2t_r1', 't2i_r1']) if d else 'NA | NA | NA | NA | NA') + ' |')
open(os.path.join(HERE, 'R30_FULL40.md'), 'w').write('\n'.join(L) + '\n')
print('not finished', len(missing))
