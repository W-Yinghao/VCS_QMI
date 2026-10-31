"""Aggregate the offline FIT correspondence-quality evaluations (fit_eval/FIT_EVAL_<cell>_<arm>.json, produced by
tools/fit_eval_offline.py on each finished R27-Q final state; truth only inside the evaluator) into one numbers-only table.
Each file scores all four weighting schemes (cos / vcs / bce / snce) on that unit's final state. Usage: aggregate_fit_eval.py"""
import json, glob, os
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KEYS = ['frac_true_in_bank', 'frac_true_in_top5', 'in_top5_alpha_mass_on_true', 'in_top5_alpha_argmax_hit', 'uniform_mass_on_true',
        'absent_max_alpha', 'absent_entropy', 'mean_entropy', 'mean_max_alpha', 'perm_alpha_maxdiff', 'query_swap_alpha_l1', 'mass_routeB']
rows, out = [], {}
for p in sorted(glob.glob(os.path.join(HERE, 'fit_eval', 'FIT_EVAL_*.json'))):
    d = json.load(open(p)); name = os.path.basename(p)[9:-5]
    out[name] = {k: d.get(k) for k in ('saved_arm', 'stage', 'epoch_reconstructed', 'n_batches', 'P', 'beta')}
    out[name]['scorers'] = d['scorers']
    for sc, m in d['scorers'].items():
        rows.append([name, sc] + [m.get(k) for k in KEYS])
json.dump(out, open(os.path.join(HERE, 'FIT_EVAL_SUMMARY.json'), 'w'), indent=1)
f = lambda v: 'NA' if v is None else (f'{v:.4f}' if isinstance(v, float) else str(v))
L = ['# R27-Q 离线 FIT 对应质量（自动生成；只列数字，不含解读）', '',
     '每个单元末轮状态上，固定 FIT batch，用同一候选（C-native，K=5）评价四种加权（cos=另一 peer 直接跨模态 cosine；vcs/bce/snce=该状态冻结 teacher 上按冻结配置训练的关系头）。真值只进入评价器。in_top5_* 只在真图位于 5 个候选内的 query 上计；absent_* 只在不在时计；uniform_mass_on_true=0.2 为均匀权重参照；perm_alpha_maxdiff 为候选顺序置换后的权重差；query_swap_alpha_l1 为换 query 后的权重变化。', '',
     '| 单元（末轮状态） | 加权 | ' + ' | '.join(KEYS) + ' |', '|---|---|' + '---|' * len(KEYS)]
L += ['| ' + ' | '.join(f(x) for x in r) + ' |' for r in rows]
open(os.path.join(HERE, 'FIT_EVAL_TABLES.md'), 'w').write('\n'.join(L) + '\n')
print(len(rows), 'rows')
