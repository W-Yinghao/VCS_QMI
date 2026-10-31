"""Per-query rank transitions for the R27-S / R27-D / R27-O / R27-O3 / R27-CS comparisons (numbers only, no TEST).
Reads the audit-3 rank arrays OUT3/<cell>/<arm>/{i2t_rank,t2i_rank}.npy (0-based rank of the GT at each unit's selected
best full-DEV checkpoint, same fixed DEV order; A1/A0 from the original export, new arms from audit_ranks_x.py).
Buckets (1-based): 1 / 2-10 / >10. "new/ref" counts queries moving ref -> new; only pairs with both units exported.
Writes SDO_TRANSITIONS.json and SDO_TRANSITIONS.md. Usage: transitions_sdo.py"""
import json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT3 = '/projects/EEG-foundation-model/yinghao/outputs_v9/r27_audit_20261029/audit3_ranks'
CELLS = ('f40_s1001', 'f60_s1001')
A0 = 'H0_A0'
PAIRS = [('a1_iface', A0), ('s1_balanced_dual', 'a1_iface'), ('s2_single_free_vcs', 'a1_iface'), ('s3_single_cap_delta', 'a1_iface'),
         ('s4_single_cap_vcs', 'a1_iface'), ('s5_single_cap_bce', 'a1_iface'), ('s6_single_cap_snce', 'a1_iface'),
         ('s4_single_cap_vcs', 's6_single_cap_snce'), ('s2_single_free_vcs', 'a3_vcs'),
         ('d1_vcs_i2t', 'a1_iface'), ('d2_vcs_t2i', 'a1_iface'), ('d3_snce_i2t', 'a1_iface'), ('d4_snce_t2i', 'a1_iface'),
         ('d1_vcs_i2t', 'd3_snce_i2t'), ('d2_vcs_t2i', 'd4_snce_t2i'), ('s4_single_cap_vcs', 'd1_vcs_i2t'), ('s4_single_cap_vcs', 'd2_vcs_t2i'),
         ('s6_single_cap_snce', 'd3_snce_i2t'), ('s6_single_cap_snce', 'd4_snce_t2i'),
         ('o1_rescue_native_t2i', 'a1_iface'), ('o2_rescue_vcs_t2i', 'd2_vcs_t2i'), ('o2_rescue_vcs_t2i', 'o1_rescue_native_t2i'),
         ('o3_true_target_t2i', 'o1_rescue_native_t2i'), ('o3_true_target_t2i', 'a1_iface'),
         ('bn_cross_t2i', 'a1_iface'), ('cn_mix_t2i', 'a1_iface'), ('bn_cross_t2i', 'cn_mix_t2i')]
COLS = ['1->2-10', '1->>10', '2-10->1', '>10->1', '>10->2-10']


def bucket(r):
    return np.where(r == 0, 0, np.where(r < 10, 1, 2))


def trans(ref, new):
    a, b = bucket(ref), bucket(new)
    m = np.zeros((3, 3), dtype=np.int64); np.add.at(m, (a, b), 1)
    return {'1->2-10': int(m[0, 1]), '1->>10': int(m[0, 2]), '2-10->1': int(m[1, 0]), '>10->1': int(m[2, 0]), '>10->2-10': int(m[2, 1]),
            'r1_rescued': int(m[1:, 0].sum()), 'r1_lost': int(m[0, 1:].sum()), 'r1_net': int(m[1:, 0].sum() - m[0, 1:].sum()),
            'r1_ref': int(m[0].sum()), 'r1_new': int(m[:, 0].sum()), 'n': int(len(ref)), 'matrix_ref_rows_new_cols': m.tolist()}


def load(cell, arm, d):
    p = os.path.join(OUT3, cell, arm, f'{d}_rank.npy')
    return np.load(p) if os.path.exists(p) else None


res, missing = {}, []
for cell in CELLS:
    res[cell] = {}
    for new, ref in PAIRS:
        k = f'{new}/{ref}'
        e = {}
        for d in ('i2t', 't2i'):
            r, n = load(cell, ref, d), load(cell, new, d)
            if r is None or n is None:
                e = None; break
            assert r.shape == n.shape
            e[d] = trans(r, n)
        if e is None:
            missing.append(f'{cell}:{k}')
        res[cell][k] = e
json.dump({'scope': 'Flickr30K seed1001 full DEV (1,014 image queries i2t, 5,070 caption queries t2i), ranks at each unit best full-DEV checkpoint; no TEST',
           'missing_pairs': missing, 'transitions': res}, open(os.path.join(HERE, 'SDO_TRANSITIONS.json'), 'w'), indent=1)
L = ['# 逐 query 排名迁移（自动生成；只列计数，不含解读）', '',
     '来源：审计 3 保存的 rank（各单元 best full-DEV checkpoint，同一固定 DEV 顺序）。“新/参照”= 参照 → 新 的 query 数；分桶 1 / 2–10 / >10（1 起计）。i2t 每格 1,014 个图像 query，t2i 5,070 个 caption query。单 seed。', '',
     f'缺失比较（任一单元未完成或未导出）：{", ".join(missing) if missing else "无"}', '']
for cell in CELLS:
    L += [f'## {cell}', '', '| 新/参照 | 方向 | ' + ' | '.join(COLS) + ' | 救回 R@1 | 损失 R@1 | 净 | R@1 计数 参照 | R@1 计数 新 |', '|---|---|' + '---|' * (len(COLS) + 5)]
    for k, e in res[cell].items():
        if e is None:
            L.append(f'| {k} | — | ' + ' | '.join(['NA'] * (len(COLS) + 5)) + ' |'); continue
        for d in ('i2t', 't2i'):
            v = e[d]
            L.append(f'| {k} | {d} | ' + ' | '.join(str(v[c]) for c in COLS) + f" | {v['r1_rescued']} | {v['r1_lost']} | {v['r1_net']:+d} | {v['r1_ref']} | {v['r1_new']} |")
    L.append('')
open(os.path.join(HERE, 'SDO_TRANSITIONS.md'), 'w').write('\n'.join(L) + '\n')
print('\n'.join(L[:40]))
