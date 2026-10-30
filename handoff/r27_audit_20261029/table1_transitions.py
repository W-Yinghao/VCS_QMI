"""R27 audit table 1 (SERVER_ADJUSTMENTS_CN.md §4 表一): per-query DEV rank transitions between paired arms. CPU only.

Reads the saved audit-3 arrays OUT3/<cell>/<arm>/{i2t_rank,t2i_rank}.npy (0-based ranks of the selected best full-DEV
checkpoints, same fixed DEV order for every unit). Rank buckets (1-based): '1' (rank0 == 0), '2-10' (1 <= rank0 <= 9),
'>10' (rank0 >= 10). For a pair (new, ref) and a seed-paired cell, counts of queries ref-bucket -> new-bucket:
1->2-10, 1->>10, 2-10->1, >10->1, >10->2-10 (plus the full 3x3 matrix). i2t: 1,014 image queries; t2i: 5,070 caption
queries. Per cell, per noise level (sum over seeds) and overall. A pair is reported for a cell only when both units exist.
Usage: table1_transitions.py [OUT3] [OUT_JSON]
"""
import os, sys, json
import numpy as np

OUT3 = sys.argv[1] if len(sys.argv) > 1 else '/projects/EEG-foundation-model/yinghao/outputs_v9/r27_audit_20261029/audit3_ranks'
OUTJ = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(os.path.abspath(__file__)), 'TABLE1.json')
CELLS = ('f40_s1001', 'f40_s1002', 'f40_s1003', 'f60_s1001', 'f60_s1002', 'f60_s1003')
NAME = {'H0_A0': 'A0', 'a1_iface': 'A1', 'a2_mean': 'A2', 'a3_vcs': 'A3', 'a4_bce': 'A4', 'a5_snce': 'A5', 'a6_delta': 'A6'}
PAIRS = (('a1_iface', 'H0_A0'), ('a3_vcs', 'a1_iface'), ('a2_mean', 'a1_iface'), ('a3_vcs', 'a6_delta'), ('a3_vcs', 'a4_bce'), ('a3_vcs', 'a5_snce'))
BUCKETS = ('1', '2-10', '>10')
KEYS = ('1->2-10', '1->>10', '2-10->1', '>10->1', '>10->2-10')


def bucket(r):
    r = np.asarray(r)
    return np.where(r == 0, 0, np.where(r <= 9, 1, 2))


def transitions(ref, new):
    """ref, new: 0-based rank arrays over the same queries. Returns dict of the five counts + 3x3 matrix."""
    a, b = bucket(ref), bucket(new)
    assert a.shape == b.shape
    m = np.zeros((3, 3), dtype=np.int64)
    np.add.at(m, (a, b), 1)
    out = {'1->2-10': int(m[0, 1]), '1->>10': int(m[0, 2]), '2-10->1': int(m[1, 0]), '>10->1': int(m[2, 0]), '>10->2-10': int(m[2, 1]),
           'matrix_ref_rows_new_cols': m.tolist(), 'n': int(a.size), 'r1_ref': int((a == 0).sum()), 'r1_new': int((b == 0).sum())}
    return out


def add(acc, t):
    for k in KEYS + ('n', 'r1_ref', 'r1_new'):
        acc[k] = acc.get(k, 0) + t[k]
    acc['n_cells'] = acc.get('n_cells', 0) + 1
    return acc


def main():
    res = dict(source=OUT3, buckets='rank (1-based) 1 / 2-10 / >10', pairs={})
    for new, ref in PAIRS:
        pn = f'{NAME[new]}/{NAME[ref]}'
        P = dict(per_cell={}, per_noise={}, total={})
        for d in ('i2t', 't2i'):
            for cell in CELLS:
                fa, fb = os.path.join(OUT3, cell, ref, f'{d}_rank.npy'), os.path.join(OUT3, cell, new, f'{d}_rank.npy')
                if not (os.path.exists(fa) and os.path.exists(fb)):
                    continue
                t = transitions(np.load(fa), np.load(fb))
                P['per_cell'].setdefault(cell, {})[d] = t
                add(P['per_noise'].setdefault(cell[:3], {}).setdefault(d, {}), t)
                add(P['total'].setdefault(d, {}), t)
        res['pairs'][pn] = P
    json.dump(res, open(OUTJ, 'w'), indent=1)
    print(json.dumps({k: {d: v['total'].get(d) for d in ('i2t', 't2i')} for k, v in res['pairs'].items()}, indent=1))


if __name__ == '__main__':
    main()
