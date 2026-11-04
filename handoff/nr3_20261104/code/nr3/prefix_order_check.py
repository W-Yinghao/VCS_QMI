#!/usr/bin/env python3
"""Label-free check: DEV orders/scores of the folded prefix (separate evaluator run) vs its parent's registered eval_008000
predictions. Reads PREDICTIONS.npz only (written before truth); reports differing query rows and the score fold error."""
import argparse, json
import numpy as np
from nr1.io import read_json, write_json
from nr3.paths import OUT, cond_paths


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--cond', required=True); c = ap.parse_args().cond
    od = OUT/'prep'/c; rec = read_json(od/'PREP_RECEIPT.json'); a, b = rec['B']['fit']['a'], rec['B']['fit']['b']
    f = np.load(od/'eval_prefix/PREDICTIONS.npz'); p = np.load(cond_paths(c)['parent']('NCE')/'eval_008000/PREDICTIONS.npz')
    assert np.array_equal(f['pairs'], p['pairs'])
    err = np.abs(f['scores'] - (a * p['scores'] + b)); res = dict(condition=c, n_pairs=int(len(err)), max_abs_score_err=float(err.max()),
                                                                     max_rel_score_err=float((err / np.maximum(1, np.abs(a * p['scores'] + b))).max()))
    for k in ('i_order', 't_order'):
        diff = (f[k] != p[k]).any(1); res[f'{k}_rows_differing'] = int(diff.sum()); res[f'{k}_rows'] = int(len(diff))
    write_json(od/'PREFIX_ORDER_CHECK.json', res); print(json.dumps(res))


if __name__ == '__main__':
    main()
