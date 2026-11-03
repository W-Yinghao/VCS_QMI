"""R39 Flickr oracle runner (separate process; ORACLE_MISSING units only). Sets the oracle-process flag, imports the truth
module, refuses every non-oracle unit (train_unit.main(oracle=...) asserts cfg['oracle']), runs the shared R39 trainer with the
oracle label hook; outputs go to OUT/oracle_runs/<unit> (refused by non-oracle processes).
Usage: python -B train_oracle.py --units u1,u2,...   |   python -B train_oracle.py --selftest"""
import os
import sys
from pathlib import Path

os.environ['R39_ORACLE_PROCESS'] = '1'
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from common import ORACLE_RUNS, CONDS, install_guards, write_json, utc
import oracle_truth
import train_unit


def selftest():
    """CPU: toy rows (i2t multi-true, t2i single true, observed-present rows, null rows without true, permuted null slot)."""
    import json
    caps = np.array([[10, 11, -1, 99], [10, -1, 12, 13], [20, 21, 22, -1], [-1, 5, 6, 7]])
    a = np.array([[0, 0, 1, 0], [.5, 0, .5, 0], [0, 0, 0, 1], [1, 0, 0, 0]], np.float32)
    qi = np.array([2, 2, 2, 9])
    ti = ((caps // 5) == qi[:, None]) & (caps != -1)
    out, c = oracle_truth.missing_relabel(a, caps, ti)
    ok = dict(row0=bool(np.allclose(out[0], [.25, .25, .5, 0])), row1_present_unchanged=bool(np.array_equal(out[1], a[1])),
              row2_no_true_unchanged=bool(np.array_equal(out[2], a[2])), row3_no_true_unchanged=bool(np.array_equal(out[3], a[3])),
              counts=c)
    imgs = np.array([[3, -1, 4], [-1, 3, 8]]); at = np.array([[0, 1, 0], [1, 0, 0]], np.float32); qc = np.array([17, 50])   # row 1: true image 10 not in the list -> unchanged
    ti2, tt2 = oracle_truth.truth_masks(qi[:2], caps[:2], qc, imgs)
    o2, _ = oracle_truth.missing_relabel(at, imgs, tt2)
    ok['t2i_single_true'] = bool(np.allclose(o2[0], [.5, .5, 0]) and np.array_equal(o2[1], at[1]))
    ok['ALL_PASS'] = all(v for k, v in ok.items() if isinstance(v, bool))
    print('R39_ORACLE_SELFTEST ' + json.dumps(ok, default=int))
    return ok


if __name__ == '__main__' and '--selftest' in sys.argv:
    install_guards(role='oracle')
    selftest()
elif __name__ == '__main__':
    install_guards(role='oracle')
    truth = oracle_truth.truth_map()
    obs = np.load(CONDS['n60']['noise_file']).astype(np.int64)
    ORACLE_RUNS.mkdir(parents=True, exist_ok=True)
    write_json(ORACLE_RUNS / 'ORACLE_DIAGNOSTICS.json', dict(utc=utc(), observed_mismatch_fraction_full_pool=float((obs != truth).mean()),
               note='evaluator statistic computed in the oracle process only'))
    train_unit.main(oracle=oracle_truth)
