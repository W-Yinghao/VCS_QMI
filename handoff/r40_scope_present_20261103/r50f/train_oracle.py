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
    ORACLE_RUNS.mkdir(parents=True, exist_ok=True)
    # R50: per job / per cond diagnostics (no shared file between parallel oracle jobs); exposure of the ORACLE_MISSING rule over the full FIT pool under
    # the N query distribution (record draws: i2t query = observed image of the record, t2i query = the record's caption)
    from common import ROOT, read_json
    _units = sys.argv[sys.argv.index('--units') + 1].split(',') if '--units' in sys.argv else []
    _diag = {}
    for _u in _units:
        _cond = read_json(ROOT / 'configs' / f'{_u}.json')['cond']
        if _cond in _diag:
            continue
        _c = CONDS[_cond]; obs = np.load(_c['noise_file']).astype(np.int64)
        _z = np.load(Path(_c['prep_dir']) / f"candidates_{_c['prep_key']}.npz"); ci = _z['i2t_top32'].astype(np.int64); ct = _z['t2i_top32'].astype(np.int64)
        _img = np.arange(len(ci))
        pres_i = (obs[ci] == _img[:, None]).any(1); true_i = (truth[ci] == _img[:, None]).any(1)
        pres_t = (ct == obs[:, None]).any(1); true_t = (ct == truth[:, None]).any(1)
        acc_i = (~pres_i & true_i)[obs]; acc_t = ~pres_t & true_t
        _diag[_cond] = dict(noise_file=_c['noise_file'], observed_mismatch_fraction_full_pool=float((obs != truth).mean()),
                            true_partner_in_top32=dict(i2t=float(true_i[obs].mean()), t2i=float(true_t.mean()), i2t_per_image=float(true_i.mean())),
                            observed_partner_in_top32=dict(i2t=float(pres_i[obs].mean()), t2i=float(pres_t.mean())),
                            null_rows=dict(i2t=float((~pres_i)[obs].mean()), t2i=float((~pres_t).mean())),
                            relabelled_rows=dict(i2t=float(acc_i.mean()), t2i=float(acc_t.mean())),
                            relabelled_fraction_of_null_rows=dict(i2t=float(acc_i.sum() / max(1, (~pres_i)[obs].sum())), t2i=float(acc_t.sum() / max(1, (~pres_t).sum()))),
                            mass_added_per_relabelled_row=oracle_truth.REAL_MASS,
                            expected_added_true_mass_per_row=dict(i2t=float(oracle_truth.REAL_MASS * acc_i.mean()), t2i=float(oracle_truth.REAL_MASS * acc_t.mean())))
    write_json(ORACLE_RUNS / f"ORACLE_DIAGNOSTICS_R50_{os.environ.get('SLURM_JOB_ID', 'local')}.json", dict(utc=utc(), units=_units, conds=_diag,
               note='evaluator statistics computed in the oracle process only (R50: per job file; full FIT pool, record-induced query distribution)'))
    train_unit.main(oracle=oracle_truth)
