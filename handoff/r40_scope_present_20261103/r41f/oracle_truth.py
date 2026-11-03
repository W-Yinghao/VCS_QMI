"""R39 Flickr oracle diagnostic (ORACLE_MISSING; package docs/SERVER_PLAN_CN.md section 5). Flickr30K precomp truth: FIT caption
record k belongs to image k//5 (pre-injection pairing). Importable ONLY inside the oracle runner process (train_oracle.py sets
R39_ORACLE_PROCESS=1); every other R39 process installs a meta-path blocker that refuses this import, and non-oracle runners assert
it is not loaded. Outputs live only under OUT/oracle_runs (refused by trainer processes) and can never initialise / label a
non-oracle unit.
  ORACLE_MISSING: only rows whose N label is all-null change: if the row's (natural target-H top-32) list holds >= 1 true slot,
  A = 0.5 * uniform(true slots in the list) + 0.5 * null; else unchanged. Rows with an observed positive keep their N label.
  No insertion, same lists/permutations/Q = Uniform(33) as N."""
import os
import sys
import numpy as np

if os.environ.get('R39_ORACLE_PROCESS') != '1' or not os.path.basename(sys.argv[0]).startswith('train_oracle'):
    raise ImportError('oracle_truth may only be imported by train_oracle.py')

N_TRAIN_CAP = 145000
NULL_ID = -1
REAL_MASS = 0.5


def truth_map():
    return np.arange(N_TRAIN_CAP, dtype=np.int64) // 5


def truth_masks(qi, caps, qc, imgs):
    """i2t: caption slot true iff caption//5 == query image; t2i: image slot true iff == query caption//5 (null slot never true)."""
    ti = ((caps // 5) == qi[:, None]) & (caps != NULL_ID)
    tt = (imgs == (qc // 5)[:, None]) & (imgs != NULL_ID)
    return ti, tt


def missing_relabel(a, ids, truth):
    a = np.asarray(a)   # keep the N label dtype (float64 from data36.null_lists): unchanged rows stay bitwise
    null = ids == NULL_ID
    assert (null.sum(1) == 1).all()
    miss = a[null] == 1.0
    n = truth.sum(1)
    sel = miss & (n > 0)
    out = a.copy()
    out[sel] = 0.0
    out[sel] = np.where(truth[sel], REAL_MASS / np.maximum(n[sel], 1)[:, None], 0.0)
    out[sel] = np.where(null[sel], 1.0 - REAL_MASS, out[sel])
    assert np.array_equal(out[~sel], a[~sel]) and np.allclose(out.sum(1), 1.0, atol=1e-6)
    return out, dict(rows=int(len(a)), null_rows=int(miss.sum()), relabelled_rows=int(sel.sum()), null_rows_no_true_in_list=int((miss & (n == 0)).sum()),
                     true_slots_used=int(truth[sel].sum()))


def apply(sup, lists):
    assert sup == 'ORACLE_MISSING', sup
    (qi, caps, ai), (qc, imgs, at) = lists
    ti, tt = truth_masks(qi, caps, qc, imgs)
    out, cnt = [], {}
    for dn, a, ids, t in (('i2t', ai, caps, ti), ('t2i', at, imgs, tt)):
        a2, c = missing_relabel(a, ids, t)
        out.append((a2, None, None))
        cnt.update({f'{dn}_{k}': float(x) for k, x in c.items()})
    return out, cnt
