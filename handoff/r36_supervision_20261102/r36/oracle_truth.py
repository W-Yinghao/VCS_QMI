"""R36 M-oracle ground truth (Flickr30K precomp: FIT caption record k belongs to image k//5) and the two oracle
diagnostic supervisions. Importable ONLY inside the oracle runner process (train_oracle.py sets
R36_ORACLE_PROCESS=1 before importing); every other R36 process installs a meta-path blocker that refuses this
import, and the non-oracle runner asserts the module is not loaded. Its outputs (masks, relabelled A, checkpoints)
live only in the oracle process and under OUT/oracle_runs, which non-oracle processes refuse to open.
  SHIELD : F = {slot: truth and A == 0}; P = A unchanged; Q uniform on the non-F slots (J/BCE); NCE drops F from
           the denominator. Wrong observed positives keep their mass. Every row keeps >= 1 valid slot (A>0 is never in F).
  RELABEL: if the list holds >= 1 true slot, A = uniform over the true slots (i2t: all true captions in the list;
           t2i: the unique true image); otherwise A unchanged (counted). Q uniform(32). No insertion/dropping."""
import os
import sys
import numpy as np

if os.environ.get('R36_ORACLE_PROCESS') != '1' or not os.path.basename(sys.argv[0]).startswith('train_oracle'):
    raise ImportError('oracle_truth may only be imported by train_oracle.py')

N_TRAIN_CAP = 145000


def truth_map():
    return np.arange(N_TRAIN_CAP, dtype=np.int64) // 5


def truth_masks(qi, caps, qc, imgs):
    """i2t: caption slot true iff caption//5 == query image; t2i: image slot true iff == query caption//5."""
    return (caps // 5) == qi[:, None], imgs == (qc // 5)[:, None]


def shield(a, truth):
    a = np.asarray(a, np.float64)
    F = truth & (a == 0)
    valid = ~F
    assert valid.any(1).all()
    q = valid / valid.sum(1, keepdims=True)
    return a, q, valid, dict(rows=int(len(a)), rows_with_shielded=int(F.any(1).sum()), shielded_slots=int(F.sum()))


def relabel(a, truth):
    a = np.asarray(a, np.float64)
    n = truth.sum(1, keepdims=True)
    new = np.where(n > 0, truth / np.maximum(n, 1), a)
    return new, dict(rows=int(len(a)), rows_relabelled=int((n[:, 0] > 0).sum()), rows_unchanged_no_true=int((n[:, 0] == 0).sum()))


def apply(sup, lists):
    """oracle supervision for one step: returns per-direction (a, q or None, valid or None) and counts."""
    (qi, caps, ai), (qc, imgs, at) = lists
    ti, tt = truth_masks(qi, caps, qc, imgs)
    out, cnt = [], {}
    for dn, a, t in (('i2t', ai, ti), ('t2i', at, tt)):
        if sup == 'SHIELD_ORACLE':
            a2, q, v, c = shield(a, t)
        elif sup == 'RELABEL_AVAILABLE_ORACLE':
            (a2, c), q, v = relabel(a, t), None, None
        else:
            raise ValueError(sup)
        out.append((a2.astype(np.float32), None if q is None else q.astype(np.float32), v))
        cnt.update({f'{dn}_{k}': x for k, x in c.items()})
    return out, cnt
