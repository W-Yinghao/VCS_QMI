"""R39: R38 data36.py + COMPLETE_XCHECK / ORACLE_MISSING in NULL_SUPS (identical N lists; labels changed only by the R39 completion hook / the oracle process). R37C: byte-level port of R37 data36.py (only this note and the presence rule text changed; constants come from R37C common: 150000 records, identity map). R37 (copy of R36 data36.py + TRUST_NATIVE_NULL and N_PRESENT_ONLY). R36 candidate-list samplers (non-oracle; benchmark truth is never read here).
Per step and direction 16 query records; the i2t query is the OBSERVED image of the record, the t2i query is the
record's caption (R35 C convention). Draw order on the main stream is identical to R35 data35.sample_lists
(rec_i, rec_t, perm_i, perm_t), so for the same seed list FORCED / RANDOM / SHIELD / RELABEL see exactly the R35 C
queries and slot permutations.

  FORCED  (R35 C, also T, M-SHIELD/M-RELABEL bases): observed positive + first 31 other frozen-H top-32 candidates.
  RANDOM  (M-RANDOM): observed positive + 31 identities drawn uniformly without replacement from the opposite FIT
          identity pool excluding the forced partner (i2t: captions 0..144999; t2i: images 0..28999), from a
          separate candidate stream (rng2) so queries/permutations stay those of C. Excludes by identity only.
  NULL    (N): the unmodified frozen-H top-32 real candidates of the query (no insertion, no replacement, no
          exclusion) + one symbolic null slot; A = observed positives in the list (i2t: every caption whose observed
          image is the query image, uniform; t2i: the observed image), or A_null = 1 if none; all 33 slots permuted.
A = observed positives (row-normalised); TRUST_HARD anchors come from the frozen R35 B-TRUST subset only.
R37:
  TRUST_NATIVE_NULL (B1): anchors (rec_i, rec_t) drawn from the parent's frozen B-TRUST subset exactly as TRUST_HARD,
          lists = the full N protocol (NULL: native top-32, no insertion, 33 slots incl. observed-missing null);
          positive labels from the full-pool observed map (unchanged N label definition).
  N_PRESENT_ONLY (B2): N lists of all-pool queries; rows whose observed partner is absent from the native top-32
          (A_null = 1) are masked to zero loss and not redrawn; per direction normalised by the fixed training-pool
          presence probability p_d (presence_probs). Same 33-slot output and Q = Uniform(33) as N."""
from __future__ import annotations
import numpy as np
from data35 import build_lists, N_SLOTS, N_Q
from common import N_TRAIN_CAP, N_TRAIN_IMG

NULL_SUPS = ('OBS_MISSING_NULL', 'TRUST_NATIVE_NULL', 'N_PRESENT_ONLY', 'COMPLETE_XCHECK', 'ORACLE_MISSING')   # R39: + COMPLETE (fixed cross-checked completion of observed-missing rows) and the Flickr oracle diagnostic (same N lists)
TRUST_SUPS = ('TRUST_HARD', 'TRUST_NATIVE_NULL')

NULL_ID = -1
K_NULL = N_SLOTS + 1


def draw_main(rng, n, k, pool=None):
    """rec_i, rec_t, perm_i, perm_t in R35 order. pool: anchor record pool (TRUST) or None (all records)."""
    if pool is None:
        rec_i = rng.integers(0, N_TRAIN_CAP, n); rec_t = rng.integers(0, N_TRAIN_CAP, n)
    else:
        rec_i = pool[rng.integers(0, len(pool), n)]; rec_t = pool[rng.integers(0, len(pool), n)]
    perm_i = np.stack([rng.permutation(k) for _ in range(n)]); perm_t = np.stack([rng.permutation(k) for _ in range(n)])
    return rec_i, rec_t, perm_i, perm_t


def random_lists(rec_i, rec_t, obs, perm_i, perm_t, rng2):
    n = len(rec_i)
    qi = obs[rec_i]
    caps = np.zeros((n, N_SLOTS), np.int64); imgs = np.zeros((n, N_SLOTS), np.int64)
    for r in range(n):
        o = rng2.choice(N_TRAIN_CAP - 1, N_SLOTS - 1, replace=False); o = o + (o >= rec_i[r])
        caps[r, 0] = rec_i[r]; caps[r, 1:] = o
    pi = obs[rec_t]
    for r in range(n):
        o = rng2.choice(N_TRAIN_IMG - 1, N_SLOTS - 1, replace=False); o = o + (o >= pi[r])
        imgs[r, 0] = pi[r]; imgs[r, 1:] = o
    caps = np.take_along_axis(caps, perm_i, 1); imgs = np.take_along_axis(imgs, perm_t, 1)
    ai = (obs[caps] == qi[:, None]).astype(np.float32); ai /= ai.sum(1, keepdims=True)
    at = (imgs == pi[:, None]).astype(np.float32); at /= at.sum(1, keepdims=True)
    return (qi, caps, ai), (rec_t.copy(), imgs, at)


def null_lists(rec_i, rec_t, obs, cand_i2t, cand_t2i, perm_i, perm_t):
    """33-slot lists; id NULL_ID marks the null slot (after permutation). A rows sum to 1."""
    qi = obs[rec_i]
    caps = np.concatenate([cand_i2t[qi].astype(np.int64), np.full((len(qi), 1), NULL_ID)], 1)
    imgs = np.concatenate([cand_t2i[rec_t].astype(np.int64), np.full((len(rec_t), 1), NULL_ID)], 1)
    pi = obs[rec_t]
    mi = np.zeros(caps.shape, bool); mi[:, :N_SLOTS] = obs[caps[:, :N_SLOTS]] == qi[:, None]
    mt = np.zeros(imgs.shape, bool); mt[:, :N_SLOTS] = imgs[:, :N_SLOTS] == pi[:, None]
    def tgt(m):
        n = m.sum(1, keepdims=True); a = m.astype(np.float32) / np.maximum(n, 1)
        a[:, N_SLOTS] = (n[:, 0] == 0).astype(np.float32); return a
    ai, at = tgt(mi), tgt(mt)
    caps = np.take_along_axis(caps, perm_i, 1); ai = np.take_along_axis(ai, perm_i, 1)
    imgs = np.take_along_axis(imgs, perm_t, 1); at = np.take_along_axis(at, perm_t, 1)
    return (qi, caps, ai), (rec_t.copy(), imgs, at)


def sample(sup, rng, rng2, d, n=N_Q, pool=None):
    """returns lists ((qi, caps, ai), (qc, imgs, at)) and the drawn indices (rec_i, rec_t, perm_i, perm_t)."""
    k = K_NULL if sup in NULL_SUPS else N_SLOTS
    rec_i, rec_t, perm_i, perm_t = draw_main(rng, n, k, pool)
    obs = d['obs_np']
    if sup in NULL_SUPS:
        lists = null_lists(rec_i, rec_t, obs, d['cand_i2t'], d['cand_t2i'], perm_i, perm_t)
    elif sup == 'RANDOM':
        lists = random_lists(rec_i, rec_t, obs, perm_i, perm_t, rng2)
    else:   # OBS_HARD (C / V), TRUST_HARD, SHIELD_ORACLE, RELABEL_AVAILABLE_ORACLE: R35 forced lists
        lists = build_lists(rec_i, rec_t, obs, d['cand_i2t'], d['cand_t2i'], perm_i, perm_t)
    return lists, (rec_i, rec_t, perm_i, perm_t)


def locked_null_lists(d, key_i, key_t):
    """N readout on the R35 locked FIT list-eval records (lev_*): same records, native top-32 + null, fixed
    permutation derived from the locked 32-slot permutation (null kept last: J is permutation invariant)."""
    L = d['locks']
    k = np.arange(K_NULL)[None].repeat(len(L[key_i]), 0)
    return null_lists(L[key_i], L[key_t], d['obs_np'], d['cand_i2t'], d['cand_t2i'], k, k[:len(L[key_t])])


def presence_probs(d):
    """fixed training-pool probability that the observed partner is in the native top-32 list, per direction, under the
    N query distribution (uniform record draws): i2t query = observed image of the record (present iff some top-32
    caption has that observed image); t2i query = the record's caption (present iff its observed image is in top-32).
    Observed map + frozen candidate tables only (no truth)."""
    obs = d['obs_np']; ci = d['cand_i2t'].astype(np.int64); ct = d['cand_t2i'].astype(np.int64)
    img_present = (obs[ci] == np.arange(N_TRAIN_IMG)[:, None]).any(1)
    cap_present = (ct == obs[:, None]).any(1)
    return dict(p_i2t=float(img_present[obs].mean()), p_t2i=float(cap_present.mean()),
                p_i2t_per_image=float(img_present.mean()), n_records=int(len(obs)),
                rule='p_d = mean over the 150000 CC152K FIT records of [observed partner in native top-32] for the query that record induces')


def valid_rows(lists):
    """N_PRESENT_ONLY mask: rows with an observed positive among the real slots (A_null == 0)."""
    (qi, caps, ai), (qc, imgs, at) = lists
    return (ai[caps == NULL_ID] == 0), (at[imgs == NULL_ID] == 0)
