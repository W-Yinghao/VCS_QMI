"""r7 R7-30 D1 — nested visual-token mask mechanisms (pure numpy; package directions/D1_token_evidence §B–C).

LLaVA-1.5 feeds its 576 visual tokens to the LLM as a 24 × 24 raster (row-major).  A mechanism draws ONE ordering of the 576 tokens per
(question, trial); the kept set at a budget is the ordering's prefix, so the compressed observations are nested (A_k = g_k(B), and
A_12.5 ⊂ A_25 ⊂ A_50 ⊂ A_75 ⊂ B).  Budgets 12.5 / 25 / 50 / 75 / 100 % = 72 / 144 / 288 / 432 / 576 tokens.  Kept tokens stay in raster order.
Three fixed trials per question, one per mechanism (package: "每个训练问题固定3种mask试验"; no mask search):
  random  uniform random ordering;                                   log P(kept set at k) = −log C(576, k)
  grid    the four 2-stride lattices (144 tokens each) in a random order, the 72-token set a checkerboard of the first lattice's 12 × 12
          coarse grid; within-block order random;                    log P(set) = −log 8, −log 4, −log 6, −log 4 at 72 / 144 / 288 / 432
  qsim    Plackett–Luce (Gumbel top-k) on z-scored question similarity scores (temperature 1); log P of the ORDERED prefix is recorded
          (the set probability of a Plackett–Luce prefix has no cheap closed form)
Seeds come from sha256 of (trial, question id), never from a process-local hash.
"""
from __future__ import annotations

import hashlib
import math

import numpy as np

N, GRID = 576, 24
BUDGETS = (72, 144, 288, 432)                                       # 12.5 / 25 / 50 / 75 %; 576 = full, 0 = text-only diagnostic
FRACTIONS = {72: 0.125, 144: 0.25, 288: 0.5, 432: 0.75, 576: 1.0}
MECHANISMS = ("random", "grid", "qsim")
SEED = 20261012


def rng_for(mechanism: str, qid: str) -> np.random.Generator:
    h = int(hashlib.sha256(f"r7-30/d1-mask/{mechanism}/{qid}".encode()).hexdigest()[:16], 16)
    return np.random.default_rng([SEED, h])


def lattice(ox: int, oy: int) -> np.ndarray:
    r, c = np.divmod(np.arange(N), GRID)
    return np.flatnonzero((r % 2 == oy) & (c % 2 == ox))


def order_random(qid: str):
    order = rng_for("random", qid).permutation(N)
    logp = {k: -(math.lgamma(N + 1) - math.lgamma(k + 1) - math.lgamma(N - k + 1)) for k in BUDGETS}
    return order.astype(np.int64), logp


def order_grid(qid: str):
    g = rng_for("grid", qid); offs = [(0, 0), (1, 1), (0, 1), (1, 0)]; first, *rest = [offs[i] for i in g.permutation(4)]
    lat = lattice(*first); r, c = np.divmod(lat, GRID); parity = int(g.integers(2))
    sub = lat[((r // 2) + (c // 2)) % 2 == parity]; other = np.setdiff1d(lat, sub)
    blocks = [sub, other] + [lattice(*o) for o in rest]
    order = np.concatenate([b[g.permutation(len(b))] for b in blocks])
    return order.astype(np.int64), {72: -math.log(8), 144: -math.log(4), 288: -math.log(6), 432: -math.log(4)}


def order_qsim(qid: str, scores: np.ndarray):
    """Gumbel top-k == Plackett–Luce sampling with weights exp(z); returns the ordering and log P of each ordered prefix."""
    s = np.asarray(scores, dtype=np.float64)
    if s.shape != (N,) or not np.isfinite(s).all():
        raise ValueError("qsim needs 576 finite scores")
    z = (s - s.mean()) / (s.std() + 1e-12)
    g = rng_for("qsim", qid); order = np.argsort(-(z + g.gumbel(size=N)), kind="stable")
    return order.astype(np.int64), pl_prefix_logp(z, order, BUDGETS)


def pl_prefix_logp(z, order, ks):
    """log P(first k items of a Plackett–Luce draw equal order[:k], in that order), weights exp(z)."""
    zo = np.asarray(z, dtype=np.float64)[np.asarray(order)]; m = zo.max()
    tail = np.cumsum(np.exp(zo - m)[::-1])[::-1]                                      # tail[j] = Σ_{t ≥ j} exp(z_{π_t} − m)
    cum = np.cumsum(zo - m - np.log(tail))
    return {k: float(cum[k - 1]) for k in ks}


def orderings(qid: str, qsim_scores: np.ndarray):
    """The three fixed trials of one question: {mechanism: (ordering[576], {budget: log-prob})}."""
    return {"random": order_random(qid), "grid": order_grid(qid), "qsim": order_qsim(qid, qsim_scores)}


def kept(order: np.ndarray, k: int) -> np.ndarray:
    """Kept token indices at budget k, in raster order (the order the LLM sees them)."""
    return np.sort(np.asarray(order[:k]))
