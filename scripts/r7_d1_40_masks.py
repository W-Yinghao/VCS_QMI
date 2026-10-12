"""r7 R7-40a — the D1-10 nested visual-token masks on Qwen2.5-VL's merged token grid (pure numpy; plan reports/R7/R7_40_PLAN_DRAFT.yaml).

Qwen2.5-VL feeds the LLM one merged token per 2 × 2 block of 14-px patches: an image resized to H × W (multiples of 28) gives a rows × cols grid
(rows = H / 28, cols = W / 28) and N = rows · cols visual tokens, in raster order (verified in the host module and on the GPU).  N varies per
image (256 ≤ N ≤ 1280 with the R7-20 pixel bounds), so budgets are fractions of each image's own N (coordinator decision, OPEN-R740-5):
    k(f, N) = max(1, floor(f · N + 1/2)),  f ∈ {0.125 (stress), 0.25, 0.5, 0.75};  f = 1 is the full pass.
Since k is non-decreasing in f, the kept sets of one ordering are nested: A_0.125 ⊂ A_0.25 ⊂ A_0.5 ⊂ A_0.75 ⊂ B.
Three fixed trials per question, one per mechanism (the frozen D1-10 rule), on the rows × cols grid:
  random  uniform random ordering of the N tokens;  log P(kept set) = −log C(N, k)
  grid    the four 2-stride lattices in a random order, the first one split into the checkerboard of its 2 × 2 coarse grid (first) and the
          rest; random order inside each block (the frozen D1-10 rule).  With N = 576 on 24 × 24 every budget is a block boundary and the
          frozen log P = −log 8 / 4 / 6 / 4 is recovered; on other grids a budget can end inside a block, whose kept part is then a uniform random
          subset of that block.  log P(kept set) is exact (enumeration of the 4! × 2 block configurations, test-checked by brute force)
  qsim    Plackett–Luce (Gumbel top-k) on z-scored cosine(merged visual token, mean question word embedding); log P of the ORDERED prefix
Seeds come from sha256 of ('r7-40/d1-mask', mechanism, question id) — a separate stream from the LLaVA masks (different grids).
"""
from __future__ import annotations

import functools
import hashlib
import itertools
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import r7_d1_masks as MK  # noqa: E402  (frozen D1-10 module: Plackett–Luce prefix probabilities)

FRACTIONS = (0.125, 0.25, 0.5, 0.75)
FKEYS = tuple(f"{f:g}" for f in FRACTIONS)                           # record keys "0.125", "0.25", "0.5", "0.75"
MECHANISMS = MK.MECHANISMS                                           # ("random", "grid", "qsim")
SEED = MK.SEED
TAG = "r7-40/d1-mask"
OFFS = ((0, 0), (1, 1), (0, 1), (1, 0))                             # lattice offsets (ox, oy), as the frozen D1-10 grid mechanism


def budget(f: float, n: int) -> int:
    """Kept tokens at fraction f of an image with n visual tokens (round half up, at least 1)."""
    return max(1, int(math.floor(f * n + 0.5)))


def budgets(n: int) -> dict:
    return {key: budget(f, n) for f, key in zip(FRACTIONS, FKEYS)}


def rng_for(mechanism: str, qid: str) -> np.random.Generator:
    h = int(hashlib.sha256(f"{TAG}/{mechanism}/{qid}".encode()).hexdigest()[:16], 16)
    return np.random.default_rng([SEED, h])


def lattice(rows: int, cols: int, ox: int, oy: int) -> np.ndarray:
    r, c = np.divmod(np.arange(rows * cols), cols)
    return np.flatnonzero((r % 2 == oy) & (c % 2 == ox))


def log_comb(n: int, k: int) -> float:
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def order_random(qid: str, rows: int, cols: int):
    n = rows * cols; order = rng_for("random", qid).permutation(n).astype(np.int64)
    return order, {key: -log_comb(n, k) for key, k in budgets(n).items()}


def grid_blocks(rows: int, cols: int, perm, parity: int):
    """The five blocks of one grid configuration: checkerboard part of the first lattice, its rest, then the other three lattices."""
    first, *rest = [OFFS[i] for i in perm]
    lat = lattice(rows, cols, *first); r, c = np.divmod(lat, cols)
    sub = lat[((r // 2) + (c // 2)) % 2 == parity]; other = np.setdiff1d(lat, sub)
    return [sub, other] + [lattice(rows, cols, *o) for o in rest]


CONFIGS = tuple((perm, par) for perm in itertools.permutations(range(4)) for par in (0, 1))   # 48 equally likely grid configurations


@functools.lru_cache(maxsize=256)
def config_blocks(rows: int, cols: int):
    """The five blocks of every one of the 48 configurations of a rows × cols grid (independent of the question)."""
    return tuple(tuple(grid_blocks(rows, cols, perm, par)) for perm, par in CONFIGS)


def grid_logp(rows: int, cols: int, order: np.ndarray, ks=None) -> dict:
    """Exact log P(kept set = order[:k]) under the grid mechanism, for every budget: average over the 48 equally likely (lattice order,
    parity) configurations of P(set | configuration) = 1 / C(|boundary block|, r) when the set is the union of the configuration's first j
    blocks plus r tokens of block j + 1, else 0."""
    n = rows * cols; ks = ks or budgets(n); out = {}; blocks_of = config_blocks(rows, cols)
    for key, k in ks.items():
        S = np.zeros(n, dtype=bool); S[np.asarray(order[:k])] = True; tot = 0.0
        for blocks in blocks_of:
            cum = 0; j = 0
            while j < len(blocks) and cum + len(blocks[j]) <= k:
                cum += len(blocks[j]); j += 1
            if not all(S[b].all() for b in blocks[:j]):
                continue
            r = k - cum
            if r == 0:                                                   # the first j blocks fill the budget exactly (|S| = k = cum)
                tot += 1.0
                continue
            if int(S[blocks[j]].sum()) == r:                             # the r remaining kept tokens all lie in the boundary block
                tot += math.exp(-log_comb(len(blocks[j]), r))
        out[key] = math.log(tot / len(CONFIGS)) if tot > 0 else float("-inf")
    return out


def order_grid(qid: str, rows: int, cols: int, with_logp: bool = True):
    g = rng_for("grid", qid); perm = g.permutation(4); parity = int(g.integers(2))       # draw order as the frozen D1-10 order_grid
    blocks = config_blocks(rows, cols)[CONFIGS.index((tuple(int(i) for i in perm), parity))]
    order = np.concatenate([b[g.permutation(len(b))] for b in blocks]).astype(np.int64)
    return order, (grid_logp(rows, cols, order) if with_logp else None)


def order_qsim(qid: str, scores: np.ndarray, n: int):
    """Gumbel top-k == Plackett–Luce sampling with weights exp(z); returns the ordering and log P of each ordered prefix."""
    s = np.asarray(scores, dtype=np.float64)
    if s.shape != (n,) or not np.isfinite(s).all():
        raise ValueError(f"qsim needs {n} finite scores")
    z = (s - s.mean()) / (s.std() + 1e-12)
    g = rng_for("qsim", qid); order = np.argsort(-(z + g.gumbel(size=n)), kind="stable").astype(np.int64)
    ks = budgets(n); lp = MK.pl_prefix_logp(z, order, tuple(ks.values()))
    return order, {key: lp[k] for key, k in ks.items()}


def orderings(qid: str, rows: int, cols: int, qsim_scores: np.ndarray):
    """The three fixed trials of one question: {mechanism: (ordering[N], {fraction key: log-prob})}."""
    n = rows * cols
    return {"random": order_random(qid, rows, cols), "grid": order_grid(qid, rows, cols), "qsim": order_qsim(qid, qsim_scores, n)}


def kept(order: np.ndarray, k: int) -> np.ndarray:
    """Kept token indices at budget k, in raster order (the order the LLM sees them)."""
    return np.sort(np.asarray(order[:k]))


def ranks(order: np.ndarray) -> np.ndarray:
    """rank[token] = position of the token in the ordering (kept at budget k  <=>  rank < k)."""
    r = np.empty(len(order), dtype=np.int64); r[np.asarray(order)] = np.arange(len(order))
    return r
