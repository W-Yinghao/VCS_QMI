"""P157 (SEP-2): confirmatory unit for P156's secondary finding — in parametric critic classes (cos-2, the application form; quad-3) under
contamination, the quadratic rule (VCS) has the lower posterior error than matched logistic (JS).  Reuses the frozen P156 trainer and readouts
(sep.run_cell) unchanged; only the cell list, a wider lr grid and a finite-guarded re-selection are new."""
from __future__ import annotations

import math

from . import sep

PROTOCOL = "P157"
LRS = (1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0)                       # P156's parametric grid widened by two half-decades at the top
SEEDS = tuple(range(200, 208))                                        # fresh (P156 used 100-107)
CLASSES, CTYPES, EPS, DS, IS, NS = ("cos2", "quad3"), ("independent", "false_negative"), (0.1, 0.3), (20, 64), (4.0, 8.0), (4096, 16384)
POP_SEED, POP_N, POP_U = 200, 131072, 8000                           # descriptive large-sample block (block S2P), one seed


def configs():
    return [(cls, ct, e, d, I) for cls in CLASSES for ct in CTYPES for e in EPS for d in DS for I in IS]


def cells_for_seed(seed: int) -> list[dict]:
    """Block S2: 2 classes x 2 contamination types x 2 eps x 2 d x 2 I x 2 N = 64 cells per seed; arms js / vcs; 2,000 updates."""
    return [{**sep.cell("S2", cls, "product", ct, e, d, I, N, 2000, sep.ARMS_PAIR, LRS, seed), "tag": "p157"}
            for N in NS for (cls, ct, e, d, I) in configs()]


def population_cells() -> list[dict]:
    """Block S2P (descriptive): the same 32 configurations at N 131,072 and 8,000 updates, seed 200."""
    return [{**sep.cell("S2P", cls, "product", ct, e, d, I, POP_N, POP_U, sep.ARMS_PAIR, LRS, POP_SEED), "tag": "p157"}
            for (cls, ct, e, d, I) in configs()]


def reselect_finite(R: dict) -> dict:
    """Re-make both selections with non-finite scores never selected; identical to run_cell's choice when every score is finite."""
    rows = R["rows"]; grid = sorted({r["lr"] for r in rows}); fin = lambda v, bad: v if (v is not None and math.isfinite(v)) else bad
    for arm in {r["arm"] for r in rows}:
        g = [r for r in rows if r["arm"] == arm]
        for r in g:
            r["selected"] = False; r["selected_commonJ"] = False
            r["lr_edge"] = "low" if r["lr"] == grid[0] else "high" if r["lr"] == grid[-1] else None
        min(g, key=lambda r: fin(r["select"]["select_risk"], math.inf))["selected"] = True
        max(g, key=lambda r: fin(r["select"]["commonJ_select"], -math.inf))["selected_commonJ"] = True
    R["cell"]["protocol_id"] = PROTOCOL; R["cell"]["reselected_finite"] = True
    return R
