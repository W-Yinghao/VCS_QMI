"""V10 T03 CAL selection (decision 9.2, protocol P2 t03.tuning_suite.selection), sample-only.

Criterion: maximise the complete empirical U-J of the frozen critic on the independent CAL samples (negatives kept).  Ties (exactly
equal float64 values, no tolerance) -> the earlier checkpoint -> the frozen candidate ordinal (learning rate ascending; MF_VCS_M1 cap
ascending within each learning rate).
  CAL_UJ_SELECT    tuning suite: over (candidate, checkpoint) of one (condition, method); the selected lr / cap are frozen for every
                   formal replicate of that (condition, method); the tuning suite's selected checkpoint is not reused.
  OWN_CAL_UJ_CKPT  formal replicate: hyperparameters fixed from the tuning suite; checkpoint chosen only on the replicate's own CAL.
Selection reads only result.json records (no evaluator output: that lives in the evaluator-only directories, denied by the truth barrier).
Every candidate must be COMPLETE under the same T03 code identity; a FAIL / missing candidate blocks the selection (no silent
exclusion), and non-finite CAL values block it as well.
"""
from __future__ import annotations

import math

import t03_common as C


def candidate_ordinal(method: str, lr: float, cap: int | None) -> int:
    li = C.LEARNING_RATES.index(float(lr))
    if method == "MF_VCS_M1":
        if cap is None:
            raise ValueError("MF_VCS_M1 candidates need a cap")
        return li * len(C.M1_CAPS) + C.M1_CAPS.index(int(cap))
    if cap is not None:
        raise ValueError("only MF_VCS_M1 has a cap")
    return li


def candidates_for(method: str) -> list[tuple[int, float, int | None]]:
    """The frozen candidate order of one method: [(ordinal, lr, cap)]."""
    if method == "MF_VCS_M1":
        out = [(lr, cap) for lr in C.LEARNING_RATES for cap in C.M1_CAPS]
    else:
        out = [(lr, None) for lr in C.LEARNING_RATES]
    return [(candidate_ordinal(method, lr, cap), lr, cap) for lr, cap in out]


class SelectionBlocked(RuntimeError):
    pass


def _finite_trace(trace: list[dict], unit_id: str) -> list[tuple[int, float]]:
    pts = [(int(c["step"]), c["CAL_UJ"]) for c in trace]
    bad = [s for s, v in pts if v is None or not math.isfinite(float(v))]
    if bad:
        raise SelectionBlocked(f"{unit_id}: non-finite CAL U-J at checkpoints {bad}")
    if [s for s, _ in pts] != sorted({s for s, _ in pts}):
        raise SelectionBlocked(f"{unit_id}: checkpoint steps not strictly increasing")
    return [(s, float(v)) for s, v in pts]


def select_checkpoint(trace: list[dict], unit_id: str = "") -> dict:
    """OWN_CAL_UJ_CKPT: argmax CAL U-J over the unit's checkpoints; ties -> the earliest checkpoint."""
    pts = _finite_trace(trace, unit_id)
    best = max(v for _, v in pts)
    tied = [s for s, v in pts if v == best]
    return {"rule": "max CAL U-J; ties -> earlier checkpoint (exact float64 equality)", "step": min(tied), "CAL_UJ": best,
            "tied_steps": tied, "n_checkpoints": len(pts)}


def select_group(cands: list[dict]) -> dict:
    """CAL_UJ_SELECT over [{'ordinal', 'unit_id', 'lr', 'cap', 'trace'}] of ONE (condition, method)."""
    if not cands:
        raise SelectionBlocked("no candidates")
    ords = [c["ordinal"] for c in cands]
    if len(set(ords)) != len(ords):
        raise SelectionBlocked("duplicate candidate ordinals")
    pts = [(c["ordinal"], s, v) for c in cands for s, v in _finite_trace(c["trace"], c["unit_id"])]
    best = max(v for _, _, v in pts)
    tied = [(s, o) for o, s, v in pts if v == best]
    step, ordinal = min(tied)                                       # earlier checkpoint first, then the frozen ordinal
    win = next(c for c in cands if c["ordinal"] == ordinal)
    return {"rule": "max CAL U-J over candidates x checkpoints; ties -> earlier checkpoint -> frozen candidate ordinal (exact float64 equality)",
            "selected": {"ordinal": ordinal, "unit_id": win["unit_id"], "lr": win["lr"], "cap": win["cap"], "step": step, "CAL_UJ": best},
            "tied": [{"step": s, "ordinal": o} for s, o in sorted(tied)], "n_candidates": len(cands),
            "table": [{"ordinal": c["ordinal"], "unit_id": c["unit_id"], "lr": c["lr"], "cap": c["cap"],
                       "CAL_UJ_by_step": {str(s): v for s, v in _finite_trace(c["trace"], c["unit_id"])}} for c in sorted(cands, key=lambda c: c["ordinal"])]}
