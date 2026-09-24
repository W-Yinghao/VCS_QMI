"""Per-step linear warmup + cosine to ``min_lr_ratio`` (spec §8).

    f(s) = (s+1)/W                                             s <  W
         = r + (1-r) * [1 + cos(pi (s-W)/(T-1-W))] / 2          s >= W
Edge cases: W = 0 (no warmup) and T-1-W <= 0 (horizon too short for a cosine) never divide by zero.
"""
from __future__ import annotations

import math


def lr_factor(step: int, total_steps: int, warmup_steps: int, min_lr_ratio: float) -> float:
    if total_steps < 1:
        raise ValueError("total_steps must be >= 1")
    if step < 0 or step >= total_steps:
        raise ValueError(f"step {step} outside [0, {total_steps})")
    if warmup_steps < 0 or warmup_steps > total_steps:
        raise ValueError("warmup_steps must be within [0, total_steps]")
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    denom = total_steps - 1 - warmup_steps
    progress = 0.0 if denom <= 0 else (step - warmup_steps) / denom
    return min_lr_ratio + (1.0 - min_lr_ratio) * (1.0 + math.cos(math.pi * progress)) / 2.0


def warmup_steps_for(*, warmup_epochs: float, epochs: int, steps_per_epoch: int, total_steps: int) -> int:
    """W = warmup_epochs * steps_per_epoch for the full horizon; for a shortened smoke horizon the same *fraction*
    of the horizon is used and recorded (spec §16: smoke has its own scheduler config)."""
    full_total = epochs * steps_per_epoch
    full_w = int(round(warmup_epochs * steps_per_epoch))
    if total_steps == full_total:
        return full_w
    return int(round(total_steps * (full_w / full_total))) if full_total > 0 else 0
