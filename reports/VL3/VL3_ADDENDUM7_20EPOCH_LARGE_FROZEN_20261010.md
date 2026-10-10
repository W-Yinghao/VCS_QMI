# VL3 addendum 7 — the 20-epoch schedule on the six Large cells, three seeds (completes the 12-cell table at 20 epochs) — FROZEN 2026-10-10

Why: add. 4 (B/16, three seeds) shows, with 5 of 6 cells complete, that softmax loses with the longer shared schedule (20 − 10: −0.24
[−0.47, −0.02]).  VCS − softmax at 20 epochs is +0.37 [+0.08, +0.66].  The paired schedule effect on VCS − softmax (+0.41 [−0.20, +1.03]) still
contains 0.  Whether "the estimator objective leads at the longer shared schedule" holds at scale is untested.  This unit runs the same 20-epoch
schedule on the Large cells.  Together with add. 4, every one of the 12 cells is then measured at both schedules.  The schedule stays shared by all
objectives (not per-objective tuning).  All three seeds run directly, without a seed rule (lesson of add. 3).  **10 epochs stays the primary VL3
schedule.  Both schedules are reported in full, whatever this unit and add. 4 find.**

## Units
3 datasets × 2 Large backbones (CLIP L/14@336, SigLIP 2 L/16) × 3 objectives × seeds 0–2 at `--epochs 20 --tag-suffix _e20 --grad-ckpt` = 54 runs.
Recipe otherwise as VL3 add. 1.  Gate 2 as VL3.

## Readings (fixed now; `scripts/vl3_aggregate.py --suffix _e20` and the "paired" block of `scripts/vl3_add2_aggregate.py`, both extended to Large before results)
1. At 20 epochs, per Large cell: VCS − JS and VCS − softmax paired by seed (95 % t).
2. Pooled over all 12 cells at 20 epochs (with add. 4): VCS − softmax and VCS − JS (95 % t), the per-cell interval counts, and the split B/16 vs Large.
3. **Schedule effect** [(VCS − softmax)_20 − (VCS − softmax)_10], paired by seed, pooled over the 12 cells and split B/16 vs Large; per-objective
   20 − 10 changes.
4. Wording (fixed in advance; it supersedes add. 4's reading 3 once all 12 cells are complete).
   - If the 12-cell pooled schedule effect contains 0: "the comparison does not depend on the schedule".  The 20-epoch pooled VCS − softmax is
     then reported beside it.
   - If it lies above 0: "a longer shared schedule favours the estimator objectives by x".  The 10-epoch tie stays the primary result.
   - If it lies below 0: the reverse.
5. No further schedules, seeds or recipe changes follow without a new pre-registration.
