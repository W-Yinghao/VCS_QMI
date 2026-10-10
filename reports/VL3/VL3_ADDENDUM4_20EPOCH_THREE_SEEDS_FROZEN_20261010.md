# VL3 addendum 4 — the 20-epoch schedule at three seeds (B/16, six cells) — FROZEN 2026-10-10

Why: add. 2 (one seed) found a small shift toward the estimator objectives with the longer schedule (mean VCS − softmax −0.01 → +0.29).  VCS / JS
select epochs 12–15 of 20 while softmax peaks around epoch 4.  The shift is smaller than the seed noise, so one seed cannot say whether the
comparison depends on the schedule.  Reviewers will ask the question, because the estimator objectives keep improving after softmax has stopped.
The schedule is shared by all objectives (not per-objective tuning).  **The primary VL3 schedule stays 10 epochs.  Both schedules are reported in
full, whatever this unit finds.**

## Units
Seeds 1 and 2 × 3 objectives × 6 B/16 cells (RefCOCOg / RefCOCO / RefCOCO+ × CLIP B/16 / SigLIP 2 B/16) at 20 epochs (`--epochs 20 --tag-suffix
_e20`) = 36 runs; seed 0 exists (add. 2).  Recipe otherwise as VL3.  Gate 2 as VL3.  Fed after the add. 3 runs.

## Readings (fixed now; `scripts/vl3_aggregate.py --suffix _e20` and the "paired" block of `scripts/vl3_add2_aggregate.py`)
1. At 20 epochs, per cell: VCS − JS and VCS − softmax paired by seed (95 % t), and pooled over the six cells (cells as units).
2. **Schedule effect:** per cell and seed, [(VCS − softmax)_20 − (VCS − softmax)_10], pooled over the six cells (95 % t), and the same for VCS − JS.
   Per objective, the pooled 20 − 10 change.
3. Wording fixed in advance.
   - If the pooled schedule effect on VCS − softmax contains 0: "the comparison does not depend on the schedule (10 vs 20 epochs)".
   - If it lies above 0: "a longer shared schedule favours the estimator objectives by x".  The 10-epoch numbers remain the primary result,
     and this is stated next to them.
   - If it lies below 0: "a longer schedule favours softmax by x".
4. No further schedules, seeds or recipe changes follow from this unit without a new pre-registration.
