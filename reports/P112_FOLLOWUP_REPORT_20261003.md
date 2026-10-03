# P112 — strong-augmentation follow-up seeds (addendum 1) — report — 2026-10-03

Addendum `P112_ADDENDUM1_FOLLOWUP_FROZEN_20261003.md`; results `P112_table.md` (d85dd9f).  Paired with P111 G2-strong (same seed, same P89 strong block, only (a, b) differ).

| strong augmentation | seed 0 | seed 1 | seed 2 | mean ± sd | paired Δ vs G2-strong [95 %] |
|---|---|---|---|---|---|
| G2 (2, κ 0.5) | 86.86 | 87.70 | 87.80 | 87.45 | — |
| (2, κ 0.75) linear | 88.22 | 88.28 | 87.96 | **88.15 ± 0.17** | +1.36 / +0.58 / +0.16 → **+0.70 [−0.81, +2.21]** |
| (2, κ 0.25) linear | 87.88 | 87.70 | 87.34 | 87.64 ± 0.27 | +1.02 / 0.00 / −0.46 → +0.19 [−1.69, +2.07] |
| (2, κ 0.75) kNN | 85.74 | 85.86 | 85.56 | 85.72 | +0.84 / +0.56 / +0.04 → +0.48 [−0.53, +1.49] |
| (2, κ 0.25) kNN | 84.56 | 85.08 | 85.16 | 84.93 | −0.34 / −0.22 / −0.36 → −0.31 [−0.49, −0.12] |

**Reading.**
- κ 0.75 is above G2-strong on all three seeds (linear and kNN), but the margin shrinks seed by seed (+1.36 → +0.16) and the 3-seed interval includes 0: the
  screen's +1.36 was mostly the low G2-strong seed 0, as the screen report cautioned.  Labelled by the P114 convention: **inconclusive at 3 seeds** (|mean| ≥ 0.3,
  interval includes 0).
- κ 0.25's screen gain does not replicate (+0.19 linear, interval wide; kNN slightly below G2-strong on every seed).
- Even the better cell (88.15) stays far below SimCLR under the same strong block (89.49) and below A-P3-strong's standard-augmentation level; a retuned fixed
  κ recovers at most part of the strong-augmentation loss.  Consistent with P123's read-only finding that κ moves the positive-gradient peak into the positive
  bulk (lower low-IoU share) — a hypothesis for full training, not a demonstrated cause.
- v6 §3: P112 thresholds are not mixed into P115; no further κ grid is opened.
