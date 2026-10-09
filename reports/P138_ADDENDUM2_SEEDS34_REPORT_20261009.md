# P138 addendum 2 — K = 16 sampled pairs vs all pairs at five seeds — report — 2026-10-09

Pre-registration `P138_ADDENDUM2_SEEDS34_*FROZEN*.md`; results-only commit `63e7516` (`reports/P138A2_results.json`, `scripts/p138a2_aggregate.py`,
job 1031585).  Per cell, K16 − parent (all pairs) paired by seed over seeds 0–4, 95 % t interval; non-inferior if the lower bound > −0.30
(CIFAR-10) / −0.50 (CIFAR-100).  Parents: P114 JS-AP3 C10, P107 / P120A1 A-P3 C100, P120 JS-AP3 C100.  Development split, frozen-h linear.

## 1. Result (linear; kNN alongside)
| cell | Δ linear per seed (0–4) | Δ linear, n 5 | margin | reading | Δ kNN | addendum 1 (n 3) |
|---|---|---|---|---|---|---|
| JS CIFAR-10 | +0.08 / −0.56 / −0.02 / +0.08 / −0.34 | −0.15 [−0.51, +0.20] | −0.30 | non-inferiority not shown | −0.20 [−0.50, +0.09] | −0.17 [−1.02, +0.69] |
| VCS CIFAR-100 | −0.20 / +0.54 / +0.82 / −0.46 / −0.60 | +0.02 [−0.76, +0.80] | −0.50 | non-inferiority not shown | −0.30 [−0.63, +0.02] | +0.39 [−0.92, +1.70] |
| JS CIFAR-100 | +0.38 / −0.22 / −0.66 / −0.32 / −0.06 | −0.18 [−0.65, +0.30] | −0.50 | non-inferiority not shown | −0.43 [−0.88, +0.02] | −0.17 [−1.46, +1.13] |
VCS CIFAR-10 was shown non-inferior at three seeds (addendum 1: +0.03 [−0.28, +0.33]) and was not extended.

## 2. Reading
- **Non-inferiority is shown only for VCS CIFAR-10.**  The other three cells stay "not shown", but their means are within ±0.2 of zero and the
  interval half-widths shrank by a factor of 1.7–2.7 from n = 3.  The lower bounds miss their margins by 0.21 (JS C10), 0.26 (VCS C100) and 0.15 (JS C100).  The
  remaining uncertainty is seed variance at this n, not a visible cost of sampling.
- kNN leans slightly negative in all three cells (−0.20 to −0.43, intervals touching 0).
- Allowed wording for the paper: "K = 16 sampled pairs per anchor match all pairs within ±0.2 linear on average (five seeds); non-inferiority at
  the pre-set margins is shown for VCS on CIFAR-10 only".  Not allowed: "sampling is free" or "non-inferior" for the other cells.
- No further seeds (frozen: "no further seeds after this addendum"; owner 2026-10-09: no new SSL experiments).
