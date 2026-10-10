# VL3 addendum 4 — the 20-epoch schedule at three seeds (B/16, six cells) — report — 2026-10-10

Protocol `VL3_ADDENDUM4_20EPOCH_THREE_SEEDS_FROZEN_20261010.md`; results-only commit `67b9688` (`reports/VL3/VL3_results_e20.json`, `reports/VL3/VL3_add2_results.json`,
job 1033507).  Shared schedule 10 → 20 epochs for all three objectives; everything else as VL3.  **Gate 2: 54 / 54 runs at 20 epochs pass.**
The 10-epoch numbers stay the primary VL3 result.  Add. 7 (Large at 20 epochs, running) extends the schedule reading to all 12 cells and
supersedes reading 3 below when complete.

## 1. Per cell at 20 epochs (mean ± sd over 3 seeds; paired 95 % t) and the schedule effect
| cell | VCS | JS | softmax | VCS − softmax @20 | VCS − softmax @10 | schedule effect |
|---|---|---|---|---|---|---|
| RefCOCOg × CLIP | 80.44 ± 0.22 | 80.72 ± 0.39 | 79.90 ± 0.44 | +0.54 [−0.02, +1.10] | +0.89 [+0.03, +1.75] | −0.35 [−1.38, +0.69] |
| RefCOCOg × SigLIP 2 | 85.27 ± 0.14 | 85.16 ± 0.26 | 84.92 ± 0.37 | +0.34 [−0.63, +1.32] | −0.05 [−1.22, +1.11] | +0.40 [−1.43, +2.23] |
| RefCOCO × CLIP | 85.36 ± 0.26 | 84.50 ± 0.30 | 84.71 ± 0.27 | **+0.65 [+0.16, +1.14]** | −0.31 [−0.94, +0.31] | +0.96 [−0.07, +1.99] |
| RefCOCO × SigLIP 2 | 88.54 ± 0.05 | 88.44 ± 0.15 | 88.48 ± 0.42 | +0.06 [−0.86, +0.98] | −0.28 [−2.17, +1.60] | +0.34 [−0.98, +1.66] |
| RefCOCO+ × CLIP | 82.02 ± 0.10 | 81.73 ± 0.06 | 81.77 ± 0.11 | +0.25 [−0.14, +0.63] | −0.48 [−1.76, +0.80] | +0.72 [−0.92, +2.37] |
| RefCOCO+ × SigLIP 2 | 87.04 ± 0.33 | 87.04 ± 0.14 | 87.44 ± 0.22 | −0.40 [−1.68, +0.87] | **−0.54 [−0.88, −0.21]** | +0.14 [−1.46, +1.74] |

## 2. Pre-registered readings (pooled over the six cells, 95 % t)
| quantity | pooled |
|---|---|
| VCS − softmax at 20 epochs | +0.24 [−0.16, +0.64] (per-cell intervals > 0: 1, < 0: 0) |
| VCS − softmax at 10 epochs (same cells) | −0.13 [−0.68, +0.42] |
| **Schedule effect on VCS − softmax** | **+0.37 [−0.11, +0.85]** |
| VCS − JS at 20 / 10 epochs | +0.18 [−0.22, +0.58] / +0.12 [+0.04, +0.19] |
| Schedule effect on VCS − JS | +0.06 [−0.34, +0.47] |
| 20 − 10, per objective | VCS +0.16 [−0.18, +0.51]; JS +0.10 [−0.18, +0.38]; softmax −0.20 [−0.41, −0.00] |
| J_recal VCS / softmax at 20 epochs | 1.33 |

**Pre-fixed wording (schedule-effect interval contains 0): "the comparison does not depend on the schedule (10 vs 20 epochs)"**, for the B/16
cells.

## 3. Readings
1. The longer schedule moves VCS − softmax by +0.37 on average.  This is in the direction add. 2 suggested, but the interval contains 0.  At 20
   epochs VCS − softmax is +0.24 [−0.16, +0.64]: still a tie, now with a positive point estimate.
2. **Interim pools mislead.**  With 5 of 6 cells, VCS − softmax at 20 epochs was +0.37 [+0.08, +0.66] (excluding 0).  The sixth cell (RefCOCO+ ×
   SigLIP 2, −0.40) brought it to +0.24 [−0.16, +0.64].  Only complete-unit pools are read.
3. Softmax is the only objective whose mean falls with the longer schedule (−0.20, upper bound −0.00).  This matches its early CAL-selected
   epochs (median 4 of 20 in add. 2).  The estimator objectives hold or gain slightly.
4. The estimator view is unchanged: VCS / JS encoders expose 1.33× the critic-fittable dependence of softmax encoders at 20 epochs too.
