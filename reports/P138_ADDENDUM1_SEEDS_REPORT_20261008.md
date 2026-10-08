# P138 addendum 1 — K = 16 sampled pairs vs all pairs, seeds 0–2, four cells — report — 2026-10-08

Pre-registration `P138_ADDENDUM1_SEEDS_FROZEN_20261007.md` (trigger: development retention in all four seed-0 cells); results-only commit `a4194a0`
(`reports/P138A1_results.json`, `scripts/p138a1_aggregate.py`).  Δ = K16 − parent (all pairs), paired by seed; parents P107 A-P3 (C10 / C100),
P114 JS-AP3 (C10), P120 JS-AP3 (C100).  Development split, frozen-h linear (kNN alongside); 95 % t interval over 3 seeds.

| cell | K16 linear (s0 / s1 / s2) | parent linear | Δ linear per seed | Δ linear, 95 % CI | margin | reading | Δ kNN, 95 % CI |
|---|---|---|---|---|---|---|---|
| VCS C10 | 89.22 / 88.88 / 89.06 | 89.06 / 88.96 / 89.06 | +0.16 / −0.08 / +0.00 | **+0.03 [−0.28, +0.33]** | −0.30 | **non-inferior** | +0.35 [−0.16, +0.85] |
| JS C10 | 88.80 / 88.74 / 88.40 | 88.72 / 89.30 / 88.42 | +0.08 / −0.56 / −0.02 | −0.17 [−1.02, +0.69] | −0.30 | not shown | −0.27 [−0.58, +0.05] |
| VCS C100 | 60.00 / 60.62 / 60.54 | 60.20 / 60.08 / 59.72 | −0.20 / +0.54 / +0.82 | +0.39 [−0.92, +1.70] | −0.50 | not shown | −0.27 [−1.09, +0.55] |
| JS C100 | 59.14 / 59.46 / 58.50 | 58.76 / 59.68 / 59.16 | +0.38 / −0.22 / −0.66 | −0.17 [−1.46, +1.13] | −0.50 | not shown | −0.49 [−1.67, +0.70] |

## Reading
- **VCS on CIFAR-10 is non-inferior with K = 16** sampled positive shifts per anchor instead of all pairs (15.3× fewer pair scores, P138): all
  three seeds within ±0.16, interval inside the 0.30 margin.
- For the other three cells non-inferiority is **not shown** at three seeds — not "worse": the mean differences are small (−0.17 / +0.39 / −0.17)
  but seed-to-seed swings of 0.6–1.0 points make the intervals (t multiplier 4.30 at n = 3) wider than the margins.  VCS on CIFAR-100 is on the
  favourable side in two of three seeds (+0.54, +0.82); JS on CIFAR-100 drifts negative over seeds 1–2.
- No wall-time gain was measured for K = 16 (P138 seed 0), so the practical case for sampling rests on the score count, and the accuracy case is
  established only for VCS on CIFAR-10.  More seeds would be needed to settle the other cells; not proposed (the cost is ≈ 5–9 GPU-h per run and the
  question is secondary to the paper's claims).
