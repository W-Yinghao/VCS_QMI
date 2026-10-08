# P145 addendum 1 — STRESS ε = 0.10 on CIFAR-100, seeds 0–2, VCS / JS / SimCLR — report — 2026-10-08

Pre-registration `P145_ADDENDUM1_C100_SEEDS_FROZEN_20261007.md` (trigger: seed-0 |Δ_VCS − Δ_JS| = 1.08 ≥ 1.00); results-only commit `609a21f`
(`reports/P145A1_results.json`, `scripts/p145a1_aggregate.py`).  Δ = STRESS − clean, paired by seed; clean runs P107 A-P3, P120 JS-AP3, P91 SimCLR
(same seeds).  Development split, frozen-h linear (kNN alongside); 95 % t intervals over 3 seeds; P114 labels.

| method | clean (mean) | STRESS (mean) | Δ linear per seed (0 / 1 / 2) | Δ linear, 95 % CI | label | Δ kNN | label |
|---|---|---|---|---|---|---|---|
| VCS (A-P3) | 60.00 | 59.93 | +0.10 / −0.30 / +0.00 | **−0.07 [−0.58, +0.45]** | close | +0.05 [−1.45, +1.56] | close |
| matched JS | 59.20 | 59.53 | +1.18 / −0.86 / +0.66 | +0.33 [−2.31, +2.96] | inconclusive | −0.04 [−2.34, +2.26] | close |
| SimCLR | 58.25 | 57.76 | −1.30 / +0.14 / −0.32 | −0.49 [−2.32, +1.33] | inconclusive | −0.43 [−1.77, +0.90] | inconclusive |

Contamination interaction (frozen reading, Δ_VCS − Δ_m paired by seed): **VCS − JS −0.39 [−2.51, +1.72] inconclusive** (kNN +0.09, close);
**VCS − SimCLR +0.43 [−1.87, +2.72] inconclusive** (kNN +0.49, inconclusive).

## Reading
- The seed-0 trigger does not replicate: JS's +1.18 at seed 0 is followed by −0.86 and +0.66, and SimCLR's −1.30 by +0.14 and −0.32.  Neither
  interaction is established; no claim that one objective tolerates replaced pairs better than another.
- VCS is the only method whose response to 10 % replaced pairs is consistently near zero (all three seeds within ±0.30; interval half-width 0.52).
  JS and SimCLR respond with seed-to-seed swings of ±1 point, which is why their intervals are wide — a variance statement, not an effect.
  Note that the P114 "close" label for the kNN rows follows from |mean| < 0.3 alone; those intervals are wide (±1.5 to ±2.3).
- Combined with P145 seed 0 on CIFAR-10 (all three methods within ±0.3), the STRESS axis at ε = 0.10 shows **no robustness difference** between
  the objectives at this power.  ε = 0.20 stays unsubmitted (needs a separate budget decision, frozen rule).
