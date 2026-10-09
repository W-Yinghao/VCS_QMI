# P145 addendum 2 — STRESS ε = 0.20, seed 0 — report — 2026-10-09

Pre-registration `P145_ADDENDUM2_EPS020_PREREG_FROZEN_20261008.md`; results-only commit `d0ed119` (`reports/P145A2_results.json`,
`scripts/p145a2_aggregate.py`, aggregation job 1030275).  Development split, frozen-h linear (kNN alongside), seed 0; clean parents as in P145
(A-P3, JS-AP3, SimCLR seed 0).

## 1. Dose curve (linear top-1, seed 0)
| dataset | method | ε 0 | ε 0.10 | ε 0.20 | Δ(0.20) linear | Δ(0.20) kNN |
|---|---|---|---|---|---|---|
| CIFAR-10 | VCS (A-P3) | 89.06 | 89.08 | 88.68 | −0.38 | +0.14 |
| CIFAR-10 | JS-AP3 | 88.72 | 88.44 | 88.70 | −0.02 | +0.10 |
| CIFAR-10 | SimCLR | 88.20 | 88.12 | 87.98 | −0.22 | +0.34 |
| CIFAR-100 | VCS (A-P3) | 60.20 | 60.30 | 59.86 | −0.34 | +0.06 |
| CIFAR-100 | JS-AP3 | 58.76 | 59.94 | 59.54 | +0.78 | +0.18 |
| CIFAR-100 | SimCLR | 58.66 | 57.36 | 58.00 | −0.66 | −0.28 |

## 2. Frozen trigger
| contrast | value | threshold | fires |
|---|---|---|---|
| CIFAR-10 Δ_VCS − Δ_JS | −0.36 | 0.50 | no |
| CIFAR-10 Δ_VCS − Δ_SimCLR | −0.16 | 0.50 | no |
| CIFAR-100 Δ_VCS − Δ_JS | **−1.12** | 1.00 | **yes** |
| CIFAR-100 Δ_VCS − Δ_SimCLR | +0.32 | 1.00 | no |

By the letter of the frozen rule, CIFAR-100 seeds 1–2 for all three methods are triggered (6 runs; ≈ 4.5 GPU-h each at the observed speed,
≈ 30 GPU-h).  As planned in `EXPERIMENT_ORDER_20261008.md` (A2: "if its trigger fires I report it and ask"), **nothing was submitted**.

## 3. Reading
- **CIFAR-10:** at ε = 0.20 every method stays within 0.4 linear point of its clean run (kNN within +0.35); no difference to follow up.
- **CIFAR-100:** the trigger comes from the same source as the ε = 0.10 seed-0 trigger: the JS-AP3 seed-0 clean parent (58.76) is the lowest of
  the three JS clean seeds (58.76 / 59.68 / 59.16), so any JS run at seed 0 tends to show a positive Δ.  JS at ε = 0.20 (59.54) lies inside the
  clean JS seed range; VCS (59.86) inside its clean range (59.72–60.20); SimCLR (58.00) at the lower edge of its clean range (58.02–58.66).
  The ε = 0.10 trigger with the same parent did not replicate over seeds 1–2 (JS Δ +1.18 → −0.86, +0.66; addendum 1, VCS − JS −0.39
  [−2.51, +1.72] inconclusive).
- Descriptive dose: VCS loses 0.3–0.4 point between ε 0.10 and 0.20 on both datasets (single seed; within one clean-seed sd on CIFAR-100).
  No method shows a dose response that separates it from the others at this power.
- No claim that any objective tolerates contaminated positives better; the STRESS axis stays an appendix robustness check
  ("no robustness difference detected at ε ≤ 0.20, seed 0 / three seeds at ε 0.10").

## 4. Decision needed (owner)
Run the triggered CIFAR-100 seeds 1–2 (frozen rule; ≈ 30 GPU-h, low priority behind the VL / P138 / P130 work), or record the trigger as fired
but not followed, with the parent explanation above (a disclosed deviation from the frozen rule).
