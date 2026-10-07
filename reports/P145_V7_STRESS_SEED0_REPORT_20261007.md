# P145 — v7 V7-STRESS (ε = 0.10 contaminated training positives), seed 0 — report — 2026-10-07

Pre-registration `P145_V7_STRESS_PREREG_FROZEN_20261007.md`; results-only commit `6686849` (`P145_results.json`).  Development split; clean baselines =
the seed-0 parents.  Five runs were interrupted by the 2026-10-07 home-quota stop and resumed from last.pt (same configs, same per-step replacement
draws, which depend only on seed and step).

## 1. Accuracy (Δ = contaminated − clean, seed 0)
| dataset | method | ε = 0.10 linear / kNN | clean | Δ linear | Δ kNN |
|---|---|---|---|---|---|
| CIFAR-10 | VCS (A-P3) | 89.08 / 87.80 | 89.06 / 87.30 | +0.02 | +0.50 |
| CIFAR-10 | JS-AP3 | 88.44 / 87.24 | 88.72 / 87.50 | −0.28 | −0.26 |
| CIFAR-10 | SimCLR | 88.12 / 87.54 | 88.20 / 87.20 | −0.08 | +0.34 |
| CIFAR-100 | VCS (A-P3) | 60.30 / 55.82 | 60.20 / 55.92 | +0.10 | −0.10 |
| CIFAR-100 | JS-AP3 | 59.94 / 55.62 | 58.76 / 54.80 | **+1.18** | +0.82 |
| CIFAR-100 | SimCLR | 57.36 / 56.34 | 58.66 / 57.38 | **−1.30** | −1.04 |

Interaction Δ_VCS − Δ_m: CIFAR-10 +0.30 (JS), +0.10 (SimCLR) — below 0.50, **no trigger**; CIFAR-100 −1.08 (JS), +1.40 (SimCLR) — ≥ 1.00, **trigger**
→ addendum 1 (seeds 1–2 for all three methods on CIFAR-100; frozen and running).

## 2. Process (step logs, second half of training)
- Replacement rate 0.100 in every run; the three methods of a dataset saw identical replacement indices (same seed and step).
- Replaced positives end at cosine ≈ 0.00 (kept: 0.93–0.95) and, for VCS / JS, T ≈ −0.74 (kept +0.68 to +0.70): all three methods treat the
  wrong pairs like negatives rather than pulling them together.
- Gradient share of the replaced positives (norm ratio to the kept positives, w.r.t. the projector outputs): VCS 0.49–0.50, SimCLR 0.48–0.50,
  **JS 0.74–0.76**; cosine of the replaced part with the total gradient: VCS 0.49–0.50, SimCLR 0.43–0.45, JS 0.62–0.64.  The bounded VCS score gives the
  contaminated pairs about the same weight as SimCLR's softmax and less than the logistic loss.
- h effective rank changes little (CIFAR-100: VCS 146 vs 156 clean, JS 147 vs 157, SimCLR 193 vs 193); no failures or non-finite steps.

## 3. Reading (frozen; single seed)
- CIFAR-10: all three methods tolerate 10 % contaminated positives within ±0.3 linear; no difference large enough to follow up.
- CIFAR-100: the methods move in opposite directions at seed 0 (JS +1.18, VCS +0.10, SimCLR −1.30).  A JS *gain* under contamination is not
  expected from the construction and is a single-seed value (CIFAR-100 seed sd ≈ 0.2–0.4); it is not interpreted before seeds 1–2.  VCS is not
  called "more robust": its change is the middle one, and a smaller drop from a different clean start is not a robustness claim.
- The gradient-share difference (JS gives contaminated pairs ~1.5× the relative weight of VCS / SimCLR) is a measured mechanism record, not yet
  linked to the accuracy changes.
