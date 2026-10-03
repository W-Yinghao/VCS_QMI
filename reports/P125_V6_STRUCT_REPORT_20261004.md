# P125 — structured measurement critics on the P108 / P117 generators (v6 V6-STRUCT) — report — 2026-10-04

Pre-registration `P125_V6_STRUCT_PREREG_FROZEN_20261003.md` (pilot disclosed); jobs 1020324 / 1020325 (CPU) exit 0; results-only commit `726a919`
(`P125_struct_results.{md,json}`, per-seed rule table `P125_per_seed_rule.txt`, cell files `reports/P125/`).  Fresh fit seeds 2–6; paired vs JointMLP
(same seed, same roles).  Frozen rule 1: "reduces / increases posterior error" only if all 5 paired differences share the sign.

## 1. Rule outcomes (Δ posterior MSE vs JointMLP, mean over 5 seeds; VCS and JS fitted, same verdict unless noted)

| condition (N) | Interaction-MLP, original coords | Interaction-MLP, rotated | quad_full | quad_r4 / r16 |
|---|---|---|---|---|
| C1 Gaussian (4096) | **reduces** (−0.034 / −0.042) | **increases** (+0.12) | no consistent difference | — |
| C1 Gaussian (16384) | **reduces** (−0.009 / −0.011) | **increases** (+0.05) | **reduces** (−0.008 / −0.009) | — |
| C2 Gaussian 2-of-100 (4096) | **reduces** (−0.44) | ≈ 0 (JS none; VCS +0.004 increases) | increases | both increase |
| C2 Gaussian 2-of-100 (16384) | **reduces** (−0.09) | **increases** (+0.13 / +0.16) | increases (+0.42) | r4 **reduces** (−0.029 / −0.016); r16 increases |
| C3 xor (4096) | **reduces** (−0.28) | **increases** (+0.37) | increases (+0.27) | — |
| C3 xor (16384) | none (JS) / increases (VCS +0.010) | **increases** (+0.14 / +0.17) | increases (+0.68) | — |

JointMLP itself is rotation-insensitive (|rot − orig| ≤ 0.012 except C3 4096, where rotation helps it by −0.10 / −0.12).

## 2. Reading (frozen rules 2–4)
- **Interaction-MLP's gains are a coordinate-dependent inductive bias (rule 3), not a structural advantage:** in original coordinates it cuts the posterior
  error a lot (C2 4096: 0.546 → 0.115; C3 4096: 0.57 → 0.29), but after a fixed random rotation of each side it is worse than JointMLP in every condition.
  The [x, y, x⊙y, |x−y|] features help only when the dependence is axis-aligned.
- **The quadratic logit has the expected Gaussian-structure advantage only where the target is low-dimensional Gaussian:** full quadratic reduces error at
  C1 N 16384 and low-rank r = 4 at C2 N 16384; elsewhere (C2 small N, all of C3 xor) it is much worse — no cross-condition generalisation (rule 2).
- **VCS and JS agree on every verdict** except two borderline cells (C2 4096 rotated Interaction-MLP; C3 16384 original) — no ranking (rule 4).
- **For v6 §9:** none of the structured families improves the measurement critic in a rotation-robust, condition-general way; adding expressiveness of these
  kinds is not the lever.  Together with P117 (calibration does not help) this points back to fitting budget / sample size and the critic's general
  function class rather than hand-structured pair features.  Not claimed: anything about image data or online SSL.
