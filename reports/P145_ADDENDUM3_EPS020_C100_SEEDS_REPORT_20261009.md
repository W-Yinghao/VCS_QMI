# P145 addendum 3 — STRESS ε = 0.20 on CIFAR-100, seeds 0–2, VCS / JS / SimCLR — report — 2026-10-09

Pre-registration `P145_ADDENDUM3_EPS020_C100_SEEDS_FROZEN_20261009.md` (trigger from addendum 2, run on the owner's decision); results-only commit
`9e43a25` (`reports/P145A3_results.json`, `scripts/p145a3_aggregate.py`, job 1032116).  Δ = STRESS − clean, paired by seed; clean runs P107 A-P3,
P120 JS-AP3, P91 SimCLR.  Development split, frozen-h linear (kNN alongside); 95 % t intervals over 3 seeds; P114 labels.  **This is the last SSL
unit** (owner 2026-10-09: no new SSL experiments).

## 1. Result
| method | clean (mean) | ε 0.20 (mean) | Δ linear per seed (0 / 1 / 2) | Δ linear, 95 % CI | label | Δ kNN | label |
|---|---|---|---|---|---|---|---|
| VCS (A-P3) | 60.00 | 60.68 | −0.34 / +0.98 / +1.40 | +0.68 [−1.58, +2.94] | inconclusive | +0.19 [−0.27, +0.65] | close |
| matched JS | 59.20 | 59.39 | +0.78 / −0.22 / +0.00 | +0.19 [−1.12, +1.49] | close | −0.33 [−1.55, +0.89] | inconclusive |
| SimCLR | 58.25 | 58.17 | −0.66 / +0.26 / +0.16 | −0.08 [−1.33, +1.17] | close | −0.09 [−1.33, +1.15] | close |
Contamination interaction (Δ_VCS − Δ_m, paired): **VCS − JS +0.49 [−2.99, +3.97] inconclusive** (kNN +0.52); **VCS − SimCLR +0.76
[−0.39, +1.91] inconclusive** (kNN +0.27).

## 2. Reading
- **The seed-0 trigger did not replicate**, for the second time with the same JS parent: at seed 0, VCS − JS was −1.12; over three seeds it is
  +0.49 with an interval of ±3.5.  The trigger was a single low clean JS run, as at ε 0.10.
- No method loses accuracy under 20 % replaced positives on CIFAR-100: every Δ interval contains 0.  VCS's two later seeds even sit 1–1.4 above
  their clean runs.  That is not read as a benefit; the interval is −1.6 to +2.9.
- **STRESS axis, final (ε 0.10 and 0.20, CIFAR-100, three seeds each; CIFAR-10 seed 0):** no robustness difference between VCS, matched JS and
  SimCLR is established at this power.  Allowed wording: "replacing 10–20 % of positive pairs with random images changes CIFAR-100 linear accuracy
  by less than the seed spread for all three objectives".  Not allowed: any claim that one objective is more robust.
