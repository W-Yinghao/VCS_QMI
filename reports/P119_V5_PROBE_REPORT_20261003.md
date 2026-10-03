# P119 — label efficiency paired by encoder seed, and readout-symmetric transfer (v5 NEXT-V-PROBE) — report — 2026-10-03

Pre-registration `P119_V5_PROBE_PREREG_FROZEN_20261003.md`; jobs 1019948 / 1019949 exit 0; results-only commit `2a00815` (`P119_v5_probe_results.{md,json}`).
Frozen encoders: recipe VCS (P35), SimCLR (P41), G2, U2, A-P3 — seeds 0–2 each; float32 features (part 3 rule); selection split; official test closed.

## 1. CIFAR-10 label efficiency (paired by encoder seed; draws averaged within a seed)
Rule: "A ahead of B at f" only if all three seed-paired differences are positive **and** the 95 % t interval excludes 0.

| comparison | 1 % labels | 10 % | 100 % |
|---|---|---|---|
| A-P3 − SimCLR | +1.17 [−0.08, +2.42] (+1.66 / +1.19 / +0.65) | +0.66 [−0.33, +1.66] | **+0.65 [+0.07, +1.22] → ahead** |
| A-P3 − G2 | +0.12 [−0.28, +0.52] | −0.05 [−0.34, +0.23] | +0.26 [−0.57, +1.09] |

At 1 % and 10 % labels A-P3 is above SimCLR on every seed but the 3-seed interval includes 0 → **no ordering claim at low labels** (the larger 1 % mean,
+1.17, has a correspondingly larger spread: within-seed draw sd 0.84).  Only the 100 % lead meets the rule.  A-P3 and G2 are indistinguishable at every
fraction.  (This is downstream label efficiency, not estimator sample efficiency.)

## 2. CIFAR-10 → CIFAR-100 frozen transfer, identical readouts for every method (100 % labels)

| readout | A-P3 | G2 | U2 | recipe VCS | SimCLR |
|---|---|---|---|---|---|
| raw_unstd (P110 anchor) | 46.81 | 47.67 | 45.75 | **52.68** | 38.15 |
| raw_std | 45.53 | 46.26 | 44.32 | **51.22** | 36.75 |
| l2_std (direction) | 45.23 | 45.91 | 44.19 | **51.13** | 36.67 |
| log‖h‖ alone | 2.19 | 2.20 | 2.07 | 2.01 | 2.21 |
| [direction, log‖h‖] | 45.39 | 45.87 | 44.23 | **51.09** | 36.85 |
| kNN | 39.89 | 38.97 | 35.46 | **44.93** | 34.26 |

- **Readout-robust ordering (frozen rule: same sign and |Δ| ≥ 1.0 under raw_unstd, raw_std, l2_std and kNN): met** for recipe VCS > SimCLR (+14.5 / +14.5 /
  +14.5 / +10.7) and A-P3 > SimCLR (+8.7 / +8.8 / +8.6 / +5.6), all paired intervals excluding 0.  The transfer reversal (recipe VCS best, SimCLR last) is
  therefore **not an artefact of the linear readout**: it holds for the standardised, direction-only and kNN readouts alike.  The 10 % table has the same order.
- **Norm diagnostics:** ‖h‖ alone carries ≈ 2 % (chance 1 %) for every family, and adding log‖h‖ back to the direction changes nothing (≤ 0.2) → the norm
  is not where any family's transfer information sits.
- **Probe lr:** the raw_unstd probe picks the grid edge (lr 0.3) in 12/12 rows for A-P3 and SimCLR (6/12 for the recipe) — the grid bound binds on the
  anchor readout; the standardised readouts pick interior values (mostly 0.03) and give the same ordering, so the binding grid edge does not explain it.
- Not claimed (pre-stated): any mechanism for the reversal (colour retention, probe under-fitting); it is reported as a robust observation of these
  CIFAR-10-pretrained encoders on CIFAR-100.

## 3. Feature precision (part 3)
float16 vs float32 changed the anchor readout by ≤ 0.08 for all 15 encoders (P110 numbers stand); one standardised readout moved 1.52 (disclosed in the prereg);
all part-1/2 numbers above use float32.
