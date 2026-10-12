# r6 REG-P1 — learned local-relation energy for BF/SHG registration (translation only) — report — 2026-10-12

Protocol `reports/R6/R6_REG_P1_PROTOCOL_FROZEN_20261011.yaml` (frozen before any natural-data result).  Results-only commits `f4a8ef8` (primary)
and the secondary < 9 px contrasts commit.
- **Data:** BF/SHG TMA cores (open; official split): FIT 32 / CAL 8 / DEV 32 image pairs.  10 seeded translation perturbations per CAL / DEV image
  (|t*| <= 51 px at 512, three magnitude strata).
- **Methods:**
  - Learned critics: VCS, logistic and InfoNCE, all with one architecture and equal inputs (BF RGB, SHG log1p).
  - CoMIR representations: official code and recipe.
  - Per-image MI / NMI, and a closed-form RFF kernel critic.
- **One shared optimiser:** coarse grid, then 3 Adam-refined starts.  Selection on CAL; DEV evaluated once.
- **Execution notes:**
  - The first submission driver lost 20 jobs; they were resubmitted unchanged.
  - A CoMIR setup bug, reachable only on the recipe path, was fixed before any CoMIR run.

## 1. Results (DEV image-macro success, % of each image's 10 registrations, then mean over 32 images; mean ± sd over seeds)
| method | < 42 px | < 9 px | median error (px) | p90 error (px) | s / registration |
|---|---|---|---|---|---|
| **VCS critic** | **100.0 ± 0.0** | **75.0 ± 3.3** | 5.4 | 14.8 | 0.53 |
| logistic critic | 98.9 ± 1.7 | 65.8 ± 4.5 | 5.7 | 16.1 | 0.52 |
| InfoNCE critic | 95.2 ± 2.3 | 57.4 ± 9.4 | 7.6 | 18.3 | 0.77 |
| CoMIR features + our optimiser | 90.9 ± 0.3 | 63.9 ± 2.5 | 5.7 | 21.1 | 0.16 |
| MI (per image) | 81.9 | 55.6 | 7.5 | 169.9 | 0.45 |
| NMI (per image) | 80.3 | 55.3 | 6.9 | 171.0 | 0.45 |
| RFF kernel critic (closed form) | 57.2 | 36.2 | 21.3 | 255.5 | 0.42 |

Contrasts VCS − X (percentage points; 95 % t over seeds; image bootstrap shared across methods):
| X | success < 42 px (frozen) | success < 9 px (secondary analysis) |
|---|---|---|
| **MI (primary)** | **+18.1** (seeds constant; image [+7.5, +30.3]) | +19.4 [+11.3, +27.5]; image [+9.1, +31.5] |
| **CoMIR (primary)** | **+9.1 [+8.3, +9.8]; image [0.0, +18.5]** | +11.1 [+8.0, +14.3]; image [+2.0, +21.2] |
| NMI | +19.7 (seeds constant; image [+8.4, +31.6]) | +19.7 [+11.6, +27.8]; image [+8.6, +32.0] |
| kernel | +42.8 (image [+28.1, +57.2]) | +38.8 [+30.6, +46.9]; image [+20.7, +55.9] |
| logistic | +1.1 [−3.1, +5.4]; image [0.0, +3.4] | +9.2 [−7.8, +26.1]; image [+1.2, +18.0] |
| InfoNCE | +4.8 [−0.8, +10.4]; image [0.0, +12.2] | +17.6 [−9.7, +45.0]; image [+9.5, +26.2] |

**Reading rule.**
- The protocol lists two intervals (seeds and images) without naming a primary one.
- A difference is read as detected only when both exclude 0.  This conservative rule was fixed at reading time, and each interval is shown.
- Where VCS succeeds at 100 % for every seed, the seed interval of a contrast with a deterministic method collapses to a point.  The image bootstrap
  then carries the uncertainty.

## 2. Pre-fixed readings (frozen wording)
1. **Ceiling readout not triggered.**  CAL success < 42 px is 0.875 for MI and NMI and 0.50 for the kernel, below 0.95, so translation-only
   BF/SHG discriminates between methods.
2. **VCS − MI (primary): "VCS registers 18.1 percentage points more DEV registrations (< 42 px)"** (image bootstrap [+7.5, +30.3]).
3. **VCS − CoMIR (primary): +9.1 points.**  The seed interval excludes 0, but the image bootstrap reaches 0 ([0.0, +18.5]).  Under the reading
   rule: **"interval containing 0 → no detectable difference"** at < 42 px.
   - CoMIR's misses concentrate on a few DEV images.
   - At < 9 px (secondary analysis) both intervals exclude 0: +11.1 points.
4. **"VCS ≈ logistic ≈ InfoNCE → a learned-critic class effect; no VCS-specific claim."**  VCS − logistic and VCS − InfoNCE contain 0 at
   < 42 px, and their seed intervals contain 0 at < 9 px.  The gain over MI / NMI belongs to the learned-critic class, not to the VCS objective.
   VCS has the highest mean at both thresholds (75.0 vs 65.8 / 57.4 at < 9 px), which is descriptive only.
5. **"A stronger CoMIR is reported as is":** CoMIR is not stronger here.

## 3. Secondary observations (descriptive)
- **Failure shape.**
  - MI / NMI fail catastrophically on about one registration in five (p90 error ≈ 170 px).
  - The learned critics' 90th percentile stays at 15–21 px, and no critic seed failed a CAL image.
- **The closed-form raw-patch kernel critic is the worst energy**, consistent with the earlier P60 negative for raw-pixel closed-form VCS on
  EPFL RGB-NIR.  What works is a learned local relation, not the closed-form estimator on raw patches.
- **InfoNCE varies most across seeds** (< 9 px: 60.3 / 65.0 / 46.9 %).  VCS varies least (71.3 / 76.6 / 77.2 %).
- **Speed:** CoMIR registers fastest (0.16 s, a single latent per image).  Critic registrations take 0.5–0.8 s on GPU.

## 4. Caveats
- 32 DEV images; translation only; one optimiser for every energy.
- CoMIR is evaluated as "CoMIR features + our optimiser", not its native SIFT / α-AMD pipeline.
- The < 9 px contrasts were computed after the results.  The 9 px success rate was a pre-registered primary metric, but its contrasts were not.
- The implementing agent saw the MI result line before REG-P2's design was final (disclosed; P2's design did not change).

## 5. Next
- **REG-P2 (rigid: ±30° rotation + the same translations; same models, re-selected on P2's CAL; encode-once CoMIR primary,
  warp-then-encode CoMIR as a sensitivity row)** is frozen and launched once its smoke completes.  It tests whether the class effect survives a
  harder transform family, where the CoMIR paper reports MI failing more often.
