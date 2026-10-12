# R5-X1 — R4-VL1 (proposal-aware full fine-tuning) replicated on RefCOCO — report — 2026-10-12

Protocol `reports/R5/R5_X1_REFCOCO_PROTOCOL_FROZEN_20261011.yaml` (owner's standing authorization; frozen before any result).  Results-only
commit `9964113` (`reports/R4/R4_VL1_results_refcoco.json`, `R4_VL1_tables_refcoco.md`; aggregate job 1034958).
- **Runs:** 9 full fine-tuning runs: RefCOCO × SigLIP 2 B/16 × {VCS, matched JS, set-event softmax} × seeds 0–2.
- **Fixed from R4-VL1:** recipe, roles, detector proposals (VL1-21), laws (uniform ν) and selection (CAL all-proposal Acc@0.5).
- **Gates:** 9 / 9 pass.  No run missing.  DEV exploratory; official val / test untouched.

## 1. Results (DEV image-macro %, mean ± sd over 3 seeds)
| quantity | VCS | JS | softmax |
|---|---|---|---|
| **all-proposal Acc@0.5, proposal-trained (primary)** | **66.60 ± 0.63** | **66.54 ± 0.45** | **75.45 ± 0.30** |
| same proposals, given-box-trained VL3 checkpoints (Table C) | 56.56 | 55.82 | 57.76 |
| adaptation gain (paired by seed) | +10.04 [+7.45, +12.63] | +10.72 [+9.88, +11.56] | +17.69 [+16.63, +18.75] |
| zero-shot SigLIP 2 B/16 (step 0) | 28.7 | 28.7 | 28.7 |
| Acc@0.75 | 53.69 | 53.51 | 62.53 |
| query-level Acc@0.5 | 65.15 | 65.13 | 74.00 |
| given-box Top-1 of the new encoder | 75.42 | 74.51 | 83.33 |
| J_recal × 100 (same p / q) | 61.00 | 61.73 | 58.20 |
| margin best-valid − best-invalid (× 100 cos) | 5.89 | 5.35 | 13.75 |
| detector top-K, new encoder (K = 32 chosen on CAL) | 67.07 | 66.87 | 75.59 |
| CAL-selected epoch | 8, 10, 7 | 8, 8, 4 | 5, 7, 5 |

| contrast (paired by seed, 95 % t) | RefCOCO |
|---|---|
| **VCS − softmax** | **−8.84 [−10.57, −7.12]** |
| VCS − JS | +0.06 [−1.36, +1.49] |

## 2. Pre-fixed readings (R4-VL1 wording, with the r5 erratum)
1. **Candidate-support adaptation works for every objective** (+10.0 / +10.7 / +17.7 points; intervals far from 0).  It is a candidate-support
   adaptation result, not a VCS-only advantage.
2. **"Softmax leads by 8.84 points on proposals."**
3. **VCS − JS: the interval contains 0** (+0.06 [−1.36, +1.49]).  Containment, not equivalence (r5 erratum).

## 3. Three datasets side by side (pooling descriptive)
| | RefCOCOg | RefCOCO+ | RefCOCO |
|---|---|---|---|
| VCS − softmax, all proposals | −1.85 [−2.41, −1.29] | −3.22 [−5.08, −1.37] | **−8.84 [−10.57, −7.12]** |
| VCS − JS | +0.38 | +0.17 | +0.06 |
| adaptation gain VCS / softmax | +15.5 / +16.7 | +19.6 / +20.6 | **+10.0 / +17.7** |
| given-box Top-1 of the new encoder, VCS / softmax | 81.47 / 82.16 | 81.66 / 83.23 | **75.42 / 83.33** |

## 4. Secondary observations (descriptive)
- **RefCOCO is the extreme case.**  Its expressions are location-heavy and its images hold many objects of the referred category.
  - The softmax lead is 2.7–4.8× larger than on the other two datasets.
  - VCS / JS gain only +10 points from proposal-aware training, against +17.7 for softmax.  On the other datasets the gains were similar.
- **The deficit appears on annotated boxes too.**  On RefCOCO, the proposal-trained VCS encoder trails softmax by 7.9 points of given-box Top-1,
  against 0.7 / 1.6 on RefCOCOg / RefCOCO+.  It is therefore not only a many-proposal effect.
  - This is consistent with RefCOCO's many same-category annotated objects.
  - The R5-20 nested-curve diagnostics have not been run for RefCOCO.
- **Same estimator-side pattern as before:** VCS / JS encoders expose more critic-fittable dependence (J_recal 61.0 / 61.7 vs 58.2) while
  ranking worse; margins are less than half of softmax's.
- **Selection:** VCS seed 1 selected the last epoch (10).  The recipe is fixed (no schedule change), as in R4-VL1.
- **Not computed:** detector top-K with the VL3 checkpoints.  R4-VL0 scored VL3 checkpoints only for RefCOCOg and RefCOCO+.

## 5. What tests the explanation here
R5-QN showed that within-query normalization closes 61–69 % of the gap on RefCOCOg / RefCOCO+.  **R5-QX1** (VQ / JQ / distribution softmax on
RefCOCO, frozen 2026-10-12, 9 runs queued) applies the same test where the gap is largest and completes the critic-class / criterion / target
decomposition on a third dataset.

## 6. Caveats
- One backbone; three seeds (df 2); DEV read repeatedly (exploratory).
- RefCOCO proposals come from the same detector, which saw the FIT images (recall recorded in VL1-21).
