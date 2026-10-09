# VL1-12 addendum 3 — Table A on FG-CLIP v1 Base (CLIP-initialised) — report — 2026-10-09

Addendum `VL1_12_ADDENDUM3_FGCLIP1_FROZEN_20261009.md`; results-only commit `f788a09` (`VL1_12f1_table.json`, `VL1_12f1_fgclip1_minus_clip.json`,
`VL1_12f1_strata.json`; job 1031446).  FG-CLIP v1 Base @ 454d763, RoIAlign region interface (224 squash, 14 × 14 grid boxes); COCO box check
52.28 (CLIP B/16 44.2).  N all × 3 seeds; same roles / law / F2r / selection as VL1-10.

## 1. Results (DEV image-macro Top-1 %, CAL common J × 100)
| route | CLIP ViT-B/16 crops | **FG-CLIP v1 region API** | FG-CLIP v1 − CLIP (paired) Top-1 | CAL J |
|---|---|---|---|---|
| raw cosine | 65.99 | 70.57 | +4.58 (no seeds) | — |
| VCS (estimator-selected) | 68.80 / J 7.20 | **73.78 ± 0.06 / J 9.79** | +4.98 [+3.67, +6.29] | **+2.59 [+2.20, +2.98]** |
| matched JS | 68.34 / J 7.05 | 73.38 ± 0.35 / J 9.53 | +5.04 [+4.36, +5.71] | **+2.48 [+2.44, +2.52]** |
| RFF ridge-tanh | 67.40 / J 5.23 | 71.86 ± 0.17 / J 6.36 | +4.46 [+4.03, +4.89] | **+1.13 [+0.93, +1.32]** |
| candidate softmax (DEV-selected, optimistic) | 70.75 | 76.13 ± 0.15 | +5.37 [+4.62, +6.13] | — |

On FG-CLIP v1: learned − raw = **VCS +3.21 [+3.06, +3.36]**, JS +2.81, RFF +1.29 [+0.87, +1.71], softmax +5.56 (optimistic).  VCS − JS: Top-1
+0.40 [−0.54, +1.35], CAL J +0.26 [+0.01, +0.51] (close on Top-1).  Strata: VCS − raw positive in every stratum (same-category +3.10,
other-category +1.92); softmax − VCS +2.55 (query-weighted).  The fits again picked lr 2e-3 (grid edge), with the CAL-J optimum at 2 850–2 950
of the 3 000 updates.

## 2. Frozen readings
1. **Same-initialisation contrast:** FG-CLIP's fine-grained training + region interface raises the dependence every estimator reads
   (J +2.6 / +2.5 / +1.1; all three agree in sign) and the learned Top-1 by ≈ 5 points.  Unlike FG-CLIP 2 − SigLIP 2, this contrast is
   positive.  The released FG-CLIP 2 checkpoint is the outlier, not the fine-grained recipe.
2. **The critic adds value on these features** (+3.2 over raw), as on CLIP (+2.8) and unlike FG-CLIP 2 (+0.2) / SigLIP 2 (+0.5).
3. VCS − JS close (same posterior).  4. Table B row: raw FG-CLIP v1 = 70.57 image-macro.

## 3. Correction to the probe statement (VL1-12 add. 2 report §3)
With five feature sets the neural-estimator J order is **SigLIP 2 (11.6) > FG-CLIP v1 (9.8) > FG-CLIP 2 (8.8) > CLIP (7.2)** (ReCLIP features:
JS 9.6, VCS pending).  The zero-shot order is SigLIP 2 (75.5) > FG-CLIP 2 (72.3) > ReCLIP (70.8) ≈ FG-CLIP v1 (70.6) > CLIP (66.0).  **J does not
follow the zero-shot accuracy** (FG-CLIP 2 vs FG-CLIP v1 swap).  It follows the **learned** Top-1 of the same critic: VCS 76.0 > 73.8 > 72.5 >
68.8.  The add. 2 sentence "in the order of their zero-shot grounding accuracy" is withdrawn.  The supportable statement: "the fitted
critic's J ranks frozen feature sets in the order of the ranking accuracy the same critic reaches after fitting".  This is a consistency
property of the estimator, not an external validation.
