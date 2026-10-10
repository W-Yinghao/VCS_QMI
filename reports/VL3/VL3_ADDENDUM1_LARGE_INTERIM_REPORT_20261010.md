# VL3 addendum 1 — full fine-tuning on Large backbones: interim report (RefCOCOg cells complete at three seeds) — 2026-10-10

Protocol `VL3_ADDENDUM1_LARGE_FROZEN_20261010.md`; results-only commit `5338ee5` (`reports/VL3/VL3_results.json`, job 1033167).  CLIP ViT-L/14@336
and SigLIP 2 L/16, all encoder parameters trained with gradient checkpointing; recipe and selection as VL3 (AdamW 1e-5, 10 epochs, CAL-Top-1
selection).  **Gate 2 (step-0 = frozen zero-shot within 0.3): every Large run passes** (76 / 76 VL3 runs in the aggregation).

## 1. RefCOCOg × Large — complete (3 seeds; DEV image-macro Top-1, mean ± sd; DEV J_recal × 100)
| cell | VCS | matched JS | candidate softmax | VCS − JS | VCS − softmax |
|---|---|---|---|---|---|
| RefCOCOg × CLIP-L/14@336 | **83.59 ± 0.19** · J 16.88 | 83.44 ± 0.24 · J 16.97 | 82.51 ± 0.60 · J 12.10 | +0.14 [−0.06, +0.35] | +1.08 [−0.41, +2.56] |
| RefCOCOg × SigLIP 2 L/16 | **87.26 ± 0.18** · J 20.38 | 87.13 ± 0.20 · J 20.44 | 86.79 ± 0.12 · J 15.20 | +0.13 [−0.45, +0.71] | **+0.47 [+0.18, +0.76]** |

Fine-tuned − frozen-feature Table A (same objective): CLIP-L VCS +14.4, JS +14.3, softmax +10.6; SigLIP 2-L VCS +11.2, JS +11.5, softmax +9.3.
Large − B/16 (RefCOCOg, three-seed means): CLIP VCS +2.68, JS +2.58, softmax +2.49; SigLIP 2 VCS +2.25, JS +2.37, softmax +1.72.

## 2. Seed-0 status of the other Large cells (seed rule; add. 3 now completes every cell at three seeds)
| cell | VCS | JS | softmax | status |
|---|---|---|---|---|
| RefCOCO × CLIP-L | 89.21 (s1 89.33) | 89.11 | 88.87 | triggered (VCS best); seeds running |
| RefCOCO × SigLIP 2-L | 90.74 | 91.34 | 91.74 | not triggered (−1.00) → add. 3 |
| RefCOCO+ × CLIP-L | 83.75 | 84.61 | 84.60 | not triggered (−0.87) → add. 3 |
| RefCOCO+ × SigLIP 2-L | 89.40 | 89.38 | 90.04 | not triggered (−0.64) → add. 3 |

## 3. Interim reading (two complete Large cells; the pooled reading waits for add. 3)
- **VCS ≈ matched JS** at scale (both intervals contain 0; +0.13 / +0.14).
- **VCS vs softmax on RefCOCOg:** VCS is ahead in both Large cells.  The SigLIP 2-L interval excludes 0 (+0.47 [+0.18, +0.76]).  This is the
  second such interval in VL3 after RefCOCOg × CLIP B/16 (+0.89 [+0.03, +1.75]), and both favour VCS.  CLIP-L is +1.08 but its interval contains 0.
  On RefCOCO / RefCOCO+ at seed 0, softmax or JS leads by 0.6–1.0 in three Large cells.  So the RefCOCOg advantage is not yet a general one.  The
  complete 12-cell table (add. 3) decides the pooled statement with wording fixed in advance.
- **Estimator view:** VCS / JS fine-tuned encoders expose 34–39 % more critic-fittable dependence (J_recal) than softmax fine-tuned ones in both
  Large cells.  This matches the B/16 range (21–43 %).
- **Fine-tuning value grows with scale for every objective, slightly more for the estimator objectives on SigLIP 2** (Large − B/16 +2.3 vs +1.7).
- Gradient checkpointing does not move step 0 (gate 2 passes within 0.20 for every Large run; largest gap RefCOCO × CLIP-L 58.92 vs 59.12).
