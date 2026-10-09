# VL1-12 — Table A on FG-CLIP 2 Base region features (estimator as a probe) — report — 2026-10-09

Addendum `VL1_12_ADDENDUM_FROZEN_20261009.md`; results-only commits `d0ed119` (`VL1_12_table.json`, `VL1_12_results.json`) and `6c5d7d7`
(`VL1_12_strata.json`).  Same roles / laws / scorer / selection as VL1-10; features = FG-CLIP 2 Base (released checkpoint @ 430fbc8, official
RoIAlign region API; COCO box classification 64.8 under the official script settings vs 74.9 reported, discrepancy documented).  N all × 3 seeds.

## 1. Results (DEV image-macro Top-1 %, CAL common J × 100; mean over seeds)
| route | CLIP ViT-B/16 crops | FG-CLIP 2 region API | FG-CLIP 2 − CLIP (paired) Top-1 | FG-CLIP 2 − CLIP CAL J |
|---|---|---|---|---|
| raw cosine | 65.99 | **72.29** | +6.30 (no seeds) | — |
| VCS (estimator-selected) | 68.80 / J 7.20 | 72.51 / J 8.76 | +3.71 [+2.57, +4.86] | **+1.56 [+0.90, +2.23]** |
| matched JS (estimator-selected) | 68.34 / J 7.05 | 72.12 / J 8.59 | +3.78 [+2.80, +4.77] | **+1.54 [+1.20, +1.88]** |
| RFF ridge-tanh | 67.40 / J 5.23 | 72.23 / J 4.28 | +4.83 [+3.94, +5.73] | **−0.95 [−1.02, −0.89]** |
| candidate softmax (DEV-selected, optimistic) | 70.75 | 76.14 | +5.38 [+4.27, +6.50] | — |

On FG-CLIP 2 features: learned − raw = VCS +0.23 [+0.13, +0.33], JS −0.17 [−0.98, +0.65], RFF −0.05 [−0.43, +0.32], softmax +3.85 [+2.92,
+4.78] (optimistic).  VCS − JS: Top-1 +0.39 [−0.51, +1.30] (task-selected +0.09), CAL J +0.17 [−0.39, +0.73] → close (same posterior).

## 2. Reading (frozen; descriptive)
1. **Raw features:** FG-CLIP 2's official region path ranks the referred objects 6.3 points better than CLIP ViT-B/16 crops (query 71.11 vs 64.22),
   with no task training.
2. **Estimator as a probe:** the two neural estimators agree that FG-CLIP 2 region–phrase features carry more of the conditional relation than
   CLIP crops: CAL J +1.5 (×100; 7.2 → 8.8, about +22 %), with intervals well above 0 and the same size for VCS and JS.  **The kernel route
   disagrees in sign** (5.23 → 4.28).  Its selected configuration sits at the corner of its grid on both feature sets (largest bandwidth 2 × median
   distance, smallest ridge 1e-4), so it is grid-limited.  Writing item W11 applies here as well.  Fitted J across feature sets compares estimation
   difficulty as well as relation strength.  The cross-feature statement is therefore made with the neural routes only, and the RFF disagreement is
   reported beside it.
3. **Learning on top of stronger features:** with FG-CLIP 2 the estimator routes add ≤ 0.2 point Top-1 over raw (VCS +0.23, JS −0.17), while their
   J still rises with training.  The task loss still adds ~+3.9 (optimistic).  The strata say the same: VCS − raw is +0.21 [+0.02, +0.39] on
   same-category queries and −0.43 on other-category ones, while softmax − VCS is +3.72 [+2.74, +4.71] (query-weighted).  **On strong
   features, a better estimate of the dependence (higher J) does not turn into a better ranking**.  The VCS critic's improvement is in calibration
   on the balanced P / Q law, not in the per-image argmax.  This is the real-data analogue of writing item W9 (detection power ≠ estimable
   dependence): here estimation quality ≠ ranking gain.
4. **Budget:** as in VL1-10, the 3 000-update cap binds for VCS and JS (17 of 18 estimator-selected checkpoints in the last 250 updates).

## 3. Consequences for the plan
- The probe statement for the paper is **"two neural estimators agree that the fine-grained-trained region features carry more region–phrase
  dependence (+22 % J)"**, worded as a fit-limited comparison, never as an S comparison.  The kernel route is shown as the counter-reading.
- SigLIP 2 Base crops (addendum 2, cache job queued) separate the backbone family from FG-CLIP 2's fine-grained training plus region interface
  (confounded, stated in the addendum).
- The two optional items below are not submitted.  (a) VCS / JS at N all with a 10 000-update cap would show whether the budget-limited J keeps
  rising (CPU only).  (b) A wider RFF grid would test whether the kernel route's opposite reading survives.
