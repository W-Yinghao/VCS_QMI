# VL3 addendum 6 — cross-dataset transfer of the fine-tuned encoders — report — 2026-10-10

Protocol `VL3_ADDENDUM6_TRANSFER_FROZEN_20261010.md`; results-only commit `b67dc7b` (`reports/VL3/VL3_add6_results.json`, `scripts/vl3_add6_aggregate.py`,
job 1033567).  Every one of the 108 VL3 checkpoints is evaluated on the DEV split of the two other datasets (given boxes).  Target images the
source model trained on or was selected on (source FIT ∪ CAL) are excluded.  **Gate X (zero-shot full-DEV within 0.3 of the frozen raw value):
24 / 24 pass.**

**Correction to the protocol text.**  The protocol stated that RefCOCO and RefCOCO+ share the same seeded partition, so their transfer would keep
every DEV image.  That is wrong.  The two datasets' train image lists differ slightly, and the seeded permutation of a different sorted list assigns
roles differently.  The exclusion therefore keeps 389 / 390 of ≈ 1 590 DEV images for RefCOCO ↔ RefCOCO+ (RefCOCOg → RefCOCO / + keeps 680 / 706;
RefCOCO / + → RefCOCOg keeps 556 / 565).  The exclusion rule was applied exactly as specified; only the expected number of kept images was
misstated.

## 1. Pre-registered readings (24 directed source → target × backbone cells, paired by seed, 95 % t)
| contrast | all 24 cells | UNC pair (8) | RefCOCOg ↔ UNC (16) |
|---|---|---|---|
| **VCS − softmax, transfer Top-1** | **−0.10 [−0.39, +0.18]** (cells > 0: 0, < 0: 1) | −0.14 [−0.79, +0.52] | −0.09 [−0.43, +0.25] |
| VCS − JS, transfer Top-1 | +0.07 [−0.07, +0.21] | +0.15 [−0.14, +0.44] | +0.03 [−0.14, +0.20] |
| transfer gap (in-domain − transfer), VCS − softmax | −0.20 [−0.57, +0.17] | −0.66 [−1.45, +0.14] | +0.02 [−0.39, +0.43] |
| transfer gap, VCS − JS | −0.03 [−0.18, +0.11] | −0.06 [−0.45, +0.32] | −0.02 [−0.17, +0.13] |

**Pre-fixed wording (interval contains 0): "transfer does not differ between the estimator objective and the task loss."**  The only per-cell
interval excluding 0 is RefCOCOg → RefCOCO+ × SigLIP 2 B/16 (−0.42 [−0.56, −0.27]).

## 2. Descriptive
- **Transfer works for every objective.**  Transfer − zero-shot on the same kept images is +7.9 to +19.0 points (VCS), with similar values for JS
  and softmax.
- **What does not transfer is dataset-specific, and it is the same for all objectives.**  In-domain − transfer on the same images is 12–16 points
  into RefCOCO, but only 0.5–3.3 points into RefCOCO+ and RefCOCOg.  RefCOCO is the dataset with heavy location wording ("left man").  A model
  trained on RefCOCO+ (where location words are forbidden) or on RefCOCOg (long descriptive expressions) does not learn it.  That matches the
  location-word interaction in add. 5.
- The extra critic-fittable dependence of VCS / JS encoders (J_recal ×1.33 in-domain) does not show up as better cross-dataset ranking.

## 3. Reading
After full fine-tuning, VCS and the task loss are equivalent in-domain (VL3, +0.01 [−0.35, +0.36]) and across datasets (−0.10 [−0.39, +0.18]).
The estimator objectives' advantage is in what the encoder exposes to a critic (J_recal), not in ranking accuracy, in-domain or transferred.
