# VL3 — full end-to-end fine-tuning, complete table: 3 datasets × 4 backbones × 3 objectives × 3 seeds — report — 2026-10-10

Protocols `VL3_FULL_FINETUNE_FROZEN_20261010.md` (B/16), `VL3_ADDENDUM1_LARGE_FROZEN_20261010.md` (Large), `VL3_ADDENDUM3_ALL_CELLS_THREE_SEEDS_FROZEN_20261010.md`
(three seeds everywhere; pooled reading fixed in advance).  Results-only commit `b690bdb` (`reports/VL3/VL3_results.json`, `scripts/vl3_aggregate.py`,
job 1033353).  All encoder parameters trained; one shared recipe (AdamW 1e-5, 10 epochs, batch 32 images, critic f = a·cos + b init (10, −2.5);
Large with gradient checkpointing); DEV image-macro Top-1 at the CAL-Top-1-selected epoch.  **Gate 2 (step-0 = frozen zero-shot within 0.3):
108 / 108 runs pass.**  This report supersedes the B/16 report and the add. 1 interim for the comparison claims.

## 1. Results (mean ± sd over 3 seeds; paired by seed, 95 % t)
| cell | VCS | matched JS | candidate softmax | VCS − JS | VCS − softmax |
|---|---|---|---|---|---|
| RefCOCOg × CLIP B/16 | **80.91 ± 0.32** | 80.86 ± 0.20 | 80.02 ± 0.06 | +0.05 [−0.99, +1.10] | **+0.89 [+0.03, +1.75]** |
| RefCOCOg × SigLIP 2 B/16 | 85.01 ± 0.23 | 84.76 ± 0.12 | 85.07 ± 0.23 | +0.26 [−0.35, +0.86] | −0.05 [−1.22, +1.11] |
| RefCOCOg × CLIP L/14@336 | **83.59 ± 0.19** | 83.44 ± 0.24 | 82.51 ± 0.60 | +0.14 [−0.06, +0.35] | +1.08 [−0.41, +2.56] |
| RefCOCOg × SigLIP 2 L/16 | **87.26 ± 0.18** | 87.13 ± 0.20 | 86.79 ± 0.12 | +0.13 [−0.45, +0.71] | **+0.47 [+0.18, +0.76]** |
| RefCOCO × CLIP B/16 | 84.88 ± 0.19 | 84.79 ± 0.47 | 85.19 ± 0.07 | +0.09 [−0.81, +1.00] | −0.31 [−0.94, +0.31] |
| RefCOCO × SigLIP 2 B/16 | 88.27 ± 0.42 | 88.22 ± 0.32 | 88.55 ± 0.69 | +0.06 [−0.44, +0.55] | −0.28 [−2.17, +1.60] |
| RefCOCO × CLIP L/14@336 | **89.28 ± 0.06** | 89.05 ± 0.07 | 88.86 ± 0.07 | +0.23 [−0.06, +0.51] | **+0.42 [+0.17, +0.67]** |
| RefCOCO × SigLIP 2 L/16 | 90.91 ± 0.27 | 91.00 ± 0.38 | 91.16 ± 0.51 | −0.09 [−1.18, +1.00] | −0.25 [−2.05, +1.55] |
| RefCOCO+ × CLIP B/16 | 81.70 ± 0.32 | 81.59 ± 0.04 | 82.18 ± 0.22 | +0.11 [−0.77, +0.99] | −0.48 [−1.76, +0.80] |
| RefCOCO+ × SigLIP 2 B/16 | 86.90 ± 0.18 | 86.77 ± 0.05 | **87.44 ± 0.16** | +0.12 [−0.44, +0.68] | **−0.54 [−0.88, −0.21]** |
| RefCOCO+ × CLIP L/14@336 | 84.18 ± 0.40 | 84.29 ± 0.28 | 84.57 ± 0.22 | −0.12 [−1.76, +1.53] | −0.39 [−1.73, +0.95] |
| RefCOCO+ × SigLIP 2 L/16 | 89.43 ± 0.02 | 89.55 ± 0.15 | **89.90 ± 0.12** | −0.12 [−0.43, +0.19] | **−0.48 [−0.82, −0.13]** |

## 2. Pre-registered pooled reading (12 cells as units, 95 % t)
| contrast | pooled | B/16 (6 cells) | Large (6 cells) | per-cell intervals > 0 / < 0 |
|---|---|---|---|---|
| **VCS − softmax** | **+0.01 [−0.35, +0.36]** | −0.13 [−0.68, +0.42] | +0.14 [−0.50, +0.78] | 3 / 2 |
| VCS − JS | +0.07 [−0.01, +0.15] | +0.12 [+0.04, +0.19] | +0.03 [−0.13, +0.19] | 0 / 0 |
| J_recal ratio VCS / softmax (JS / softmax) | 1.33 (1.33) | | | |

**Wording fixed in advance (pooled interval contains 0): "after full end-to-end fine-tuning, VCS matches the task-matched candidate softmax
on grounding accuracy".**  The point estimate is +0.01 point over 12 cells, with a pooled interval of ±0.36 point.

## 3. Readings
1. **VCS matches the task loss.**  Over 3 datasets × 4 backbones, VCS and softmax are tied (+0.01 [−0.35, +0.36]).  On frozen features the task
   loss led in every cell (VL1 / VL2, by 1–10 points).  Fine-tuning the encoders closes that gap completely.
2. **Same posterior.**  VCS − JS is +0.07 [−0.01, +0.15] pooled, and every per-cell interval contains 0.  The B/16 subset shows a small, consistent
   VCS edge (+0.12 [+0.04, +0.19]); Large does not (+0.03).
3. **Estimator view.**  Encoders fine-tuned with VCS or JS expose **33 % more critic-fittable dependence** (DEV J_recal) than encoders fine-tuned with
   softmax, at equal ranking accuracy.  The ratio is 1.24–1.43 per cell and holds in all 12 cells.
4. **Fine-tuning value.**  Fine-tuned − frozen-feature critic (same objective): VCS +9.0 to +23.2, softmax +6.9 to +20.0.  The gains are largest on
   RefCOCO (dense same-category scenes).  Large − B/16 (VCS): +2.3 to +4.4 in every dataset × family.
5. **Dataset pattern (descriptive, not pre-registered; hypothesis only).**
   - Mean VCS − softmax is **+0.60 on RefCOCOg** (4 cells; 2 intervals above 0, none below), −0.11 on RefCOCO (1 above), and **−0.47 on RefCOCO+**
     (all 4 cells negative; 2 intervals below 0).
   - RefCOCOg has long, relational expressions.  RefCOCO+ forbids location words, so its expressions are short and appearance-only.
   - One possible reading: the task loss's sharper ranking pressure helps when the expression names a single attribute, and the estimator
     objective's two-sided (P vs Q) signal helps on long expressions.  This is not tested, and nothing here licenses a claim about it.
6. **The seed rule biased the pooled view, as feared.**
   - Pooling only the 7 cells the seed rule had selected (VCS within 0.5 of the best at seed 0) gives +0.29.  All 12 cells give +0.01.
   - The five cells completed by add. 3 moved toward 0 from their seed-0 values (−0.60…−1.00 → −0.25…−0.54) but stayed negative.
   - Method lesson for later units: do not condition seed counts on the seed-0 outcome when a pooled statement is planned.

## 4. Caveats
- One shared recipe, not tuned per objective (no per-objective search, including for the task baseline).  The 20-epoch schedule check (add. 2,
  one seed) moved the comparison by ≈ +0.3 toward the estimator objectives.  Add. 4 (three seeds at 20 epochs, running) gives the paired
  schedule effect.  10 epochs stays the primary schedule.
- DEV is a development split.  Official val / test are untouched until the final pass (owner decision).
- Cells are not independent (they share datasets and backbone families), so the pooled interval over cells is a descriptive summary, not a
  population inference.
