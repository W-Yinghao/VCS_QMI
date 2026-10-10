# VL3 — full fine-tuning: interim report (RefCOCOg × CLIP complete at three seeds; other cells at seed 0) — 2026-10-10

Protocol `VL3_FULL_FINETUNE_FROZEN_20261010.md`; interim results-only commit `dca0a68` (`reports/VL3/VL3_results.json`, `scripts/vl3_aggregate.py`).  DEV
image-macro Top-1 at the CAL-Top-1-selected epoch; all encoder parameters trained; shared recipe (AdamW 1e-5, 10 epochs, critic f = a·cos + b).
**Gate 2 (step-0 = frozen zero-shot within 0.3): 23 / 23 runs pass.**

## 1. RefCOCOg × CLIP ViT-B/16 — complete (3 seeds)
| objective | fine-tuned DEV Top-1 | − zero-shot (66.0) | frozen-feature Table A (same objective) | − frozen | DEV J_recal × 100 |
|---|---|---|---|---|---|
| **VCS** | **80.91 ± 0.32** | +14.8 | 68.80 | **+12.1** | 14.45 |
| matched JS | 80.86 ± 0.20 | +14.7 | 68.34 | +12.5 | 14.52 |
| candidate softmax | 80.02 ± 0.06 | +13.9 | 70.75 | +9.3 | 10.14 |
Paired by seed: **VCS − JS +0.05 [−0.99, +1.10]** (close); **VCS − softmax +0.89 [+0.03, +1.75]**.

## 2. Seed-0 status of the other cells (seeds 1–2 running where the pre-stated rule triggered)
| cell | VCS | JS | softmax | rule |
|---|---|---|---|---|
| RefCOCOg × SigLIP 2 | 84.80 (s1 84.98) | 84.65 | 85.30 | triggered (−0.50) |
| RefCOCO × CLIP | 84.66 | 84.36 | 85.26 | **not triggered (−0.60)** → reported at one seed |
| RefCOCO × SigLIP 2 | 88.28 | 88.39 | 87.76 | triggered (−0.11) |
| RefCOCO+ × CLIP | 82.06 | 81.55 | 82.01 | triggered (VCS best) |
| RefCOCO+ × SigLIP 2 | running | 86.82 | running | — |

## 3. Interim reading (one complete cell; not a conclusion)
- **Full fine-tuning changes the picture from the frozen-feature tables.**  On frozen features the task loss (softmax) was the best ranker by ≈ 2 points
  in every cell.  Fine-tuning the encoders, the estimator objectives catch up and, in this cell, VCS ranks above softmax.  The interval just
  excludes 0 at n = 3, so the paired interval of the remaining cells decides whether this generalises.  The seed-0 signs are mixed (softmax ahead
  on RefCOCO × CLIP and RefCOCOg × SigLIP 2; VCS ahead on RefCOCO × SigLIP 2 and RefCOCO+ × CLIP).
- **VCS ≈ matched JS** after fine-tuning as on frozen features (same posterior).
- **Estimator view:** encoders fine-tuned with VCS / JS expose about 40 % more critic-fittable dependence on DEV (J_recal 14.5) than encoders fine-tuned with
  softmax (10.1), at a comparable or better Top-1.  Softmax's own critic is uncalibrated (J_own ≈ 0 or negative), as expected for a ranking loss.
- All objectives peak at epochs 3–10.  The recipe was shared and not tuned per objective (no per-objective search, as for SimCLR in SSL).

## Update — RefCOCOg × SigLIP 2 B/16 complete (3 seeds; results-only commit `548a230` — committed after this section was first pushed)
| objective | fine-tuned DEV Top-1 | frozen-feature Table A | − frozen | DEV J_recal × 100 |
|---|---|---|---|---|
| VCS | 85.01 ± 0.23 | 76.00 | +9.0 | 18.20 |
| matched JS | 84.76 ± 0.12 | 76.43 | +8.3 | 18.16 |
| candidate softmax | 85.07 ± 0.23 | 78.22 | +6.9 | 13.98 |
Paired: **VCS − JS +0.26 [−0.35, +0.86]**; **VCS − softmax −0.05 [−1.22, +1.11]** (tied).
Two complete cells so far: after full fine-tuning **VCS matches or exceeds the task loss on ranking** (CLIP +0.89 [+0.03, +1.75]; SigLIP 2 −0.05
[−1.22, +1.11]), where on frozen features softmax led by ≈ 2 points.  **VCS / JS fine-tuned encoders expose 30–40 % more critic-fittable
dependence** than softmax fine-tuned ones in both cells.  The fine-tuning gain over frozen features is largest for the estimator objectives (+8 to
+12) and smallest for softmax (+7 to +9).
