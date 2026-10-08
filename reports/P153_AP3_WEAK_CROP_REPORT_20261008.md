# P153 — A-P3 with weaker crops (RandomResizedCrop scale min 0.20 → 0.35), CIFAR-10 / CIFAR-100, seed 0 — report — 2026-10-08

Pre-registration `P153_AP3_WEAK_CROP_PREREG_FROZEN_20261008.md`; results-only commit `16bb443` (`P153_results.json`).

| dataset | crop min 0.35: linear / kNN | A-P3 (crop min 0.20) | Δ linear / kNN | threshold |
|---|---|---|---|---|
| CIFAR-10 | 86.86 / 86.46 | 89.06 / 87.30 | −2.20 / −0.84 | +0.30 |
| CIFAR-100 | 55.00 / 52.80 | 60.20 / 55.92 | −5.20 / −3.12 | +0.50 |

Reading (frozen rule): **not promising on either dataset — the crop axis ends at 0.20.**  Weaker crops cost far more than stronger crops did
(P115: crop min 0.08 −0.19 linear on C10), and more on the finer-grained dataset; consistent with the stage-B weak-augmentation cell (−6 on the old
recipe).  With P115 this brackets the crop scale from both sides: 0.20 is the best tested value.  Single seed; jitter not varied here.
