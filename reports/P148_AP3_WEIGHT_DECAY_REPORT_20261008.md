# P148 — A-P3 matrix weight decay 1e-4 → {5e-4, 1e-5}, CIFAR-10 / CIFAR-100, seed 0 — report — 2026-10-08

Pre-registration `P148_AP3_WEIGHT_DECAY_PREREG_FROZEN_20261007.md`; results-only commit `276d7f5` (`P148_results.json`).  Only
`optimizer.matrix_weight_decay_encoder_projector` changed; 800 epochs, development split, final frozen-h linear (kNN alongside).

| dataset | wd 5e-4 linear / kNN | Δ vs A-P3 | wd 1e-5 linear / kNN | Δ vs A-P3 | A-P3 (wd 1e-4) | threshold |
|---|---|---|---|---|---|---|
| CIFAR-10 | 89.28 / 87.80 | +0.22 / +0.50 | 88.92 / 87.24 | −0.14 / −0.06 | 89.06 / 87.30 | +0.30 |
| CIFAR-100 | 60.46 / 56.06 | +0.26 / +0.14 | 60.52 / 55.80 | +0.32 / −0.12 | 60.20 / 55.92 | +0.50 |

Reading (frozen rule): **no cell is promising** — the largest linear gains (+0.22 CIFAR-10 at 5e-4; +0.32 CIFAR-100 at 1e-5) are below their thresholds
and the two datasets do not agree on a direction (CIFAR-10 prefers more decay, CIFAR-100 is flat between 1e-5 and 5e-4).  All four cells are within
the seed-0 → 5-seed spread of A-P3 on CIFAR-100 (sd 0.38).  **The weight-decay axis ends at 1e-4.**  Single seed; other decays and decoupled schedules
not tested.  With P147 (batch) this closes the last open optimiser axes of the exploration backlog: the A-P3 recipe is flat in lr, batch, epochs,
views, projector width, momentum keys and weight decay at the registered thresholds.
