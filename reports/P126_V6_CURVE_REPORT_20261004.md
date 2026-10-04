# P126 — fixed-endpoint curvature scorer f = a[(s − κ) + λ s(1 − s)] at (a, κ) = (2, 0.5), λ = ±0.25, CIFAR-10 / CIFAR-100, seed 0 — report — 2026-10-04

Pre-registration `P126_V6_CURVE_PREREG_FROZEN_20261003.md` (owner batch approval "同意进行item1"); results-only commit `9a445dc` (`P126_results.txt`,
`reports/P126/`).  Same J, P / Q weights, all-view tokens, full gradients, encoder / projector / augmentation / schedule as A-P3; only the scorer's
curvature differs (f(0), f(1) fixed; actual zero s0 = 0.4384 for λ = +0.25, 0.5616 for λ = −0.25).  One seed per cell (screen).  Development
validation split; official test closed.  Two of the four units were moved between GPUs mid-run (requeue + epoch-boundary resume from last.pt, see
`slurm_logs/gpu_upgrade.log`); step times therefore mix GPU types and are not compared.  Peak memory 5.1–5.3 GB (as A-P3).

## 1. Results (linear / kNN %, epoch 800)

| dataset | λ = 0 (A-P3 seed 0) | λ = −0.25 | λ = +0.25 | candidate threshold |
|---|---|---|---|---|
| CIFAR-10 | 89.06 / 87.30 | 89.08 / 87.44 (+0.02 / +0.14) | 88.98 / 87.12 (−0.08 / −0.18) | +0.30 |
| CIFAR-100 | 60.20 / 55.92 | 59.34 / 55.70 (−0.86 / −0.22) | 59.86 / 56.26 (−0.34 / +0.34) | +0.50 |

## 2. Reading (frozen rule)
- **No candidate on either dataset → "no curvature gain at this (a, κ)".**  No λ cell exceeds its baseline by the threshold; on CIFAR-10 both are
  within ±0.1 of A-P3, on CIFAR-100 both are below it on linear (−0.86, −0.34; one seed, A-P3 C100 seed sd 0.25).
- Per the prereg: no seeds 1–2, no matched-JS of the same shape, no further λ values; the trade-off readouts (strong augmentation, transfer, V1) are
  not run.  Consistent with P123 (read-only: ±0.25 curvature changes the low-IoU positive-gradient share far less than κ does).
- Not claimed: that the affine scorer is optimal, or anything about other (a, κ) — the per-dataset (a, κ) grid is P127 (running).
