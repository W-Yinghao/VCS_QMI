# P127 — v6 §8.2 per-dataset (a, κ) grid for VCS on the A-P3 structure, CIFAR-10 and CIFAR-100, seed 0 — report — 2026-10-05

Pre-registration `P127_V6_AFFINE_GRID_PREREG_FROZEN_20261003.md`; results-only commit `df246a3` (`P127_results.txt`, `reports/P127/`).  Each cell = that
dataset's A-P3 seed-0 config with only (a, b = −aκ) changed; (2, 0.5) = A-P3 seed 0.  Development validation split; official test closed.  Several
units were moved between GPUs mid-run (requeue + epoch-boundary resume; `slurm_logs/gpu_upgrade.log`).  SimCLR grid (P128) withdrawn by the owner,
so this is a VCS dataset-selection table only (no tuned-vs-tuned VCS-vs-SimCLR statement); the matched-JS grid is P129 (running).

## 1. Results (linear / kNN %, seed 0; Δ vs (2, 0.5))

| (a, κ) | CIFAR-10 | Δ lin / kNN | CIFAR-100 | Δ lin / kNN |
|---|---|---|---|---|
| **(2, 0.5)** A-P3 | 89.06 / 87.30 | — | 60.20 / **55.92** | — |
| (1.5, 0.5) | 88.98 / **87.42** | −0.08 / +0.12 | 58.92 / 54.92 | −1.28 / −1.00 |
| (2, 0.25) | 88.36 / 86.38 | −0.70 / −0.92 | 60.30 / 55.24 | +0.10 / −0.68 |
| (2, 0.75) | **89.30** / 87.24 | **+0.24** / −0.06 | 58.34 / 53.10 | −1.86 / −2.82 |
| (3, 0.5) | 88.10 / 86.58 | −0.96 / −0.72 | **60.64** / 55.70 | **+0.44** / −0.22 |
| (3, 0.25) | 87.28 / 85.22 | −1.78 / −2.08 | 60.18 / 55.42 | −0.02 / −0.50 |

## 2. Reading (frozen rule)
- **Dataset-selected cell = highest linear:** CIFAR-10 (2, 0.75) +0.24 (< 0.30), CIFAR-100 (3, 0.5) +0.44 (< 0.50).  Neither reaches its threshold
  (≈ 2 × A-P3 seed sd), so **(2, 0.5) stays the VCS cell on both datasets**; no seeds 1–2.
- Descriptive: the grid is flat near (2, 0.5) on CIFAR-10 (κ = 0.75 or a = 1.5 within ±0.3, kNN within ±0.15) and penalises a small κ or a large a;
  on CIFAR-100 the best cells are larger-a (3, 0.5) and (2, 0.25)/(3, 0.25) — a and κ interact (same zero-crossing scale aκ ≈ 0.5–1.5) — while
  κ = 0.75 is clearly worse.  (2, 0.5) has the best CIFAR-100 kNN of the grid.  One seed per cell: no ordering claimed among the near-zero cells.
- The CIFAR-100 coarse / fine / conditional readouts (P124 protocol, evaluation-only) for the five new cells run as job 1021612 (`reports/P127_gran`)
  and are appended as an addendum.
- Next in this line: P129 (the same grid for matched JS) → tuned-VCS vs tuned-JS (frozen in P129); P135 (lr check for both losses).
