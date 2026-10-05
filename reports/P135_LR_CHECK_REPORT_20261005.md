# P135 — optimizer learning-rate check for VCS (A-P3) and matched JS-AP3, CIFAR-10 and CIFAR-100, seed 0 — report — 2026-10-05

Pre-registration `P135_LR_CHECK_AP3_FAMILY_PREREG_FROZEN_20261004.md`; results-only commit `79127d3` (`P135_results.txt`, `reports/P135/`).  (a, κ) = (2, 0.5);
lr 1e-3 = the existing seed-0 runs.  Some units moved between GPU types (wall time not compared).

## 1. Results (linear / kNN %, seed 0)

| loss, dataset | lr 5e-4 | lr 1e-3 | lr 2e-3 | selected (Δ vs 1e-3) | rule |
|---|---|---|---|---|---|
| VCS, CIFAR-10 | 88.72 / 87.02 | **89.06** / 87.30 | 88.94 / 87.84 | 1e-3 | keeps 1e-3 |
| VCS, CIFAR-100 | 60.38 / 55.20 | 60.20 / 55.92 | **60.52** / 56.68 | 2e-3 (+0.32) | keeps 1e-3 (< 0.50) |
| JS, CIFAR-10 | 88.28 / 87.14 | 88.72 / 87.50 | **88.86** / 87.56 | 2e-3 (+0.14) | keeps 1e-3 |
| JS, CIFAR-100 | 59.30 / 54.66 | 58.76 / 54.80 | **59.36** / 55.90 | 2e-3 (**+0.60**) | **replaces 1e-3 → seeds 1–2** (addendum 1) |

## 2. JS-vs-VCS sensitivity (the check P114 / P120 require)
Seed-0 VCS − JS linear at 5e-4 / 1e-3 / 2e-3: CIFAR-10 +0.44 / +0.34 / +0.08; CIFAR-100 +1.08 / +1.44 / +1.16 → **the gap keeps its sign at all three
learning rates on both datasets ("holds across lr")**.  Hence the P120 addendum-1 result (CIFAR-100 A-P3 − JS clear at 5 seeds, +0.80) is not an
artefact of the common lr 1e-3: at the shared scorer (2, 0.5), VCS stays ahead of JS on CIFAR-100 across a 4× lr range (one seed per lr).

## 3. Reading
- VCS is lr-insensitive in this range (all cells within ±0.35 of 1e-3 on linear); JS gains 0.6 on CIFAR-100 from a larger lr — like the scorer grid
  (P129), JS benefits more from tuning than VCS.  The general statement "VCS > JS on CIFAR-100" is therefore restricted to the shared scorer
  (2, 0.5); with per-loss scorer tuning the CIFAR-100 comparison is decided by P129 addendum 3 (pending), and on CIFAR-10 tuned VCS and tuned JS
  are close (P129 addendum 1).
- Addendum 1 (frozen with this report): JS CIFAR-100 lr 2e-3 seeds 1–2, submitted now.
