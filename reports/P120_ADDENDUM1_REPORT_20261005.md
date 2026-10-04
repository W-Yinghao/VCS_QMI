# P120 addendum 1 — CIFAR-100 A-P3 (VCS) vs matched JS at 5 seeds — report — 2026-10-05

Addendum `P120_ADDENDUM1_C100_SEEDS34_FROZEN_20261004.md` (labels frozen before the seed-3/4 runs; the decision to extend was disclosed as post hoc);
results-only commit `0fce831` (`P120_addendum1_5seed.{txt,json}`).  Fresh seeds 3, 4 = seed-0 configs with only run.seed changed.  Two of the
four new runs were moved between GPUs mid-run (requeue + epoch-boundary resume; `slurm_logs/gpu_upgrade.log`).

| | A-P3 (5 seeds) | JS-AP3 (5 seeds) | A-P3 − JS, mean [95 % t] | per seed (0–4) | label |
|---|---|---|---|---|---|
| linear | 60.15 ± 0.38 | 59.35 ± 0.42 | **+0.80 [+0.06, +1.54]** | +1.44 +0.40 +0.56 +1.44 +0.18 | **clear** |
| kNN | 55.90 ± 0.31 | 55.30 ± 0.39 | **+0.60 [+0.11, +1.09]** | +1.12 +0.28 +0.14 +0.76 +0.72 | **clear** |

## Reading (frozen)
- **Clear difference on CIFAR-100 at 5 seeds, in VCS's favour, on both readouts** (A-P3 ahead on all five seeds); the fresh seeds alone agree in
  sign (+1.44 / +0.18 linear, +0.76 / +0.72 kNN).  On CIFAR-10 the same contrast was close (P114, +0.21 at 3 seeds) — no pooling across datasets.
- **Required before any general statement** (P114 / P120 rule): the lr / optimiser-budget sensitivity check for both losses.  That check is P135
  (lr 5e-4 / 2e-3 for VCS A-P3 and JS-AP3 on CIFAR-10 and CIFAR-100, seed 0; submitted 2026-10-05): the gap "holds across lr" if it keeps its sign
  at all three learning rates.  Until P135 is read, the statement is limited to "at the common recipe lr (1e-3) and (a, κ) = (2, 0.5)".
- Also pending and related: P127 / P129 per-dataset (a, κ) grids (tuned VCS vs tuned JS comparison, frozen in P129).
