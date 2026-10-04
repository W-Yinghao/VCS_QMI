# P115 diagnostics — final report (job 1 + job 2: standard / crop-only / jitter-only / both-strong, A-P3 and SimCLR, seed 0) — 2026-10-04

Pre-registration `P115_DIAG_PREREG_FROZEN_20261003.md` (reading descriptive only, no pre-written mechanism); job 1 interim report
`P115_DIAG_INTERIM_REPORT_20261003.md`; job 2 = job 1020582 (CPU, exit 0); results-only commit `3ee8fa8` (`P115_diag_results.{md,json}` over 12 runs /
44 checkpoints, `P115_diag_factor_table.md`, `scripts/p115_diag_factor_table.py`, per-run files `reports/P115_diag/`).  Accuracies are from the P115
factor report (`P115_V5_AUG_FACTORS_REPORT_20261004.md`).  One seed per cell; correlational readouts; nothing here filters or reweights pairs.

## 1. The factor cells at epoch 800 (full trajectories 20 / 100 / 400 / 800 in `P115_diag_factor_table.md`)

| method | cell | lin / kNN | s⁺ mean | s⁺ q05 | low-IoU share of positives | low-IoU share of positive-term encoder grad | IoU < 0.1 share of that grad | h eff. rank | z eff. rank |
|---|---|---|---|---|---|---|---|---|---|
| A-P3 | standard | 89.06 / 87.30 | 0.937 | 0.787 | 0.10 | 0.48 | 0.05 | 115 | 57 |
| A-P3 | crop-only | 88.80 / 86.42 | 0.883 | 0.604 | 0.30 | **0.95** | **0.56** | 90 | 30 |
| A-P3 | jitter-only | 88.78 / 87.78 | 0.927 | 0.762 | 0.10 | 0.52 | 0.08 | 103 | 47 |
| A-P3 | both | 88.66 / 86.20 | 0.864 | 0.549 | 0.30 | 0.88 | 0.42 | 88 | 26 |
| SimCLR | standard | 88.20 / 87.20 | 0.941 | 0.787 | 0.10 | 0.41 | 0.04 | 148 | 117 |
| SimCLR | crop-only | 89.86 / 88.68 | 0.870 | 0.481 | 0.30 | 0.68 | 0.23 | 134 | 83 |
| SimCLR | jitter-only | 88.24 / 88.48 | 0.926 | 0.739 | 0.10 | 0.21 | 0.00 | 140 | 105 |
| SimCLR | both | 89.74 / 88.86 | 0.844 | 0.372 | 0.30 | 0.73 | 0.34 | 129 | 76 |

(low-IoU = crop IoU < 0.25; "share of grad" = share of the total positive-term encoder gradient, shares sum to 1.)

## 2. Description (frozen reading: per factor, side by side; no mechanism asserted)
- **The crop factor sets the pair geometry, jitter does not:** small crops triple the low-overlap share of positive pairs (0.10 → 0.30) in both
  methods; jitter leaves it unchanged by construction.  Positive similarity drops with crops in both (s⁺ 0.94 → 0.87–0.88).
- **Where the gradient goes under strong crops differs by method:** in A-P3 the low-overlap pairs take 0.95 of the positive-term encoder gradient
  at epoch 800 (pairs with IoU < 0.1 — 7.6 % of positives — take 0.56); in SimCLR 0.68 (0.23).  A-P3 keeps those pairs tighter: s⁺ q05 0.60 vs
  SimCLR 0.48.  This concentration grows late in A-P3 (crop-only: 0.76 → 0.74 → 0.69 → 0.95 over 20 / 100 / 400 / 800) and stays near 0.7 in SimCLR.
- **Representation rank:** crops lower the effective rank of h and z in both methods; A-P3's z rank halves (57 → 30), SimCLR's falls by a third
  (117 → 83).  A-P3 is lower-rank than SimCLR in every cell (h 88–115 vs 129–148).
- **Accuracy alongside:** SimCLR gains from crops (+1.66 linear crop-only) while its rank falls; A-P3 does not gain (−0.26).  So "lower rank" or
  "more low-overlap gradient" alone does not track the accuracy change across methods — the same direction of change goes with opposite accuracy
  effects.  What co-occurs with A-P3's missing crop gain, descriptively, is the near-total concentration of its positive-term gradient on the
  low-overlap pairs and the tighter alignment it reaches on them.
- **Not claimed:** that the concentration causes the missing gain; that reweighting / filtering low-IoU pairs would help (not proposed); anything
  beyond seed 0.  A causal test would need an intervention unit (e.g. a capped or reweighted positive term) with its own pre-registration — listed
  here as an open option, not launched.
