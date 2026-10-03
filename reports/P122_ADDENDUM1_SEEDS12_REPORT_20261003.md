# P122 addendum 1 — CIFAR-10 encoder seeds 1–2 (seed-pooled structural increments) — report — 2026-10-03

Addendum `P122_ADDENDUM1_SEEDS12_FROZEN_20261003.md`; job 1020321 exit 0; aggregate over seeds 0–2 `P122_v6_evidence_results_all.{md,json}` (33 cell files),
seed-pooled table `P122_seed_pooled_c10.txt` (unit = encoder seed, n = 3; 95 % t over seeds).  Wording per the prereg: "additional score recovered in these
function classes and budgets".

## 1. Seed-pooled increments (CIFAR-10), Z − s and H − Z, VCS / JS

| measurement law | encoder | Ĵ_s | Z − s (VCS / JS) | H − Z (VCS / JS) |
|---|---|---|---|---|
| standard | A-P3 | 0.974 | +0.0003 / +0.0008 (all seeds +) | ≈ 0 |
| standard | A-P3 strong | 0.969 | **+0.0043 / +0.0046** (CI > 0) | +0.0008 / +0.0007 (CI > 0) |
| standard | recipe VCS | 0.993 | 0.0000 / 0.0000 | ≈ 0 |
| standard | SimCLR | 0.978 | −0.0007 / −0.0004 (CI ∋ 0) | ≈ 0 |
| standard | SimCLR strong | 0.973 | −0.0026 / +0.0012 (CI ∋ 0) | ≈ 0 |
| strong | A-P3 | 0.78 | +0.0066 / **+0.0074** | **+0.0134 / +0.0155** (CI > 0) |
| strong | A-P3 strong | 0.90 | **+0.0034 / +0.0044** | **+0.0126 / +0.0166** (CI > 0) |
| strong | recipe VCS | 0.82 | **+0.0140 / +0.0217** (CI > 0) | +0.0113 / +0.0072 (CI ∋ 0) |
| strong | SimCLR | 0.78 | **+0.0061 / +0.0081** (CI > 0) | ≈ 0 |
| strong | SimCLR strong | 0.90 | **+0.0006 / +0.0012** (CI > 0) | +0.0060 / +0.0070 |

## 2. Reading
- **Replicates across encoder seeds:** under the **strong** measurement law, full pair structure recovers score beyond the scalar s on every encoder family
  (Z − s positive for all five, H − Z additionally for A-P3 and A-P3-strong, ≈ +0.013–0.017).  Under the **standard** law the structural increment is
  essentially zero except for **A-P3-strong** (+0.004 at Z, seed-stable), i.e. the encoder trained with strong crops carries some pair evidence that a
  single cosine does not read even on standard pairs.
- The seed-0 cells that met the rule with SimCLR / SimCLR-strong under the standard law do **not** replicate (seed-pooled intervals include 0).
- Observation, not a mechanism: the strong-augmentation pairs (low crop overlap) are where a single cosine stops being a sufficient pair statistic for these
  frozen representations; A-P3's h carries extra pair evidence beyond z there (H − Z), SimCLR's does not.
