# P143 (MV6-I2) — fixed VCS measurement, vary encoder: VICReg and selected logistic on CIFAR-10, four families on CIFAR-100 — report — 2026-10-07

Pre-registration `P143_MV6_I2_RETENTION_EXTENSION_PREREG_FROZEN_20261006.md`; jobs 1024525 (extraction), 1024530 (stage 0), 1024531 (C10), 1024532
(C100), all exit 0; results-only commit `9c57e83` (`reports/P143/`, `P143_results.{md,json,csv}`).  Axis `fixed_measurement_vary_encoder`; the same
P109-I1 / P118 procedure for every encoder (critics refitted per encoder); development data only; each row = one encoder seed.

## 1. Stage 0 — reproduction gate (the deleted cache)
Regenerated A-P3 seed-1 features (CPU) reproduce the recorded colour row: H 0.60 / 0.58, logits 0.13 / 0.11 (VCS / JS statistic); max |Δpower| over
all layers × statistics × cells 0.005; Δacc −0.20 vs recorded −0.18 points (inside [−0.53, +0.15]).  **Pass** → stages 1–2 ran.

## 2. Colour 0.1 (n 2 000 × 100 draws; responses on 8 000 pairs) — VCS statistic (the Table 4 column)
| dataset | encoder | H (seed 1 / 2) | O (seed 1 / 2) | Δacc points (seed 1 / 2) | flips % | clean head acc % |
|---|---|---|---|---|---|---|
| CIFAR-10 | VCS (existing rows) | 0.60 / 0.58 | 0.13 / 0.14 | −0.18 / −0.40 | 2.8 / 2.9 | 87.7 |
| CIFAR-10 | selected logistic (2, 0.25) | 0.43 / 0.31 | 0.14 / 0.07 | −0.21 / −0.26 | 2.9 / 3.0 | 86.8 |
| CIFAR-10 | SimCLR (existing rows) | 0.08 / 0.11 | 0.11 / 0.05 | −0.25 / −0.51 | 2.2 / 2.4 | 86.7 / 87.3 |
| CIFAR-10 | VICReg | 0.10 / 0.14 | 0.08 / 0.09 | −0.43 / −0.37 | 3.0 / 3.3 | 83.8 / 83.2 |
| CIFAR-100 | VCS | 0.85 / 0.71 | 0.53 / 0.57 | −0.66 / −0.63 | 11.5 / 11.2 | 54.8 / 53.9 |
| CIFAR-100 | selected logistic (3, 0.5) | 0.90 / 0.95 | 0.65 / 0.77 | −0.68 / −0.54 | 11.3 / 12.3 | 54.1 / 53.8 |
| CIFAR-100 | SimCLR | 0.19 / 0.13 | 0.21 / 0.16 | −0.51 / −0.01 | 8.4 / 7.9 | 54.6 / 54.4 |
| CIFAR-100 | VICReg | 0.18 / 0.12 | 0.11 / 0.13 | −0.44 / −0.50 | 10.1 / 9.7 | 49.2 / 48.3 |

Blur 0.25: H 0.04–0.31 and O 0.02–0.17 for every encoder on both datasets; |Δacc| ≤ 0.16 points (full table in `P143_results.md`).

## 3. Nulls
VCS statistic, H and O columns: **no flag in any row** (both nulls, 200 draws, threshold 0.09).  Flags elsewhere, all kept in the ledger: JS
statistic on CIFAR-10 selected logistic (layer3 / h / z, 0.10) and SimCLR seed 1 (z); on CIFAR-100 VICReg seed 2 blur, the z layer of the VCS
statistic and hence its any-layer max-T (not a Table 4 column); SimCLR seed 1 blur logits / JS.  `null_all_planted` uses colour 0.2 / blur 1.0.

## 4. Reading (descriptive, per seed; frozen rules)
- **Colour 0.1 is far more detectable in the two balanced-posterior encoders than in SimCLR and VICReg, on both datasets**: H 0.31–0.95 vs
  0.08–0.19.  Within that pair the order is dataset-dependent (CIFAR-10: VCS 0.60 / 0.58 > selected logistic 0.43 / 0.31; CIFAR-100: selected
  logistic 0.90 / 0.95 ≥ VCS 0.85 / 0.71).  The pattern belongs to the shared posterior objective family, not to VCS alone.
- On CIFAR-100 the colour shift is also detectable in the logits of both balanced-posterior encoders (O 0.53–0.77; CIFAR-10 0.07–0.14).
- Accuracy changes stay below 0.7 points in every row; CIFAR-100 flip rates (8–12 %) are higher because its head accuracy is ~54 %.  More
  detectable attribute evidence again does not imply a larger accuracy change.
- Rejection rates describe detectability at n 2 000 for this critic class — not amounts of retained information, not interpretability, and
  non-rejection does not mean absence.  Two seeds per encoder; no mean ± sd is formed.
- Provenance: VICReg C10 4-view / 800-epoch seeds were scored on the official CIFAR-10 test set in P68 (training unaffected; the audit uses
  training-partition images only).

## 5. Not claimed
Ranking of encoders by interpretability; causal statements; other strengths, n or critic classes; pooling with Table 3.
