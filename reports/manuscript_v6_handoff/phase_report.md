# MV6 phase report — CVPR-MV6-HANDOFF-20261006-r2 — 2026-10-07

Intake `intake_report.md` (e476e04); owner go 2026-10-06 ("Submit all", block B, CIFAR-100).  Three measurement units ran on frozen checkpoints only
(no encoder training), all on CPU; each has a results-only commit and a report.  Ledgers: `tables/measurement_comparison.csv` (Table 3 axis, P116 rows)
+ `reports/P142_results.csv` (new rows); `tables/ssl_retention_response.csv` (Table 4 axis, P118 rows) + `reports/P143_results.csv` (new rows).

## What was done, reused, added
| logical | P | status | key result | report |
|---|---|---|---|---|
| MV6-I1 (Table 3) | P142 | COMPLETED | Same draws as P116 (gate passed in all 5 cells).  Rejection, blur 0.5 n1000 / n2000 / colour 0.2 n2000: VCS 0.75 / 0.99 / 0.97, logistic@50 0.82 / 1.00 / 0.94, same-target RFF kernel 0.59 / 0.96 / 0.80, cond. HSIC class-wise 0.19 / 0.58 / 0.34, deep 0.15 / 0.55 / 0.41; nulls 0.03–0.07 for all.  Block B (Table 4 SimCLR): same ordering, lower sensitivity (colour 0.1 at null for all) | `P142_MV6_I1_TABLE3_KERNEL_HSIC_REPORT_20261007.md` |
| MV6-R0 (Table 4) | — | REUSED | 8 rows rebuilt from raw JSON, exact; P143 stage 0 regenerated the deleted cache and reproduced the A-P3 / 1 colour row (max |Δpower| 0.005) | intake + P143 |
| MV6-I2 | P143 | COMPLETED | Colour 0.1, H (VCS statistic): CIFAR-10 VCS 0.60 / 0.58, selected logistic 0.43 / 0.31, SimCLR 0.08 / 0.11, VICReg 0.10 / 0.14; CIFAR-100 VCS 0.85 / 0.71, selected logistic 0.90 / 0.95, SimCLR 0.19 / 0.13, VICReg 0.18 / 0.12; Δacc < 0.7 pts in every row; no VCS-column null flag | `P143_MV6_I2_RETENTION_EXTENSION_REPORT_20261007.md` |
| MV6-C1 | P144 | COMPLETED (estimation-limited) | Real maps exact (r, z, logits recomputed ≤ 3e-5); J at h ≤ 0 at n 3 000 (null level, as in P118 visual), so h→r→z and h→O decrements are zero within the repeat spread; one value estimation-sensitive; bootstrap deviation disclosed | `P144_MV6_C1_TRUE_COMPRESSION_REPORT_20261007.md` |

## Statements supported / not supported
- Supported: on fixed SimCLR features, the VCS two-stage solver and matched logistic detect the planted attributes at the stated n with calibrated
  nulls; a same-target RFF critic and conditional HSIC detect less at the same n and permutations (Table 3 can gain these columns on the same draws).
- Supported: colour 0.1 is far more detectable in the representations of the two balanced-posterior encoders (VCS, selected logistic) than in
  SimCLR and VICReg, on CIFAR-10 and CIFAR-100; the order within the pair depends on the dataset; accuracy changes stay small.
- Not supported: "VCS is more interpretable"; any amount-of-information reading of rejection rates; any compression ordering (P144).

## Writing items (owner)
W1–W5 of the intake; plus: Table 3's "fixed SimCLR" can be either the P5 model (block A, original) or Table 4's model (block B); Table 4 can gain
VICReg / selected-logistic rows (P143) and a CIFAR-100 block, each encoder seed as one row; P144 supports only the methodological statement that the
real deterministic maps are measurable with the exact map definition and that the attribute dependence is below finite-sample J resolution at n 3 000.
