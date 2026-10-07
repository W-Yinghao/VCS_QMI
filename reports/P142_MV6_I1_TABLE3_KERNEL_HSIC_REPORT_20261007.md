# P142 (MV6-I1) — Table 3 completion: same-target kernel and conditional HSIC on a fixed encoder — report — 2026-10-07

Pre-registration `P142_MV6_I1_TABLE3_KERNEL_HSIC_PREREG_FROZEN_20261006.md`; jobs 1024526 / 1024527 (block A), 1024528 / 1024529 (block B), CPU, all
exit 0; results-only commit `64d371b` (`reports/P142/`, `P142_results.{md,json,csv}`).  Axis `fixed_encoder_vary_estimator`; rejection rates over
R draws at α 0.05 with B = 200 shared within-class permutations; nothing selected on EVAL.

## 1. Block A — the Table 3 fixture (P5 SimCLR, 2 views, 200 epochs), the same draws as P116
**Reproduction gate passed in all five P116 cells** (A1 and A3 per-repeat statistics within 7.6e-11, no decision changed), so the new rows are paired
with the published Table 3 rows draw by draw.

| cell | n | VCS (A1) | logistic A3@50 / 200 / 800 | same-target kernel (K1) | cond. HSIC class-wise (H1) | cond. HSIC deep (H2) |
|---|---|---|---|---|---|---|
| blur 0.5 | 1 000 | 0.75 | 0.82 / 0.82 / 0.82 | 0.59 | 0.19 | 0.15 |
| blur 0.5 | 2 000 | 0.99 | 1.00 / 1.00 / 1.00 | 0.96 | 0.58 | 0.55 |
| colour 0.2 | 2 000 | 0.97 | 0.94 / 0.94 / 0.94 | 0.80 | 0.34 | 0.41 |
| null (label only) | 1 000 | 0.04 | 0.05 / 0.04 / 0.04 | 0.05 | 0.03 | 0.04 |
| null (label only) | 2 000 | 0.04 | 0.04 / 0.04 / 0.04 | 0.06 | 0.07 | 0.03 |
| null (all planted 0.2) | 2 000 | 0.06 | 0.06 / 0.06 / 0.06 | 0.04 | 0.06 | 0.05 |

Null rejection 0.03–0.07 for every method (no flag > 0.09); the new n-1 000 null gives the blur n-1 000 cell its applicable null.

## 2. Block B — the same protocol on Table 4's SimCLR (P41 4-view / 800-epoch, seed 1; new fixture)
| cell | n | VCS (A1) | logistic A3@50 / 200 / 800 | K1 | H1 | H2 |
|---|---|---|---|---|---|---|
| blur 0.5 | 1 000 | 0.27 | 0.33 / 0.33 / 0.33 | 0.27 | 0.08 | 0.10 |
| blur 0.5 | 2 000 | 0.66 | 0.72 / 0.72 / 0.72 | 0.50 | 0.22 | 0.19 |
| colour 0.2 | 2 000 | 0.52 | 0.61 / 0.61 / 0.61 | 0.42 | 0.39 | 0.44 |
| colour 0.1 | 2 000 | 0.09 | 0.07 / 0.07 / 0.07 | 0.08 | 0.09 | 0.07 |
| nulls (3 cells) | 1 000 / 2 000 | 0.04–0.07 | 0.04–0.06 | 0.04–0.07 | 0.03–0.04 | 0.03–0.06 |

## 3. Cost (per repeat, n 2 000, from these runs)
VCS A1 ≈ 0.1 s fit + selection; logistic 0.08 / 0.4–0.5 / 1.1–1.7 s at 50 / 200 / 800 closures; K1 0.5–0.8 s (three bandwidths); H1 no fit, 1.3–2.4 s
for 200 permutations; H2 70–110 s fit (deep kernel, 300 steps) + 1–2 s permutations.  These runs shared their nodes with the H2 computation, so
absolute times are inflated relative to P116 (A1 0.04 s there); Table 3's own timings stay the reference, and only within-run ratios are read.

## 4. Reading (frozen)
- **On both fixed models the ordering is logistic ≈ VCS > same-target RFF kernel > both conditional HSIC tests**, with every method calibrated.
  VCS and logistic stay within ±0.09 of each other (logistic ahead on blur, VCS ahead on colour 0.2 in block A, logistic ahead in block B).  The
  kernel critic of the same target trails the linear-in-h two-stage solver (−0.04 to −0.23).  The HSIC statistics, on their own native scale,
  detect far less at the same n and permutations (0.15–0.58 where VCS / logistic reach 0.75–1.00).
- Logistic budget: 50 closures already reach the 800-closure power in every cell (as in P116).
- **Block B — Table 4's own SimCLR** is much less sensitive to these attributes than the Table 3 model (blur 0.5 / n 2 000: 0.66 vs 0.99; colour
  0.2: 0.52 vs 0.97), and colour 0.1 is at null level for every method (0.07–0.09) — consistent with Table 4's SimCLR seed-1 H = 0.08 under the
  P109 protocol.  The estimator ordering is the same as in block A, but with smaller margins (HSIC deep 0.44 vs VCS 0.52 on colour 0.2).
- Estimator usability only: nothing here compares encoders as such (the block A / B difference is a property of the two SimCLR models).

## 5. Writing items (for the owner)
Table 3 can gain K1 / H1 / H2 columns on the same draws (block A), with the applicable null range 0.03–0.07 including the new n-1 000 null.  If the
"fixed SimCLR" should be the Table 4 model, block B is the same protocol on it (W1 of the intake).  HSIC is a different native statistic: report its
rejection rates and cost, never in an S-error column.

## 6. Not claimed
Encoder properties; other strengths, n, kernels or bandwidth grids; HSIC values compared with J / S.
