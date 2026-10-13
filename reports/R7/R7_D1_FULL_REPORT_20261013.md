# R7 D1-FULL — LLaVA selectors trained on the full GQA FIT next to the 10k FIT: report (2026-10-13)

**Status:** complete.  Nine D1-FULL selector units (S1 VCS, S2 logistic, S3 direct risk × seeds 0–2, all at the CAL-selected learning rate
1e-4) evaluated on GQA DEV, TextVQA val and ChartQA val with D1-30's frozen evaluator; one image / chart-grouped bootstrap per benchmark
(2,000 replicates, seed 0) shared by every statistic, so both scales are paired.  Readout job 1038598 (`scripts/r7_d1full_score.py`, frozen);
results-only commit 7a89736 (`reports/R7/R7_D1_FULL_results/d1full_results.json`, unedited).  Integrity: no dropped, blank or duplicate
prediction lines; the recomputed 10k block equals the D1-30 readout on all 27 primary contrasts (largest difference 0).

**Claim rule (DF-8, frozen before any D1-30 result).**  A D1 claim is made only where the two scales (10k FIT = D1-30; full FIT = D1-FULL) agree
in sign and both pass their Holm-corrected primary tests (Holm across the 9 contrasts at each latency point); full-scale numbers are the
headline; a disagreement is reported as scale-dependent.

## 1. Primary family (quality points at matched latency L25 / L50 / L75; S1 − SparseVLM on the budget-matched D30-1 basis)

**gqa_dev** (7,986 items, 2,000 groups)

| contrast | point | full FIT: Δ [95 % CI], Holm p | 10k FIT: Δ [95 % CI], Holm p | DF-8 |
|---|---|---|---|---|
| S1 − S2 | L25 | -0.06 [-0.48, +0.37], p 1.000 | -2.86 [-3.36, -2.34], p 0.009 | no claim (Holm) |
| S1 − S3 | L25 | +0.38 [-0.17, +0.92], p 0.724 | -1.19 [-1.67, -0.71], p 0.009 | scale-dependent sign |
| S1 − SparseVLM | L25 | -3.59 [-4.26, -2.93], p 0.009 | -4.12 [-4.81, -3.42], p 0.009 | claim: S1 lower |
| S1 − S2 | L50 | +0.00 [-0.34, +0.36], p 1.000 | -1.23 [-1.59, -0.87], p 0.009 | scale-dependent sign |
| S1 − S3 | L50 | -0.49 [-0.91, -0.04], p 0.120 | -0.64 [-1.03, -0.27], p 0.009 | no claim (Holm) |
| S1 − SparseVLM | L50 | -1.19 [-1.71, -0.68], p 0.009 | -1.09 [-1.63, -0.56], p 0.009 | claim: S1 lower |
| S1 − S2 | L75 | -0.05 [-0.24, +0.15], p 1.000 | -0.19 [-0.35, -0.03], p 0.189 | no claim (Holm) |
| S1 − S3 | L75 | -0.26 [-0.51, -0.01], p 0.329 | -0.11 [-0.31, +0.09], p 1.000 | no claim (Holm) |
| S1 − SparseVLM | L75 | -0.18 [-0.54, +0.18], p 1.000 | -0.06 [-0.40, +0.30], p 1.000 | no claim (Holm) |

**textvqa_val** (4,799 items, 3,039 groups)

| contrast | point | full FIT: Δ [95 % CI], Holm p | 10k FIT: Δ [95 % CI], Holm p | DF-8 |
|---|---|---|---|---|
| S1 − S2 | L25 | +2.01 [+1.47, +2.56], p 0.009 | -4.37 [-5.05, -3.65], p 0.009 | scale-dependent sign |
| S1 − S3 | L25 | +1.18 [+0.51, +1.81], p 0.010 | +8.76 [+8.09, +9.50], p 0.009 | claim: S1 higher |
| S1 − SparseVLM | L25 | -19.59 [-20.60, -18.55], p 0.009 | -16.97 [-17.97, -15.99], p 0.009 | claim: S1 lower |
| S1 − S2 | L50 | +2.19 [+1.64, +2.77], p 0.009 | -2.10 [-2.67, -1.52], p 0.009 | scale-dependent sign |
| S1 − S3 | L50 | +1.24 [+0.50, +1.98], p 0.015 | +9.11 [+8.37, +9.84], p 0.009 | claim: S1 higher |
| S1 − SparseVLM | L50 | -11.11 [-11.97, -10.27], p 0.009 | -7.52 [-8.25, -6.81], p 0.009 | claim: S1 lower |
| S1 − S2 | L75 | +0.05 [-0.23, +0.36], p 1.000 | -0.37 [-0.60, -0.15], p 0.024 | scale-dependent sign |
| S1 − S3 | L75 | +3.46 [+2.97, +3.96], p 0.009 | +3.63 [+3.25, +4.03], p 0.009 | claim: S1 higher |
| S1 − SparseVLM | L75 | -1.09 [-1.53, -0.66], p 0.009 | -0.18 [-0.62, +0.21], p 1.000 | no claim (Holm) |

**chartqa_val** (1,920 items, 1,056 groups)

| contrast | point | full FIT: Δ [95 % CI], Holm p | 10k FIT: Δ [95 % CI], Holm p | DF-8 |
|---|---|---|---|---|
| S1 − S2 | L25 | -0.10 [-0.55, +0.35], p 1.000 | -1.13 [-1.67, -0.59], p 0.009 | no claim (Holm) |
| S1 − S3 | L25 | -0.19 [-0.82, +0.47], p 1.000 | +1.32 [+0.75, +1.90], p 0.009 | scale-dependent sign |
| S1 − SparseVLM | L25 | -3.59 [-4.67, -2.42], p 0.009 | -2.60 [-3.66, -1.45], p 0.009 | claim: S1 lower |
| S1 − S2 | L50 | +0.18 [-0.38, +0.70], p 1.000 | -0.62 [-1.13, -0.14], p 0.017 | scale-dependent sign |
| S1 − S3 | L50 | -0.64 [-1.30, +0.11], p 0.276 | +0.81 [+0.27, +1.36], p 0.014 | scale-dependent sign |
| S1 − SparseVLM | L50 | -2.62 [-3.59, -1.63], p 0.009 | -2.00 [-2.92, -1.12], p 0.009 | claim: S1 lower |
| S1 − S2 | L75 | -0.04 [-0.37, +0.26], p 1.000 | +0.00 [-0.29, +0.31], p 1.000 | scale-dependent sign |
| S1 − S3 | L75 | -0.11 [-0.55, +0.35], p 1.000 | +0.34 [+0.00, +0.67], p 0.306 | scale-dependent sign |
| S1 − SparseVLM | L75 | -0.10 [-0.73, +0.54], p 1.000 | +0.14 [-0.52, +0.79], p 1.000 | scale-dependent sign |

## 2. What DF-8 allows

- **9 claims.**  S1 is below SparseVLM at L25 and L50 on all three benchmarks (full-scale Δ: GQA −3.59 / −1.19, TextVQA −19.59 / −11.11,
  ChartQA −3.59 / −2.62).  S1 is above S3 on TextVQA at L25 / L50 / L75 (+1.18 / +1.24 / +3.46).
- **11 scale-dependent** contrasts (opposite signs at the two scales).  The most consequential is S1 − S2 on TextVQA: −4.37 at L25 at 10k
  (Holm-significant) but +2.01 at L25 and +2.19 at L50 at full scale (Holm-significant).  No claim in either direction.
- **7 without a claim** (same sign, Holm fails at one scale).  This includes S1 − S2 on GQA: −2.86 at L25 at 10k (Holm-significant), −0.06
  at full scale (Holm p 1.0).

So the 10k finding "VCS below matched logistic" does not survive at full scale; the finding "S1 below SparseVLM at tight latency" does.

## 3. Data scale (descriptive: full − 10k at matched latency; no multiplicity adjustment, no claim)

Quality full vs 10k (Δ [95 % CI], paired image bootstrap):

| benchmark | objective | L25 | L50 | L75 |
|---|---|---|---|---|
| gqa_dev | S1 | 77.80 vs 77.07 (+0.72 [+0.24, +1.20]) | 81.35 vs 81.35 (-0.01 [-0.36, +0.36]) | 83.51 vs 83.63 (-0.12 [-0.29, +0.06]) |
| gqa_dev | S2 | 77.85 vs 79.93 (-2.08 [-2.56, -1.57]) | 81.34 vs 82.59 (-1.24 [-1.60, -0.87]) | 83.55 vs 83.82 (-0.27 [-0.46, -0.07]) |
| gqa_dev | S3 | 77.42 vs 78.27 (-0.85 [-1.36, -0.35]) | 81.84 vs 81.99 (-0.16 [-0.56, +0.22]) | 83.77 vs 83.74 (+0.03 [-0.19, +0.26]) |
| textvqa_val | S1 | 24.24 vs 26.88 (-2.65 [-3.32, -1.94]) | 33.80 vs 37.37 (-3.57 [-4.24, -2.94]) | 43.59 vs 44.69 (-1.10 [-1.39, -0.80]) |
| textvqa_val | S2 | 22.22 vs 31.25 (-9.02 [-9.70, -8.32]) | 31.61 vs 39.47 (-7.86 [-8.53, -7.21]) | 43.54 vs 45.06 (-1.52 [-1.84, -1.22]) |
| textvqa_val | S3 | 23.05 vs 18.12 (+4.94 [+4.27, +5.63]) | 32.55 vs 28.26 (+4.29 [+3.53, +5.06]) | 40.13 vs 41.06 (-0.92 [-1.36, -0.48]) |
| chartqa_val | S1 | 14.73 vs 15.69 (-0.96 [-1.48, -0.43]) | 16.68 vs 17.38 (-0.69 [-1.31, -0.08]) | 18.67 vs 19.00 (-0.33 [-0.70, +0.01]) |
| chartqa_val | S2 | 14.84 vs 16.82 (-1.98 [-2.55, -1.37]) | 16.50 vs 17.99 (-1.49 [-2.06, -0.87]) | 18.71 vs 19.00 (-0.29 [-0.63, +0.07]) |
| chartqa_val | S3 | 14.93 vs 14.38 (+0.55 [-0.09, +1.14]) | 17.32 vs 16.56 (+0.75 [+0.06, +1.43]) | 18.78 vs 18.67 (+0.11 [-0.29, +0.49]) |

At the frozen recipe (20,000 + 5,000 steps; each FIT question seen about 32 times at 10k and 4.6 times at full scale, OPEN DF-4), more
training tuples did not raise the learned selectors' quality: most full − 10k differences are small or negative.  The largest drops are S2's
(GQA L25 −2.08, TextVQA L25 −9.02); S3 rises on TextVQA (+4.94 at L25).  The S1 − S2 reversal on TextVQA therefore comes mainly from S2 falling at
full scale, not from S1 improving.  This contrast measures more distinct tuples at an equal optimisation budget, not longer or better-tuned
training.

## 4. Context (not part of any claim)

- At 10k the CAL-selected phase-A learning rate differed between objectives (S1 3e-4, S2 and S3 1e-4); at full scale all three selected
  1e-4.  The soft − hard predicted-risk gap logged at the smallest budget is large only for 3e-4 critics; the REV8 diagnostics
  (`reports/REV8/REV8_DIAG_PROTOCOL_FROZEN_20261013.yaml`) decompose it.  Nothing here attributes the scale dependence to that rate.
- The official-test confirmation of these D1-FULL selectors is R7-50 phase F (GQA testdev + ChartQA test), running under its frozen protocol;
  this report reads only DEV / val.

## 5. Files

`reports/R7/R7_D1_FULL_results/d1full_results.json` (every row, contrast, per-seed delta, CI, fixed-budget table, dynamic rows, latency
points); protocol `reports/R7/R7_D1_FULL_PROTOCOL_DRAFT_20261012.yaml` with `reports/R7/R7_D1_FULL_STAGED_FREEZE_20261012.yaml` (DF-1…11,
execution notes df_1…df_5).
