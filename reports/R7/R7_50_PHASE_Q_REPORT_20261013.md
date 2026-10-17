# R7-50 phase Q — official-test evaluation of the Qwen2.5-VL-7B rows (R7-40b selectors) — report — 2026-10-13

- **Approval:** owner, 2026-10-12 10:16 UTC ("我都确认"; "两项都批准"; b22c540) — R7-50 phase Q after R7-40c.  Design: the frozen phase-Q addendum
  (reports/R7/R7_50_PHASE_Q_ADDENDUM_DRAFT_20261012.yaml, frozen 2026-10-13 02:00 UTC before any label was opened) and execution notes r7_50_4–6.
- **One read:** full-scale integrity gate on GQA DEV / ChartQA val passed on every item (inputs, answers and kept counts identical to R7-40c);
  prediction manifest (336 files, sha256 070e958e…) committed before scoring (23406a9); ALLOW_SCORING_Q named that commit; one scoring run
  (job 1037696, the frozen stage-4 command run by the coordinator, note r7_50_6); results-only commit 1bbeb8e.
- **Sets:** GQA testdev-balanced (12,578 questions, 398 images) and ChartQA test (2,500: 1,250 human + 1,250 augmented).  Exposure: Qwen2.5-VL's
  training data is undisclosed, so exposure to GQA / ChartQA (train and possibly test) cannot be excluded; project side: never used for selection.
- **Basis:** Qwen has no B*, so the primary points are fixed fractions 25 / 50 / 75 % of the merged visual tokens; no acceleration claim.
- **Primary family (Q4):** S1 − S2, S1 − S3, S1 − uniform grid × 2 benchmarks; Holm over 6 per point; 2,000-replicate
  image (chart) grouped paired bootstrap.

## 1. Primary contrasts (accuracy points; 95 % interval; n.s. = Holm-adjusted p ≥ 0.05)
| contrast | fraction | GQA test-dev | ChartQA test |
|---|---|---|---|
| S1-S2 | 25% | -1.40 [-1.94, -0.83] | -13.28 [-14.68, -11.93] |
| S1-S2 | 50% | -1.05 [-1.48, -0.60] | -9.08 [-10.32, -7.85] |
| S1-S2 | 75% | -0.53 [-0.85, -0.20] | -3.00 [-3.88, -2.21] |
| S1-S3 | 25% | -0.55 [-1.02, -0.08] (n.s.) | -0.91 [-2.21, +0.50] (n.s.) |
| S1-S3 | 50% | -0.60 [-1.03, -0.17] | -1.19 [-2.51, +0.08] (n.s.) |
| S1-S3 | 75% | -0.25 [-0.60, +0.07] (n.s.) | -1.04 [-1.96, -0.09] (n.s.) |
| S1-grid | 25% | -4.81 [-5.44, -4.15] | -16.25 [-18.09, -14.52] |
| S1-grid | 50% | -2.89 [-3.41, -2.33] | -13.73 [-15.19, -12.34] |
| S1-grid | 75% | -1.19 [-1.59, -0.78] | -5.48 [-6.57, -4.47] |

## 2. Accuracy at fixed fractions (seed means for S1–S3)
| **gqa_testdev** (n = 12578; full 60.98) | 12.5 % | 25 % | 50 % | 75 % |
|---|---|---|---|---|
| uniform grid | 55.53 | 58.48 | 60.03 | 60.68 |
| S1 VCS | 50.56 | 53.67 | 57.14 | 59.49 |
| S2 logistic | 51.84 | 55.07 | 58.20 | 60.02 |
| S3 direct risk | 50.68 | 54.22 | 57.74 | 59.74 |

| **chartqa_test** (n = 2500; full 83.88) | 12.5 % | 25 % | 50 % | 75 % |
|---|---|---|---|---|
| uniform grid | 37.80 | 54.84 | 74.20 | 81.28 |
| S1 VCS | 25.91 | 38.59 | 60.47 | 75.80 |
| S2 logistic | 36.23 | 51.87 | 69.55 | 78.80 |
| S3 direct risk | 26.44 | 39.49 | 61.65 | 76.84 |

## 3. Agreement with R7-40c (same Qwen selectors, DEV / val)
| benchmark | contrast | fraction | official test | R7-40c DEV / val | same sign | same Holm verdict |
|---|---|---|---|---|---|---|
| gqa_testdev | S1-S2 | 25% | -1.40 | -0.61 | yes | yes |
| gqa_testdev | S1-S2 | 50% | -1.05 | -0.28 (n.s.) | yes | no |
| gqa_testdev | S1-S2 | 75% | -0.53 | -0.27 (n.s.) | yes | no |
| gqa_testdev | S1-S3 | 25% | -0.55 (n.s.) | -1.24 | yes | no |
| gqa_testdev | S1-S3 | 50% | -0.60 | -1.47 | yes | yes |
| gqa_testdev | S1-S3 | 75% | -0.25 (n.s.) | -1.16 | yes | no |
| gqa_testdev | S1-grid | 25% | -4.81 | -4.47 | yes | yes |
| gqa_testdev | S1-grid | 50% | -2.89 | -3.43 | yes | yes |
| gqa_testdev | S1-grid | 75% | -1.19 | -2.00 | yes | yes |
| chartqa_test | S1-S2 | 25% | -13.28 | -12.60 | yes | yes |
| chartqa_test | S1-S2 | 50% | -9.08 | -8.37 | yes | yes |
| chartqa_test | S1-S2 | 75% | -3.00 | -2.22 | yes | yes |
| chartqa_test | S1-S3 | 25% | -0.91 (n.s.) | +0.66 (n.s.) | NO | yes |
| chartqa_test | S1-S3 | 50% | -1.19 (n.s.) | -0.99 (n.s.) | yes | yes |
| chartqa_test | S1-S3 | 75% | -1.04 (n.s.) | -0.42 (n.s.) | yes | yes |
| chartqa_test | S1-grid | 25% | -16.25 | -17.26 | yes | yes |
| chartqa_test | S1-grid | 50% | -13.73 | -13.80 | yes | yes |
| chartqa_test | S1-grid | 75% | -5.48 | -4.62 | yes | yes |

Same sign in 17 of 18 cells; same Holm verdict in 14 of 18 (families differ: 9 per point in R7-40c with TextVQA, 6 here).

## 4. Cross-host side-by-side on the official test (descriptive only; LLaVA at its B* = 4 latency points, Qwen at fixed fractions)
| benchmark | contrast | point | LLaVA (phase L, latency point) | Qwen (phase Q, fraction) |
|---|---|---|---|---|
| gqa_testdev | S1-S2 | L25 / 25% | -3.24 | -1.40 |
| gqa_testdev | S1-S2 | L50 / 50% | -1.46 | -1.05 |
| gqa_testdev | S1-S2 | L75 / 75% | -0.26 | -0.53 |
| gqa_testdev | S1-S3 | L25 / 25% | -0.99 | -0.55 (n.s.) |
| gqa_testdev | S1-S3 | L50 / 50% | -0.15 (n.s.) | -0.60 |
| gqa_testdev | S1-S3 | L75 / 75% | -0.03 (n.s.) | -0.25 (n.s.) |
| chartqa_test | S1-S2 | L25 / 25% | -0.87 | -13.28 |
| chartqa_test | S1-S2 | L50 / 50% | -0.32 (n.s.) | -9.08 |
| chartqa_test | S1-S2 | L75 / 75% | +0.12 (n.s.) | -3.00 |
| chartqa_test | S1-S3 | L25 / 25% | +1.07 | -0.91 (n.s.) |
| chartqa_test | S1-S3 | L50 / 50% | +0.76 | -1.19 (n.s.) |
| chartqa_test | S1-S3 | L75 / 75% | +0.25 (n.s.) | -1.04 (n.s.) |

## 5. Reading
- **The official test reproduces the Qwen DEV / val picture.**  S1 (VCS) is below the training-free uniform grid at every fraction on both
  benchmarks, below S2 (logistic) at every fraction on both benchmarks, and S1 − S3 is detected only on GQA at 50 % (−0.60).
- On both hosts and on both test sets, the VCS-trained selector does not beat its logistic sibling; on Qwen every learned selector is below the
  uniform grid (descriptive table), on LLaVA S1 is below SparseVLM at tight budgets.
- **For D1:** with R7-40c, this completes the 10k-image Qwen evidence; the full-data Qwen replication (E2) is running, and the D1 claim rule for
  LLaVA (DF-8) still waits for D1-FULL and R7-50 phase F.

## Corrections (2026-10-17; numbers unchanged)
1. §5, line 85 — "On both hosts and on both test sets, the VCS-trained selector does not beat its logistic sibling" was written before phase F
   and holds only for the 10k-FIT selectors read in phases L and Q.  Read as "the VCS selector never beats its logistic sibling on the official
   test", it is wrong: R7-50 phase F (full-FIT LLaVA selectors, `reports/R7/R7_50_results_F/r7_50_results.json`) detected S1 above S2 on ChartQA
   test at L25, +0.75 [+0.34, +1.17], Holm p 0.006 (per seed +2.17, −0.10, +0.18), and S1 below S2 on GQA testdev at L25, −0.78 [−1.16, −0.40],
   Holm p 0.006 (per seed −0.21, +0.21, −2.35); its other four S1 − S2 cells are not detected.  Correct statement: with the 10k-FIT selectors, S1
   is below S2 on both hosts' official tests wherever a difference is detected (phase L: GQA testdev at L25 / L50 / L75 and ChartQA test at L25;
   phase Q: every fraction on both benchmarks); with the full-FIT LLaVA selectors there is no stable ordering (two detected L25 cells of opposite
   sign, each carried mainly by one seed).  Numbers printed by `scripts/r7_d1_closeout_tables.py`; reading in
   `reports/R7/R7_D1_CLOSEOUT_REPORT_20261017.md` §3.
