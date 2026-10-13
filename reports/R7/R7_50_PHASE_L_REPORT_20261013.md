# R7-50 phase L — official-test evaluation of the LLaVA-1.5 rows (D1-20 selectors) — report — 2026-10-13

- **Approval:** owner, 2026-10-12 10:16 UTC ("我都确认"; scope confirmed "两项都批准"), recorded in b22c540.  Protocol:
  reports/R7/R7_50_PROTOCOL_FROZEN_20261012.yaml (execution notes r7_50_1–3).
- **One read:** prediction manifest (448 files, sha256 44116bd7…) committed before scoring (a59be12), identical to the pre-manifest record
  (d2c9fb3); ALLOW_SCORING_L named that commit; a single scoring run (job 1036812); results-only commit 81c89df (files unedited).
- **Sets:** GQA testdev-balanced (12,578 questions, 398 images) and ChartQA test (2,500 questions: 1,250 human + 1,250 augmented; 1,509 charts).
  Exposure check: all 398 GQA test-dev images have new ('n'-prefixed) ids and none is in LLaVA-1.5's instruction mix; ChartQA is not in the mix.
- **Latency points:** the frozen D1-30 timing of the mapped validation benchmark (GQA DEV for GQA test-dev, ChartQA val for ChartQA test) at
  B* = 4 on nodesumo01 — the same files and the same co-tenancy caveat as the D1-30 report.
- **Primary family:** S1 − S2 and S1 − S3 at L25 / L50 / L75, S1 − SparseVLM budget-matched at 144 / 288 / 432 tokens (D30-1); Holm over the 6
  contrasts of both benchmarks at each latency point; 2,000-replicate image (chart) grouped paired bootstrap; no equivalence claim.

## 1. Primary contrasts (accuracy points; 95 % interval; n.s. = Holm-adjusted p ≥ 0.05)
| contrast | point | GQA testdev | ChartQA test |
|---|---|---|---|
| S1-S2 | L25 | -3.24 [-3.76, -2.66] | -0.87 [-1.37, -0.41] |
| S1-S2 | L50 | -1.46 [-1.83, -1.09] | -0.32 [-0.74, +0.11] (n.s.) |
| S1-S2 | L75 | -0.26 [-0.43, -0.10] | +0.12 [-0.09, +0.34] (n.s.) |
| S1-S3 | L25 | -0.99 [-1.49, -0.46] | +1.07 [+0.62, +1.55] |
| S1-S3 | L50 | -0.15 [-0.52, +0.22] (n.s.) | +0.76 [+0.28, +1.18] |
| S1-S3 | L75 | -0.03 [-0.21, +0.15] (n.s.) | +0.25 [-0.00, +0.54] (n.s.) |
| S1-SparseVLM | L25 | -4.03 [-4.73, -3.30] | -1.65 [-2.53, -0.87] |
| S1-SparseVLM | L50 | -1.59 [-2.04, -1.14] | -0.88 [-1.56, -0.21] |
| S1-SparseVLM | L75 | -0.05 [-0.40, +0.27] (n.s.) | +0.04 [-0.43, +0.53] (n.s.) |

## 2. Accuracy at fixed budgets (seed means for S1–S3; tokens kept out of 576)
| **gqa_testdev** (n = 12578; full 61.95) | 12.5 % | 25 % | 50 % | 75 % |
|---|---|---|---|---|
| uniform grid | 56.26 | 58.70 | 60.29 | 60.92 |
| S1 VCS | 50.52 | 54.76 | 59.11 | 61.48 |
| S2 logistic | 54.61 | 57.74 | 60.33 | 61.56 |
| S3 direct risk | 52.45 | 55.46 | 59.17 | 61.50 |
| SparseVLM (adapted) | 54.83 | 58.79 | 60.70 | 61.53 |

| **chartqa_test** (n = 2500; full 18.16) | 12.5 % | 25 % | 50 % | 75 % |
|---|---|---|---|---|
| uniform grid | 13.52 | 14.80 | 16.92 | 17.44 |
| S1 VCS | 13.84 | 14.91 | 16.56 | 17.80 |
| S2 logistic | 14.95 | 15.57 | 16.80 | 17.57 |
| S3 direct risk | 12.84 | 13.77 | 15.89 | 17.68 |
| SparseVLM (adapted) | 14.36 | 16.56 | 17.44 | 17.76 |

## 3. Agreement with the D1-30 DEV / val reading (same selectors, same rows)
| benchmark | contrast | point | official test | D1-30 DEV / val | same sign | same Holm verdict |
|---|---|---|---|---|---|---|
| gqa_testdev | S1-S2 | L25 | -3.24 (p_holm 0.006) | -2.857 (p_holm 0.009) | yes | yes |
| chartqa_test | S1-S2 | L25 | -0.87 (p_holm 0.006) | -1.126 (p_holm 0.009) | yes | yes |
| gqa_testdev | S1-S2 | L50 | -1.46 (p_holm 0.006) | -1.234 (p_holm 0.009) | yes | yes |
| chartqa_test | S1-S2 | L50 | -0.32 (p_holm 0.254) | -0.616 (p_holm 0.017) | yes | no |
| gqa_testdev | S1-S2 | L75 | -0.26 (p_holm 0.012) | -0.194 (p_holm 0.189) | yes | no |
| chartqa_test | S1-S2 | L75 | +0.12 (p_holm 1.000) | +0.000 (p_holm 1.000) | yes | yes |
| gqa_testdev | S1-S3 | L25 | -0.99 (p_holm 0.006) | -1.192 (p_holm 0.009) | yes | yes |
| chartqa_test | S1-S3 | L25 | +1.07 (p_holm 0.006) | +1.316 (p_holm 0.009) | yes | yes |
| gqa_testdev | S1-S3 | L50 | -0.15 (p_holm 0.457) | -0.639 (p_holm 0.009) | yes | no |
| chartqa_test | S1-S3 | L50 | +0.76 (p_holm 0.012) | +0.812 (p_holm 0.014) | yes | yes |
| gqa_testdev | S1-S3 | L75 | -0.03 (p_holm 1.000) | -0.106 (p_holm 1.000) | yes | yes |
| chartqa_test | S1-S3 | L75 | +0.25 (p_holm 0.270) | +0.338 (p_holm 0.306) | yes | yes |
| gqa_testdev | S1-SparseVLM | L25 | -4.03 (p_holm 0.006) | -4.124 (p_holm 0.009) | yes | yes |
| chartqa_test | S1-SparseVLM | L25 | -1.65 (p_holm 0.006) | -2.604 (p_holm 0.009) | yes | yes |
| gqa_testdev | S1-SparseVLM | L50 | -1.59 (p_holm 0.006) | -1.089 (p_holm 0.009) | yes | yes |
| chartqa_test | S1-SparseVLM | L50 | -0.88 (p_holm 0.039) | -1.997 (p_holm 0.009) | yes | yes |
| gqa_testdev | S1-SparseVLM | L75 | -0.05 (p_holm 1.000) | -0.058 (p_holm 1.000) | yes | yes |
| chartqa_test | S1-SparseVLM | L75 | +0.04 (p_holm 1.000) | +0.139 (p_holm 1.000) | yes | yes |

Same sign in 18 of 18 cells; same Holm verdict in 15 of 18 (the Holm families differ: 9 per point in D1-30 with TextVQA, 6 here).

## 4. Reading
- **The official test reproduces the DEV / val picture.**  S1 (VCS) is below S2 (logistic) at L25 and L50 on GQA test-dev and at L25 on
  ChartQA test, and below SparseVLM at L25 and L50 on both; S1 − S3 is negative on GQA test-dev (detected at L25 only) and positive on ChartQA
  test (L25, L50); no contrast is detected at L75 except S1 − S2 on GQA test-dev (−0.26).
- The full LLaVA-1.5-7B row scores 61.95 on GQA test-dev, consistent with the published 62.0 for this model; GQA DEV (84.65) is drawn from
  GQA train-balanced, which is in LLaVA-1.5's training mix, so DEV accuracies are higher in level but the contrasts agree in sign.
- Descriptive (fixed budget): on GQA test-dev S1 is below the uniform grid at 12.5–50 %; on ChartQA test S1 is close to the grid at 12.5–25 %
  and below S2 and SparseVLM.
- **For D1:** with D1-30, this is the 10k-image half of the claim rule (DF-8): the VCS-trained selector does not beat its logistic sibling or
  SparseVLM on LLaVA-1.5 at this scale.  The final D1 statement waits for D1-FULL (and R7-50 phase F on its selectors); the Qwen replication is
  R7-40c, whose official-test rows are phase Q.
