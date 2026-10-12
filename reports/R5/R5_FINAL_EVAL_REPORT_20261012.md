# r5 final evaluation — official test, one-shot — report — 2026-10-12

- **Approval:** owner 2026-10-12 10:16 UTC ("我都确认", scope "两项都批准"), recorded in `b22c540`.
- **Freeze:** `3434bb5` (`reports/R5/R5_FINAL_EVAL_FROZEN_20261012.yaml`), before any test parse.  Code `5890434` (script sha256 fbd3051a…, the
  DEV-verified version).
- **Test sources:** hashed before parsing (job 1035972); the single build (job 1035973) re-checked them, and every count equals the official
  split size.
- **Predictions:** 108 files for the 153 rows, hash-committed before scoring (`ad1f8a9`).  Scored once (job 1036188, clean tree); results-only
  commit `337cc98` (`reports/R5/final_eval/results.json`, `tables.md`, unedited).
- **Metric (primary):** the standard query-level Acc@0.5 over ALL official expressions; a query without a proposal counts as an error.  The DEV
  image-macro metric is reported separately in `tables.md` Table 2.

## Evaluation sets
| set | images | queries | images >= 2 referred | queries >= 2 | no proposal | ceiling all / >= 2 |
|---|---|---|---|---|---|---|
| refcoco_testA | 750 | 5657 | 715 | 5564 | 0 | 98.87 / 98.90 |
| refcoco_testB | 750 | 5095 | 691 | 4939 | 0 | 97.00 / 97.10 |
| refcocog_test | 2600 | 9602 | 1580 | 7622 | 0 | 97.28 / 97.18 |
| refcocoplus_testA | 750 | 5726 | 715 | 5626 | 0 | 98.85 / 98.88 |
| refcocoplus_testB | 750 | 4889 | 689 | 4716 | 0 | 97.44 / 97.56 |


(ceiling = share of queries with at least one proposal at IoU ≥ 0.5; no query was left without a proposal.)

## 1. Primary confirmation (frozen): VQ − VCS, Holm over the 5 splits, two-interval reading rule
| set | VQ − VCS | seed-paired 95 % t | image bootstrap 95 % | p (two-sided) | Holm p | decision |
|---|---|---|---|---|---|---|
| refcoco_testA | +5.57 | [+4.22, +6.92] | [+4.52, +6.69] | 0.0032 | 0.0127 | VQ improves VCS |
| refcoco_testB | +8.73 | [+7.90, +9.55] | [+7.20, +10.12] | 0.0005 | 0.0024 | VQ improves VCS |
| refcocog_test | +0.83 | [+0.11, +1.54] | [+0.23, +1.42] | 0.0381 | 0.1143 | no detectable difference (Holm) |
| refcocoplus_testA | +1.36 | [-0.09, +2.80] | [+0.65, +2.08] | 0.0561 | 0.1143 | no detectable difference (Holm) |
| refcocoplus_testB | +2.65 | [-0.95, +6.24] | [+1.68, +3.59] | 0.0871 | 0.1143 | no detectable difference (Holm) |

**Reading (frozen rule: Holm-adjusted seed-paired t rejects AND the image bootstrap excludes 0 with the same sign):**
- **RefCOCO testA (+5.57) and testB (+8.73): "VQ improves VCS"** — both intervals exclude 0 and Holm rejects.
- **RefCOCOg (+0.83), RefCOCO+ testA (+1.36) and testB (+2.65): "no detectable difference at the family-wise level"** (Holm p 0.114).  Not
  an equivalence claim: all three estimates are positive, and the image bootstrap excludes 0 on each.  With 3 seeds the seed-paired t has 2
  degrees of freedom, so the family-wise test is conservative.
- The test estimates match DEV in direction and size: DEV VQ − VCS was +1.04 (RefCOCOg), +2.13 (RefCOCO+) and +7.69 (RefCOCO) in the integrity
  re-run.

## 2. Estimation (frozen: effect sizes with both 95 % intervals; no test, no non-inferiority margin)
| contrast | set | mean | seed-paired 95 % t | image bootstrap 95 % | p | reading (t / bootstrap) |
|---|---|---|---|---|---|---|
| VQ - D | refcoco_testA | -1.10 | [-1.82, -0.37] | [-1.72, -0.37] | 0.0229 | below 0 / below 0 |
| VQ - E | refcoco_testA | -1.44 | [-2.34, -0.55] | [-2.32, -0.57] | 0.0202 | below 0 / below 0 |
| VQ - JQ | refcoco_testA | +0.55 | [-0.95, +2.04] | [-0.04, +1.12] | 0.2558 | contains 0 / contains 0 |
| VQ - D | refcoco_testB | -1.89 | [-6.10, +2.32] | [-2.78, -0.96] | 0.1932 | contains 0 / below 0 |
| VQ - E | refcoco_testB | -1.79 | [-4.18, +0.61] | [-2.89, -0.67] | 0.0849 | contains 0 / below 0 |
| VQ - JQ | refcoco_testB | +0.01 | [-2.00, +2.02] | [-0.71, +0.68] | 0.9802 | contains 0 / contains 0 |
| VQ - D | refcocog_test | -1.23 | [-3.35, +0.89] | [-1.71, -0.74] | 0.1298 | contains 0 / below 0 |
| VQ - E | refcocog_test | -1.06 | [-1.45, -0.68] | [-1.53, -0.54] | 0.0071 | below 0 / below 0 |
| VQ - JQ | refcocog_test | -0.07 | [-1.45, +1.32] | [-0.42, +0.31] | 0.8568 | contains 0 / contains 0 |
| VQ - D | refcocoplus_testA | -0.69 | [-2.25, +0.87] | [-1.25, -0.13] | 0.1961 | contains 0 / below 0 |
| VQ - E | refcocoplus_testA | -0.73 | [-2.58, +1.11] | [-1.53, +0.05] | 0.2288 | contains 0 / contains 0 |
| VQ - JQ | refcocoplus_testA | -0.05 | [-1.45, +1.36] | [-0.51, +0.44] | 0.8995 | contains 0 / contains 0 |
| VQ - D | refcocoplus_testB | -0.37 | [-2.40, +1.65] | [-1.17, +0.38] | 0.5094 | contains 0 / contains 0 |
| VQ - E | refcocoplus_testB | +0.12 | [-0.06, +0.29] | [-0.73, +1.01] | 0.1077 | contains 0 / contains 0 |
| VQ - JQ | refcocoplus_testB | +0.33 | [-0.79, +1.45] | [-0.32, +0.99] | 0.3349 | contains 0 / contains 0 |
| VQ - D | refcoco_testA | -0.81 | [-1.20, -0.42] | [-1.46, -0.10] | 0.0125 | below 0 / below 0 |
| VQ - E | refcoco_testA | -1.07 | [-2.15, +0.02] | [-2.03, -0.15] | 0.0513 | contains 0 / below 0 |
| VQ - JQ | refcoco_testA | +0.56 | [-1.43, +2.55] | [-0.06, +1.16] | 0.3493 | contains 0 / contains 0 |
| VQ - D | refcoco_testB | -2.04 | [-7.39, +3.31] | [-2.99, -1.14] | 0.2431 | contains 0 / below 0 |
| VQ - E | refcoco_testB | -2.19 | [-5.95, +1.56] | [-3.38, -1.07] | 0.1285 | contains 0 / below 0 |
| VQ - JQ | refcoco_testB | -0.09 | [-2.25, +2.07] | [-0.84, +0.64] | 0.8752 | contains 0 / contains 0 |
| VQ - D | refcocog_test | -1.30 | [-3.46, +0.86] | [-1.87, -0.76] | 0.1220 | contains 0 / below 0 |
| VQ - E | refcocog_test | -1.04 | [-1.83, -0.24] | [-1.62, -0.44] | 0.0306 | below 0 / below 0 |
| VQ - JQ | refcocog_test | -0.14 | [-1.90, +1.63] | [-0.58, +0.31] | 0.7718 | contains 0 / contains 0 |
| VQ - D | refcocoplus_testA | -0.54 | [-2.51, +1.43] | [-1.14, +0.08] | 0.3600 | contains 0 / contains 0 |
| VQ - E | refcocoplus_testA | -0.87 | [-2.86, +1.12] | [-1.66, -0.05] | 0.2008 | contains 0 / below 0 |
| VQ - JQ | refcocoplus_testA | -0.10 | [-1.47, +1.27] | [-0.56, +0.38] | 0.7795 | contains 0 / contains 0 |
| VQ - D | refcocoplus_testB | -0.38 | [-2.44, +1.67] | [-1.26, +0.46] | 0.5073 | contains 0 / contains 0 |
| VQ - E | refcocoplus_testB | +0.19 | [-0.31, +0.69] | [-0.80, +1.18] | 0.2361 | contains 0 / contains 0 |
| VQ - JQ | refcocoplus_testB | +0.38 | [-0.64, +1.40] | [-0.35, +1.04] | 0.2499 | contains 0 / contains 0 |
| VQ - D | refcoco_testA | -1.10 | [-1.55, -0.66] | [-1.73, -0.38] | 0.0088 | below 0 / below 0 |
| VQ - E | refcoco_testA | -1.41 | [-2.37, -0.45] | [-2.28, -0.51] | 0.0241 | below 0 / below 0 |
| VQ - JQ | refcoco_testA | +0.52 | [-1.28, +2.32] | [-0.07, +1.10] | 0.3394 | contains 0 / contains 0 |
| VQ - D | refcoco_testB | -1.94 | [-6.76, +2.87] | [-2.83, -0.99] | 0.2243 | contains 0 / below 0 |
| VQ - E | refcoco_testB | -1.96 | [-5.00, +1.09] | [-3.06, -0.79] | 0.1095 | contains 0 / below 0 |
| VQ - JQ | refcoco_testB | -0.05 | [-2.29, +2.20] | [-0.77, +0.64] | 0.9361 | contains 0 / contains 0 |
| VQ - D | refcocog_test | -1.36 | [-3.83, +1.11] | [-1.92, -0.78] | 0.1414 | contains 0 / below 0 |
| VQ - E | refcocog_test | -1.15 | [-1.96, -0.33] | [-1.72, -0.58] | 0.0265 | below 0 / below 0 |
| VQ - JQ | refcocog_test | -0.12 | [-1.98, +1.75] | [-0.54, +0.33] | 0.8107 | contains 0 / contains 0 |
| VQ - D | refcocoplus_testA | -0.59 | [-2.26, +1.09] | [-1.16, -0.05] | 0.2713 | contains 0 / below 0 |
| VQ - E | refcocoplus_testA | -0.69 | [-2.60, +1.22] | [-1.51, +0.11] | 0.2616 | contains 0 / contains 0 |
| VQ - JQ | refcocoplus_testA | -0.04 | [-1.37, +1.29] | [-0.52, +0.43] | 0.9189 | contains 0 / contains 0 |
| VQ - D | refcocoplus_testB | -0.43 | [-2.55, +1.69] | [-1.27, +0.38] | 0.4741 | contains 0 / contains 0 |
| VQ - E | refcocoplus_testB | +0.07 | [-0.37, +0.51] | [-0.80, +1.01] | 0.5598 | contains 0 / contains 0 |
| VQ - JQ | refcocoplus_testB | +0.44 | [-0.79, +1.67] | [-0.23, +1.11] | 0.2650 | contains 0 / contains 0 |

**Reading (descriptive):**
- **VQ stays below the softmax controls on most splits.**  VQ − E is −0.7 to −1.8 points on four splits; both intervals are below 0 on RefCOCO
  testA and RefCOCOg, the bootstrap alone elsewhere, and both contain 0 on RefCOCO+ testB (+0.12).  VQ − D is −0.4 to −1.9, with both
  intervals below 0 on RefCOCO testA.  This matches the DEV criterion gap (VQ − D below 0 on all three DEV sets).
- **VQ ≈ JQ:** every VQ − JQ interval contains 0 (−0.07 to +0.55).  Inside the normalized model class, the VCS and matched-JS criteria do not
  separate.

## 3. Test accuracies (query-level Acc@0.5, all expressions; mean ± sd over seeds 0–2)
| method | refcoco_testA | refcoco_testB | refcocog_test | refcocoplus_testA | refcocoplus_testB |
|---|---|---|---|---|---|
| VCS | 73.07 ± 0.63 | 56.77 ± 1.09 | 72.77 ± 0.33 | 79.21 ± 0.32 | 60.71 ± 1.19 |
| JS | 72.58 ± 0.28 | 56.34 ± 0.52 | 72.38 ± 0.16 | 78.95 ± 0.54 | 60.21 ± 0.33 |
| VQ | 78.64 ± 0.11 | 65.50 ± 0.80 | 73.60 ± 0.49 | 80.56 ± 0.34 | 63.35 ± 0.49 |
| JQ | 78.09 ± 0.72 | 65.49 ± 0.78 | 73.66 ± 0.18 | 80.61 ± 0.60 | 63.03 ± 0.51 |
| D | 79.74 ± 0.34 | 67.39 ± 0.90 | 74.82 ± 0.37 | 81.26 ± 0.31 | 63.73 ± 0.43 |
| E | 80.08 ± 0.32 | 67.29 ± 0.35 | 74.66 ± 0.34 | 81.30 ± 0.50 | 63.24 ± 0.45 |
| VNU | — | — | 75.27 ± 0.25 | 82.04 ± 0.22 | 63.33 ± 0.23 |
| JNU | — | — | 74.94 ± 0.47 | 81.86 ± 0.19 | 63.05 ± 0.76 |
| VCS+F | — | — | 74.98 ± 0.54 | 81.56 ± 0.11 | 61.91 ± 0.95 |
| JS+F | — | — | 74.59 ± 0.21 | 81.45 ± 0.74 | 61.83 ± 0.45 |
| E+F | — | — | 75.79 ± 0.24 | 82.14 ± 0.59 | 63.90 ± 0.18 |
| VNU+F | — | — | 75.49 ± 0.25 | 82.45 ± 0.52 | 63.10 ± 0.52 |
| JNU+F | — | — | 75.09 ± 0.41 | 82.09 ± 0.37 | 63.09 ± 0.75 |

"—" = not evaluated: no checkpoint (VNU / JNU on RefCOCO were never trained) or no CAL-selected λ (fused rows), as frozen (decision D3).
F = + λ_CAL log s (detector score, R5-10 F1).  Secondary contrasts with their intervals: `tables.md`.

## 4. Integrity and procedure
- **Before the read:** the DEV integrity check reproduced all 51 reported DEV rows within 0.3 points (largest −0.26).
- **Build:** source sha256 matched the hashes recorded before the freeze; "checks failed: none" on every split; labels sealed (0400).
- **Detector:** the proposals ran on RTX PRO 6000, the GPU type of the DEV proposals (decision D6).
- **Code:** identical across stages (manifest `code_consistent_across_stages: true`).  The score stage verified every prediction file against
  the committed manifest.  One scoring run.

## 5. Caveats and disclosures
- **The test annotations were never sealed in closed/.**  The shared R4 / r5 loader parsed them into memory in every run and dropped the test
  records by role; no test item was scored, ranked or used for a choice before this read (R4-02 ledger, updated before the read).
- **Variance sources:** 3 seeds per method; the seed-paired t covers training randomness only, and the image bootstrap covers test-image
  sampling only.
- **Exposure:**
  - The SigLIP 2 backbone's pre-training may include COCO images, and RefCOCO / RefCOCO+ / RefCOCOg share COCO images.
  - The detector was trained on COCO-2017 minus every CAL / DEV / val / test image of the three datasets.
- **Licences:** the RefCOCO-family terms conflict between sources or require attribution, and the detector backbone's ImageNet weights have
  unknown terms (D5 register V1–V3, V17).  The owner records the publication basis.
- **Not evaluated:** 63 pre-declared rows lacked a checkpoint or a CAL λ.  None was filled after the read.

## 6. What this supports (owner decision T2 structure)
- **Confirmed at the family-wise level on RefCOCO:** within-query normalization (VQ) improves the original VCS critic by 5.6 (testA) and
  8.7 (testB) points.
- **Positive but not family-wise significant on RefCOCOg (+0.8) and RefCOCO+ (+1.4, +2.7).**
- **Estimation:** VQ trails the softmax objectives by 0.4–1.9 points on most splits, and VQ ≈ JQ.  So the normalization closes most of the
  VCS-to-softmax gap; the remainder is the criterion gap seen on DEV (tested setup only; not an unavoidable cost).
