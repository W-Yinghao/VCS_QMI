Publication copy (sample names aliased; numbers and conclusions unchanged).

# R46 results (numbers only)

> **Status note (2026-11-03):** R46 JOINT jobs (A JOINT arms, all of B) were cancelled by the user after the R44 audit (JOINT student layout leak); only units with DONE are tabulated, the rest are NA. Numbers use the native per-batch centring; layout-free re-evaluations: CORRECTED_READOUTS.md.

13/30 R46 units complete. DEV two-stage (frozen-H top-100 reranked), native 1K COCO / Flickr DEV. TEST not read.

## A: COCO 80% c80 s1001, FIT pool restriction (8000 updates)

| pool | obj | N-frozen | N-JOINT | JOINT-FROZEN |
|---|---|---|---|---|
| full (R43-A) | J | 536.72 | 545.08 | 8.36 |
| full (R43-A) | BCE | 536.46 | 544.26 | 7.80 |
| full (R43-A) | NCE | 536.82 | 543.42 | 6.60 |
| 58k | J | 537.32 | 546.46 | 9.14 |
| 58k | BCE | 536.86 | 545.18 | 8.32 |
| 58k | NCE | 537.90 | NA | NA |
| 29k | J | 537.62 | 546.30 | 8.68 |
| 29k | BCE | 537.80 | NA | NA |
| 29k | NCE | 537.72 | NA | NA |

## B: iterated N supervision, round 2 (8000 more updates from the round-1 step-8000 state)

| condition | obj | round 1 (8000) | R2 CONTINUED | R2 REBUILT | REBUILT-CONTINUED | CONTINUED-R1 | REBUILT-R1 |
|---|---|---|---|---|---|---|---|
| F60_R30_H1001 | J | 517.87 | 523.53 | 522.60 | -0.93 | 5.66 | 4.73 |
| F60_R30_H1001 | BCE | 515.68 | NA | NA | NA | NA | NA |
| F60_R30_H1001 | NCE | 515.52 | NA | NA | NA | NA | NA |
| F80_IN2R_H1001 | J | 457.55 | 450.89 | 450.57 | -0.32 | -6.67 | -6.98 |
| F80_IN2R_H1001 | BCE | 454.71 | NA | NA | NA | NA | NA |
| F80_IN2R_H1001 | NCE | 456.69 | NA | NA | NA | NA | NA |
| C80_IN2R_H1001 | J | 545.08 | NA | NA | NA | NA | NA |
| C80_IN2R_H1001 | BCE | 544.26 | NA | NA | NA | NA | NA |
| C80_IN2R_H1001 | NCE | 543.42 | NA | NA | NA | NA | NA |

## Zero-training coverage (training record pool; truth columns Flickr only, evaluator-only)

| condition | table | records | i2t partner in top32 | t2i partner in top32 | i2t observed pos slots | truth: t2i true image in top32 | truth: obs pair correct |
|---|---|---|---|---|---|---|---|
| COCO80_full | parent_H | 566435 | 0.489 | 0.165 | 0.659 | NA | NA |
| COCO80_full | round2_J | 566435 | 0.507 | 0.169 | 0.689 | NA | NA |
| COCO80_full | round2_BCE | 566435 | 0.505 | 0.167 | 0.684 | NA | NA |
| COCO80_full | round2_NCE | 566435 | 0.508 | 0.168 | 0.689 | NA | NA |
| COCO80_p58k | parent_H | 290000 | 0.536 | 0.179 | 0.737 | NA | NA |
| COCO80_p29k | parent_H | 145000 | 0.577 | 0.190 | 0.809 | NA | NA |
| Flickr80_f80_s1001 | parent_H | 145000 | 0.675 | 0.226 | 1.017 | 0.697 | 0.200 |
| Flickr80_f80_s1001 | round2_J | 145000 | 0.692 | 0.233 | 1.061 | 0.684 | 0.200 |
| Flickr80_f80_s1001 | round2_BCE | 145000 | 0.694 | 0.233 | 1.065 | 0.679 | 0.200 |
| Flickr80_f80_s1001 | round2_NCE | 145000 | 0.687 | 0.230 | 1.046 | 0.668 | 0.200 |
| Flickr60_f60_s1001 | parent_H | 145000 | 0.894 | 0.399 | 1.870 | 0.837 | 0.400 |
| Flickr60_f60_s1001 | round2_J | 145000 | 0.897 | 0.401 | 1.883 | 0.847 | 0.400 |
| Flickr60_f60_s1001 | round2_BCE | 145000 | 0.899 | 0.402 | 1.892 | 0.843 | 0.400 |
| Flickr60_f60_s1001 | round2_NCE | 145000 | 0.887 | 0.400 | 1.829 | 0.842 | 0.400 |

Files: R46_RESULTS.csv, R46_CONTRASTS.csv, R46_CURVES.csv, R46_COST.csv, R46_FAILURES.csv, R46_COVERAGE.json/csv.
