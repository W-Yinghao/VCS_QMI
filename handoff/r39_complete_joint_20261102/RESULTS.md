Publication copy (sample names aliased; numbers and conclusions unchanged).

# R39 COMPLETE x JOINT results (numbers only)

21/21 R39 units complete; 6 R38 N-shared-cosine references reused (not resubmitted). Fixed step 8000. DEV two-stage (frozen-H top-100 reranked, rest in H order). TEST not read.

## Flickr60: DEV rSum at step 8000

| cell | J | BCE | NCE | J-BCE | J-NCE |
|---|---|---|---|---|---|
| N_frozen | 509.15 | 508.40 | 506.69 | 0.75 | 2.47 |
| COMPLETE_frozen | 509.11 | 508.50 | 506.88 | 0.61 | 2.23 |
| N_joint | 517.87 | 515.68 | 515.52 | 2.19 | 2.35 |
| COMPLETE_joint | 518.13 | 516.79 | 517.12 | 1.34 | 1.01 |
| ORACLE_frozen | 509.92 | 509.94 | 509.39 | -0.02 | 0.53 |

frozen H native DEV rSum 505.21; teacher-only rerank rSum 493.93

| objective | COMPLETE-N (frozen) | JOINT-FROZEN (N) | JOINT-FROZEN (COMPLETE) | interaction |
|---|---|---|---|---|
| J | -0.04 | 8.72 | 9.01 | 0.30 |
| BCE | 0.10 | 7.28 | 8.28 | 1.01 |
| NCE | 0.20 | 8.84 | 10.24 | 1.40 |

## CC152K: DEV rSum at step 8000

| cell | J | BCE | NCE | J-BCE | J-NCE |
|---|---|---|---|---|---|
| N_frozen | 392.70 | 392.80 | 388.60 | -0.10 | 4.10 |
| COMPLETE_frozen | 392.60 | 392.60 | 388.60 | 0.00 | 4.00 |
| N_joint | 390.80 | 390.80 | 386.80 | 0.00 | 4.00 |
| COMPLETE_joint | 391.10 | 390.90 | 387.70 | 0.20 | 3.40 |

frozen H native DEV rSum 389.70; teacher-only rerank rSum 396.80

| objective | COMPLETE-N (frozen) | JOINT-FROZEN (N) | JOINT-FROZEN (COMPLETE) | interaction |
|---|---|---|---|---|
| J | -0.10 | -1.90 | -1.50 | 0.40 |
| BCE | -0.20 | -2.00 | -1.70 | 0.30 |
| NCE | -0.00 | -1.80 | -0.90 | 0.90 |

Files: RESULTS_FIXED.csv, CONTRASTS_2x2.csv (all recalls), OBJECTIVE_CONTRASTS.csv, VS_H_TEACHER.csv, RESCUED_LOST.csv, CURVES.csv, COVERAGE.csv (incl. isolated Flickr audit rows),
LABEL_SOURCES.csv (training-batch label counts), STUDENT_CHANGE.csv, BOOTSTRAP.csv (descriptive), COST_LEDGER.csv, FAILURES.csv.
J readouts: DEV list-J uses the clean DEV annotation; FIT observed-label J and completed-label J are different label objects (P changed) and are not ranked against each other.

## R39X JOINT confirmation extension (numbers only)

21/21 R39X units complete (15 JOINT on five additional parents + 3 JOINT and 3 new N-frozen cells on Flickr 0%). Fixed step 8000, DEV two-stage, TEST not read.

| parent | H | obj | N-frozen | N-JOINT | JOINT-FROZEN | frozen-H | JOINT-H | net R@1 vs H (frozen / JOINT) |
|---|---|---|---|---|---|---|---|---|
| Flickr00_R30H_s1001 | 522.07 | J | 522.78 | 527.59 | 4.81 | 0.71 | 5.52 | 43 / 146 |
| Flickr00_R30H_s1001 | 522.07 | BCE | 521.95 | 525.66 | 3.71 | -0.12 | 3.59 | 45 / 112 |
| Flickr00_R30H_s1001 | 522.07 | NCE | 524.24 | 551.08 | 26.84 | 2.17 | 29.01 | 75 / 601 |
| Flickr60_R30H_s1002 | 482.94 | J | 485.94 | 493.39 | 7.46 | 3.00 | 10.45 | 42 / 162 |
| Flickr60_R30H_s1002 | 482.94 | BCE | 486.51 | 491.79 | 5.29 | 3.57 | 8.86 | 43 / 137 |
| Flickr60_R30H_s1002 | 482.94 | NCE | 485.74 | 491.18 | 5.44 | 2.80 | 8.24 | 38 / 119 |
| Flickr60_R30H_s1003 | 496.07 | J | 498.46 | 510.47 | 12.01 | 2.39 | 14.40 | 60 / 226 |
| Flickr60_R30H_s1003 | 496.07 | BCE | 497.91 | 508.66 | 10.75 | 1.83 | 12.58 | 61 / 191 |
| Flickr60_R30H_s1003 | 496.07 | NCE | 498.48 | 508.40 | 9.92 | 2.41 | 12.33 | 39 / 181 |
| CC152K_R31H0_s1002 | 394.60 | J | 394.10 | 392.80 | -1.30 | -0.50 | -1.80 | -2 / -8 |
| CC152K_R31H0_s1002 | 394.60 | BCE | 392.90 | 391.20 | -1.70 | -1.70 | -3.40 | -5 / -14 |
| CC152K_R31H0_s1002 | 394.60 | NCE | 390.30 | 390.00 | -0.30 | -4.30 | -4.60 | -9 / -6 |
| CC152K_R31H0_s1003 | 392.60 | J | 393.50 | 391.40 | -2.10 | 0.90 | -1.20 | 20 / 10 |
| CC152K_R31H0_s1003 | 392.60 | BCE | 393.00 | 393.40 | 0.40 | 0.40 | 0.80 | 12 / 13 |
| CC152K_R31H0_s1003 | 392.60 | NCE | 391.30 | 391.10 | -0.20 | -1.30 | -1.50 | 12 / 9 |
| CC152K_R37S_s1001 | 393.20 | J | 393.40 | 390.20 | -3.20 | 0.20 | -3.00 | 9 / -10 |
| CC152K_R37S_s1001 | 393.20 | BCE | 393.20 | 391.10 | -2.10 | 0.00 | -2.10 | 9 / 1 |
| CC152K_R37S_s1001 | 393.20 | NCE | 390.30 | 389.70 | -0.60 | -2.90 | -3.50 | 14 / 3 |

| parent | cell | J-BCE | J-NCE |
|---|---|---|---|
| Flickr00_R30H_s1001 | frozen | 0.83 | -1.46 |
| Flickr00_R30H_s1001 | joint | 1.93 | -23.49 |
| Flickr60_R30H_s1002 | frozen | -0.57 | 0.20 |
| Flickr60_R30H_s1002 | joint | 1.60 | 2.21 |
| Flickr60_R30H_s1003 | frozen | 0.55 | -0.02 |
| Flickr60_R30H_s1003 | joint | 1.81 | 2.07 |
| CC152K_R31H0_s1002 | frozen | 1.20 | 3.80 |
| CC152K_R31H0_s1002 | joint | 1.60 | 2.80 |
| CC152K_R31H0_s1003 | frozen | 0.50 | 2.20 |
| CC152K_R31H0_s1003 | joint | -2.00 | 0.30 |
| CC152K_R37S_s1001 | frozen | 0.20 | 3.10 |
| CC152K_R37S_s1001 | joint | -0.90 | 0.50 |

Files: R39X_RESULTS.csv (all recalls, pair identity, student change, costs), R39X_CURVES.csv. N-frozen cells: reused R38 / R37S N-shared-cosine units; Flickr 0% frozen cells are new R39X units (no R38 reference at 0%).
Failed attempts (kept, rerun after amendment A1 of PROTOCOL_FROZEN_R39X.json; technical in-unit checks only): 4 (R39X_FAILURES.csv).
