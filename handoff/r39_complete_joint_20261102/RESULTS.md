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

3/21 R39X units complete (15 JOINT on five additional parents + 3 JOINT and 3 new N-frozen cells on Flickr 0%). Fixed step 8000, DEV two-stage, TEST not read.

| parent | H | obj | N-frozen | N-JOINT | JOINT-FROZEN | frozen-H | JOINT-H | net R@1 vs H (frozen / JOINT) |
|---|---|---|---|---|---|---|---|---|
| Flickr00_R30H_s1001 | 522.07 | J | 522.78 | NA | NA | 0.71 | NA | 43 / NA |
| Flickr00_R30H_s1001 | 522.07 | BCE | 521.95 | NA | NA | -0.12 | NA | 45 / NA |
| Flickr00_R30H_s1001 | 522.07 | NCE | 524.24 | NA | NA | 2.17 | NA | 75 / NA |
| Flickr60_R30H_s1002 | 482.94 | J | 485.94 | NA | NA | 3.00 | NA | 42 / NA |
| Flickr60_R30H_s1002 | 482.94 | BCE | 486.51 | NA | NA | 3.57 | NA | 43 / NA |
| Flickr60_R30H_s1002 | 482.94 | NCE | 485.74 | NA | NA | 2.80 | NA | 38 / NA |
| Flickr60_R30H_s1003 | 496.07 | J | 498.46 | NA | NA | 2.39 | NA | 60 / NA |
| Flickr60_R30H_s1003 | 496.07 | BCE | 497.91 | NA | NA | 1.83 | NA | 61 / NA |
| Flickr60_R30H_s1003 | 496.07 | NCE | 498.48 | NA | NA | 2.41 | NA | 39 / NA |
| CC152K_R31H0_s1002 | 394.60 | J | 394.10 | NA | NA | -0.50 | NA | -2 / NA |
| CC152K_R31H0_s1002 | 394.60 | BCE | 392.90 | NA | NA | -1.70 | NA | -5 / NA |
| CC152K_R31H0_s1002 | 394.60 | NCE | 390.30 | NA | NA | -4.30 | NA | -9 / NA |
| CC152K_R31H0_s1003 | 392.60 | J | 393.50 | NA | NA | 0.90 | NA | 20 / NA |
| CC152K_R31H0_s1003 | 392.60 | BCE | 393.00 | NA | NA | 0.40 | NA | 12 / NA |
| CC152K_R31H0_s1003 | 392.60 | NCE | 391.30 | NA | NA | -1.30 | NA | 12 / NA |
| CC152K_R37S_s1001 | 393.20 | J | 393.40 | NA | NA | 0.20 | NA | 9 / NA |
| CC152K_R37S_s1001 | 393.20 | BCE | 393.20 | NA | NA | 0.00 | NA | 9 / NA |
| CC152K_R37S_s1001 | 393.20 | NCE | 390.30 | NA | NA | -2.90 | NA | 14 / NA |

| parent | cell | J-BCE | J-NCE |
|---|---|---|---|
| Flickr00_R30H_s1001 | frozen | 0.83 | -1.46 |
| Flickr00_R30H_s1001 | joint | NA | NA |
| Flickr60_R30H_s1002 | frozen | -0.57 | 0.20 |
| Flickr60_R30H_s1002 | joint | NA | NA |
| Flickr60_R30H_s1003 | frozen | 0.55 | -0.02 |
| Flickr60_R30H_s1003 | joint | NA | NA |
| CC152K_R31H0_s1002 | frozen | 1.20 | 3.80 |
| CC152K_R31H0_s1002 | joint | NA | NA |
| CC152K_R31H0_s1003 | frozen | 0.50 | 2.20 |
| CC152K_R31H0_s1003 | joint | NA | NA |
| CC152K_R37S_s1001 | frozen | 0.20 | 3.10 |
| CC152K_R37S_s1001 | joint | NA | NA |

Files: R39X_RESULTS.csv (all recalls, pair identity, student change, costs), R39X_CURVES.csv. N-frozen cells: reused R38 / R37S N-shared-cosine units; Flickr 0% frozen cells are new R39X units (no R38 reference at 0%).
Failed attempts (kept, rerun after amendment A1 of PROTOCOL_FROZEN_R39X.json; technical in-unit checks only): 3 (R39X_FAILURES.csv).
