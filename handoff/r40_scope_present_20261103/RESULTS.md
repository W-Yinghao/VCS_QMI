Publication copy (sample names aliased; numbers and conclusions unchanged).

# R40 update scope x incomplete-correspondence supervision (numbers only)

27/27 R40 units complete. Reused read-only references: R38 N-FROZEN, R39 N-JOINT, R39 ORACLE-FROZEN. Fixed step 8000; DEV two-stage (frozen-H top-100 reranked, rest in H order). TEST not read.

## Flickr60 (frozen H DEV rSum 505.21): rSum at step 8000

| cell | J | BCE | NCE | J-BCE | J-NCE |
|---|---|---|---|---|---|
| N_frozen (ref) | 509.15 | 508.40 | 506.69 | 0.75 | 2.47 |
| image_only | 513.20 | 512.21 | 512.37 | 0.99 | 0.83 |
| text_only | 508.90 | 508.40 | 508.90 | 0.49 | -0.00 |
| N_both_joint (ref) | 517.87 | 515.68 | 515.52 | 2.19 | 2.35 |
| PRESENT_frozen | 508.30 | 508.05 | 506.00 | 0.26 | 2.31 |
| PRESENT_both_joint | 516.11 | 514.08 | 517.93 | 2.03 | -1.81 |
| ORACLE_frozen (ref) | 509.92 | 509.94 | 509.39 | -0.02 | 0.53 |
| ORACLE_both_joint | 524.67 | 524.16 | 526.07 | 0.51 | -1.40 |

Section 3 (update scope, rSum):

| objective | IMAGE-FROZEN | TEXT-FROZEN | BOTH-IMAGE | BOTH-TEXT | BOTH-IMAGE-TEXT+FROZEN |
|---|---|---|---|---|---|
| J | 4.04 | -0.26 | 4.67 | 8.97 | 4.93 |
| BCE | 3.81 | 0.00 | 3.47 | 7.28 | 3.47 |
| NCE | 5.68 | 2.21 | 3.16 | 6.63 | 0.95 |

Section 4 (PRESENT, rSum):

| objective | PRESENT-N (frozen) | PRESENT-N (joint) | [PJ-PF]-[NJ-NF] |
|---|---|---|---|
| J | -0.85 | -1.76 | -0.91 |
| BCE | -0.36 | -1.60 | -1.24 |
| NCE | -0.69 | 2.41 | 3.10 |

Section 5 (ORACLE_MISSING, rSum):

| objective | ORACLE_JOINT-N_JOINT | ORACLE_JOINT-ORACLE_FROZEN | [OJ-OF]-[NJ-NF] |
|---|---|---|---|
| J | 6.80 | 14.75 | 6.04 |
| BCE | 8.48 | 14.22 | 6.94 |
| NCE | 10.55 | 16.69 | 7.85 |

## CC152K (frozen H DEV rSum 389.70): rSum at step 8000

| cell | J | BCE | NCE | J-BCE | J-NCE |
|---|---|---|---|---|---|
| N_frozen (ref) | 392.70 | 392.80 | 388.60 | -0.10 | 4.10 |
| image_only | 391.00 | 391.50 | 388.50 | -0.50 | 2.50 |
| text_only | 390.50 | 390.50 | 388.80 | 0.00 | 1.70 |
| N_both_joint (ref) | 390.80 | 390.80 | 386.80 | 0.00 | 4.00 |
| PRESENT_frozen | 392.30 | 392.20 | 389.90 | 0.10 | 2.40 |
| PRESENT_both_joint | 391.80 | 390.70 | 388.30 | 1.10 | 3.50 |

Section 3 (update scope, rSum):

| objective | IMAGE-FROZEN | TEXT-FROZEN | BOTH-IMAGE | BOTH-TEXT | BOTH-IMAGE-TEXT+FROZEN |
|---|---|---|---|---|---|
| J | -1.70 | -2.20 | -0.20 | 0.30 | 2.00 |
| BCE | -1.30 | -2.30 | -0.70 | 0.30 | 1.60 |
| NCE | -0.10 | 0.20 | -1.70 | -2.00 | -1.90 |

Section 4 (PRESENT, rSum):

| objective | PRESENT-N (frozen) | PRESENT-N (joint) | [PJ-PF]-[NJ-NF] |
|---|---|---|---|
| J | -0.40 | 1.00 | 1.40 |
| BCE | -0.60 | -0.10 | 0.50 |
| NCE | 1.30 | 1.50 | 0.20 |

Files: RESULTS_FIXED.csv, CONTRASTS_SCOPE_2x2.csv / CONTRASTS_PRESENT.csv / CONTRASTS_ORACLE.csv (all six recalls + rSum), OBJECTIVE_CONTRASTS.csv, CURVES.csv,
RESCUED_LOST.csv (vs frozen H and between cells), BOOTSTRAP.csv (descriptive), GRAD_SPLIT.csv, PRESENT_LOGS.csv, STUDENT_CHANGE.csv, COST_LEDGER.csv, FAILURES.csv; diag7/ (section 7 readouts).
