Publication copy (sample names aliased; numbers and conclusions unchanged).

# R36 supervision / candidate-selection results (numbers only)
generated 2026-11-02T12:35:38Z; 27/27 units complete; fixed endpoints (M/T/N step 8000; V phase-2 step 4000 = total 12000, phase-1 8000 listed); full DEV 1014 img / 5070 cap; two-stage readout (frozen-H top-100 reranked by g, rest in H order); TEST not read.

## Fixed-endpoint DEV retrieval

| unit | i2t R1 | R5 | R10 | t2i R1 | R5 | R10 | rSum | Δ vs H | tie-opt | tie-pess | rescued/lost R@1 i2t | t2i | GPU |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| r36_M_n60_G_RANDOM_J | 79.09 | 94.87 | 97.83 | 58.66 | 83.49 | 89.43 | 503.37 | -1.83 | 503.37 | 503.37 | 7/23 | 72/74 | NVIDIA RTX PRO 6000 Blackwell Server Edition |
| r36_M_n60_G_RANDOM_BCE | 79.29 | 94.87 | 97.83 | 58.88 | 83.59 | 89.64 | 504.10 | -1.10 | 504.10 | 504.10 | 10/24 | 79/70 | NVIDIA RTX PRO 6000 Blackwell Server Edition |
| r36_M_n60_G_RANDOM_NCE | 79.09 | 94.18 | 98.03 | 58.34 | 83.08 | 89.05 | 501.78 | -3.43 | 501.78 | 501.78 | 21/37 | 146/164 | NVIDIA RTX PRO 6000 Blackwell Server Edition |
| r36_M_n60_G_SHIELD_ORACLE_J | 0.20 | 1.97 | 4.54 | 0.00 | 0.02 | 0.12 | 6.84 | -498.36 | 6.84 | 6.84 | 0/816 | 0/2976 | NVIDIA RTX PRO 6000 Blackwell Server Edition |
| r36_M_n60_G_SHIELD_ORACLE_BCE | 22.49 | 31.26 | 37.18 | 1.79 | 2.80 | 3.59 | 99.11 | -406.09 | 99.11 | 99.11 | 0/590 | 2/2887 | NVIDIA RTX PRO 6000 Blackwell Server Edition |
| r36_M_n60_G_SHIELD_ORACLE_NCE | 39.35 | 50.00 | 55.03 | 6.11 | 8.26 | 9.66 | 168.42 | -336.79 | 168.42 | 168.42 | 3/422 | 0/2666 | NVIDIA RTX PRO 6000 Blackwell Server Edition |
| r36_M_n60_G_RELABEL_AVAILABLE_ORACLE_J | 81.36 | 96.45 | 98.22 | 60.18 | 84.04 | 89.94 | 510.20 | 4.99 | 510.20 | 510.20 | 41/34 | 225/150 | NVIDIA RTX PRO 6000 Blackwell Server Edition |
| r36_M_n60_G_RELABEL_AVAILABLE_ORACLE_BCE | 49.51 | 66.86 | 75.94 | 18.36 | 26.47 | 31.93 | 269.07 | -236.13 | 269.07 | 269.07 | 9/325 | 3/2048 | NVIDIA RTX PRO 6000 Blackwell Server Edition |
| r36_M_n60_G_RELABEL_AVAILABLE_ORACLE_NCE | 65.88 | 79.39 | 85.31 | 28.88 | 34.91 | 38.19 | 332.54 | -172.66 | 332.54 | 332.54 | 19/169 | 15/1527 | NVIDIA RTX PRO 6000 Blackwell Server Edition |
| r36_T_n60_G_J | 80.87 | 96.15 | 98.13 | 59.43 | 83.69 | 89.70 | 507.97 | 2.76 | 507.97 | 507.97 | 37/35 | 202/165 | NVIDIA RTX PRO 6000 Blackwell Server Edition |
| r36_T_n60_G_BCE | 80.67 | 95.07 | 97.63 | 58.11 | 80.26 | 85.78 | 497.51 | -7.69 | 497.51 | 497.51 | 26/26 | 117/147 | NVIDIA RTX PRO 6000 Blackwell Server Edition |
| r36_T_n60_G_NCE | 79.88 | 95.07 | 98.13 | 58.19 | 83.02 | 88.54 | 502.82 | -2.39 | 502.82 | 502.82 | 31/39 | 202/228 | NVIDIA RTX PRO 6000 Blackwell Server Edition |
| r36_T_n60_L_J | 80.57 | 95.86 | 98.22 | 59.05 | 83.53 | 89.53 | 506.77 | 1.56 | 506.77 | 506.77 | 37/38 | 199/181 | NVIDIA RTX PRO 6000 Blackwell Server Edition |
| r36_T_n60_L_BCE | 71.10 | 85.40 | 90.43 | 49.13 | 66.39 | 72.68 | 435.15 | -70.06 | 435.15 | 435.15 | 25/122 | 129/614 | NVIDIA RTX PRO 6000 Blackwell Server Edition |
| r36_T_n60_L_NCE | 71.70 | 90.73 | 94.67 | 51.38 | 72.17 | 79.27 | 459.92 | -45.29 | 459.92 | 459.92 | 32/123 | 210/581 | NVIDIA RTX PRO 6000 Blackwell Server Edition |
| r36_N_n60_G_J | 80.97 | 96.15 | 98.22 | 59.17 | 83.57 | 89.64 | 507.73 | 2.52 | 507.73 | 507.73 | 30/27 | 200/176 | NVIDIA RTX PRO 6000 Blackwell Server Edition |
| r36_N_n60_G_BCE | 81.07 | 96.06 | 98.03 | 59.05 | 83.81 | 89.57 | 507.57 | 2.37 | 507.57 | 507.57 | 32/28 | 205/187 | NVIDIA RTX PRO 6000 Blackwell Server Edition |
| r36_N_n60_G_NCE | 80.67 | 95.46 | 98.32 | 59.05 | 83.35 | 89.63 | 506.49 | 1.28 | 506.49 | 506.49 | 27/27 | 209/191 | NVIDIA RTX PRO 6000 Blackwell Server Edition |
| r36_N_n60_L_J | 80.28 | 96.06 | 98.13 | 59.33 | 83.77 | 89.63 | 507.18 | 1.97 | 507.18 | 507.18 | 32/36 | 208/176 | NVIDIA L40S |
| r36_N_n60_L_BCE | 80.57 | 96.15 | 98.13 | 59.07 | 84.04 | 89.68 | 507.65 | 2.45 | 507.65 | 507.65 | 32/33 | 208/189 | NVIDIA L40S |
| r36_N_n60_L_NCE | 80.28 | 95.27 | 98.13 | 58.68 | 83.53 | 89.57 | 505.44 | 0.24 | 505.44 | 505.44 | 29/33 | 202/203 | NVIDIA L40S |
| r36_V_n00_h2002_L_J | 84.32 | 96.15 | 98.22 | 66.11 | 87.95 | 92.50 | 525.27 | 3.20 | 525.27 | 525.27 | 33/35 | 198/104 | NVIDIA L40S |
| r36_V_n00_h2002_L_BCE | 83.53 | 96.25 | 98.03 | 66.04 | 87.75 | 92.56 | 524.16 | 2.09 | 524.16 | 524.16 | 34/44 | 204/114 | NVIDIA L40S |
| r36_V_n00_h2002_L_NCE | 73.67 | 91.12 | 95.96 | 51.01 | 70.06 | 77.75 | 459.57 | -62.50 | 459.57 | 459.57 | 27/137 | 155/827 | NVIDIA L40S |
| r36_V_n00_h2003_L_J | 83.83 | 96.55 | 98.52 | 65.98 | 87.79 | 92.50 | 525.17 | 3.10 | 525.17 | 525.17 | 29/36 | 195/108 | NVIDIA L40S |
| r36_V_n00_h2003_L_BCE | 82.74 | 96.15 | 98.13 | 65.80 | 87.89 | 92.56 | 523.27 | 1.20 | 523.27 | 523.27 | 28/46 | 207/129 | NVIDIA L40S |
| r36_V_n00_h2003_L_NCE | 67.36 | 86.19 | 93.10 | 42.01 | 60.87 | 69.13 | 418.66 | -103.41 | 418.66 | 418.66 | 27/201 | 104/1232 | NVIDIA L40S |

## V: phase-1 step 8000 (secondary) vs phase-2 step 4000 (primary, total 12000)

| unit | rSum @8000 | rSum @12000 | phase-2 step0 == phase-1 final |
|---|---|---|---|
| r36_V_n00_h2002_L_J | 524.22 | 525.27 | True |
| r36_V_n00_h2002_L_BCE | 522.60 | 524.16 | True |
| r36_V_n00_h2002_L_NCE | 507.26 | 459.57 | True |
| r36_V_n00_h2003_L_J | 524.00 | 525.17 | True |
| r36_V_n00_h2003_L_BCE | 522.29 | 523.27 | True |
| r36_V_n00_h2003_L_NCE | 509.11 | 418.66 | True |

Frozen H native DEV rSum: n00 522.07, n60 505.21

## R35 reference rows (read-only, seed1001, fixed endpoints)

| unit | step | rSum | Δ vs H | rescued/lost R@1 i2t | t2i |
|---|---|---|---|---|---|
| r35_C_n60_s1001_G_J | 8000 | 6.35 | -498.86 | 0/816 | 0/2976 |
| r35_C_n60_s1001_G_BCE | 8000 | 92.94 | -412.27 | 0/606 | 1/2907 |
| r35_C_n60_s1001_G_NCE | 8000 | 153.59 | -351.62 | 1/458 | 0/2761 |
| r35_C_n60_s1001_L_J | 8000 | 5.58 | -499.63 | 0/818 | 0/2976 |
| r35_C_n60_s1001_L_BCE | 8000 | 52.82 | -452.39 | 1/732 | 0/2961 |
| r35_C_n60_s1001_L_NCE | 8000 | 121.68 | -383.53 | 6/581 | 0/2855 |
| r35_B_n60_s1001_TRUST_J | 8000 | 499.76 | -5.44 | 22/39 | 136/219 |
| r35_B_n60_s1001_TRUST_BCE | 8000 | 500.73 | -4.48 | 26/38 | 141/213 |
| r35_B_n60_s1001_TRUST_NCE | 8000 | 502.09 | -3.12 | 31/35 | 163/207 |
| r35_C_n00_s1001_L_J | 8000 | 523.63 | 1.56 | 22/37 | 190/97 |
| r35_C_n00_s1001_L_BCE | 8000 | 522.70 | 0.63 | 23/40 | 186/105 |
| r35_C_n00_s1001_L_NCE | 8000 | 515.05 | -7.02 | 33/55 | 263/267 |
| r35_D_n00_s1001_L_J_HEAD_CONTROL | 4000 | 524.62 | 2.54 | 32/39 | 198/100 |
| r35_D_n00_s1001_L_BCE_HEAD_CONTROL | 4000 | 522.17 | 0.10 | 28/54 | 210/135 |
| r35_D_n00_s1001_L_NCE_HEAD_CONTROL | 4000 | 404.38 | -117.69 | 20/247 | 100/1314 |

J columns (RESULTS_FIXED.csv): DEV list J (clean annotation, 32 slots), FIT forced-list J (observed), FIT null-list J (N only, 33 slots, Q=Uniform(33)) are different statistical objects and are not ranked against each other.

## Cost (formal logs)

| unit | GPU | updates | train s | eval s | s/update | peak GB |
|---|---|---|---|---|---|---|
| r36_M_n60_G_RANDOM_J | NVIDIA RTX PRO 6000 Blackwell Server Edition | 8000 | 30 | 7 | 0.0038 | 2.3 |
| r36_M_n60_G_RANDOM_BCE | NVIDIA RTX PRO 6000 Blackwell Server Edition | 8000 | 24 | 6 | 0.0030 | 2.3 |
| r36_M_n60_G_RANDOM_NCE | NVIDIA RTX PRO 6000 Blackwell Server Edition | 8000 | 23 | 9 | 0.0028 | 2.3 |
| r36_M_n60_G_SHIELD_ORACLE_J | NVIDIA RTX PRO 6000 Blackwell Server Edition | 8000 | 27 | 6 | 0.0034 | 2.3 |
| r36_M_n60_G_SHIELD_ORACLE_BCE | NVIDIA RTX PRO 6000 Blackwell Server Edition | 8000 | 23 | 7 | 0.0029 | 2.3 |
| r36_M_n60_G_SHIELD_ORACLE_NCE | NVIDIA RTX PRO 6000 Blackwell Server Edition | 8000 | 25 | 4 | 0.0032 | 2.3 |
| r36_M_n60_G_RELABEL_AVAILABLE_ORACLE_J | NVIDIA RTX PRO 6000 Blackwell Server Edition | 8000 | 26 | 6 | 0.0032 | 2.3 |
| r36_M_n60_G_RELABEL_AVAILABLE_ORACLE_BCE | NVIDIA RTX PRO 6000 Blackwell Server Edition | 8000 | 23 | 5 | 0.0028 | 2.3 |
| r36_M_n60_G_RELABEL_AVAILABLE_ORACLE_NCE | NVIDIA RTX PRO 6000 Blackwell Server Edition | 8000 | 22 | 5 | 0.0028 | 2.3 |
| r36_T_n60_G_J | NVIDIA RTX PRO 6000 Blackwell Server Edition | 8000 | 26 | 9 | 0.0032 | 2.3 |
| r36_T_n60_G_BCE | NVIDIA RTX PRO 6000 Blackwell Server Edition | 8000 | 22 | 6 | 0.0028 | 2.3 |
| r36_T_n60_G_NCE | NVIDIA RTX PRO 6000 Blackwell Server Edition | 8000 | 22 | 5 | 0.0027 | 2.3 |
| r36_T_n60_L_J | NVIDIA RTX PRO 6000 Blackwell Server Edition | 8000 | 504 | 83 | 0.0630 | 15.0 |
| r36_T_n60_L_BCE | NVIDIA RTX PRO 6000 Blackwell Server Edition | 8000 | 500 | 66 | 0.0625 | 15.2 |
| r36_T_n60_L_NCE | NVIDIA RTX PRO 6000 Blackwell Server Edition | 8000 | 501 | 67 | 0.0627 | 15.2 |
| r36_N_n60_G_J | NVIDIA RTX PRO 6000 Blackwell Server Edition | 8000 | 35 | 10 | 0.0044 | 2.3 |
| r36_N_n60_G_BCE | NVIDIA RTX PRO 6000 Blackwell Server Edition | 8000 | 28 | 6 | 0.0035 | 2.3 |
| r36_N_n60_G_NCE | NVIDIA RTX PRO 6000 Blackwell Server Edition | 8000 | 28 | 5 | 0.0035 | 2.3 |
| r36_N_n60_L_J | NVIDIA L40S | 8000 | 397 | 87 | 0.0496 | 15.0 |
| r36_N_n60_L_BCE | NVIDIA L40S | 8000 | 393 | 89 | 0.0491 | 15.2 |
| r36_N_n60_L_NCE | NVIDIA L40S | 8000 | 388 | 82 | 0.0485 | 15.2 |
| r36_V_n00_h2002_L_J | NVIDIA L40S | 12000 | 563 | 104 | 0.0469 | 15.2 |
| r36_V_n00_h2002_L_BCE | NVIDIA L40S | 12000 | 558 | 100 | 0.0465 | 15.2 |
| r36_V_n00_h2002_L_NCE | NVIDIA L40S | 12000 | 557 | 101 | 0.0464 | 15.2 |
| r36_V_n00_h2003_L_J | NVIDIA L40S | 12000 | 566 | 123 | 0.0472 | 15.2 |
| r36_V_n00_h2003_L_BCE | NVIDIA L40S | 12000 | 561 | 100 | 0.0467 | 15.2 |
| r36_V_n00_h2003_L_NCE | NVIDIA L40S | 12000 | 560 | 127 | 0.0467 | 15.2 |

Failures / retries: 0 (FAILURES.csv). Other tables: CURVES.csv, RANK_TRANSITIONS.csv, NULL_READOUTS.csv, COST_LEDGER.csv, R35_REFERENCE_ROWS.csv; evaluator-only audit: AUDIT_RESULTS.json, PROVENANCE_AUDIT.csv, CANDIDATE_CONFUSION.csv, SCORE_GRAD_AUDIT.csv.
