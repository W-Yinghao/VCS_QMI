Publication copy (sample names aliased; numbers and conclusions unchanged).

# R50 results (numbers only)

protocol_id = FIT_FIXED_CENTER_V1 on every row (PRIMARY). SECONDARY_TRANSDUCTIVE = R47 G / R48 original split-global readout, separate columns only. DEV only; no TEST read. Primary endpoint step 8000. NA = not available.

Formal units DONE: 24/24. References re-read: 18/18.

## Parent H (same protocol)

| parent | PRIMARY rSum | i2t R@1/5/10 | t2i R@1/5/10 | SECONDARY (R47 G) rSum | native-layout rSum |
|---|---:|---|---|---:|---:|
| Flickr60 R30-H s1001 | 496.23 | 76.63/94.87/97.73 | 56.29/82.25/88.46 | 496.53 | 505.21 |
| CC152K R31-H0 s1001 | 386.40 | 44.20/70.80/79.20 | 42.30/70.50/79.40 | 385.10 | 389.70 |
| Flickr80 IN2R H s1001 | 453.10 | 66.37/88.26/93.69 | 47.06/74.77/82.94 | 453.85 | 464.93 |
| Flickr0 R30-H s1001 | 512.56 | 80.47/95.27/98.22 | 61.68/85.86/91.07 | 511.85 | 522.07 |
| COCO60 IN2R H s1001 | 525.02 | 78.20/97.30/99.20 | 62.80/91.24/96.28 | 524.20 | 538.94 |

## N references re-read (R48 N-frozen / R47 G N-JOINT weights, step 8000)

| unit | PRIMARY rSum | -H | i2t R@1/5/10 | t2i R@1/5/10 | SECONDARY rSum | SECONDARY -H(sec) |
|---|---:|---:|---|---|---:|---:|
| R48_F60_R30_H1001_N_frozen_lfG_J | 492.66 | -3.57 | 75.64/94.28/97.73 | 55.62/81.36/88.03 | 492.80 | -3.73 |
| R48_F60_R30_H1001_N_frozen_lfG_BCE | 492.29 | -3.94 | 75.64/94.28/97.44 | 55.40/81.44/88.09 | 492.01 | -4.52 |
| R48_F60_R30_H1001_N_frozen_lfG_NCE | 492.35 | -3.89 | 76.04/93.79/97.73 | 55.27/81.50/88.03 | 492.50 | -4.02 |
| R47_G_F60_R30_H1001_N_joint_J | 491.60 | -4.64 | 74.65/94.08/97.73 | 55.70/81.54/87.89 | 491.68 | -4.95 |
| R47_G_F60_R30_H1001_N_joint_BCE | 489.82 | -6.41 | 74.26/93.69/97.24 | 55.38/81.42/87.83 | 490.83 | -5.80 |
| R47_G_F60_R30_H1001_N_joint_NCE | 490.14 | -6.09 | 74.56/93.98/97.04 | 55.15/81.34/88.07 | 490.77 | -5.86 |
| R48_CC31_H1001_N_frozen_lfG_J | 390.00 | 3.60 | 45.00/71.60/79.60 | 44.00/70.80/79.00 | 390.00 | 4.80 |
| R48_CC31_H1001_N_frozen_lfG_BCE | 389.10 | 2.70 | 44.90/71.30/79.70 | 44.00/70.40/78.80 | 390.10 | 4.90 |
| R48_CC31_H1001_N_frozen_lfG_NCE | 384.80 | -1.60 | 44.10/70.40/78.70 | 43.40/70.10/78.10 | 385.30 | 0.10 |
| R47_G_CC31_H1001_N_joint_J | 386.70 | 0.30 | 43.50/69.90/79.50 | 44.30/70.80/78.70 | 387.90 | 2.80 |
| R47_G_CC31_H1001_N_joint_BCE | 386.50 | 0.10 | 44.30/70.10/79.20 | 44.20/70.00/78.70 | 387.20 | 2.10 |
| R47_G_CC31_H1001_N_joint_NCE | 384.30 | -2.10 | 43.90/69.40/78.20 | 44.00/70.30/78.50 | 385.70 | 0.60 |
| R48_F80_IN2R_H1001_N_frozen_lfG_J | 441.72 | -11.38 | 64.00/87.87/92.31 | 44.99/71.46/81.08 | 442.45 | -11.40 |
| R48_F80_IN2R_H1001_N_frozen_lfG_BCE | 439.74 | -13.35 | 63.91/87.28/92.21 | 44.60/71.18/80.57 | 440.39 | -13.45 |
| R48_F80_IN2R_H1001_N_frozen_lfG_NCE | 441.62 | -11.48 | 64.99/86.98/92.70 | 44.40/71.81/80.73 | 442.05 | -11.79 |
| R47_G_F80_IN2R_H1001_N_joint_J | 434.18 | -18.92 | 62.13/86.00/92.01 | 44.00/70.24/79.80 | 434.69 | -19.15 |
| R47_G_F80_IN2R_H1001_N_joint_BCE | 432.74 | -20.36 | 61.64/86.19/91.81 | 43.53/69.96/79.61 | 432.35 | -21.50 |
| R47_G_F80_IN2R_H1001_N_joint_NCE | 434.52 | -18.58 | 62.52/85.70/92.01 | 43.71/70.47/80.10 | 434.06 | -19.78 |

## R50 units (step 8000)

| unit | rSum | -H | -N | i2t R@1/5/10 | t2i R@1/5/10 | DEV cov i2t/t2i | train s | GPU |
|---|---:|---:|---:|---|---|---|---:|---|
| R50_A_F60_R30H1001_PRESENT_ONLY_frozen_J | 491.64 | -4.60 | -1.03 | 75.54/93.98/97.34 | 55.50/81.32/87.95 | 99.90/98.64 | 98.23 | NVIDIA L40S |
| R50_A_F60_R30H1001_PRESENT_ONLY_frozen_BCE | 491.24 | -4.99 | -1.05 | 75.35/93.89/97.24 | 55.46/81.44/87.87 | 99.90/98.64 | 90.55 | NVIDIA L40S |
| R50_A_F60_R30H1001_PRESENT_ONLY_frozen_NCE | 493.27 | -2.96 | 0.93 | 75.54/94.38/97.63 | 55.78/81.60/88.34 | 99.90/98.64 | 86.09 | NVIDIA L40S |
| R50_A_F60_R30H1001_PRESENT_ONLY_both_joint_J | 490.16 | -6.07 | -1.44 | 74.46/93.79/96.84 | 55.60/81.38/88.09 | 99.90/98.64 | 1252.49 | NVIDIA RTX PRO 6000 Blackwell Server Edition |
| R50_A_F60_R30H1001_PRESENT_ONLY_both_joint_BCE | 489.63 | -6.61 | -0.20 | 73.96/93.59/97.04 | 55.64/81.48/87.91 | 99.90/98.64 | 1250.43 | NVIDIA RTX PRO 6000 Blackwell Server Edition |
| R50_A_F60_R30H1001_PRESENT_ONLY_both_joint_NCE | 489.78 | -6.45 | -0.36 | 74.65/94.08/96.94 | 54.87/81.10/88.13 | 99.90/98.64 | 1265.10 | NVIDIA RTX PRO 6000 Blackwell Server Edition |
| R50_A_CC_R31H1001_PRESENT_ONLY_frozen_J | 389.50 | 3.10 | -0.50 | 45.20/71.40/80.00 | 43.90/70.40/78.60 | 96.10/96.10 | 47.20 | NVIDIA RTX PRO 6000 Blackwell Server Edition |
| R50_A_CC_R31H1001_PRESENT_ONLY_frozen_BCE | 389.10 | 2.70 | 0.00 | 45.00/71.60/80.00 | 43.80/70.40/78.30 | 96.10/96.10 | 39.91 | NVIDIA RTX PRO 6000 Blackwell Server Edition |
| R50_A_CC_R31H1001_PRESENT_ONLY_frozen_NCE | 387.50 | 1.10 | 2.70 | 45.50/70.70/79.40 | 43.60/70.60/77.70 | 96.10/96.10 | 38.80 | NVIDIA RTX PRO 6000 Blackwell Server Edition |
| R50_A_CC_R31H1001_PRESENT_ONLY_both_joint_J | 386.40 | 0.00 | -0.30 | 44.50/70.30/79.20 | 43.80/70.20/78.40 | 96.10/96.10 | 1418.74 | NVIDIA RTX PRO 6000 Blackwell Server Edition |
| R50_A_CC_R31H1001_PRESENT_ONLY_both_joint_BCE | 386.70 | 0.30 | 0.20 | 44.40/70.40/79.20 | 44.20/69.90/78.60 | 96.10/96.10 | 1418.05 | NVIDIA RTX PRO 6000 Blackwell Server Edition |
| R50_A_CC_R31H1001_PRESENT_ONLY_both_joint_NCE | 386.80 | 0.40 | 2.50 | 44.60/70.50/78.80 | 44.00/70.90/78.00 | 96.10/96.10 | 1579.29 | NVIDIA RTX PRO 6000 Blackwell Server Edition |
| R50_B_F60_R30H1001_ORACLE_MISSING_frozen_J | 494.06 | -2.17 | 1.40 | 75.74/94.97/97.63 | 55.88/81.62/88.22 | 99.90/98.64 | 41.49 | NVIDIA RTX PRO 6000 Blackwell Server Edition |
| R50_B_F60_R30H1001_ORACLE_MISSING_frozen_BCE | 493.47 | -2.76 | 1.18 | 75.84/94.58/97.34 | 55.94/81.68/88.11 | 99.90/98.64 | 33.29 | NVIDIA RTX PRO 6000 Blackwell Server Edition |
| R50_B_F60_R30H1001_ORACLE_MISSING_frozen_NCE | 492.64 | -3.59 | 0.30 | 75.94/94.18/97.34 | 55.72/81.38/88.09 | 99.90/98.64 | 32.54 | NVIDIA RTX PRO 6000 Blackwell Server Edition |
| R50_B_F60_R30H1001_ORACLE_MISSING_both_joint_J | 493.20 | -3.04 | 1.60 | 75.54/94.08/97.34 | 56.19/81.87/88.17 | 99.90/98.64 | 1251.24 | NVIDIA RTX PRO 6000 Blackwell Server Edition |
| R50_B_F60_R30H1001_ORACLE_MISSING_both_joint_BCE | 493.29 | -2.94 | 3.47 | 75.35/93.98/97.44 | 56.37/81.89/88.26 | 99.90/98.64 | 1248.11 | NVIDIA RTX PRO 6000 Blackwell Server Edition |
| R50_B_F60_R30H1001_ORACLE_MISSING_both_joint_NCE | 494.18 | -2.05 | 4.04 | 75.84/94.28/97.53 | 55.88/82.11/88.54 | 99.90/98.64 | 1247.50 | NVIDIA RTX PRO 6000 Blackwell Server Edition |
| R50_B_F80_IN2RH1001_ORACLE_MISSING_frozen_J | 448.97 | -4.12 | 7.26 | 65.29/88.66/93.49 | 46.53/73.18/81.83 | 99.70/97.53 | 85.73 | NVIDIA L40S |
| R50_B_F80_IN2RH1001_ORACLE_MISSING_frozen_BCE | 450.06 | -3.04 | 10.32 | 65.09/88.56/93.89 | 46.67/73.69/82.17 | 99.70/97.53 | 79.83 | NVIDIA L40S |
| R50_B_F80_IN2RH1001_ORACLE_MISSING_frozen_NCE | 448.24 | -4.85 | 6.63 | 65.68/88.36/93.59 | 45.64/73.02/81.95 | 99.70/97.53 | 76.98 | NVIDIA L40S |
| R50_B_F80_IN2RH1001_ORACLE_MISSING_both_joint_J | 447.63 | -5.46 | 13.45 | 65.19/88.26/93.59 | 46.43/72.41/81.76 | 99.70/97.53 | 1253.28 | NVIDIA RTX PRO 6000 Blackwell Server Edition |
| R50_B_F80_IN2RH1001_ORACLE_MISSING_both_joint_BCE | 448.17 | -4.93 | 15.42 | 65.29/88.26/93.20 | 46.67/72.92/81.83 | 99.70/97.53 | 1250.87 | NVIDIA RTX PRO 6000 Blackwell Server Edition |
| R50_B_F80_IN2RH1001_ORACLE_MISSING_both_joint_NCE | 448.24 | -4.85 | 13.73 | 65.48/88.17/93.49 | 46.35/73.12/81.64 | 99.70/97.53 | 1248.99 | NVIDIA RTX PRO 6000 Blackwell Server Edition |

## 2x2 contrasts (rSum, step 8000)

| parent | supervision | obj | X-N frozen | X-N joint | interaction (XJ-XF)-(NJ-NF) |
|---|---|---|---:|---:|---:|
| CC152K R31-H0 s1001 | PRESENT_ONLY | J | -0.50 | -0.30 | 0.20 |
| CC152K R31-H0 s1001 | PRESENT_ONLY | BCE | 0.00 | 0.20 | 0.20 |
| CC152K R31-H0 s1001 | PRESENT_ONLY | NCE | 2.70 | 2.50 | -0.20 |
| Flickr60 R30-H s1001 | ORACLE_MISSING | J | 1.40 | 1.60 | 0.20 |
| Flickr60 R30-H s1001 | ORACLE_MISSING | BCE | 1.18 | 3.47 | 2.29 |
| Flickr60 R30-H s1001 | ORACLE_MISSING | NCE | 0.30 | 4.04 | 3.75 |
| Flickr60 R30-H s1001 | PRESENT_ONLY | J | -1.03 | -1.44 | -0.41 |
| Flickr60 R30-H s1001 | PRESENT_ONLY | BCE | -1.05 | -0.20 | 0.85 |
| Flickr60 R30-H s1001 | PRESENT_ONLY | NCE | 0.93 | -0.36 | -1.28 |
| Flickr80 IN2R H s1001 | ORACLE_MISSING | J | 7.26 | 13.45 | 6.19 |
| Flickr80 IN2R H s1001 | ORACLE_MISSING | BCE | 10.32 | 15.42 | 5.11 |
| Flickr80 IN2R H s1001 | ORACLE_MISSING | NCE | 6.63 | 13.73 | 7.10 |

## Objective contrasts

| parent | supervision | encoder | J-BCE | J-NCE |
|---|---|---|---:|---:|
| CC152K R31-H0 s1001 | PRESENT_ONLY | frozen | 0.40 | 2.00 |
| CC152K R31-H0 s1001 | PRESENT_ONLY | both_joint | -0.30 | -0.40 |
| CC152K R31-H0 s1001 | N_reference | frozen | 0.90 | 5.20 |
| CC152K R31-H0 s1001 | N_reference | both_joint | 0.20 | 2.40 |
| Flickr60 R30-H s1001 | ORACLE_MISSING | frozen | 0.59 | 1.42 |
| Flickr60 R30-H s1001 | ORACLE_MISSING | both_joint | -0.10 | -0.99 |
| Flickr60 R30-H s1001 | PRESENT_ONLY | frozen | 0.39 | -1.64 |
| Flickr60 R30-H s1001 | PRESENT_ONLY | both_joint | 0.53 | 0.37 |
| Flickr60 R30-H s1001 | N_reference | frozen | 0.37 | 0.32 |
| Flickr60 R30-H s1001 | N_reference | both_joint | 1.78 | 1.46 |
| Flickr80 IN2R H s1001 | ORACLE_MISSING | frozen | -1.08 | 0.73 |
| Flickr80 IN2R H s1001 | ORACLE_MISSING | both_joint | -0.53 | -0.61 |
| Flickr80 IN2R H s1001 | N_reference | frozen | 1.97 | 0.10 |
| Flickr80 IN2R H s1001 | N_reference | both_joint | 1.44 | -0.34 |

## Oracle exposure (full FIT pool, oracle process)

| cond | obs mismatch | true in top32 i2t/t2i | observed in top32 i2t/t2i | relabelled rows i2t/t2i | relabelled of null rows i2t/t2i | added true mass/row i2t/t2i |
|---|---:|---|---|---|---|---|
| n60_lfF | 0.6000 | 0.9690/0.8228 | 0.8900/0.3979 | 0.0812/0.4297 | 0.7377/0.7136 | 0.0406/0.2148 |
| f80in_lfF | 0.8000 | 0.8803/0.6772 | 0.6740/0.2268 | 0.2230/0.4668 | 0.6840/0.6037 | 0.1115/0.2334 |

In-training oracle counters (batch_stats sums of the per-update row counts over all updates): R50_RESULTS.csv columns oracle_*.
Curves: R50_CURVES.csv. Reuse receipt: REUSE_RECEIPT_R50.json. Normalisation: NORMALIZATION_MANIFEST.json.
