Publication copy (sample names aliased; numbers and conclusions unchanged).

# R41 extension results (numbers only)

51/51 R41 units complete (A-E 39, amendments F 3 and G 9). DEV two-stage (frozen-H top-100 reranked). TEST not read.

## A: budget 16000 vs 8000 (N x both_joint)

| parent | obj | N-JOINT 8000 (ref) | 16k @8000 | 16k @16000 | 16k@16000 - ref8000 | N-frozen (ref) |
|---|---|---|---|---|---|---|
| F60_R30_H1001 | J | 517.87 | 521.30 | 524.02 | 6.15 | 509.15 |
| F60_R30_H1001 | BCE | 515.68 | 519.31 | 520.79 | 5.11 | 508.40 |
| F60_R30_H1001 | NCE | 515.52 | 516.84 | 518.70 | 3.18 | 506.69 |
| F60_R30_H1002 | J | 493.39 | 496.79 | 500.20 | 6.80 | 485.94 |
| F60_R30_H1002 | BCE | 491.79 | 494.26 | 498.40 | 6.61 | 486.51 |
| F60_R30_H1002 | NCE | 491.18 | 494.87 | 497.24 | 6.06 | 485.74 |
| F60_R30_H1003 | J | 510.47 | 514.77 | 518.78 | 8.30 | 498.46 |
| F60_R30_H1003 | BCE | 508.66 | 513.12 | 516.31 | 7.65 | 497.91 |
| F60_R30_H1003 | NCE | 508.40 | 513.00 | 514.46 | 6.06 | 498.48 |
| F00_R30_H1001 | J | 527.59 | 539.01 | 550.65 | 23.06 | 522.78 |
| F00_R30_H1001 | BCE | 525.66 | 535.23 | 548.22 | 22.56 | 521.95 |
| F00_R30_H1001 | NCE | 551.08 | 562.33 | 567.73 | 16.65 | 524.24 |

## B: training seed 2002 vs seed 1001 (Flickr60 s1001 N-JOINT, 8000)

| obj | seed1001 (R39) | seed2002 | diff |
|---|---|---|---|
| J | 517.87 | 517.02 | -0.85 |
| BCE | 515.68 | 516.92 | 1.24 |
| NCE | 515.52 | 517.10 | 1.58 |

J-BCE: seed1001 2.19, seed2002 0.10; J-NCE: seed1001 2.35, seed2002 -0.08

## C: IN2R-native Flickr parents (8000)

| parent | H | obj | N-frozen | N-JOINT | JOINT-FROZEN |
|---|---|---|---|---|---|
| F40_IN2R_H1001 | 514.18 | J | 518.11 | 536.90 | 18.80 |
| F40_IN2R_H1001 | 514.18 | BCE | 518.56 | 535.78 | 17.22 |
| F40_IN2R_H1001 | 514.18 | NCE | 517.16 | 538.05 | 20.89 |
| F80_IN2R_H1001 | 464.93 | J | 458.05 | 457.55 | -0.49 |
| F80_IN2R_H1001 | 464.93 | BCE | 456.33 | 454.71 | -1.62 |
| F80_IN2R_H1001 | 464.93 | NCE | 456.82 | 456.69 | -0.14 |

## D: ORACLE_MISSING on Flickr60 s1002 (8000)

| obj | N-frozen (R38) | ORACLE-frozen | N-JOINT (R39X) | ORACLE-JOINT | OJ-NJ | OF-NF | [OJ-OF]-[NJ-NF] |
|---|---|---|---|---|---|---|---|
| J | 485.94 | 487.18 | 493.39 | 501.64 | 8.24 | 1.24 | 7.00 |
| BCE | 486.51 | 487.93 | 491.79 | 500.65 | 8.86 | 1.42 | 7.44 |
| NCE | 485.74 | 487.22 | 491.18 | 502.56 | 11.38 | 1.48 | 9.90 |

## E: image_only (8000)

| parent | obj | N-frozen | IMAGE_ONLY | N-JOINT | IMAGE-FROZEN | BOTH-IMAGE |
|---|---|---|---|---|---|---|
| F60_R30_H1002 | J | 485.94 | 489.33 | 493.39 | 3.39 | 4.06 |
| F60_R30_H1002 | BCE | 486.51 | 489.80 | 491.79 | 3.29 | 1.99 |
| F60_R30_H1002 | NCE | 485.74 | 489.17 | 491.18 | 3.43 | 2.01 |
| F60_R30_H1003 | J | 498.46 | 505.86 | 510.47 | 7.40 | 4.62 |
| F60_R30_H1003 | BCE | 497.91 | 505.70 | 508.66 | 7.79 | 2.96 |
| F60_R30_H1003 | NCE | 498.48 | 504.46 | 508.40 | 5.98 | 3.94 |

## F (amendment): CC152K CC31 N-JOINT with student lr 1e-6 -> 1e-7 (10x lower), 8000

| obj | N-frozen (R38) | N-JOINT (R39, enc lr x1) | N-JOINT enc lr x0.1 | F-NF | F-NJ |
|---|---|---|---|---|---|
| J | 392.70 | 390.80 | 392.30 | -0.40 | 1.50 |
| BCE | 392.80 | 390.80 | 391.90 | -0.90 | 1.10 |
| NCE | 388.60 | 386.80 | 387.50 | -1.10 | 0.70 |

## G1 (amendment): Flickr 0% R30-H f00_s1001 N-JOINT, training seed 2002 vs 1001 (8000)

| obj | seed1001 (R39X) | seed2002 | diff |
|---|---|---|---|
| J | 527.59 | 528.48 | 0.89 |
| BCE | 525.66 | 524.24 | -1.42 |
| NCE | 551.08 | 550.77 | -0.32 |

J-BCE: seed1001 1.93, seed2002 4.24; J-NCE: seed1001 -23.49, seed2002 -22.29

## G2 (amendment): ORACLE_MISSING on Flickr60 s1003 (8000)

| obj | N-frozen (R38) | ORACLE-frozen | N-JOINT (R39X) | ORACLE-JOINT | OJ-NJ | OF-NF | [OJ-OF]-[NJ-NF] |
|---|---|---|---|---|---|---|---|
| J | 498.46 | 499.80 | 510.47 | 517.69 | 7.22 | 1.34 | 5.88 |
| BCE | 497.91 | 499.90 | 508.66 | 517.10 | 8.44 | 1.99 | 6.45 |
| NCE | 498.48 | 499.98 | 508.40 | 520.67 | 12.27 | 1.50 | 10.77 |

Files: R41_RESULTS.csv (six recalls, step-8000 rSum for A, in-unit check flags), R41_CONTRASTS.csv, R41_CURVES.csv, R41_COST.csv, R41_FAILURES.csv.
