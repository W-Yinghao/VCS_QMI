Publication copy (sample names aliased; numbers and conclusions unchanged).

# R44 results (numbers only)

> **R44 audit concluded (2026-11-03): the B/C ROUTE_PRESENT numbers come from a layout leak through the JOINT student centring (R44_AUDIT.md); do not use them.**

> **Readout note (2026-11-03, R44 audit):** numbers in this file use the native IN²R per-batch centring over the sequential native layout (true image-caption pairs share a batch mean); JOINT students since R39 were also trained with that layout. Layout-free re-evaluations of the same checkpoints: CORRECTED_READOUTS.md; audit: R44_AUDIT.md; layout-free retraining: RESULTS_R47.md.

39/39 R44 units complete. DEV two-stage (frozen-H top-100 reranked). TEST not read.

## A: Flickr60 R30-H N-JOINT 16000 updates, training seed 1001 (R41-A) vs 2002 (R44)

| parent | obj | s1001 @8000 | s2002 @8000 | s1001 @12000 | s2002 @12000 | s1001 @16000 | s2002 @16000 | 2002-1001 @16000 |
|---|---|---|---|---|---|---|---|---|
| F60_R30_H1001 | J | 521.30 | 520.63 | NA | 522.62 | 524.02 | 523.69 | -0.34 |
| F60_R30_H1001 | BCE | 519.31 | 519.49 | NA | 521.46 | 520.79 | 521.42 | 0.63 |
| F60_R30_H1001 | NCE | 516.84 | 519.33 | NA | 519.21 | 518.70 | 519.45 | 0.75 |
| F60_R30_H1002 | J | 496.79 | 497.67 | NA | 498.86 | 500.20 | 500.04 | -0.16 |
| F60_R30_H1002 | BCE | 494.26 | 494.97 | NA | 497.53 | 498.40 | 498.80 | 0.39 |
| F60_R30_H1002 | NCE | 494.87 | 495.27 | NA | 497.02 | 497.24 | 497.59 | 0.36 |
| F60_R30_H1003 | J | 514.77 | 518.32 | NA | 521.16 | 518.78 | 521.28 | 2.50 |
| F60_R30_H1003 | BCE | 513.12 | 516.57 | NA | 518.68 | 516.31 | 519.57 | 3.25 |
| F60_R30_H1003 | NCE | 513.00 | 515.35 | NA | 517.95 | 514.46 | 518.11 | 3.65 |

### A: nested mean at 16000 (mean over parents of the per-parent seed mean) and objective gaps per seed

| obj | nested mean | mean s1001 | mean s2002 |
|---|---|---|---|
| J | 514.67 | 514.33 | 515.00 |
| BCE | 512.55 | 511.83 | 513.26 |
| NCE | 510.92 | 510.13 | 511.72 |

| parent | seed | J-BCE @16000 | J-NCE @16000 |
|---|---|---|---|
| F60_R30_H1001 | 1001 | 3.23 | 5.33 |
| F60_R30_H1001 | 2002 | 2.27 | 4.24 |
| F60_R30_H1002 | 1001 | 1.79 | 2.96 |
| F60_R30_H1002 | 2002 | 1.24 | 2.45 |
| F60_R30_H1003 | 1001 | 2.47 | 4.32 |
| F60_R30_H1003 | 2002 | 1.72 | 3.18 |

## B/C: gradient routing (8000 updates, training seed 1001)

| condition | obj | ALL | PRESENT_ALL | ROUTE_PRESENT | ROUTE_RANDOM | RP-ALL | RP-PA | RP-RR | RR-ALL |
|---|---|---|---|---|---|---|---|---|---|
| F00_R30_H1001 | J | 527.59 | 527.28 | 532.52 | 527.87 | 4.93 | 5.25 | 4.65 | 0.28 |
| F00_R30_H1001 | BCE | 525.66 | 525.64 | 530.02 | 525.66 | 4.36 | 4.38 | 4.36 | 0.00 |
| F00_R30_H1001 | NCE | 551.08 | 550.37 | 558.74 | 550.65 | 7.65 | 8.36 | 8.09 | -0.43 |
| F60_R30_H1001 | J | 517.87 | 516.11 | 567.50 | 515.35 | 49.63 | 51.38 | 52.15 | -2.52 |
| F60_R30_H1001 | BCE | 515.68 | 514.08 | 567.48 | 514.60 | 51.79 | 53.39 | 52.88 | -1.08 |
| F60_R30_H1001 | NCE | 515.52 | 517.93 | 569.76 | 514.42 | 54.24 | 51.83 | 55.35 | -1.10 |
| F80_IN2R_H1001 | J | 457.55 | 455.03 | 542.50 | 457.91 | 84.95 | 87.48 | 84.60 | 0.36 |
| F80_IN2R_H1001 | BCE | 454.71 | 454.64 | 536.79 | 456.06 | 82.07 | 82.15 | 80.73 | 1.34 |
| F80_IN2R_H1001 | NCE | 456.69 | 447.14 | 539.29 | 456.31 | 82.60 | 92.15 | 82.98 | -0.37 |
| CC_R31_H1001 | J | 390.80 | 391.80 | 393.30 | 390.00 | 2.50 | 1.50 | 3.30 | -0.80 |
| CC_R31_H1001 | BCE | 390.80 | 390.70 | 393.30 | 391.20 | 2.50 | 2.60 | 2.10 | 0.40 |
| CC_R31_H1001 | NCE | 386.80 | 388.30 | 394.20 | 387.90 | 7.40 | 5.90 | 6.30 | 1.10 |

### B/C: objective gaps per arm (J-BCE / J-NCE)

| condition | arm | J-BCE | J-NCE |
|---|---|---|---|
| F00_R30_H1001 | ALL | 1.93 | -23.49 |
| F00_R30_H1001 | PRESENT_ALL | 1.64 | -23.10 |
| F00_R30_H1001 | ROUTE_PRESENT | 2.50 | -26.21 |
| F00_R30_H1001 | ROUTE_RANDOM | 2.21 | -22.78 |
| F60_R30_H1001 | ALL | 2.19 | 2.35 |
| F60_R30_H1001 | PRESENT_ALL | 2.03 | -1.81 |
| F60_R30_H1001 | ROUTE_PRESENT | 0.02 | -2.27 |
| F60_R30_H1001 | ROUTE_RANDOM | 0.75 | 0.93 |
| F80_IN2R_H1001 | ALL | 2.84 | 0.87 |
| F80_IN2R_H1001 | PRESENT_ALL | 0.39 | 7.89 |
| F80_IN2R_H1001 | ROUTE_PRESENT | 5.72 | 3.21 |
| F80_IN2R_H1001 | ROUTE_RANDOM | 1.85 | 1.60 |
| CC_R31_H1001 | ALL | 0.00 | 4.00 |
| CC_R31_H1001 | PRESENT_ALL | 1.10 | 3.50 |
| CC_R31_H1001 | ROUTE_PRESENT | -0.00 | -0.90 |
| CC_R31_H1001 | ROUTE_RANDOM | -1.20 | 2.10 |

Files: R44_RESULTS.csv, R44_CONTRASTS.csv, R44_CURVES.csv, R44_EXPOSURE.csv (routing mask / present rows, empty batches, masked-row gradient counters), R44_COST.csv, R44_FAILURES.csv.
