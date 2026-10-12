# R5-QX1 — RefCOCO: VCS / JS with the query-normalized critic vs the R4-VL1 / R5-30 objectives (DEV all-proposal image-macro Acc@0.5; exploratory)

Gates: 9 / 9 pass.  Missing: none.

## refcoco

| arm | seed 0 | seed 1 | seed 2 | mean ± sd | Acc@0.75 | query | given-box | J_recal | epochs |
|---|---|---|---|---|---|---|---|---|---|
| VQ | 74.02 | 74.55 | 73.93 | 74.17 ± 0.33 | 60.22 ± 0.25 | 72.52 ± 0.37 | 81.82 ± 0.80 | 59.69 ± 0.37 | [8, 8, 8] |
| JQ | 72.44 | 74.51 | 72.84 | 73.26 ± 1.10 | 59.19 ± 0.82 | 71.61 ± 1.01 | 81.28 ± 0.39 | 60.12 ± 0.26 | [7, 9, 7] |
| E | 75.76 | 75.42 | 75.16 | 75.45 ± 0.30 | 62.53 ± 0.35 | 74.00 ± 0.13 | 83.33 ± 0.21 | 58.20 ± 0.34 | [5, 7, 5] |
| VCS | 66.29 | 67.33 | 66.20 | 66.60 ± 0.63 | 53.69 ± 0.48 | 65.15 ± 0.43 | 75.42 ± 0.43 | 61.00 ± 0.90 | [8, 10, 7] |
| JS | 66.02 | 66.82 | 66.78 | 66.54 ± 0.45 | 53.51 ± 0.08 | 65.13 ± 0.51 | 74.51 ± 0.52 | 61.73 ± 1.65 | [8, 8, 4] |
| D | 75.82 | 75.82 | 76.06 | 75.90 ± 0.14 | 61.27 ± 0.12 | 74.34 ± 0.16 | 83.51 ± 0.52 | 60.37 ± 0.22 | [9, 8, 9] |

| contrast (paired by seed) | value | reading |
|---|---|---|
| VQ_minus_E (primary) | -1.28 [-2.35, -0.21] | below 0 |
| VQ_minus_VCS (primary) | +7.56 [+6.82, +8.30] | above 0 |
| JQ_minus_E | -2.18 [-5.19, +0.83] | interval contains 0 |
| JQ_minus_JS | +6.72 [+4.58, +8.86] | above 0 |
| VQ_minus_JQ | +0.90 [-1.07, +2.87] | interval contains 0 |
| VQ_minus_D | -1.73 [-2.80, -0.66] | below 0 |
| D_minus_E | +0.45 [-0.59, +1.49] | interval contains 0 |

Decomposition (means, exact): VCS − E -8.84 = (VCS − VQ) -7.56 + (VQ − D) -1.73 + (D − E) +0.45
