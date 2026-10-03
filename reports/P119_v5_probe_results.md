# P119 — v5 NEXT-V-PROBE results (frozen encoders; selection split; official test closed)

## 1. CIFAR-10 label efficiency, paired by encoder seed (FIT-selected probe; draws averaged within a seed; draws are not seeds)

### primary — P110 draws 0–2 (re-summary of reported data)

| family | 1 % | 10 % | 100 % |
|---|---|---|---|
| A-P3 | 85.09 ± 0.12 | 87.74 ± 0.08 | 88.93 ± 0.22 |
| G2 | 84.97 ± 0.04 | 87.79 ± 0.11 | 88.67 ± 0.12 |
| U2 | 83.14 ± 0.51 | 86.39 ± 0.12 | 87.34 ± 0.36 |
| recipe VCS | 80.39 ± 0.30 | 85.41 ± 0.09 | 87.23 ± 0.36 |
| SimCLR | 83.92 ± 0.50 | 87.07 ± 0.38 | 88.29 ± 0.34 |

| paired difference (by encoder seed) | mean [95 % t CI over seeds] |
|---|---|
| A-P3 - SimCLR @ 0.01 | +1.17 [-0.08, +2.42] (per seed +1.66, +1.19, +0.65; draw sd 0.84) |
| A-P3 - SimCLR @ 0.1 | +0.66 [-0.33, +1.66] (per seed +0.59, +1.09, +0.30; draw sd 0.34) |
| A-P3 - SimCLR @ 1.0 | +0.65 [+0.07, +1.22] (per seed +0.86, +0.68, +0.40; draw sd nan) |
| G2 - SimCLR @ 0.01 | +1.05 [-0.23, +2.32] (per seed +1.62, +0.89, +0.63; draw sd 0.62) |
| G2 - SimCLR @ 0.1 | +0.71 [-0.50, +1.92] (per seed +0.73, +1.19, +0.22; draw sd 0.37) |
| G2 - SimCLR @ 1.0 | +0.39 [-0.74, +1.52] (per seed +0.46, +0.80, -0.10; draw sd nan) |
| U2 - SimCLR @ 0.01 | -0.78 [-3.31, +1.75] (per seed +0.33, -0.99, -1.67; draw sd 0.73) |
| U2 - SimCLR @ 0.1 | -0.69 [-1.72, +0.35] (per seed -0.52, -0.38, -1.16; draw sd 0.32) |
| U2 - SimCLR @ 1.0 | -0.95 [-2.33, +0.43] (per seed -0.44, -0.86, -1.54; draw sd nan) |
| recipe VCS - SimCLR @ 0.01 | -3.53 [-5.41, -1.65] (per seed -2.66, -3.90, -4.03; draw sd 0.68) |
| recipe VCS - SimCLR @ 0.1 | -1.66 [-2.78, -0.54] (per seed -1.55, -1.27, -2.15; draw sd 0.35) |
| recipe VCS - SimCLR @ 1.0 | -1.06 [-2.23, +0.11] (per seed -1.38, -0.52, -1.28; draw sd nan) |
| A-P3 - G2 @ 0.01 | +0.12 [-0.28, +0.52] (per seed +0.04, +0.31, +0.02; draw sd 0.51) |
| A-P3 - G2 @ 0.1 | -0.05 [-0.34, +0.23] (per seed -0.13, -0.10, +0.08; draw sd 0.32) |
| A-P3 - G2 @ 1.0 | +0.26 [-0.57, +1.09] (per seed +0.40, -0.12, +0.50; draw sd nan) |

### secondary — with P119 extra draws 3–5

| family | 1 % | 10 % | 100 % |
|---|---|---|---|
| A-P3 | 85.02 ± 0.16 | 87.69 ± 0.04 | 88.93 ± 0.22 |
| G2 | 84.77 ± 0.17 | 87.62 ± 0.11 | 88.67 ± 0.12 |
| U2 | 83.29 ± 0.28 | 86.28 ± 0.07 | 87.34 ± 0.36 |
| recipe VCS | 80.27 ± 0.29 | 85.31 ± 0.15 | 87.23 ± 0.36 |
| SimCLR | 84.07 ± 0.60 | 86.90 ± 0.39 | 88.29 ± 0.34 |

| paired difference (by encoder seed) | mean [95 % t CI over seeds] |
|---|---|
| A-P3 - SimCLR @ 0.01 | +0.95 [-0.72, +2.63] (per seed +1.55, +1.09, +0.22; draw sd 0.65) |
| A-P3 - SimCLR @ 0.1 | +0.78 [-0.20, +1.76] (per seed +0.77, +1.18, +0.39; draw sd 0.28) |
| A-P3 - SimCLR @ 1.0 | +0.65 [+0.07, +1.22] (per seed +0.86, +0.68, +0.40; draw sd nan) |
| G2 - SimCLR @ 0.01 | +0.71 [-0.43, +1.84] (per seed +1.21, +0.59, +0.32; draw sd 0.57) |
| G2 - SimCLR @ 0.1 | +0.71 [-0.42, +1.84] (per seed +0.64, +1.20, +0.30; draw sd 0.34) |
| G2 - SimCLR @ 1.0 | +0.39 [-0.74, +1.52] (per seed +0.46, +0.80, -0.10; draw sd nan) |
| U2 - SimCLR @ 0.01 | -0.77 [-2.88, +1.33] (per seed +0.16, -1.00, -1.49; draw sd 0.62) |
| U2 - SimCLR @ 0.1 | -0.62 [-1.57, +0.32] (per seed -0.51, -0.31, -1.05; draw sd 0.30) |
| U2 - SimCLR @ 1.0 | -0.95 [-2.33, +0.43] (per seed -0.44, -0.86, -1.54; draw sd nan) |
| recipe VCS - SimCLR @ 0.01 | -3.80 [-5.90, -1.71] (per seed -2.86, -4.07, -4.48; draw sd 0.59) |
| recipe VCS - SimCLR @ 0.1 | -1.59 [-2.78, -0.40] (per seed -1.40, -1.24, -2.14; draw sd 0.31) |
| recipe VCS - SimCLR @ 1.0 | -1.06 [-2.23, +0.11] (per seed -1.38, -0.52, -1.28; draw sd nan) |
| A-P3 - G2 @ 0.01 | +0.25 [-0.51, +1.01] (per seed +0.34, +0.50, -0.09; draw sd 0.44) |
| A-P3 - G2 @ 0.1 | +0.07 [-0.12, +0.26] (per seed +0.13, -0.02, +0.10; draw sd 0.28) |
| A-P3 - G2 @ 1.0 | +0.26 [-0.57, +1.09] (per seed +0.40, -0.12, +0.50; draw sd nan) |

## 2. CIFAR-10 → CIFAR-100 frozen transfer, identical readouts for every method (mean ± sd over encoder seeds; draws averaged within seed)

Norm readouts (`lognorm_std`, `l2_lognorm_std`) are diagnostics and do not replace the standard table (`raw_unstd` = the P110 readout).

### 100 % labels

| readout | A-P3 | G2 | U2 | recipe VCS | SimCLR |
|---|---|---|---|---|---|
| raw_unstd | 46.81 ± 0.40 | 47.67 ± 0.84 | 45.75 ± 0.55 | 52.68 ± 0.44 | 38.15 ± 0.38 |
| raw_std | 45.53 ± 0.37 | 46.26 ± 0.73 | 44.32 ± 0.67 | 51.22 ± 0.16 | 36.75 ± 0.45 |
| l2_std | 45.23 ± 0.31 | 45.91 ± 0.57 | 44.19 ± 0.51 | 51.13 ± 0.29 | 36.67 ± 0.53 |
| lognorm_std | 2.19 ± 0.17 | 2.20 ± 0.05 | 2.07 ± 0.07 | 2.01 ± 0.16 | 2.21 ± 0.32 |
| l2_lognorm_std | 45.39 ± 0.29 | 45.87 ± 0.69 | 44.23 ± 0.53 | 51.09 ± 0.15 | 36.85 ± 0.54 |
| knn | 39.89 ± 0.33 | 38.97 ± 0.04 | 35.46 ± 0.26 | 44.93 ± 0.75 | 34.26 ± 1.60 |

### 10 % labels

| readout | A-P3 | G2 | U2 | recipe VCS | SimCLR |
|---|---|---|---|---|---|
| raw_unstd | 35.42 ± 0.17 | 36.14 ± 0.15 | 33.64 ± 0.18 | 39.96 ± 0.49 | 27.45 ± 0.27 |
| raw_std | 33.18 ± 0.30 | 33.86 ± 0.03 | 31.92 ± 0.35 | 37.76 ± 0.39 | 26.16 ± 0.70 |
| l2_std | 33.41 ± 0.34 | 33.89 ± 0.32 | 31.65 ± 0.38 | 37.70 ± 0.37 | 25.94 ± 0.47 |
| lognorm_std | 2.08 ± 0.19 | 2.06 ± 0.06 | 1.92 ± 0.21 | 2.05 ± 0.08 | 2.03 ± 0.14 |
| l2_lognorm_std | 33.31 ± 0.45 | 33.77 ± 0.27 | 31.89 ± 0.45 | 37.71 ± 0.35 | 26.17 ± 0.47 |
| knn | 30.99 ± 0.20 | 30.23 ± 0.40 | 26.98 ± 0.25 | 35.23 ± 0.18 | 24.93 ± 1.99 |

### paired differences by encoder seed (100 % labels)

| readout | contrast | mean [95 % t CI] |
|---|---|---|
| raw_unstd | A-P3 - SimCLR | +8.67 [+7.85, +9.49] |
| raw_unstd | G2 - SimCLR | +9.52 [+8.11, +10.93] |
| raw_unstd | U2 - SimCLR | +7.60 [+6.52, +8.68] |
| raw_unstd | recipe VCS - SimCLR | +14.53 [+13.72, +15.34] |
| raw_unstd | A-P3 - recipe VCS | -5.87 [-5.99, -5.74] |
| raw_unstd | G2 - recipe VCS | -5.01 [-6.07, -3.96] |
| raw_unstd | U2 - recipe VCS | -6.93 [-7.23, -6.63] |
| raw_unstd | SimCLR - recipe VCS | -14.53 [-15.34, -13.72] |
| raw_std | A-P3 - SimCLR | +8.79 [+8.14, +9.44] |
| raw_std | G2 - SimCLR | +9.51 [+8.40, +10.62] |
| raw_std | U2 - SimCLR | +7.57 [+6.87, +8.28] |
| raw_std | recipe VCS - SimCLR | +14.47 [+13.71, +15.23] |
| raw_std | A-P3 - recipe VCS | -5.69 [-6.46, -4.92] |
| raw_std | G2 - recipe VCS | -4.96 [-6.40, -3.52] |
| raw_std | U2 - recipe VCS | -6.90 [-8.28, -5.52] |
| raw_std | SimCLR - recipe VCS | -14.47 [-15.23, -13.71] |
| l2_std | A-P3 - SimCLR | +8.57 [+6.71, +10.42] |
| l2_std | G2 - SimCLR | +9.24 [+8.25, +10.23] |
| l2_std | U2 - SimCLR | +7.53 [+6.26, +8.79] |
| l2_std | recipe VCS - SimCLR | +14.47 [+12.93, +16.01] |
| l2_std | A-P3 - recipe VCS | -5.90 [-6.29, -5.51] |
| l2_std | G2 - recipe VCS | -5.23 [-6.34, -4.11] |
| l2_std | U2 - recipe VCS | -6.94 [-7.74, -6.14] |
| l2_std | SimCLR - recipe VCS | -14.47 [-16.01, -12.93] |
| lognorm_std | A-P3 - SimCLR | -0.02 [-1.19, +1.15] |
| lognorm_std | G2 - SimCLR | -0.01 [-0.85, +0.84] |
| lognorm_std | U2 - SimCLR | -0.14 [-0.84, +0.56] |
| lognorm_std | recipe VCS - SimCLR | -0.20 [-0.60, +0.20] |
| lognorm_std | A-P3 - recipe VCS | +0.18 [-0.64, +1.00] |
| lognorm_std | G2 - recipe VCS | +0.19 [-0.30, +0.68] |
| lognorm_std | U2 - recipe VCS | +0.06 [-0.24, +0.36] |
| lognorm_std | SimCLR - recipe VCS | +0.20 [-0.20, +0.60] |
| l2_lognorm_std | A-P3 - SimCLR | +8.55 [+6.96, +10.14] |
| l2_lognorm_std | G2 - SimCLR | +9.03 [+8.37, +9.69] |
| l2_lognorm_std | U2 - SimCLR | +7.38 [+6.43, +8.33] |
| l2_lognorm_std | recipe VCS - SimCLR | +14.24 [+13.12, +15.36] |
| l2_lognorm_std | A-P3 - recipe VCS | -5.69 [-6.25, -5.13] |
| l2_lognorm_std | G2 - recipe VCS | -5.21 [-6.61, -3.82] |
| l2_lognorm_std | U2 - recipe VCS | -6.86 [-7.82, -5.90] |
| l2_lognorm_std | SimCLR - recipe VCS | -14.24 [-15.36, -13.12] |
| knn | A-P3 - SimCLR | +5.63 [+2.34, +8.93] |
| knn | G2 - SimCLR | +4.71 [+0.84, +8.59] |
| knn | U2 - SimCLR | +1.20 [-2.11, +4.51] |
| knn | recipe VCS - SimCLR | +10.67 [+4.88, +16.47] |
| knn | A-P3 - recipe VCS | -5.04 [-7.55, -2.53] |
| knn | G2 - recipe VCS | -5.96 [-7.90, -4.02] |
| knn | U2 - recipe VCS | -9.47 [-11.98, -6.96] |
| knn | SimCLR - recipe VCS | -10.67 [-16.47, -4.88] |

### chosen probe lr (grid 0.03 / 0.1 / 0.3; an edge choice means the grid bound binds)

| family | raw_unstd | raw_std | l2_std | lognorm_std | l2_lognorm_std |
|---|---|---|---|---|---|
| A-P3 | lr0.3×12 | lr0.03×7, lr0.1×4, lr0.3×1 | lr0.03×11, lr0.1×1 | lr0.03×5, lr0.1×1, lr0.3×6 | lr0.03×10, lr0.1×1, lr0.3×1 |
| G2 | lr0.3×12 | lr0.03×10, lr0.1×2 | lr0.03×10, lr0.1×2 | lr0.03×4, lr0.1×4, lr0.3×4 | lr0.03×9, lr0.1×2, lr0.3×1 |
| U2 | lr0.3×12 | lr0.03×12 | lr0.03×9, lr0.1×3 | lr0.03×1, lr0.1×3, lr0.3×8 | lr0.03×10, lr0.1×2 |
| recipe VCS | lr0.1×6, lr0.3×6 | lr0.03×8, lr0.1×2, lr0.3×2 | lr0.03×10, lr0.1×1, lr0.3×1 | lr0.03×5, lr0.1×5, lr0.3×2 | lr0.03×11, lr0.3×1 |
| SimCLR | lr0.3×12 | lr0.03×11, lr0.3×1 | lr0.03×9, lr0.1×2, lr0.3×1 | lr0.03×6, lr0.1×3, lr0.3×3 | lr0.03×9, lr0.1×2, lr0.3×1 |

