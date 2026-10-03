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

