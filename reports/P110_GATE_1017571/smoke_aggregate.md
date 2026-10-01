# P110 — package v4 V1 / V2 / V3 results (selection split; official test closed)

## V1 — label efficiency (CIFAR-10) and frozen transfer (CIFAR-100): selection-split top-1, FIT-selected probe (recipe-fixed probe)
mean ± sd over encoder seeds × subset draws

| dataset | labels | P104_G2_views4_800ep | P35_vcs_a5_views4_800ep |
|---|---|---|---|
| cifar10 | 1 % | 56.07 ± 0.70 (55.93 ± 0.31) | 41.07 ± 2.73 (40.53 ± 2.91) |
| cifar10 | 10 % | 81.40 ± 0.92 (80.87 ± 0.64) | 73.47 ± 2.08 (73.13 ± 2.39) |
| cifar10 | 100 % | 86.40 (85.80) | 81.20 (81.20) |
| cifar100 | 10 % | 10.53 ± 0.81 (9.93 ± 0.42) | 11.33 ± 1.50 (12.27 ± 0.12) |
| cifar100 | 100 % | 25.40 (25.40) | 31.80 (31.80) |

## V2 — development corruptions (6 families × 5 severities, selection images; recipe probe on clean FIT)

| encoder family | clean | mCA | mean relative drop % | mean consistency % | brightness | contrast | gaussian_blur | gaussian_noise | jpeg | pixelate |
|---|---|---|---|---|---|---|---|---|---|---|
| P104_G2_views4_800ep | 85.80 | 61.14 | 28.74 | 66.31 | 82.52 | 82.88 | 56.84 | 32.96 | 66.64 | 45.00 |
| P35_vcs_a5_views4_800ep | 81.20 | 56.24 | 30.74 | 62.79 | 79.36 | 78.44 | 48.88 | 27.88 | 59.48 | 43.40 |

## V3 — audit-assisted selection: reversal-test accuracy (worst-group) of the selected member

| cue | ρ | encoder family | A | B_vcs_closed_J | B_js_exact | B_hsic_class | C_random_eligible | pool best |
|---|---|---|---|---|---|---|---|---|
| colour | 0.8 | P104_G2_views4_800ep | 81.33 (60.00) | 84.67 (62.30) | 84.67 (62.30) | 82.67 (61.82) | 83.17 (61.68) | 84.67 |
| colour | 0.8 | P35_vcs_a5_views4_800ep | 74.33 (47.17) | 74.33 (47.17) | 74.33 (47.17) | 75.00 (58.18) | 73.56 (50.27) | 75.00 |
| colour | 0.95 | P104_G2_views4_800ep | 81.33 (60.00) | 84.67 (62.30) | 83.67 (61.82) | 84.67 (62.30) | 83.22 (61.37) | 84.67 |
| colour | 0.95 | P35_vcs_a5_views4_800ep | 74.17 (47.17) | 74.17 (47.17) | 74.50 (56.60) | 74.50 (56.60) | 73.44 (49.74) | 75.33 |
| tag | 0.8 | P104_G2_views4_800ep | 84.50 (61.82) | 83.17 (67.27) | 83.17 (67.27) | 84.50 (61.82) | 83.72 (62.70) | 85.50 |
| tag | 0.8 | P35_vcs_a5_views4_800ep | 75.17 (54.10) | 76.33 (52.83) | 76.33 (52.83) | 75.17 (54.10) | 76.00 (52.00) | 77.17 |
| tag | 0.95 | P104_G2_views4_800ep | 84.50 (61.82) | 83.67 (60.66) | 83.17 (67.27) | 83.67 (60.66) | 84.21 (65.01) | 85.50 |
| tag | 0.95 | P35_vcs_a5_views4_800ep | 75.17 (54.10) | 75.17 (54.10) | 76.17 (52.83) | 75.17 (54.10) | 75.94 (52.00) | 76.67 |

**Pre-stated reading (pooled over encoders, cues, ρ; 95 % bootstrap CI resampling encoder runs):** Δ_k = test acc(B_k) − test acc(A).

| rule | mean Δ vs A (pts) | 95 % CI |
|---|---|---|
| B_vcs_closed_J | +0.71 | [+0.29, +1.13] |
| B_js_exact | +0.69 | [+0.62, +0.75] |
| B_hsic_class | +0.60 | [+0.25, +0.96] |
| C_random_eligible | +0.35 | [+0.03, +0.66] |
