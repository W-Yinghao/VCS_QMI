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
| colour | 0.8 | P104_G2_views4_800ep | 81.33 (60.00) | 81.33 (60.00) | 81.33 (60.00) | 82.67 (61.82) | 83.17 (61.68) | 84.67 |
| colour | 0.8 | P35_vcs_a5_views4_800ep | 74.33 (47.17) | 71.33 (45.45) | 74.33 (47.17) | 75.17 (58.49) | 73.61 (50.37) | 75.17 |
| colour | 0.95 | P104_G2_views4_800ep | 81.33 (60.00) | 83.67 (61.82) | 83.67 (61.82) | 84.67 (62.30) | 83.22 (61.37) | 84.67 |
| colour | 0.95 | P35_vcs_a5_views4_800ep | 74.17 (47.17) | 74.17 (47.17) | 74.50 (56.60) | 74.50 (56.60) | 73.44 (49.74) | 75.33 |
| tag | 0.8 | P104_G2_views4_800ep | 84.67 (65.45) | 83.50 (67.27) | 83.50 (67.27) | 84.67 (65.45) | 83.78 (66.10) | 86.17 |
| tag | 0.8 | P35_vcs_a5_views4_800ep | 74.83 (52.73) | 77.83 (58.49) | 77.83 (58.49) | 74.83 (52.73) | 76.33 (55.61) | 77.83 |
| tag | 0.95 | P104_G2_views4_800ep | 84.67 (65.45) | 84.67 (65.45) | 83.50 (67.27) | 83.17 (65.57) | 83.78 (66.10) | 86.17 |
| tag | 0.95 | P35_vcs_a5_views4_800ep | 75.17 (52.73) | 77.00 (54.55) | 75.17 (52.73) | 77.00 (54.55) | 76.67 (55.25) | 77.83 |

**Pre-stated reading (pooled over encoders, cues, ρ; 95 % bootstrap CI resampling encoder runs):** Δ_k = test acc(B_k) − test acc(A).

| rule | mean Δ vs A (pts) | 95 % CI |
|---|---|---|
| B_vcs_closed_J | +0.38 | [+0.29, +0.46] |
| B_js_exact | +0.42 | [+0.00, +0.83] |
| B_hsic_class | +0.77 | [+0.75, +0.79] |
| C_random_eligible | +0.44 | [+0.39, +0.49] |
