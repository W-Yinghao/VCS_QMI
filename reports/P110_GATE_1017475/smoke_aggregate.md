# P110 — package v4 V1 / V2 / V3 results (selection split; official test closed)

## V1 — label efficiency (CIFAR-10) and frozen transfer (CIFAR-100): selection-split top-1, FIT-selected probe (recipe-fixed probe)
mean ± sd over encoder seeds × subset draws

| dataset | labels | P104_G2_views4_800ep | P35_vcs_a5_views4_800ep |
|---|---|---|---|
| cifar10 | 1 % | 56.07 ± 0.70 (55.93 ± 0.31) | 41.07 ± 2.73 (40.60 ± 2.88) |
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
| colour | 0.9 | P104_G2_views4_800ep | 84.00 (81.41) | 82.17 (79.55) | 83.67 (81.41) | 83.67 (81.41) | 84.17 (82.11) | 85.50 |
| colour | 0.9 | P35_vcs_a5_views4_800ep | 76.00 (74.72) | 74.17 (64.00) | 76.00 (74.72) | 76.00 (74.72) | 75.08 (69.36) | 76.50 |
| colour | 0.97 | P104_G2_views4_800ep | 83.17 (76.47) | 85.17 (79.41) | 85.17 (77.78) | 85.17 (79.41) | 84.00 (77.43) | 85.17 |
| colour | 0.97 | P35_vcs_a5_views4_800ep | 74.17 (62.96) | 74.17 (62.96) | 74.17 (62.96) | 74.17 (62.96) | 73.53 (68.72) | 76.67 |
| tag | 0.9 | P104_G2_views4_800ep | 84.83 (72.73) | 83.17 (73.08) | 84.83 (72.73) | 83.50 (73.08) | 83.83 (72.96) | 85.33 |
| tag | 0.9 | P35_vcs_a5_views4_800ep | 75.33 (57.69) | 75.33 (57.69) | 75.33 (57.69) | 75.33 (57.69) | 75.33 (57.69) | 77.17 |
| tag | 0.97 | P104_G2_views4_800ep | 84.83 (82.86) | 85.83 (84.87) | 85.83 (84.87) | 83.33 (81.25) | 84.42 (83.05) | 85.83 |
| tag | 0.97 | P35_vcs_a5_views4_800ep | 75.67 (73.28) | 76.67 (68.57) | 75.67 (73.28) | 75.67 (73.28) | 76.17 (70.93) | 77.17 |

**Pre-stated reading (pooled over encoders, cues, ρ; 95 % bootstrap CI resampling encoder runs):** Δ_k = test acc(B_k) − test acc(A).

| rule | mean Δ vs A (pts) | 95 % CI |
|---|---|---|
| B_vcs_closed_J | -0.17 | [-0.21, -0.13] |
| B_js_exact | +0.33 | [+0.00, +0.67] |
| B_hsic_class | -0.15 | [-0.29, +0.00] |
| C_random_eligible | -0.18 | [-0.26, -0.10] |
