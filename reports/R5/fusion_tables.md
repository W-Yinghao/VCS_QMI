# r5 R5-10 — R4-VL2 with symmetric CAL fusion (F1); exploratory

F0 (frozen R4-VL2 readings): complete.  F2: not_identifiable_from_saved_states.  CAL evaluations added: 180.  an interval containing 0 only contains 0; no equivalence margin was pre-specified.

## refcocog

| model | seed | λ chosen on CAL (label) | DEV λ = 0 | DEV λ = 1 | DEV at CAL λ | reproduction diff |
|---|---|---|---|---|---|---|
| VCS | 0 | 2 (test_time_fusion) | 71.80 | 74.10 | 73.62 | +0.00 |
| VCS | 1 | 1 (test_time_fusion) | 71.56 | 74.02 | 74.02 | +0.06 |
| VCS | 2 | 1 (test_time_fusion) | 71.89 | 73.97 | 73.97 | -0.01 |
| JS | 0 | 1 (test_time_fusion) | 71.13 | 73.55 | 73.55 | +0.08 |
| JS | 1 | 1 (test_time_fusion) | 71.65 | 73.82 | 73.82 | +0.11 |
| JS | 2 | 1 (test_time_fusion) | 71.24 | 73.42 | 73.42 | -0.24 |
| set softmax | 0 | 1 (test_time_fusion) | 73.46 | 74.45 | 74.45 | +0.03 |
| set softmax | 1 | 1 (test_time_fusion) | 73.36 | 74.12 | 74.12 | +0.04 |
| set softmax | 2 | 0.5 (test_time_fusion) | 74.01 | 74.39 | 74.37 | +0.02 |
| VCS(nu) | 0 | 2 (posthoc_task_calibration) | 67.24 | 73.98 | 73.86 | +0.11 |
| VCS(nu) | 1 | 2 (posthoc_task_calibration) | 67.53 | 74.22 | 74.43 | +0.18 |
| VCS(nu) | 2 | 2 (posthoc_task_calibration) | 66.54 | 73.44 | 73.91 | -0.07 |
| JS(nu) | 0 | 2 (posthoc_task_calibration) | 65.83 | 73.87 | 74.10 | +0.00 |
| JS(nu) | 1 | 1 (native) | 66.23 | 73.58 | 73.58 | -0.13 |
| JS(nu) | 2 | 2 (posthoc_task_calibration) | 66.41 | 73.42 | 73.80 | -0.03 |

| contrast (paired by seed) | at the CAL-chosen λ | at fixed λ = 1 |
|---|---|---|
| C1_F1 VCS(nu) - set softmax | -0.25 [-1.46, +0.97] | -0.44 [-1.74, +0.87] |
| C2_F1 VCS(nu) - VCS | +0.20 [-0.40, +0.80] | -0.15 [-1.06, +0.77] |
| C4_F1 VCS(nu) - JS(nu) | +0.24 [-1.14, +1.63] | +0.26 [-0.57, +1.09] |
| JS(nu) - JS | +0.23 [-0.81, +1.27] | +0.03 [-0.68, +0.74] |
| JS(nu) - set softmax | -0.49 [-0.79, -0.19] | -0.70 [-1.29, -0.11] |
| VCS - set softmax (both fused on CAL) | -0.45 [-1.37, +0.47] | -0.29 [-0.70, +0.12] |

## refcocoplus

| model | seed | λ chosen on CAL (label) | DEV λ = 0 | DEV λ = 1 | DEV at CAL λ | reproduction diff |
|---|---|---|---|---|---|---|
| VCS | 0 | 0.5 (test_time_fusion) | 72.96 | 74.66 | 74.68 | +0.07 |
| VCS | 1 | 1 (test_time_fusion) | 73.46 | 75.67 | 75.67 | +0.05 |
| VCS | 2 | 1 (test_time_fusion) | 72.84 | 74.94 | 74.94 | +0.06 |
| JS | 0 | 1 (test_time_fusion) | 73.00 | 74.51 | 74.51 | +0.07 |
| JS | 1 | 1 (test_time_fusion) | 72.97 | 74.78 | 74.78 | -0.00 |
| JS | 2 | 1 (test_time_fusion) | 72.66 | 75.47 | 75.47 | -0.01 |
| set softmax | 0 | 1 (test_time_fusion) | 76.14 | 76.83 | 76.83 | +0.04 |
| set softmax | 1 | 1 (test_time_fusion) | 75.98 | 76.53 | 76.53 | +0.08 |
| set softmax | 2 | 1 (test_time_fusion) | 76.81 | 77.21 | 77.21 | +0.06 |
| VCS(nu) | 0 | 2 (posthoc_task_calibration) | 68.96 | 75.46 | 76.13 | +0.08 |
| VCS(nu) | 1 | 2 (posthoc_task_calibration) | 66.70 | 75.48 | 75.58 | -0.01 |
| VCS(nu) | 2 | 1 (native) | 68.12 | 75.64 | 75.64 | +0.03 |
| JS(nu) | 0 | 1 (native) | 68.71 | 75.57 | 75.57 | +0.07 |
| JS(nu) | 1 | 1 (native) | 66.76 | 75.62 | 75.62 | +0.11 |
| JS(nu) | 2 | 2 (posthoc_task_calibration) | 67.19 | 75.68 | 76.04 | -0.03 |

| contrast (paired by seed) | at the CAL-chosen λ | at fixed λ = 1 |
|---|---|---|
| C1_F1 VCS(nu) - set softmax | -1.07 [-2.19, +0.04] | -1.33 [-1.99, -0.67] |
| C2_F1 VCS(nu) - VCS | +0.69 [-1.22, +2.59] | +0.44 [-0.91, +1.78] |
| C4_F1 VCS(nu) - JS(nu) | +0.04 [-1.16, +1.24] | -0.10 [-0.22, +0.03] |
| JS(nu) - JS | +0.82 [+0.21, +1.44] | +0.70 [-0.40, +1.80] |
| JS(nu) - set softmax | -1.11 [-1.57, -0.66] | -1.23 [-2.01, -0.46] |
| VCS - set softmax (both fused on CAL) | -1.76 [-3.70, +0.18] | -1.77 [-3.72, +0.19] |

