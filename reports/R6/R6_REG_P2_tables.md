# R6 REG-P2 — BF/SHG rigid registration (θ* ∈ ±30°, REG-P1 translations): DEV results

Missing: none.

| method | seeds | success < 42 px (%) | success < 9 px (%) | median corner error (px) | p90 (px) | median rotation error (°) | median translation error (px) | failures (energy) | s / registration | REG-P1 → P2 success < 42 (pp) |
|---|---|---|---|---|---|---|---|---|---|---|
| vcs | [0, 1, 2] | 94.2 ± 2.0 | 68.2 ± 0.7 | 5.7 | 18.5 | 0.06 | 5.6 | 56 (39) | 1.54 | -5.8 [-10.9, -0.8] |
| logistic | [0, 1, 2] | 93.1 ± 1.4 | 64.9 ± 1.5 | 5.8 | 20.0 | 0.07 | 5.7 | 66 (39) | 1.04 | -5.7 [-6.9, -4.5] |
| infonce | [0, 1, 2] | 94.9 ± 1.7 | 59.6 ± 2.5 | 6.9 | 18.5 | 0.16 | 6.3 | 49 (49) | 1.26 | -0.3 [-9.0, +8.3] |
| comir | [0, 1, 2] | 90.0 ± 1.1 | 62.0 ± 1.9 | 5.8 | 93.6 | 0.10 | 5.6 | 96 (94) | 0.27 | -0.9 [-3.0, +1.1] |
| comir_wte | [0, 1, 2] | 86.4 ± 4.6 | 60.2 ± 3.7 | 6.2 | 191.1 | 0.03 | 6.0 | 131 (102) | 10.99 | — |
| mi | [None] | 74.1 | 49.4 | 9.3 | 269.2 | 0.19 | 7.8 | 83 (69) | 0.83 | -7.8 |
| nmi | [None] | 75.9 | 50.6 | 8.9 | 261.6 | 0.19 | 8.2 | 77 (67) | 0.83 | -4.4 |
| kernel | [None] | 42.2 | 24.4 | 117.8 | 384.0 | 3.23 | 72.7 | 185 (174) | 4.70 | -15.0 |

comir = CoMIR encode-once (primary row); comir_wte = CoMIR warp-then-encode (sensitivity row, DEV only).

| contrast (image-macro success) | value |
|---|---|
| vcs_minus_mi@42 | +20.1 [+15.1, +25.2]; image bootstrap [+8.8, +32.9] |
| vcs_minus_nmi@42 | +18.2 [+13.2, +23.3]; image bootstrap [+7.7, +30.3] |
| vcs_minus_kernel@42 | +52.0 [+46.9, +57.0]; image bootstrap [+37.7, +65.8] |
| vcs_minus_logistic@42 | +1.0 [-6.9, +9.0]; image bootstrap [+0.0, +2.9] |
| vcs_minus_infonce@42 | -0.7 [-9.8, +8.3]; image bootstrap [-4.2, +3.0] |
| vcs_minus_comir@42 | +4.2 [-3.6, +11.9]; image bootstrap [+0.3, +9.3] |
| vcs_minus_comir_wte@42 | +7.8 [-6.2, +21.8]; image bootstrap [+3.1, +14.0] |
| comir_wte_minus_comir@42 | -3.6 [-13.5, +6.2]; image bootstrap [-6.2, -1.4] |
| vcs_minus_mi@9 | +18.9 [+17.2, +20.5]; image bootstrap [+6.9, +31.6] |
| vcs_minus_nmi@9 | +17.6 [+16.0, +19.2]; image bootstrap [+6.5, +30.2] |
| vcs_minus_kernel@9 | +43.9 [+42.2, +45.5]; image bootstrap [+25.7, +62.3] |
| vcs_minus_logistic@9 | +3.3 [-2.0, +8.6]; image bootstrap [+1.0, +6.0] |
| vcs_minus_infonce@9 | +8.6 [+2.6, +14.7]; image bootstrap [+2.8, +15.2] |
| vcs_minus_comir@9 | +6.2 [+2.9, +9.6]; image bootstrap [+2.7, +10.4] |
| vcs_minus_comir_wte@9 | +8.0 [-2.9, +18.9]; image bootstrap [+3.4, +13.1] |
| comir_wte_minus_comir@9 | -1.8 [-15.4, +11.9]; image bootstrap [-5.5, +1.9] |
