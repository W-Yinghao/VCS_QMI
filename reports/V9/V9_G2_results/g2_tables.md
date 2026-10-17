# V9-G2 — EPFL RGB–NIR rigid registration: training-sample fraction × training mismatch

Missing units: none.  Ceiling readout: {'all_cal_success42_ge_0.95': False, 'min_cal_success42': 0.7, 'rule': "if every method's CAL unit success < 10.5 px >= 0.95 the < 2.25 px success is the discriminating readout (REG-P3's pre-registered readout)"}

| cell | seeds | n FIT units | success < 10.5 px (%) | success < 2.25 px (%) | median corner error (err_full px) | failures (energy) | s / registration |
|---|---|---|---|---|---|---|---|
| vcs|frac0.25|eps0 | [901, 902, 903] | [82, 82, 82] | 99.3 ± 0.0 | 87.9 ± 1.0 | 2.8 | 15 (15) | 0.78 |
| vcs|frac0.25|eps0.2 | [901, 902, 903] | [82, 82, 82] | 99.3 ± 0.2 | 87.0 ± 0.8 | 2.9 | 16 (16) | 0.76 |
| vcs|frac1|eps0 | [901, 902, 903] | [330, 330, 330] | 99.2 ± 0.4 | 85.9 ± 1.3 | 2.6 | 18 (18) | 1.48 |
| vcs|frac1|eps0.2 | [901, 902, 903] | [330, 330, 330] | 99.6 ± 0.4 | 86.6 ± 1.5 | 3.2 | 8 (8) | 1.48 |
| logistic|frac0.25|eps0 | [901, 902, 903] | [82, 82, 82] | 99.2 ± 0.4 | 86.8 ± 1.0 | 2.6 | 17 (17) | 0.75 |
| logistic|frac0.25|eps0.2 | [901, 902, 903] | [82, 82, 82] | 99.2 ± 0.2 | 86.6 ± 0.7 | 2.6 | 17 (17) | 0.75 |
| logistic|frac1|eps0 | [901, 902, 903] | [330, 330, 330] | 98.9 ± 0.3 | 86.8 ± 1.1 | 2.7 | 24 (24) | 0.74 |
| logistic|frac1|eps0.2 | [901, 902, 903] | [330, 330, 330] | 99.6 ± 0.4 | 87.7 ± 1.0 | 2.9 | 8 (8) | 0.75 |
| infonce|frac0.25|eps0 | [901, 902, 903] | [82, 82, 82] | 99.6 ± 0.2 | 90.2 ± 3.0 | 2.1 | 8 (5) | 1.99 |
| infonce|frac0.25|eps0.2 | [901, 902, 903] | [82, 82, 82] | 98.7 ± 0.2 | 88.5 ± 2.6 | 2.8 | 29 (25) | 1.85 |
| infonce|frac1|eps0 | [901, 902, 903] | [330, 330, 330] | 99.3 ± 0.4 | 88.2 ± 1.6 | 2.9 | 15 (15) | 1.96 |
| infonce|frac1|eps0.2 | [901, 902, 903] | [330, 330, 330] | 98.9 ± 0.3 | 88.1 ± 2.2 | 3.5 | 23 (23) | 1.90 |
| comir|frac0.25|eps0 | [901, 902, 903] | [82, 82, 82] | 96.4 ± 1.8 | 85.6 ± 1.0 | 2.1 | 77 (69) | 0.20 |
| comir|frac0.25|eps0.2 | [901, 902, 903] | [82, 82, 82] | 96.7 ± 0.0 | 86.1 ± 1.0 | 2.1 | 75 (67) | 0.19 |
| comir|frac1|eps0 | [901, 902, 903] | [330, 330, 330] | 96.7 ± 1.0 | 85.3 ± 1.3 | 2.2 | 72 (71) | 0.32 |
| comir|frac1|eps0.2 | [901, 902, 903] | [330, 330, 330] | 96.6 ± 0.9 | 86.3 ± 1.8 | 2.1 | 81 (75) | 0.32 |
| kernel|frac0.25|eps0 | [901] | [82] | 87.2 | 75.7 | 2.5 | 97 (94) | 4.66 |
| kernel|frac0.25|eps0.2 | [901] | [82] | 87.3 | 76.3 | 2.4 | 96 (83) | 4.66 |
| kernel|frac1|eps0 | [901] | [330] | 87.8 | 76.5 | 2.5 | 92 (86) | 2.17 |
| kernel|frac1|eps0.2 | [901] | [330] | 88.1 | 76.7 | 2.4 | 90 (79) | 2.17 |
| mi | — | — | 93.8 | 82.9 | 1.9 | 45 (22) | — |
| nmi | — | — | 94.0 | 82.9 | 1.9 | 44 (21) | — |

## Primary family (Holm over 4; detected = seed t and unit bootstrap exclude 0 and Holm p < 0.05)

| contrast | value | boot p | Holm p | detected | complete |
|---|---|---|---|---|---|
| vcs_minus_mi|frac0.25|eps0.2@42 | +5.4 [+5.0, +5.8]; unit bootstrap [+1.2, +10.4] | 0.0040 | 0.0080 | True | True |
| vcs_minus_comir|frac0.25|eps0.2@42 | +2.5 [+2.2, +2.9]; unit bootstrap [+0.5, +5.3] | 0.0100 | 0.0100 | True | True |
| vcs_minus_mi|frac1|eps0.2@42 | +5.8 [+4.8, +6.9]; unit bootstrap [+1.8, +11.0] | 0.0000 | 0.0000 | True | True |
| vcs_minus_comir|frac1|eps0.2@42 | +3.1 [+1.1, +5.0]; unit bootstrap [+0.6, +6.4] | 0.0010 | 0.0030 | True | True |

## All contrasts

| contrast | value |
|---|---|
| vcs_minus_mi|frac0.25|eps0@42 | +5.5 [+5.5, +5.5]; unit bootstrap [+1.4, +10.2] |
| vcs_minus_nmi|frac0.25|eps0@42 | +5.3 [+5.3, +5.3]; unit bootstrap [+1.2, +10.3] |
| vcs_minus_kernel|frac0.25|eps0@42 | +12.2 [+12.2, +12.2]; unit bootstrap [+5.7, +19.0] |
| vcs_minus_logistic|frac0.25|eps0@42 | +0.1 [-1.0, +1.2]; unit bootstrap [-0.0, +0.3] |
| vcs_minus_infonce|frac0.25|eps0@42 | -0.3 [-0.9, +0.2]; unit bootstrap [-1.5, +0.4] |
| vcs_minus_comir|frac0.25|eps0@42 | +2.9 [-1.7, +7.5]; unit bootstrap [+0.6, +5.8] |
| vcs_minus_mi|frac0.25|eps0.2@42 | +5.4 [+5.0, +5.8]; unit bootstrap [+1.2, +10.4] |
| vcs_minus_nmi|frac0.25|eps0.2@42 | +5.3 [+4.9, +5.7]; unit bootstrap [+0.8, +10.4] |
| vcs_minus_kernel|frac0.25|eps0.2@42 | +12.0 [+11.6, +12.4]; unit bootstrap [+5.6, +19.3] |
| vcs_minus_logistic|frac0.25|eps0.2@42 | +0.0 [-0.4, +0.4]; unit bootstrap [+0.0, +0.1] |
| vcs_minus_infonce|frac0.25|eps0.2@42 | +0.6 [+0.2, +1.0]; unit bootstrap [+0.0, +1.4] |
| vcs_minus_comir|frac0.25|eps0.2@42 | +2.5 [+2.2, +2.9]; unit bootstrap [+0.5, +5.3] |
| vcs_minus_mi|frac1|eps0@42 | +5.3 [+4.3, +6.4]; unit bootstrap [+1.2, +10.0] |
| vcs_minus_nmi|frac1|eps0@42 | +5.2 [+4.2, +6.2]; unit bootstrap [+0.8, +9.9] |
| vcs_minus_kernel|frac1|eps0@42 | +11.3 [+10.3, +12.4]; unit bootstrap [+5.0, +18.0] |
| vcs_minus_logistic|frac1|eps0@42 | +0.3 [-0.6, +1.2]; unit bootstrap [+0.0, +0.7] |
| vcs_minus_infonce|frac1|eps0@42 | -0.1 [-1.4, +1.1]; unit bootstrap [-1.0, +0.6] |
| vcs_minus_comir|frac1|eps0@42 | +2.5 [+0.7, +4.2]; unit bootstrap [+0.1, +5.4] |
| vcs_minus_mi|frac1|eps0.2@42 | +5.8 [+4.8, +6.9]; unit bootstrap [+1.8, +11.0] |
| vcs_minus_nmi|frac1|eps0.2@42 | +5.7 [+4.6, +6.7]; unit bootstrap [+1.5, +10.6] |
| vcs_minus_kernel|frac1|eps0.2@42 | +11.6 [+10.5, +12.6]; unit bootstrap [+5.1, +18.8] |
| vcs_minus_logistic|frac1|eps0.2@42 | +0.0 [+0.0, +0.0]; unit bootstrap [+0.0, +0.0] |
| vcs_minus_infonce|frac1|eps0.2@42 | +0.7 [-1.1, +2.5]; unit bootstrap [+0.0, +2.1] |
| vcs_minus_comir|frac1|eps0.2@42 | +3.1 [+1.1, +5.0]; unit bootstrap [+0.6, +6.4] |
| vcs_minus_mi|frac0.25|eps0@9 | +5.0 [+2.5, +7.4]; unit bootstrap [-1.8, +11.8] |
| vcs_minus_nmi|frac0.25|eps0@9 | +5.0 [+2.5, +7.4]; unit bootstrap [-1.6, +11.5] |
| vcs_minus_kernel|frac0.25|eps0@9 | +12.2 [+9.7, +14.7]; unit bootstrap [+6.3, +18.8] |
| vcs_minus_logistic|frac0.25|eps0@9 | +1.1 [-1.9, +4.0]; unit bootstrap [-1.1, +3.8] |
| vcs_minus_infonce|frac0.25|eps0@9 | -2.3 [-8.8, +4.1]; unit bootstrap [-7.4, +2.0] |
| vcs_minus_comir|frac0.25|eps0@9 | +2.3 [-2.3, +7.0]; unit bootstrap [-1.3, +6.2] |
| vcs_minus_mi|frac0.25|eps0.2@9 | +4.1 [+2.0, +6.2]; unit bootstrap [-1.2, +10.0] |
| vcs_minus_nmi|frac0.25|eps0.2@9 | +4.1 [+2.0, +6.2]; unit bootstrap [-1.3, +10.0] |
| vcs_minus_kernel|frac0.25|eps0.2@9 | +10.7 [+8.6, +12.8]; unit bootstrap [+4.6, +17.5] |
| vcs_minus_logistic|frac0.25|eps0.2@9 | +0.4 [-3.1, +3.9]; unit bootstrap [-2.1, +2.7] |
| vcs_minus_infonce|frac0.25|eps0.2@9 | -1.6 [-8.7, +5.6]; unit bootstrap [-4.0, +0.6] |
| vcs_minus_comir|frac0.25|eps0.2@9 | +0.9 [-0.1, +1.9]; unit bootstrap [-2.7, +4.3] |
| vcs_minus_mi|frac1|eps0@9 | +3.0 [-0.2, +6.2]; unit bootstrap [-2.5, +9.2] |
| vcs_minus_nmi|frac1|eps0@9 | +3.0 [-0.2, +6.2]; unit bootstrap [-2.5, +8.6] |
| vcs_minus_kernel|frac1|eps0@9 | +9.4 [+6.2, +12.6]; unit bootstrap [+3.7, +15.4] |
| vcs_minus_logistic|frac1|eps0@9 | -0.9 [-2.2, +0.4]; unit bootstrap [-2.2, +0.1] |
| vcs_minus_infonce|frac1|eps0@9 | -2.3 [-9.4, +4.8]; unit bootstrap [-5.0, +0.1] |
| vcs_minus_comir|frac1|eps0@9 | +0.6 [-3.8, +5.0]; unit bootstrap [-1.9, +3.4] |
| vcs_minus_mi|frac1|eps0.2@9 | +3.7 [-0.2, +7.5]; unit bootstrap [-2.6, +10.4] |
| vcs_minus_nmi|frac1|eps0.2@9 | +3.7 [-0.2, +7.5]; unit bootstrap [-2.5, +10.1] |
| vcs_minus_kernel|frac1|eps0.2@9 | +9.9 [+6.1, +13.7]; unit bootstrap [+3.6, +17.2] |
| vcs_minus_logistic|frac1|eps0.2@9 | -1.1 [-6.8, +4.6]; unit bootstrap [-3.8, +1.2] |
| vcs_minus_infonce|frac1|eps0.2@9 | -1.6 [-4.2, +1.1]; unit bootstrap [-5.5, +2.3] |
| vcs_minus_comir|frac1|eps0.2@9 | +0.3 [-0.6, +1.2]; unit bootstrap [-3.5, +3.6] |

## Interaction: change of success < 10.5 px from ε 0 to 0.2 (seed-paired)

| method, fraction | Δ (pp) |
|---|---|
| vcs|frac0.25|delta_0_to_0.2@42 | -0.0 [-0.4, +0.4] |
| vcs|frac1|delta_0_to_0.2@42 | +0.5 [-1.6, +2.5] |
| logistic|frac0.25|delta_0_to_0.2@42 | +0.0 [-0.9, +0.9] |
| logistic|frac1|delta_0_to_0.2@42 | +0.7 [-0.7, +2.2] |
| infonce|frac0.25|delta_0_to_0.2@42 | -1.0 [-1.9, -0.1] |
| infonce|frac1|delta_0_to_0.2@42 | -0.4 [-1.8, +1.0] |
| comir|frac0.25|delta_0_to_0.2@42 | +0.3 [-4.4, +4.9] |
| comir|frac1|delta_0_to_0.2@42 | -0.1 [-3.9, +3.6] |
| kernel|frac0.25|delta_0_to_0.2@42 | +0.1 |
| kernel|frac1|delta_0_to_0.2@42 | +0.2 |
