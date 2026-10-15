# V9-G1 — BF/SHG rigid registration: training-sample fraction × training mismatch

Missing units: none.

| cell | seeds | n FIT | success < 42 px (%) | success < 9 px (%) | median corner error (px) | failures (energy) | s / registration |
|---|---|---|---|---|---|---|---|
| vcs|frac0.25|eps0 | [901, 902, 903] | [8, 8, 8] | 93.6 ± 0.8 | 60.6 ± 4.4 | 6.8 | 61 (53) | 1.32 |
| vcs|frac0.25|eps0.2 | [901, 902, 903] | [8, 8, 8] | 92.1 ± 2.4 | 61.8 ± 1.5 | 6.3 | 76 (53) | 0.77 |
| vcs|frac1|eps0 | [901, 902, 903] | [32, 32, 32] | 95.9 ± 3.3 | 65.9 ± 3.3 | 5.7 | 39 (29) | 0.75 |
| vcs|frac1|eps0.2 | [901, 902, 903] | [32, 32, 32] | 96.6 ± 2.0 | 66.9 ± 0.8 | 5.5 | 33 (23) | 0.76 |
| logistic|frac0.25|eps0 | [901, 902, 903] | [8, 8, 8] | 94.1 ± 2.0 | 62.4 ± 3.3 | 6.5 | 57 (53) | 1.62 |
| logistic|frac0.25|eps0.2 | [901, 902, 903] | [8, 8, 8] | 91.8 ± 2.2 | 60.4 ± 3.0 | 6.0 | 79 (56) | 0.77 |
| logistic|frac1|eps0 | [901, 902, 903] | [32, 32, 32] | 93.8 ± 0.3 | 64.2 ± 0.8 | 5.8 | 60 (32) | 0.75 |
| logistic|frac1|eps0.2 | [901, 902, 903] | [32, 32, 32] | 96.7 ± 0.7 | 65.8 ± 2.0 | 6.0 | 32 (29) | 0.76 |
| infonce|frac0.25|eps0 | [901, 902, 903] | [8, 8, 8] | 92.7 ± 2.6 | 58.2 ± 7.7 | 7.6 | 70 (63) | 1.62 |
| infonce|frac0.25|eps0.2 | [901, 902, 903] | [8, 8, 8] | 93.5 ± 3.0 | 60.2 ± 4.9 | 6.9 | 62 (62) | 0.78 |
| infonce|frac1|eps0 | [901, 902, 903] | [32, 32, 32] | 95.5 ± 1.8 | 64.3 ± 0.4 | 5.8 | 43 (42) | 0.75 |
| infonce|frac1|eps0.2 | [901, 902, 903] | [32, 32, 32] | 93.5 ± 1.3 | 61.6 ± 3.8 | 6.6 | 62 (57) | 0.76 |
| comir|frac0.25|eps0 | [901, 902, 903] | [0, 0, 0] | 74.3 ± 14.7 | 47.6 ± 10.0 | 10.2 | 247 (232) | 0.33 |
| comir|frac0.25|eps0.2 | [901, 902, 903] | [0, 0, 0] | 64.1 ± 25.3 | 43.4 ± 18.9 | 81.1 | 345 (339) | 0.33 |
| comir|frac1|eps0 | [901, 902, 903] | [0, 0, 0] | 84.7 ± 3.5 | 58.5 ± 3.5 | 6.9 | 147 (144) | 0.50 |
| comir|frac1|eps0.2 | [901, 902, 903] | [0, 0, 0] | 61.4 ± 23.4 | 41.2 ± 16.7 | 81.7 | 371 (357) | 0.50 |
| kernel|frac0.25|eps0 | [901] | [8] | 48.4 | 25.3 | 126.1 | 165 (161) | 4.71 |
| kernel|frac0.25|eps0.2 | [901] | [8] | 47.5 | 23.1 | 140.5 | 168 (160) | 4.67 |
| kernel|frac1|eps0 | [901] | [32] | 37.5 | 21.9 | 149.6 | 200 (187) | 4.93 |
| kernel|frac1|eps0.2 | [901] | [32] | 40.0 | 23.4 | 119.4 | 192 (181) | 4.96 |
| mi | — | — | 74.1 | 49.4 | 9.3 | 83 (69) | — |
| nmi | — | — | 75.9 | 50.6 | 8.9 | 77 (67) | — |

## Primary family (Holm over 4; detected = seed t and unit bootstrap exclude 0 and Holm p < 0.05)

| contrast | value | boot p | Holm p | detected | complete |
|---|---|---|---|---|---|
| vcs_minus_mi|frac0.25|eps0.2@42 | +18.0 [+12.1, +23.9]; unit bootstrap [+6.2, +32.3] | 0.0020 | 0.0020 | True | True |
| vcs_minus_comir|frac0.25|eps0.2@42 | +28.0 [-33.7, +89.7]; unit bootstrap [+20.1, +35.6] | 0.0000 | 0.0000 | False | True |
| vcs_minus_mi|frac1|eps0.2@42 | +22.5 [+17.7, +27.3]; unit bootstrap [+11.2, +35.0] | 0.0000 | 0.0000 | True | True |
| vcs_minus_comir|frac1|eps0.2@42 | +35.2 [-25.3, +95.7]; unit bootstrap [+27.0, +43.9] | 0.0000 | 0.0000 | False | True |

## All contrasts

| contrast | value |
|---|---|
| vcs_minus_mi|frac0.25|eps0@42 | +19.6 [+17.6, +21.5]; unit bootstrap [+6.7, +33.0] |
| vcs_minus_nmi|frac0.25|eps0@42 | +17.7 [+15.8, +19.7]; unit bootstrap [+5.6, +32.2] |
| vcs_minus_kernel|frac0.25|eps0@42 | +45.2 [+43.3, +47.2]; unit bootstrap [+29.5, +60.7] |
| vcs_minus_logistic|frac0.25|eps0@42 | -0.4 [-5.9, +5.0]; unit bootstrap [-3.4, +1.9] |
| vcs_minus_infonce|frac0.25|eps0@42 | +0.9 [-3.8, +5.7]; unit bootstrap [-2.4, +4.7] |
| vcs_minus_comir|frac0.25|eps0@42 | +19.4 [-15.5, +54.3]; unit bootstrap [+10.8, +28.2] |
| vcs_minus_mi|frac0.25|eps0.2@42 | +18.0 [+12.1, +23.9]; unit bootstrap [+6.2, +32.3] |
| vcs_minus_nmi|frac0.25|eps0.2@42 | +16.1 [+10.3, +22.0]; unit bootstrap [+4.3, +30.1] |
| vcs_minus_kernel|frac0.25|eps0.2@42 | +44.6 [+38.7, +50.5]; unit bootstrap [+28.7, +60.4] |
| vcs_minus_logistic|frac0.25|eps0.2@42 | +0.3 [-2.5, +3.1]; unit bootstrap [-3.0, +3.3] |
| vcs_minus_infonce|frac0.25|eps0.2@42 | -1.5 [-8.6, +5.7]; unit bootstrap [-5.6, +2.1] |
| vcs_minus_comir|frac0.25|eps0.2@42 | +28.0 [-33.7, +89.7]; unit bootstrap [+20.1, +35.6] |
| vcs_minus_mi|frac1|eps0@42 | +21.9 [+13.8, +30.0]; unit bootstrap [+10.5, +34.2] |
| vcs_minus_nmi|frac1|eps0@42 | +20.0 [+11.9, +28.1]; unit bootstrap [+9.1, +32.3] |
| vcs_minus_kernel|frac1|eps0@42 | +58.4 [+50.3, +66.5]; unit bootstrap [+44.1, +72.3] |
| vcs_minus_logistic|frac1|eps0@42 | +2.2 [-5.9, +10.3]; unit bootstrap [+0.0, +5.2] |
| vcs_minus_infonce|frac1|eps0@42 | +0.4 [-5.4, +6.2]; unit bootstrap [-0.9, +2.2] |
| vcs_minus_comir|frac1|eps0@42 | +11.2 [+9.9, +12.6]; unit bootstrap [+4.8, +18.8] |
| vcs_minus_mi|frac1|eps0.2@42 | +22.5 [+17.7, +27.3]; unit bootstrap [+11.2, +35.0] |
| vcs_minus_nmi|frac1|eps0.2@42 | +20.6 [+15.8, +25.5]; unit bootstrap [+9.2, +33.0] |
| vcs_minus_kernel|frac1|eps0.2@42 | +56.6 [+51.7, +61.4]; unit bootstrap [+41.2, +70.8] |
| vcs_minus_logistic|frac1|eps0.2@42 | -0.1 [-6.4, +6.2]; unit bootstrap [-2.2, +2.2] |
| vcs_minus_infonce|frac1|eps0.2@42 | +3.0 [-3.0, +9.1]; unit bootstrap [+0.0, +6.6] |
| vcs_minus_comir|frac1|eps0.2@42 | +35.2 [-25.3, +95.7]; unit bootstrap [+27.0, +43.9] |
| vcs_minus_mi|frac0.25|eps0@9 | +11.2 [+0.3, +22.2]; unit bootstrap [-0.5, +24.4] |
| vcs_minus_nmi|frac0.25|eps0@9 | +10.0 [-1.0, +21.0]; unit bootstrap [-0.7, +22.4] |
| vcs_minus_kernel|frac0.25|eps0@9 | +35.3 [+24.4, +46.3]; unit bootstrap [+16.9, +52.6] |
| vcs_minus_logistic|frac0.25|eps0@9 | -1.8 [-6.0, +2.5]; unit bootstrap [-5.0, +1.6] |
| vcs_minus_infonce|frac0.25|eps0@9 | +2.4 [-13.5, +18.3]; unit bootstrap [-3.4, +7.9] |
| vcs_minus_comir|frac0.25|eps0@9 | +13.0 [-8.8, +34.8]; unit bootstrap [+6.5, +21.1] |
| vcs_minus_mi|frac0.25|eps0.2@9 | +12.4 [+8.6, +16.2]; unit bootstrap [+0.7, +25.3] |
| vcs_minus_nmi|frac0.25|eps0.2@9 | +11.1 [+7.3, +15.0]; unit bootstrap [-0.2, +23.3] |
| vcs_minus_kernel|frac0.25|eps0.2@9 | +38.6 [+34.8, +42.5]; unit bootstrap [+19.7, +57.7] |
| vcs_minus_logistic|frac0.25|eps0.2@9 | +1.4 [-3.6, +6.3]; unit bootstrap [+0.0, +3.2] |
| vcs_minus_infonce|frac0.25|eps0.2@9 | +1.6 [-7.4, +10.5]; unit bootstrap [-1.6, +5.1] |
| vcs_minus_comir|frac0.25|eps0.2@9 | +18.3 [-29.4, +66.0]; unit bootstrap [+10.3, +27.1] |
| vcs_minus_mi|frac1|eps0@9 | +16.6 [+8.4, +24.7]; unit bootstrap [+3.3, +30.6] |
| vcs_minus_nmi|frac1|eps0@9 | +15.3 [+7.1, +23.5]; unit bootstrap [+3.6, +27.4] |
| vcs_minus_kernel|frac1|eps0@9 | +44.1 [+35.9, +52.2]; unit bootstrap [+26.6, +60.9] |
| vcs_minus_logistic|frac1|eps0@9 | +1.8 [-7.5, +11.0]; unit bootstrap [-0.9, +4.8] |
| vcs_minus_infonce|frac1|eps0@9 | +1.7 [-7.3, +10.6]; unit bootstrap [-1.6, +5.1] |
| vcs_minus_comir|frac1|eps0@9 | +7.4 [-3.9, +18.7]; unit bootstrap [+1.9, +13.9] |
| vcs_minus_mi|frac1|eps0.2@9 | +17.5 [+15.4, +19.6]; unit bootstrap [+4.4, +32.1] |
| vcs_minus_nmi|frac1|eps0.2@9 | +16.2 [+14.2, +18.3]; unit bootstrap [+4.3, +29.3] |
| vcs_minus_kernel|frac1|eps0.2@9 | +43.4 [+41.4, +45.5]; unit bootstrap [+24.6, +61.9] |
| vcs_minus_logistic|frac1|eps0.2@9 | +1.0 [-4.0, +6.1]; unit bootstrap [-3.3, +5.3] |
| vcs_minus_infonce|frac1|eps0.2@9 | +5.3 [-6.3, +16.9]; unit bootstrap [+1.4, +9.8] |
| vcs_minus_comir|frac1|eps0.2@9 | +25.6 [-14.5, +65.8]; unit bootstrap [+16.8, +35.1] |

## Interaction: change of success < 42 px from ε 0 to 0.2 (seed-paired)

| method, fraction | Δ (pp) |
|---|---|
| vcs|frac0.25|delta_0_to_0.2@42 | -1.6 [-9.0, +5.8] |
| vcs|frac1|delta_0_to_0.2@42 | +0.6 [-11.6, +12.8] |
| logistic|frac0.25|delta_0_to_0.2@42 | -2.3 [-8.0, +3.4] |
| logistic|frac1|delta_0_to_0.2@42 | +2.9 [+2.0, +3.8] |
| infonce|frac0.25|delta_0_to_0.2@42 | +0.8 [-5.2, +6.9] |
| infonce|frac1|delta_0_to_0.2@42 | -2.0 [-7.4, +3.5] |
| comir|frac0.25|delta_0_to_0.2@42 | -10.2 [-44.4, +23.9] |
| comir|frac1|delta_0_to_0.2@42 | -23.3 [-73.1, +26.4] |
| kernel|frac0.25|delta_0_to_0.2@42 | -0.9 |
| kernel|frac1|delta_0_to_0.2@42 | +2.5 |
