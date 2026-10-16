# V9-G1 — BF/SHG rigid registration: training-sample fraction × training mismatch

Missing units: none.

| cell | seeds | n FIT | success < 42 px (%) | success < 9 px (%) | median corner error (px) | failures (energy) | s / registration |
|---|---|---|---|---|---|---|---|
| vcs|frac0.25|eps0 | [901, 902, 903] | [8, 8, 8] | 97.2 ± 0.4 | 71.6 ± 2.7 | 5.3 | 112 (97) | 1.56 |
| vcs|frac0.25|eps0.2 | [901, 902, 903] | [8, 8, 8] | 97.4 ± 1.2 | 73.7 ± 1.0 | 5.1 | 103 (91) | 0.77 |
| vcs|frac1|eps0 | [901, 902, 903] | [32, 32, 32] | 98.3 ± 0.5 | 76.7 ± 1.7 | 4.9 | 68 (64) | 0.78 |
| vcs|frac1|eps0.2 | [901, 902, 903] | [32, 32, 32] | 98.7 ± 0.6 | 76.6 ± 1.4 | 4.8 | 54 (53) | 0.77 |
| logistic|frac0.25|eps0 | [901, 902, 903] | [8, 8, 8] | 97.5 ± 1.0 | 74.6 ± 0.8 | 5.4 | 102 (95) | 1.98 |
| logistic|frac0.25|eps0.2 | [901, 902, 903] | [8, 8, 8] | 97.4 ± 0.3 | 74.6 ± 0.7 | 5.0 | 105 (96) | 0.78 |
| logistic|frac1|eps0 | [901, 902, 903] | [32, 32, 32] | 98.4 ± 0.7 | 75.0 ± 1.0 | 4.8 | 66 (62) | 0.77 |
| logistic|frac1|eps0.2 | [901, 902, 903] | [32, 32, 32] | 98.2 ± 0.4 | 76.4 ± 2.2 | 4.9 | 73 (72) | 0.78 |
| infonce|frac0.25|eps0 | [901, 902, 903] | [8, 8, 8] | 96.8 ± 2.5 | 70.3 ± 1.1 | 6.2 | 130 (117) | 1.98 |
| infonce|frac0.25|eps0.2 | [901, 902, 903] | [8, 8, 8] | 97.6 ± 0.7 | 73.4 ± 4.3 | 5.5 | 96 (96) | 0.78 |
| infonce|frac1|eps0 | [901, 902, 903] | [32, 32, 32] | 98.4 ± 0.1 | 74.3 ± 2.6 | 5.5 | 63 (63) | 0.77 |
| infonce|frac1|eps0.2 | [901, 902, 903] | [32, 32, 32] | 98.2 ± 0.6 | 72.1 ± 3.3 | 5.7 | 71 (67) | 0.78 |
| comir|frac0.25|eps0 | [901, 902, 903] | [0, 0, 0] | 76.9 ± 17.3 | 56.6 ± 13.1 | 8.6 | 929 (905) | 0.49 |
| comir|frac0.25|eps0.2 | [901, 902, 903] | [0, 0, 0] | 68.4 ± 25.6 | 52.3 ± 18.0 | 73.0 | 1272 (1229) | 0.51 |
| comir|frac1|eps0 | [901, 902, 903] | [0, 0, 0] | 84.8 ± 5.4 | 66.6 ± 3.4 | 5.8 | 612 (591) | 0.51 |
| comir|frac1|eps0.2 | [901, 902, 903] | [0, 0, 0] | 62.9 ± 21.5 | 47.0 ± 17.5 | 68.2 | 1493 (1429) | 0.51 |
| kernel|frac0.25|eps0 | [901] | [8] | 48.4 | 26.3 | 49.7 | 691 (662) | 4.66 |
| kernel|frac0.25|eps0.2 | [901] | [8] | 47.8 | 25.4 | 58.6 | 699 (666) | 4.66 |
| kernel|frac1|eps0 | [901] | [32] | 34.3 | 18.5 | 179.4 | 881 (840) | 4.67 |
| kernel|frac1|eps0.2 | [901] | [32] | 36.2 | 19.8 | 164.4 | 855 (819) | 4.68 |
| mi | — | — | 78.4 | 57.6 | 6.6 | 289 (189) | — |
| nmi | — | — | 79.0 | 58.2 | 6.5 | 281 (192) | — |

## Primary family (Holm over 4; detected = seed t and unit bootstrap exclude 0 and Holm p < 0.05)

| contrast | value | boot p | Holm p | detected | complete |
|---|---|---|---|---|---|
| vcs_minus_mi|frac0.25|eps0.2@42 | +19.0 [+16.1, +21.9]; unit bootstrap [+13.6, +24.9] | 0.0000 | 0.0000 | True | True |
| vcs_minus_comir|frac0.25|eps0.2@42 | +29.1 [-32.7, +90.8]; unit bootstrap [+24.9, +33.1] | 0.0000 | 0.0000 | False | True |
| vcs_minus_mi|frac1|eps0.2@42 | +20.2 [+18.8, +21.6]; unit bootstrap [+14.5, +26.2] | 0.0000 | 0.0000 | True | True |
| vcs_minus_comir|frac1|eps0.2@42 | +35.8 [-16.1, +87.7]; unit bootstrap [+30.9, +40.6] | 0.0000 | 0.0000 | False | True |

## All contrasts

| contrast | value |
|---|---|
| vcs_minus_mi|frac0.25|eps0@42 | +18.8 [+17.9, +19.7]; unit bootstrap [+13.1, +24.4] |
| vcs_minus_nmi|frac0.25|eps0@42 | +18.2 [+17.3, +19.1]; unit bootstrap [+12.8, +23.8] |
| vcs_minus_kernel|frac0.25|eps0@42 | +48.8 [+47.9, +49.7]; unit bootstrap [+41.4, +56.2] |
| vcs_minus_logistic|frac0.25|eps0@42 | -0.2 [-2.1, +1.6]; unit bootstrap [-1.0, +0.5] |
| vcs_minus_infonce|frac0.25|eps0@42 | +0.4 [-4.8, +5.7]; unit bootstrap [-0.3, +1.3] |
| vcs_minus_comir|frac0.25|eps0@42 | +20.3 [-21.9, +62.6]; unit bootstrap [+16.1, +24.9] |
| vcs_minus_mi|frac0.25|eps0.2@42 | +19.0 [+16.1, +21.9]; unit bootstrap [+13.6, +24.9] |
| vcs_minus_nmi|frac0.25|eps0.2@42 | +18.4 [+15.5, +21.3]; unit bootstrap [+13.0, +24.1] |
| vcs_minus_kernel|frac0.25|eps0.2@42 | +49.6 [+46.7, +52.5]; unit bootstrap [+42.7, +56.6] |
| vcs_minus_logistic|frac0.25|eps0.2@42 | +0.0 [-2.5, +2.6]; unit bootstrap [-1.1, +1.1] |
| vcs_minus_infonce|frac0.25|eps0.2@42 | -0.2 [-2.2, +1.9]; unit bootstrap [-0.9, +0.6] |
| vcs_minus_comir|frac0.25|eps0.2@42 | +29.1 [-32.7, +90.8]; unit bootstrap [+24.9, +33.1] |
| vcs_minus_mi|frac1|eps0@42 | +19.9 [+18.5, +21.2]; unit bootstrap [+14.6, +25.8] |
| vcs_minus_nmi|frac1|eps0@42 | +19.3 [+17.9, +20.6]; unit bootstrap [+13.7, +25.2] |
| vcs_minus_kernel|frac1|eps0@42 | +64.1 [+62.7, +65.4]; unit bootstrap [+56.8, +70.3] |
| vcs_minus_logistic|frac1|eps0@42 | -0.0 [-1.9, +1.8]; unit bootstrap [-0.4, +0.3] |
| vcs_minus_infonce|frac1|eps0@42 | -0.1 [-1.3, +1.1]; unit bootstrap [-0.5, +0.1] |
| vcs_minus_comir|frac1|eps0@42 | +13.5 [-1.2, +28.3]; unit bootstrap [+9.3, +18.3] |
| vcs_minus_mi|frac1|eps0.2@42 | +20.2 [+18.8, +21.6]; unit bootstrap [+14.5, +26.2] |
| vcs_minus_nmi|frac1|eps0.2@42 | +19.6 [+18.2, +21.0]; unit bootstrap [+14.0, +25.3] |
| vcs_minus_kernel|frac1|eps0.2@42 | +62.5 [+61.1, +63.9]; unit bootstrap [+55.3, +69.4] |
| vcs_minus_logistic|frac1|eps0.2@42 | +0.5 [-0.7, +1.6]; unit bootstrap [+0.0, +1.2] |
| vcs_minus_infonce|frac1|eps0.2@42 | +0.4 [+0.0, +0.8]; unit bootstrap [+0.0, +1.2] |
| vcs_minus_comir|frac1|eps0.2@42 | +35.8 [-16.1, +87.7]; unit bootstrap [+30.9, +40.6] |
| vcs_minus_mi|frac0.25|eps0@9 | +14.0 [+7.2, +20.7]; unit bootstrap [+8.6, +19.5] |
| vcs_minus_nmi|frac0.25|eps0@9 | +13.4 [+6.6, +20.1]; unit bootstrap [+8.0, +19.1] |
| vcs_minus_kernel|frac0.25|eps0@9 | +45.2 [+38.5, +52.0]; unit bootstrap [+37.8, +53.2] |
| vcs_minus_logistic|frac0.25|eps0@9 | -3.0 [-8.0, +2.0]; unit bootstrap [-5.0, -1.2] |
| vcs_minus_infonce|frac0.25|eps0@9 | +1.3 [-2.7, +5.3]; unit bootstrap [-1.5, +4.0] |
| vcs_minus_comir|frac0.25|eps0@9 | +15.0 [-15.8, +45.8]; unit bootstrap [+10.6, +19.5] |
| vcs_minus_mi|frac0.25|eps0.2@9 | +16.0 [+13.6, +18.5]; unit bootstrap [+11.1, +21.4] |
| vcs_minus_nmi|frac0.25|eps0.2@9 | +15.4 [+13.0, +17.9]; unit bootstrap [+10.6, +20.7] |
| vcs_minus_kernel|frac0.25|eps0.2@9 | +48.2 [+45.8, +50.6]; unit bootstrap [+41.3, +55.4] |
| vcs_minus_logistic|frac0.25|eps0.2@9 | -0.9 [-4.9, +3.0]; unit bootstrap [-2.7, +0.7] |
| vcs_minus_infonce|frac0.25|eps0.2@9 | +0.3 [-10.6, +11.2]; unit bootstrap [-1.4, +2.2] |
| vcs_minus_comir|frac0.25|eps0.2@9 | +21.4 [-22.2, +65.0]; unit bootstrap [+17.6, +25.8] |
| vcs_minus_mi|frac1|eps0@9 | +19.1 [+14.8, +23.4]; unit bootstrap [+13.9, +24.5] |
| vcs_minus_nmi|frac1|eps0@9 | +18.5 [+14.2, +22.8]; unit bootstrap [+13.4, +23.8] |
| vcs_minus_kernel|frac1|eps0@9 | +58.2 [+53.9, +62.5]; unit bootstrap [+50.8, +65.4] |
| vcs_minus_logistic|frac1|eps0@9 | +1.7 [-4.7, +8.2]; unit bootstrap [+0.3, +3.3] |
| vcs_minus_infonce|frac1|eps0@9 | +2.4 [-8.2, +13.1]; unit bootstrap [+0.8, +4.1] |
| vcs_minus_comir|frac1|eps0@9 | +10.0 [+2.7, +17.4]; unit bootstrap [+6.0, +14.4] |
| vcs_minus_mi|frac1|eps0.2@9 | +19.0 [+15.5, +22.6]; unit bootstrap [+13.8, +24.8] |
| vcs_minus_nmi|frac1|eps0.2@9 | +18.4 [+14.9, +22.0]; unit bootstrap [+13.0, +24.3] |
| vcs_minus_kernel|frac1|eps0.2@9 | +56.9 [+53.3, +60.4]; unit bootstrap [+49.6, +64.6] |
| vcs_minus_logistic|frac1|eps0.2@9 | +0.2 [-4.4, +4.9]; unit bootstrap [-1.2, +1.7] |
| vcs_minus_infonce|frac1|eps0.2@9 | +4.5 [-6.3, +15.4]; unit bootstrap [+2.3, +6.8] |
| vcs_minus_comir|frac1|eps0.2@9 | +29.6 [-11.8, +71.1]; unit bootstrap [+24.7, +34.6] |

## Interaction: change of success < 42 px from ε 0 to 0.2 (seed-paired)

| method, fraction | Δ (pp) |
|---|---|
| vcs|frac0.25|delta_0_to_0.2@42 | +0.2 [-2.8, +3.3] |
| vcs|frac1|delta_0_to_0.2@42 | +0.3 [-2.4, +3.0] |
| logistic|frac0.25|delta_0_to_0.2@42 | -0.1 [-2.8, +2.7] |
| logistic|frac1|delta_0_to_0.2@42 | -0.2 [-3.0, +2.6] |
| infonce|frac0.25|delta_0_to_0.2@42 | +0.8 [-4.5, +6.1] |
| infonce|frac1|delta_0_to_0.2@42 | -0.2 [-1.8, +1.4] |
| comir|frac0.25|delta_0_to_0.2@42 | -8.5 [-34.6, +17.6] |
| comir|frac1|delta_0_to_0.2@42 | -21.9 [-61.8, +18.0] |
| kernel|frac0.25|delta_0_to_0.2@42 | -0.6 |
| kernel|frac1|delta_0_to_0.2@42 | +1.9 |
