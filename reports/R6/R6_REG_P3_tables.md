# R6 REG-P3 — EPFL RGB-NIR Scene (nirscene1; Brown & Süsstrunk, CVPR 2011): RGB reference, NIR moving: rigid registration (θ* ∈ ±30°, |t*| <= 51 px, 512-px windows): DEV results

Missing: none (not run by protocol: 3 CoMIR warp-then-encode files).  DEV: 72 units, 74 windows.  Success thresholds: mean corner error < 10.5 / < 2.25 native px (REG-P2's 42 / 9 px in REG-P2-equivalent units).  Unit-macro success.

| method | seeds | success < 10.5 px (%) | success < 2.25 px (%) | median corner error (px) | p90 (px) | median rotation error (°) | median translation error (px) | failures (energy) | CAL success < 10.5 (%) | s / registration |
|---|---|---|---|---|---|---|---|---|---|---|
| vcs | [0, 1, 2] | 99.3 ± 0.2 | 87.6 ± 1.7 | 0.79 | 2.4 | 0.14 | 0.65 | 15 (15) | 100.0 ± 0.0 | 0.98 |
| logistic | [0, 1, 2] | 99.2 ± 0.3 | 87.3 ± 1.4 | 0.67 | 2.4 | 0.11 | 0.55 | 17 (16) | 100.0 ± 0.0 | 1.23 |
| infonce | [0, 1, 2] | 99.4 ± 0.7 | 90.0 ± 1.1 | 0.58 | 2.3 | 0.09 | 0.49 | 13 (8) | 100.0 ± 0.0 | 1.01 |
| comir | [0, 1, 2] | 96.2 ± 1.2 | 85.3 ± 1.0 | 0.54 | 2.6 | 0.07 | 0.46 | 86 (78) | 100.0 ± 0.0 | 0.19 |
| mi | [None] | 93.8 | 82.9 | 0.47 | 3.0 | 0.06 | 0.42 | 45 (22) | 70.0 | 0.51 |
| nmi | [None] | 94.0 | 82.9 | 0.47 | 3.0 | 0.06 | 0.42 | 44 (21) | 72.2 | 0.50 |
| kernel | [None] | 87.8 | 76.0 | 0.65 | 41.5 | 0.09 | 0.56 | 93 (82) | 88.9 | 2.17 |

comir = CoMIR encode-once (the CoMIR row); comir_wte (CoMIR warp-then-encode) is not run in REG-P3.
Ceiling readout (every method's CAL success < 10.5 px >= 0.95): not triggered.

| contrast (unit-macro success; @42 = < 10.5 px, @9 = < 2.25 px) | value |
|---|---|
| vcs_minus_mi@42 | +5.5 [+4.9, +6.1]; unit bootstrap [+1.5, +10.8] |
| vcs_minus_nmi@42 | +5.3 [+4.7, +5.9]; unit bootstrap [+1.2, +10.6] |
| vcs_minus_kernel@42 | +11.5 [+10.9, +12.1]; unit bootstrap [+5.3, +18.8] |
| vcs_minus_logistic@42 | +0.1 [-1.3, +1.5]; unit bootstrap [+0.0, +0.3] |
| vcs_minus_infonce@42 | -0.1 [-1.5, +1.3]; unit bootstrap [-1.6, +1.0] |
| vcs_minus_comir@42 | +3.1 [+0.7, +5.5]; unit bootstrap [+0.5, +6.3] |
| vcs_minus_mi@9 | +4.7 [+0.4, +9.0]; unit bootstrap [-1.2, +10.8] |
| vcs_minus_nmi@9 | +4.7 [+0.4, +9.0]; unit bootstrap [-1.0, +10.7] |
| vcs_minus_kernel@9 | +11.7 [+7.3, +16.0]; unit bootstrap [+5.8, +18.4] |
| vcs_minus_logistic@9 | +0.3 [-2.3, +3.0]; unit bootstrap [-1.6, +2.4] |
| vcs_minus_infonce@9 | -2.4 [-9.3, +4.5]; unit bootstrap [-6.4, +1.3] |
| vcs_minus_comir@9 | +2.4 [-4.2, +8.9]; unit bootstrap [-1.6, +6.4] |
