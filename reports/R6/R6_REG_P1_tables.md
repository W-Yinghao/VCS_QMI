# R6 REG-P1 — BF/SHG translation-only registration: DEV results

Missing: none.

| method | seeds | success < 42 px (%) | success < 9 px (%) | median error (px) | p90 error (px) | s / registration |
|---|---|---|---|---|---|---|
| vcs | [0, 1, 2] | 100.0 ± 0.0 | 75.0 ± 3.3 | 5.4 | 14.8 | 0.53 |
| logistic | [0, 1, 2] | 98.9 ± 1.7 | 65.8 ± 4.5 | 5.7 | 16.1 | 0.52 |
| infonce | [0, 1, 2] | 95.2 ± 2.3 | 57.4 ± 9.4 | 7.6 | 18.3 | 0.77 |
| comir | [0, 1, 2] | 90.9 ± 0.3 | 63.9 ± 2.5 | 5.7 | 21.1 | 0.16 |
| mi | [None] | 81.9 | 55.6 | 7.5 | 169.9 | 0.45 |
| nmi | [None] | 80.3 | 55.3 | 6.9 | 171.0 | 0.45 |
| kernel | [None] | 57.2 | 36.2 | 21.3 | 255.5 | 0.42 |

| contrast (success < 42 px, image-macro) | value |
|---|---|
| vcs_minus_mi | +18.1 [+18.1, +18.1]; image bootstrap [+7.5, +30.3] |
| vcs_minus_nmi | +19.7 [+19.7, +19.7]; image bootstrap [+8.4, +31.6] |
| vcs_minus_kernel | +42.8 [+42.8, +42.8]; image bootstrap [+28.1, +57.2] |
| vcs_minus_logistic | +1.1 [-3.1, +5.4]; image bootstrap [+0.0, +3.4] |
| vcs_minus_infonce | +4.8 [-0.8, +10.4]; image bootstrap [+0.0, +12.2] |
| vcs_minus_comir | +9.1 [+8.3, +9.8]; image bootstrap [+0.0, +18.5] |
