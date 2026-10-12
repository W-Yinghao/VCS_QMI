# R6 REG-P1 — BF/SHG translation-only registration: DEV results — contrasts at < 9 px (secondary)

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

| contrast (success < 9 px, image-macro) — secondary analysis | value |
|---|---|
| vcs_minus_mi | +19.4 [+11.3, +27.5]; image bootstrap [+9.1, +31.5] |
| vcs_minus_nmi | +19.7 [+11.6, +27.8]; image bootstrap [+8.6, +32.0] |
| vcs_minus_kernel | +38.8 [+30.6, +46.9]; image bootstrap [+20.7, +55.9] |
| vcs_minus_logistic | +9.2 [-7.8, +26.1]; image bootstrap [+1.2, +18.0] |
| vcs_minus_infonce | +17.6 [-9.7, +45.0]; image bootstrap [+9.5, +26.2] |
| vcs_minus_comir | +11.1 [+8.0, +14.3]; image bootstrap [+2.0, +21.2] |
