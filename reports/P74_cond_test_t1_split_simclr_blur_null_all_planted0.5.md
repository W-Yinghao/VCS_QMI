# T1 — within-class conditional test with fair controls, N | Y re-drawn per repeat — 2026-09-29T20:28:05Z

Families: blur (P45_precheck_D_simclr_blur); δ = 0.05; 200 shared within-class permutations; critics / classifier / deep kernel: 300 steps, fit on 80 % of FIT, selected on 20 %.

| case | mode | s | n | R | vcs_perm | vcs_hoeff | hsic_perm | hsic_class | hsic_deep | c2st_logit | js_perm | hoeff lin / mlp / closed | mean J (VCS) | mean JS |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| blur_null_all_planted0.5 | null_all_planted | 0.5 | 200 | 100 | 0.08 | 0.00 | 0.07 | 0.04 | 0.03 | 0.03 | 0.07 | 0.00 / 0.00 / 0.00 | -0.0113 ± 0.0281 | -0.0056 |
| blur_null_all_planted0.5 | null_all_planted | 0.5 | 500 | 100 | 0.08 | 0.00 | 0.07 | 0.08 | 0.08 | 0.05 | 0.06 | 0.00 / 0.00 / 0.00 | -0.0031 ± 0.0050 | -0.0044 |
| blur_null_all_planted0.5 | null_all_planted | 0.5 | 1000 | 100 | 0.07 | 0.00 | 0.06 | 0.07 | 0.02 | 0.03 | 0.04 | 0.00 / 0.00 / 0.00 | -0.0017 ± 0.0026 | -0.0014 |
| blur_null_all_planted0.5 | null_all_planted | 0.5 | 2000 | 100 | 0.01 | 0.00 | 0.06 | 0.09 | 0.08 | 0.06 | 0.02 | 0.00 / 0.00 / 0.00 | -0.0011 ± 0.0018 | -0.0005 |

## Smallest n with power ≥ 0.8 (planted cases)

| case | vcs_perm | vcs_hoeff | hsic_perm | hsic_class | hsic_deep | c2st_logit | js_perm |
|---|---|---|---|---|---|---|---|
