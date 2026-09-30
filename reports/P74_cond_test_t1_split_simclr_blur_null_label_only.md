# T1 — within-class conditional test with fair controls, N | Y re-drawn per repeat — 2026-09-29T20:25:59Z

Families: blur (P45_precheck_D_simclr_blur); δ = 0.05; 200 shared within-class permutations; critics / classifier / deep kernel: 300 steps, fit on 80 % of FIT, selected on 20 %.

| case | mode | s | n | R | vcs_perm | vcs_hoeff | hsic_perm | hsic_class | hsic_deep | c2st_logit | js_perm | hoeff lin / mlp / closed | mean J (VCS) | mean JS |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| blur_null_label_only | null_label_only | 0 | 200 | 100 | 0.03 | 0.00 | 0.07 | 0.03 | 0.06 | 0.05 | 0.05 | 0.00 / 0.00 / 0.00 | -0.0067 ± 0.0120 | -0.0079 |
| blur_null_label_only | null_label_only | 0 | 500 | 100 | 0.01 | 0.00 | 0.06 | 0.06 | 0.05 | 0.03 | 0.04 | 0.00 / 0.00 / 0.00 | -0.0037 ± 0.0060 | -0.0038 |
| blur_null_label_only | null_label_only | 0 | 1000 | 100 | 0.04 | 0.00 | 0.04 | 0.03 | 0.04 | 0.08 | 0.06 | 0.00 / 0.00 / 0.00 | -0.0021 ± 0.0033 | -0.0009 |
| blur_null_label_only | null_label_only | 0 | 2000 | 100 | 0.08 | 0.00 | 0.02 | 0.06 | 0.05 | 0.04 | 0.09 | 0.00 / 0.00 / 0.00 | -0.0010 ± 0.0016 | -0.0004 |

## Smallest n with power ≥ 0.8 (planted cases)

| case | vcs_perm | vcs_hoeff | hsic_perm | hsic_class | hsic_deep | c2st_logit | js_perm |
|---|---|---|---|---|---|---|---|
