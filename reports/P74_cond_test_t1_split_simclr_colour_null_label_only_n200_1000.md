# T1 — within-class conditional test with fair controls, N | Y re-drawn per repeat — 2026-09-29T20:07:11Z

Families: colour (P45_precheck_D_simclr); δ = 0.05; 200 shared within-class permutations; critics / classifier / deep kernel: 300 steps, fit on 80 % of FIT, selected on 20 %.

| case | mode | s | n | R | vcs_perm | vcs_hoeff | hsic_perm | hsic_class | hsic_deep | c2st_logit | js_perm | hoeff lin / mlp / closed | mean J (VCS) | mean JS |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| colour_null_label_only | null_label_only | 0 | 200 | 100 | 0.01 | 0.00 | 0.03 | 0.02 | 0.07 | 0.04 | 0.04 | 0.00 / 0.00 / 0.00 | -0.0105 ± 0.0248 | -0.0097 |
| colour_null_label_only | null_label_only | 0 | 1000 | 100 | 0.10 | 0.00 | 0.01 | 0.08 | 0.03 | 0.05 | 0.07 | 0.00 / 0.00 / 0.00 | -0.0022 ± 0.0033 | -0.0008 |

## Smallest n with power ≥ 0.8 (planted cases)

| case | vcs_perm | vcs_hoeff | hsic_perm | hsic_class | hsic_deep | c2st_logit | js_perm |
|---|---|---|---|---|---|---|---|
