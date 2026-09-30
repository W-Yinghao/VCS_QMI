# T1 — within-class conditional test with fair controls, N | Y re-drawn per repeat — 2026-09-29T22:33:35Z

Families: colour (P45_precheck_D_vcs4v800); δ = 0.05; 200 shared within-class permutations; critics / classifier / deep kernel: 300 steps, fit on 80 % of FIT, selected on 20 %.

| case | mode | s | n | R | vcs_perm | vcs_hoeff | hsic_perm | hsic_class | hsic_deep | c2st_logit | js_perm | hoeff lin / mlp / closed | mean J (VCS) | mean JS |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| colour_null_label_only | null_label_only | 0 | 200 | 100 | 0.03 | 0.00 | 0.03 | 0.06 | 0.06 | 0.03 | 0.03 | 0.00 / 0.00 / 0.00 | -0.0122 ± 0.0290 | -0.0081 |
| colour_null_label_only | null_label_only | 0 | 1000 | 100 | 0.01 | 0.00 | 0.04 | 0.06 | 0.07 | 0.04 | 0.02 | 0.00 / 0.00 / 0.00 | -0.0020 ± 0.0031 | -0.0007 |

## Smallest n with power ≥ 0.8 (planted cases)

| case | vcs_perm | vcs_hoeff | hsic_perm | hsic_class | hsic_deep | c2st_logit | js_perm |
|---|---|---|---|---|---|---|---|
