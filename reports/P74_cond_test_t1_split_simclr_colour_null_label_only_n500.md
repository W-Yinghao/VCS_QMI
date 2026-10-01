# T1 — within-class conditional test with fair controls, N | Y re-drawn per repeat — 2026-09-30T15:57:21Z

Families: colour (P45_precheck_D_simclr); δ = 0.05; 200 shared within-class permutations; critics / classifier / deep kernel: 300 steps, fit on 80 % of FIT, selected on 20 %.

| case | mode | s | n | R | vcs_perm | vcs_hoeff | hsic_perm | hsic_class | hsic_deep | c2st_logit | js_perm | hoeff lin / mlp / closed | mean J (VCS) | mean JS |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| colour_null_label_only | null_label_only | 0 | 500 | 1000 | 0.04 | 0.00 | 0.04 | 0.05 | 0.05 | 0.05 | 0.04 | 0.00 / 0.00 / 0.00 | -0.0038 ± 0.0063 | -0.0029 |

## Smallest n with power ≥ 0.8 (planted cases)

| case | vcs_perm | vcs_hoeff | hsic_perm | hsic_class | hsic_deep | c2st_logit | js_perm |
|---|---|---|---|---|---|---|---|
