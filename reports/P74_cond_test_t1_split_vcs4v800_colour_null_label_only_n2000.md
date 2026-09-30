# T1 — within-class conditional test with fair controls, N | Y re-drawn per repeat — 2026-09-28T19:40:47Z

Families: colour (P45_precheck_D_vcs4v800); δ = 0.05; 200 shared within-class permutations; critics / classifier / deep kernel: 300 steps, fit on 80 % of FIT, selected on 20 %.

| case | mode | s | n | R | vcs_perm | vcs_hoeff | hsic_perm | hsic_class | hsic_deep | c2st_logit | js_perm | hoeff lin / mlp / closed | mean J (VCS) | mean JS |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| colour_null_label_only | null_label_only | 0 | 2000 | 1000 | 0.05 | 0.00 | 0.06 | 0.06 | 0.06 | 0.05 | 0.05 | 0.00 / 0.00 / 0.00 | -0.0007 ± 0.0013 | -0.0006 |

## Smallest n with power ≥ 0.8 (planted cases)

| case | vcs_perm | vcs_hoeff | hsic_perm | hsic_class | hsic_deep | c2st_logit | js_perm |
|---|---|---|---|---|---|---|---|
