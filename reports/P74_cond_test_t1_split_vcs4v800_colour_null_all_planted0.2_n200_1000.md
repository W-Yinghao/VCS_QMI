# T1 — within-class conditional test with fair controls, N | Y re-drawn per repeat — 2026-09-29T20:53:38Z

Families: colour (P45_precheck_D_vcs4v800); δ = 0.05; 200 shared within-class permutations; critics / classifier / deep kernel: 300 steps, fit on 80 % of FIT, selected on 20 %.

| case | mode | s | n | R | vcs_perm | vcs_hoeff | hsic_perm | hsic_class | hsic_deep | c2st_logit | js_perm | hoeff lin / mlp / closed | mean J (VCS) | mean JS |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| colour_null_all_planted0.2 | null_all_planted | 0.2 | 200 | 100 | 0.04 | 0.00 | 0.05 | 0.09 | 0.04 | 0.00 | 0.06 | 0.00 / 0.00 / 0.00 | -0.0061 ± 0.0173 | -0.0082 |
| colour_null_all_planted0.2 | null_all_planted | 0.2 | 1000 | 100 | 0.07 | 0.00 | 0.03 | 0.08 | 0.09 | 0.04 | 0.04 | 0.00 / 0.00 / 0.00 | -0.0021 ± 0.0031 | -0.0006 |

## Smallest n with power ≥ 0.8 (planted cases)

| case | vcs_perm | vcs_hoeff | hsic_perm | hsic_class | hsic_deep | c2st_logit | js_perm |
|---|---|---|---|---|---|---|---|
