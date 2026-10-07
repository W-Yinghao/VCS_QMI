# P142 results — fixed encoder, vary estimator (Table 3 axis)

P116 reproduction gate (block A, A1 and A3 per repeat): colour_s0.2 n2000: pass=True (max |Δ| 7.6e-11, mismatches 0); blur_s0.5 n1000: pass=True (max |Δ| 1.1e-11, mismatches 0); blur_s0.5 n2000: pass=True (max |Δ| 5.6e-12, mismatches 0); colour_null_label_only n2000: pass=True (max |Δ| 7.2e-12, mismatches 0); colour_null_all_planted0.2 n2000: pass=True (max |Δ| 9.5e-12, mismatches 0)

## Block A — F-SOLVER-P116 (P5_simclr_seed0, 2 views, 200 epochs)

| cell | n | R | A1 | A3@50 | A3@200 | A3@800 | K1 | H1 | H2 |
|---|---|---|---|---|---|---|---|---|---|
| colour_s0.2 | 2000 | 100 | 0.97 | 0.94 | 0.94 | 0.94 | 0.80 | 0.34 | 0.41 |
| blur_s0.5 | 1000 | 100 | 0.75 | 0.82 | 0.82 | 0.82 | 0.59 | 0.19 | 0.15 |
| blur_s0.5 | 2000 | 100 | 0.99 | 1.00 | 1.00 | 1.00 | 0.96 | 0.58 | 0.55 |
| colour_null_label_only | 2000 | 200 | 0.04 | 0.04 | 0.04 | 0.04 | 0.06 | 0.07 | 0.03 |
| colour_null_all_planted0.2 | 2000 | 200 | 0.06 | 0.06 | 0.06 | 0.06 | 0.04 | 0.06 | 0.05 |
| colour_null_label_only | 1000 | 200 | 0.04 | 0.05 | 0.04 | 0.04 | 0.05 | 0.03 | 0.04 |

Cost per repeat (n 2 000 planted cells, mean seconds; fit includes selection; permutation = 200 permutations):

| method | fit s | perm s |
|---|---|---|
| VCS (ridge-tanh two-stage) | 0.11 | 0.003 |
| matched logistic, 50 closures | 0.08 | 0.004 |
| matched logistic, 200 | 0.42 | 0.003 |
| matched logistic, 800 | 1.65 | 0.003 |
| same-target kernel (RFF, A1 solver) | 0.46 | 0.003 |
| conditional HSIC, class-wise median kernel | nan | 1.347 |
| conditional HSIC, deep kernel | 69.61 | 0.951 |

## Block B — F-SOLVER-P142B (P41_simclr_views4_800ep_seed1 = Table 4 SimCLR / 1)

| cell | n | R | A1 | A3@50 | A3@200 | A3@800 | K1 | H1 | H2 |
|---|---|---|---|---|---|---|---|---|---|
| blur_s0.5 | 1000 | 100 | 0.27 | 0.33 | 0.33 | 0.33 | 0.27 | 0.08 | 0.10 |
| blur_s0.5 | 2000 | 100 | 0.66 | 0.72 | 0.72 | 0.72 | 0.50 | 0.22 | 0.19 |
| colour_s0.2 | 2000 | 100 | 0.52 | 0.61 | 0.61 | 0.61 | 0.42 | 0.39 | 0.44 |
| colour_s0.1 | 2000 | 100 | 0.09 | 0.07 | 0.07 | 0.07 | 0.08 | 0.09 | 0.07 |
| colour_null_label_only | 1000 | 200 | 0.04 | 0.04 | 0.04 | 0.04 | 0.04 | 0.04 | 0.06 |
| colour_null_label_only | 2000 | 200 | 0.06 | 0.06 | 0.06 | 0.06 | 0.04 | 0.03 | 0.04 |
| colour_null_all_planted0.2 | 2000 | 200 | 0.07 | 0.06 | 0.06 | 0.06 | 0.07 | 0.03 | 0.03 |

Cost per repeat (n 2 000 planted cells, mean seconds; fit includes selection; permutation = 200 permutations):

| method | fit s | perm s |
|---|---|---|
| VCS (ridge-tanh two-stage) | 0.09 | 0.004 |
| matched logistic, 50 closures | 0.09 | 0.007 |
| matched logistic, 200 | 0.50 | 0.007 |
| matched logistic, 800 | 1.11 | 0.007 |
| same-target kernel (RFF, A1 solver) | 0.80 | 0.004 |
| conditional HSIC, class-wise median kernel | nan | 2.351 |
| conditional HSIC, deep kernel | 110.14 | 2.026 |
