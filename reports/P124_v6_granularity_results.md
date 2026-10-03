# P124 — v6 V6-GRANULARITY results (frozen CIFAR-100 SSL encoders; selection split; official test never opened)

Readouts identical for every method; PRIMARY = `recipe_raw` (the original frozen-h linear readout).  Conditional = coarse label given at test
time, 5-way within the coarse group (dedicated probe per group) — never an unconditional 100-way accuracy.  mean ± sd over encoder seeds.

## Anchor: fine 100-way `recipe_raw` vs the stored training-evaluation linear accuracy (must agree within GPU noise, |Δ| ≤ 0.3)

| run | stored | recomputed | Δ |
|---|---|---|---|
| P107_AP3_c100_views4_800ep_seed0 | 60.20 | 60.20 | +0.00 |
| P107_AP3_c100_views4_800ep_seed1 | 60.08 | 60.08 | +0.00 |
| P107_AP3_c100_views4_800ep_seed2 | 59.72 | 59.72 | +0.00 |
| P111_G2_c100_views4_800ep_seed0 | 59.46 | 59.44 | -0.02 |
| P111_G2_c100_views4_800ep_seed1 | 59.88 | 59.90 | +0.02 |
| P111_G2_c100_views4_800ep_seed2 | 60.46 | 60.44 | -0.02 |
| P91_c100_vcs_a5_views4_800ep_seed0 | 60.00 | 60.00 | +0.00 |
| P91_c100_vcs_a5_views4_800ep_seed1 | 59.68 | 59.68 | +0.00 |
| P91_c100_vcs_a5_views4_800ep_seed2 | 60.04 | 60.04 | +0.00 |
| P91_c100_simclr_views4_800ep_seed0 | 58.66 | 58.64 | -0.02 |
| P91_c100_simclr_views4_800ep_seed1 | 58.02 | 58.06 | +0.04 |
| P91_c100_simclr_views4_800ep_seed2 | 58.08 | 58.04 | -0.04 |

## coarse accuracy (%)

| readout | A-P3 | G2 | recipe VCS | SimCLR |
|---|---|---|---|---|
| recipe_raw (primary) | 71.82 ± 0.27 | 70.91 ± 0.35 | 68.77 ± 0.73 | 70.70 ± 0.53 |
| raw_unstd | 72.00 ± 0.44 | 71.63 ± 0.71 | 69.11 ± 0.16 | 70.73 ± 0.48 |
| raw_std | 71.41 ± 0.47 | 70.50 ± 0.49 | 68.73 ± 0.67 | 69.64 ± 0.48 |
| l2_std | 71.26 ± 0.43 | 70.47 ± 0.66 | 68.61 ± 0.60 | 69.99 ± 0.63 |
| knn | 71.47 ± 0.53 | 70.24 ± 0.09 | 68.90 ± 0.39 | 71.97 ± 0.53 |

## fine accuracy (%)

| readout | A-P3 | G2 | recipe VCS | SimCLR |
|---|---|---|---|---|
| recipe_raw (primary) | 60.00 ± 0.25 | 59.93 ± 0.50 | 59.91 ± 0.20 | 58.25 ± 0.34 |
| raw_unstd | 60.00 ± 0.25 | 59.93 ± 0.50 | 59.93 ± 0.11 | 58.18 ± 0.23 |
| raw_std | 59.34 ± 0.27 | 59.08 ± 0.16 | 59.17 ± 0.21 | 56.59 ± 0.22 |
| l2_std | 58.92 ± 0.32 | 58.69 ± 0.38 | 58.96 ± 0.10 | 56.93 ± 0.19 |
| knn | 55.81 ± 0.25 | 54.13 ± 0.11 | 54.95 ± 0.41 | 57.21 ± 0.36 |

## conditional_macro accuracy (%)

| readout | A-P3 | G2 | recipe VCS | SimCLR |
|---|---|---|---|---|
| recipe_raw (primary) | 76.34 ± 0.51 | 75.87 ± 0.40 | 76.61 ± 0.16 | 74.90 ± 0.18 |
| raw_unstd | 76.20 ± 0.70 | 75.71 ± 0.29 | 76.81 ± 0.52 | 74.72 ± 0.27 |
| raw_std | 72.71 ± 0.35 | 73.15 ± 0.33 | 74.16 ± 0.05 | 70.05 ± 0.81 |
| l2_std | 72.83 ± 0.46 | 73.08 ± 0.65 | 74.19 ± 0.18 | 70.27 ± 0.98 |
| knn | 72.94 ± 0.18 | 71.68 ± 0.28 | 73.03 ± 0.08 | 73.61 ± 0.33 |

## conditional_overall accuracy (%)

| readout | A-P3 | G2 | recipe VCS | SimCLR |
|---|---|---|---|---|
| recipe_raw (primary) | 76.34 ± 0.51 | 75.87 ± 0.40 | 76.61 ± 0.16 | 74.90 ± 0.18 |
| raw_unstd | 76.20 ± 0.70 | 75.71 ± 0.29 | 76.81 ± 0.52 | 74.72 ± 0.27 |
| raw_std | 72.71 ± 0.35 | 73.15 ± 0.33 | 74.16 ± 0.05 | 70.05 ± 0.81 |
| l2_std | 72.83 ± 0.46 | 73.08 ± 0.65 | 74.19 ± 0.18 | 70.27 ± 0.98 |
| knn | 72.94 ± 0.18 | 71.68 ± 0.28 | 73.03 ± 0.08 | 73.61 ± 0.33 |

## Seed-paired contrasts (95 % t interval, df = n − 1; labels: close |Δ| < 0.3 / clear (|Δ| ≥ 0.3 and interval excludes 0) / inconclusive)

| contrast | readout | Δ coarse | Δ fine | Δ conditional macro | Δ conditional overall |
|---|---|---|---|---|---|
| A-P3 − recipe VCS | recipe_raw (primary) | +3.05 [+0.62, +5.47] clear + | +0.09 [-0.83, +1.02] close | -0.27 [-1.61, +1.08] close | -0.27 [-1.61, +1.08] close |
| A-P3 − recipe VCS | raw_unstd | +2.89 [+1.49, +4.28] clear + | +0.07 [-0.77, +0.90] close | -0.61 [-3.05, +1.83] inconclusive | -0.61 [-3.05, +1.83] inconclusive |
| A-P3 − recipe VCS | raw_std | +2.69 [+0.09, +5.28] clear + | +0.17 [-0.67, +1.00] close | -1.45 [-2.20, -0.69] clear − | -1.45 [-2.20, -0.69] clear − |
| A-P3 − recipe VCS | l2_std | +2.65 [+0.33, +4.96] clear + | -0.04 [-0.82, +0.74] close | -1.36 [-2.88, +0.16] inconclusive | -1.36 [-2.88, +0.16] inconclusive |
| A-P3 − recipe VCS | knn | +2.57 [+2.14, +2.99] clear + | +0.85 [-0.33, +2.04] inconclusive | -0.09 [-0.51, +0.32] close | -0.09 [-0.51, +0.32] close |
| A-P3 − SimCLR | recipe_raw (primary) | +1.12 [-0.54, +2.78] inconclusive | +1.75 [+1.16, +2.35] clear + | +1.44 [-0.05, +2.93] inconclusive | +1.44 [-0.05, +2.93] inconclusive |
| A-P3 − SimCLR | raw_unstd | +1.27 [-0.86, +3.41] inconclusive | +1.82 [+1.38, +2.26] clear + | +1.48 [-0.70, +3.66] inconclusive | +1.48 [-0.70, +3.66] inconclusive |
| A-P3 − SimCLR | raw_std | +1.77 [+0.06, +3.48] clear + | +2.75 [+2.18, +3.32] clear + | +2.67 [-0.13, +5.47] inconclusive | +2.67 [-0.13, +5.47] inconclusive |
| A-P3 − SimCLR | l2_std | +1.27 [-0.65, +3.20] inconclusive | +1.99 [+1.61, +2.36] clear + | +2.55 [-1.01, +6.12] inconclusive | +2.55 [-1.01, +6.12] inconclusive |
| A-P3 − SimCLR | knn | -0.50 [-1.75, +0.75] inconclusive | -1.41 [-2.80, -0.01] clear − | -0.67 [-1.95, +0.61] inconclusive | -0.67 [-1.95, +0.61] inconclusive |
| G2 − recipe VCS | recipe_raw (primary) | +2.14 [-0.51, +4.79] inconclusive | +0.02 [-1.25, +1.29] close | -0.74 [-1.77, +0.29] inconclusive | -0.74 [-1.77, +0.29] inconclusive |
| G2 − recipe VCS | raw_unstd | +2.51 [+0.58, +4.45] clear + | -0.01 [-1.14, +1.13] close | -1.11 [-2.08, -0.13] clear − | -1.11 [-2.08, -0.13] clear − |
| G2 − recipe VCS | raw_std | +1.77 [-0.66, +4.21] inconclusive | -0.09 [-0.91, +0.72] close | -1.01 [-1.93, -0.08] clear − | -1.01 [-1.93, -0.08] clear − |
| G2 − recipe VCS | l2_std | +1.85 [-0.68, +4.39] inconclusive | -0.27 [-1.39, +0.84] close | -1.11 [-2.31, +0.10] inconclusive | -1.11 [-2.31, +0.10] inconclusive |
| G2 − recipe VCS | knn | +1.34 [+0.15, +2.53] clear + | -0.82 [-2.04, +0.40] inconclusive | -1.35 [-2.10, -0.61] clear − | -1.35 [-2.10, -0.61] clear − |

## Pattern statement (frozen rule, PRIMARY readout only; Δcoarse vs Δconditional-macro labels)

- **A-P3 - recipe VCS:** coarse-specific gain (coarse up, within-coarse not)
- **A-P3 - SimCLR:** no pattern (inconclusive at 3 seeds)
- **G2 - recipe VCS:** no pattern (inconclusive at 3 seeds)

## 100-way recipe head: implied-coarse accuracy × fine accuracy given implied coarse correct (= fine accuracy); masked-head conditional

| family | implied coarse | fine given coarse correct | fine | masked-head conditional macro | masked-head overall |
|---|---|---|---|---|---|
| A-P3 | 73.49 ± 0.13 | 81.65 ± 0.46 | 60.00 ± 0.25 | 76.39 ± 0.53 | 76.39 ± 0.53 |
| G2 | 73.19 ± 0.63 | 81.87 ± 0.02 | 59.93 ± 0.50 | 76.25 ± 0.25 | 76.25 ± 0.25 |
| recipe VCS | 72.76 ± 0.29 | 82.33 ± 0.08 | 59.91 ± 0.20 | 76.99 ± 0.25 | 76.99 ± 0.25 |
| SimCLR | 71.41 ± 0.36 | 81.56 ± 0.17 | 58.25 ± 0.34 | 74.95 ± 0.18 | 74.95 ± 0.18 |

## Chosen probe (lr, wd) per family and readout (selected readouts; grid lr 0.03 / 0.1 / 0.3 × wd 0 / 5e-4)

- A-P3 / coarse: raw_unstd: {'lr': 0.1, 'weight_decay': 0.0005}, {'lr': 0.03, 'weight_decay': 0.0}, {'lr': 0.03, 'weight_decay': 0.0}; raw_std: {'lr': 0.03, 'weight_decay': 0.0005}, {'lr': 0.1, 'weight_decay': 0.0005}, {'lr': 0.03, 'weight_decay': 0.0005}; l2_std: {'lr': 0.03, 'weight_decay': 0.0005}, {'lr': 0.03, 'weight_decay': 0.0005}, {'lr': 0.03, 'weight_decay': 0.0005}
- A-P3 / fine: raw_unstd: {'lr': 0.1, 'weight_decay': 0.0}, {'lr': 0.1, 'weight_decay': 0.0}, {'lr': 0.1, 'weight_decay': 0.0}; raw_std: {'lr': 0.03, 'weight_decay': 0.0005}, {'lr': 0.03, 'weight_decay': 0.0005}, {'lr': 0.03, 'weight_decay': 0.0005}; l2_std: {'lr': 0.03, 'weight_decay': 0.0005}, {'lr': 0.03, 'weight_decay': 0.0005}, {'lr': 0.03, 'weight_decay': 0.0005}
- G2 / coarse: raw_unstd: {'lr': 0.3, 'weight_decay': 0.0005}, {'lr': 0.03, 'weight_decay': 0.0005}, {'lr': 0.03, 'weight_decay': 0.0}; raw_std: {'lr': 0.03, 'weight_decay': 0.0005}, {'lr': 0.03, 'weight_decay': 0.0005}, {'lr': 0.3, 'weight_decay': 0.0005}; l2_std: {'lr': 0.03, 'weight_decay': 0.0005}, {'lr': 0.03, 'weight_decay': 0.0005}, {'lr': 0.1, 'weight_decay': 0.0005}
- G2 / fine: raw_unstd: {'lr': 0.1, 'weight_decay': 0.0}, {'lr': 0.1, 'weight_decay': 0.0}, {'lr': 0.1, 'weight_decay': 0.0}; raw_std: {'lr': 0.03, 'weight_decay': 0.0005}, {'lr': 0.03, 'weight_decay': 0.0005}, {'lr': 0.03, 'weight_decay': 0.0005}; l2_std: {'lr': 0.03, 'weight_decay': 0.0005}, {'lr': 0.03, 'weight_decay': 0.0005}, {'lr': 0.03, 'weight_decay': 0.0005}
- recipe VCS / coarse: raw_unstd: {'lr': 0.03, 'weight_decay': 0.0}, {'lr': 0.3, 'weight_decay': 0.0005}, {'lr': 0.03, 'weight_decay': 0.0}; raw_std: {'lr': 0.1, 'weight_decay': 0.0005}, {'lr': 0.03, 'weight_decay': 0.0005}, {'lr': 0.03, 'weight_decay': 0.0005}; l2_std: {'lr': 0.03, 'weight_decay': 0.0005}, {'lr': 0.03, 'weight_decay': 0.0005}, {'lr': 0.03, 'weight_decay': 0.0005}
- recipe VCS / fine: raw_unstd: {'lr': 0.03, 'weight_decay': 0.0}, {'lr': 0.03, 'weight_decay': 0.0}, {'lr': 0.1, 'weight_decay': 0.0}; raw_std: {'lr': 0.03, 'weight_decay': 0.0005}, {'lr': 0.03, 'weight_decay': 0.0005}, {'lr': 0.03, 'weight_decay': 0.0005}; l2_std: {'lr': 0.03, 'weight_decay': 0.0005}, {'lr': 0.03, 'weight_decay': 0.0005}, {'lr': 0.03, 'weight_decay': 0.0005}
- SimCLR / coarse: raw_unstd: {'lr': 0.1, 'weight_decay': 0.0}, {'lr': 0.1, 'weight_decay': 0.0}, {'lr': 0.03, 'weight_decay': 0.0}; raw_std: {'lr': 0.03, 'weight_decay': 0.0005}, {'lr': 0.03, 'weight_decay': 0.0005}, {'lr': 0.3, 'weight_decay': 0.0005}; l2_std: {'lr': 0.03, 'weight_decay': 0.0005}, {'lr': 0.03, 'weight_decay': 0.0005}, {'lr': 0.03, 'weight_decay': 0.0005}
- SimCLR / fine: raw_unstd: {'lr': 0.3, 'weight_decay': 0.0}, {'lr': 0.1, 'weight_decay': 0.0}, {'lr': 0.1, 'weight_decay': 0.0}; raw_std: {'lr': 0.03, 'weight_decay': 0.0005}, {'lr': 0.03, 'weight_decay': 0.0005}, {'lr': 0.03, 'weight_decay': 0.0005}; l2_std: {'lr': 0.03, 'weight_decay': 0.0005}, {'lr': 0.03, 'weight_decay': 0.0005}, {'lr': 0.03, 'weight_decay': 0.0005}
