# P124 — v6 V6-GRANULARITY results (frozen CIFAR-100 SSL encoders; selection split; official test never opened)

Readouts identical for every method; PRIMARY = `recipe_raw` (the original frozen-h linear readout).  Conditional = coarse label given at test
time, 5-way within the coarse group (dedicated probe per group) — never an unconditional 100-way accuracy.  mean ± sd over encoder seeds.

## Anchor: fine 100-way `recipe_raw` vs the stored training-evaluation linear accuracy (must agree within GPU noise, |Δ| ≤ 0.3)

| run | stored | recomputed | Δ |
|---|---|---|---|

## coarse accuracy (%)

| readout | A-P3 | SimCLR |
|---|---|---|
| recipe_raw (primary) | 63.20 | 64.60 |
| raw_unstd | 63.80 | 63.20 |
| raw_std | 62.80 | 60.80 |
| l2_std | 62.60 | 63.40 |
| knn | 64.80 | 67.80 |

## fine accuracy (%)

| readout | A-P3 | SimCLR |
|---|---|---|
| recipe_raw (primary) | 49.20 | 46.40 |
| raw_unstd | 49.40 | 46.20 |
| raw_std | 48.20 | 43.00 |
| l2_std | 47.60 | 48.00 |
| knn | 46.80 | 48.00 |

## conditional_macro accuracy (%)

| readout | A-P3 | SimCLR |
|---|---|---|
| recipe_raw (primary) | 64.84 | 62.69 |
| raw_unstd | 64.90 | 63.44 |
| raw_std | 63.57 | 60.58 |
| l2_std | 64.01 | 61.69 |
| knn | 64.95 | 62.70 |

## conditional_overall accuracy (%)

| readout | A-P3 | SimCLR |
|---|---|---|
| recipe_raw (primary) | 65.60 | 63.40 |
| raw_unstd | 65.60 | 64.20 |
| raw_std | 64.20 | 61.00 |
| l2_std | 64.80 | 62.00 |
| knn | 65.80 | 63.80 |

## Seed-paired contrasts (95 % t interval, df = n − 1; labels: close |Δ| < 0.3 / clear (|Δ| ≥ 0.3 and interval excludes 0) / inconclusive)

| contrast | readout | Δ coarse | Δ fine | Δ conditional macro | Δ conditional overall |
|---|---|---|---|---|---|
| A-P3 − SimCLR | recipe_raw (primary) | — | — | — | — |
| A-P3 − SimCLR | raw_unstd | — | — | — | — |
| A-P3 − SimCLR | raw_std | — | — | — | — |
| A-P3 − SimCLR | l2_std | — | — | — | — |
| A-P3 − SimCLR | knn | — | — | — | — |

## Pattern statement (frozen rule, PRIMARY readout only; Δcoarse vs Δconditional-macro labels)


## 100-way recipe head: implied-coarse accuracy × fine accuracy given implied coarse correct (= fine accuracy); masked-head conditional

| family | implied coarse | fine given coarse correct | fine | masked-head conditional macro | masked-head overall |
|---|---|---|---|---|---|
| A-P3 | 65.20 | 75.46 | 49.20 | 66.21 | 67.00 |
| SimCLR | 62.80 | 73.89 | 46.40 | 65.45 | 66.00 |

## Chosen probe (lr, wd) per family and readout (selected readouts; grid lr 0.03 / 0.1 / 0.3 × wd 0 / 5e-4)

- A-P3 / coarse: raw_unstd: {'lr': 0.3, 'weight_decay': 0.0005}; raw_std: {'lr': 0.03, 'weight_decay': 0.0}; l2_std: {'lr': 0.03, 'weight_decay': 0.0}
- A-P3 / fine: raw_unstd: {'lr': 0.1, 'weight_decay': 0.0005}; raw_std: {'lr': 0.3, 'weight_decay': 0.0005}; l2_std: {'lr': 0.03, 'weight_decay': 0.0}
- SimCLR / coarse: raw_unstd: {'lr': 0.3, 'weight_decay': 0.0}; raw_std: {'lr': 0.03, 'weight_decay': 0.0}; l2_std: {'lr': 0.03, 'weight_decay': 0.0}
- SimCLR / fine: raw_unstd: {'lr': 0.3, 'weight_decay': 0.0}; raw_std: {'lr': 0.3, 'weight_decay': 0.0}; l2_std: {'lr': 0.1, 'weight_decay': 0.0}
