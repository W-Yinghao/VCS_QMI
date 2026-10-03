# P124 — v6 V6-GRANULARITY results (frozen CIFAR-100 SSL encoders; selection split; official test never opened)

Readouts identical for every method; PRIMARY = `recipe_raw` (the original frozen-h linear readout).  Conditional = coarse label given at test
time, 5-way within the coarse group (dedicated probe per group) — never an unconditional 100-way accuracy.  mean ± sd over encoder seeds.

## Anchor: fine 100-way `recipe_raw` vs the stored training-evaluation linear accuracy (must agree within GPU noise, |Δ| ≤ 0.3)

| run | stored | recomputed | Δ |
|---|---|---|---|
| P107_AP3_c100_views4_800ep_seed0 | 60.20 | 60.20 | +0.00 |

## coarse accuracy (%)

| readout | A-P3 |
|---|---|
| recipe_raw (primary) | 72.06 |
| raw_unstd | 72.18 |
| raw_std | 71.94 |
| l2_std | 71.76 |
| knn | 72.08 |

## fine accuracy (%)

| readout | A-P3 |
|---|---|
| recipe_raw (primary) | 60.20 |
| raw_unstd | 60.20 |
| raw_std | 59.58 |
| l2_std | 59.16 |
| knn | 55.92 |

## conditional_macro accuracy (%)

| readout | A-P3 |
|---|---|
| recipe_raw (primary) | 76.84 |
| raw_unstd | 76.94 |
| raw_std | 73.08 |
| l2_std | 73.08 |
| knn | 72.78 |

## conditional_overall accuracy (%)

| readout | A-P3 |
|---|---|
| recipe_raw (primary) | 76.84 |
| raw_unstd | 76.94 |
| raw_std | 73.08 |
| l2_std | 73.08 |
| knn | 72.78 |

## Seed-paired contrasts (95 % t interval, df = n − 1; labels: close |Δ| < 0.3 / clear (|Δ| ≥ 0.3 and interval excludes 0) / inconclusive)

| contrast | readout | Δ coarse | Δ fine | Δ conditional macro | Δ conditional overall |
|---|---|---|---|---|---|

## Pattern statement (frozen rule, PRIMARY readout only; Δcoarse vs Δconditional-macro labels)


## 100-way recipe head: implied-coarse accuracy × fine accuracy given implied coarse correct (= fine accuracy); masked-head conditional

| family | implied coarse | fine given coarse correct | fine | masked-head conditional macro | masked-head overall |
|---|---|---|---|---|---|
| A-P3 | 73.34 | 82.08 | 60.20 | 76.74 | 76.74 |

## Chosen probe (lr, wd) per family and readout (selected readouts; grid lr 0.03 / 0.1 / 0.3 × wd 0 / 5e-4)

- A-P3 / coarse: raw_unstd: {'lr': 0.1, 'weight_decay': 0.0005}; raw_std: {'lr': 0.03, 'weight_decay': 0.0005}; l2_std: {'lr': 0.03, 'weight_decay': 0.0005}
- A-P3 / fine: raw_unstd: {'lr': 0.1, 'weight_decay': 0.0}; raw_std: {'lr': 0.03, 'weight_decay': 0.0005}; l2_std: {'lr': 0.03, 'weight_decay': 0.0005}
