# P141 — CIFAR-100 four-site (layer3 / h / r / z) readouts: results

2 encoders.  Primary readout recipe_raw (P124 rule); conditional = macro 5-way given the coarse label.  h is the backbone metric.

## 1. Per method and site (mean ± sd over seeds)

| method | site | dim | n | coarse (recipe_raw) | coarse (raw_std) | coarse (knn) | fine (recipe_raw) | fine (raw_std) | fine (knn) | conditional (recipe_raw) | conditional (raw_std) | conditional (knn) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A-P3 | layer3 | 256 | 1 | 48.00 ± 0.00 | 46.00 ± 0.00 | 32.20 ± 0.00 | 31.60 ± 0.00 | 34.60 ± 0.00 | 18.60 ± 0.00 | 58.85 ± 0.00 | 59.09 ± 0.00 | 44.81 ± 0.00 |
| A-P3 | h | 512 | 1 | 63.20 ± 0.00 | 62.80 ± 0.00 | 64.80 ± 0.00 | 49.20 ± 0.00 | 48.20 ± 0.00 | 46.80 ± 0.00 | 64.84 ± 0.00 | 63.57 ± 0.00 | 64.95 ± 0.00 |
| A-P3 | r | 128 | 1 | 58.80 ± 0.00 | 61.60 ± 0.00 | 64.40 ± 0.00 | 41.40 ± 0.00 | 45.60 ± 0.00 | 45.80 ± 0.00 | 59.36 ± 0.00 | 63.92 ± 0.00 | 62.27 ± 0.00 |
| A-P3 | z | 128 | 1 | 61.80 ± 0.00 | 62.60 ± 0.00 | 64.40 ± 0.00 | 33.00 ± 0.00 | 44.00 ± 0.00 | 45.80 ± 0.00 | 64.21 ± 0.00 | 63.93 ± 0.00 | 62.27 ± 0.00 |
| SimCLR | layer3 | 256 | 1 | 45.40 ± 0.00 | 48.40 ± 0.00 | 31.40 ± 0.00 | 32.00 ± 0.00 | 34.00 ± 0.00 | 19.00 ± 0.00 | 58.79 ± 0.00 | 58.92 ± 0.00 | 43.74 ± 0.00 |
| SimCLR | h | 512 | 1 | 64.60 ± 0.00 | 60.80 ± 0.00 | 67.80 ± 0.00 | 46.40 ± 0.00 | 43.00 ± 0.00 | 48.00 ± 0.00 | 62.69 ± 0.00 | 60.58 ± 0.00 | 62.70 ± 0.00 |
| SimCLR | r | 128 | 1 | 59.80 ± 0.00 | 63.00 ± 0.00 | 63.80 ± 0.00 | 42.80 ± 0.00 | 47.20 ± 0.00 | 45.40 ± 0.00 | 58.99 ± 0.00 | 59.33 ± 0.00 | 60.01 ± 0.00 |
| SimCLR | z | 128 | 1 | 63.20 ± 0.00 | 64.40 ± 0.00 | 63.80 ± 0.00 | 33.00 ± 0.00 | 47.20 ± 0.00 | 45.40 ± 0.00 | 63.03 ± 0.00 | 58.69 ± 0.00 | 60.01 ± 0.00 |

## 2. Seed-paired contrasts, recipe_raw (A-P3 − JS-AP3 and A-P3 − SimCLR frozen labels; vs tuned JS descriptive)

| contrast | site | task | seeds | mean Δ | 95 % CI | label |
|---|---|---|---|---|---|---|
| A-P3 − SimCLR | layer3 | coarse | 1 | +2.60 | [+nan, +nan] | single seed |
| A-P3 − SimCLR | layer3 | fine | 1 | -0.40 | [+nan, +nan] | single seed |
| A-P3 − SimCLR | layer3 | conditional | 1 | +0.06 | [+nan, +nan] | single seed |
| A-P3 − SimCLR | h | coarse | 1 | -1.40 | [+nan, +nan] | single seed |
| A-P3 − SimCLR | h | fine | 1 | +2.80 | [+nan, +nan] | single seed |
| A-P3 − SimCLR | h | conditional | 1 | +2.15 | [+nan, +nan] | single seed |
| A-P3 − SimCLR | r | coarse | 1 | -1.00 | [+nan, +nan] | single seed |
| A-P3 − SimCLR | r | fine | 1 | -1.40 | [+nan, +nan] | single seed |
| A-P3 − SimCLR | r | conditional | 1 | +0.37 | [+nan, +nan] | single seed |
| A-P3 − SimCLR | z | coarse | 1 | -1.40 | [+nan, +nan] | single seed |
| A-P3 − SimCLR | z | fine | 1 | +0.00 | [+nan, +nan] | single seed |
| A-P3 − SimCLR | z | conditional | 1 | +1.18 | [+nan, +nan] | single seed |

## QC

max |h − P124-cached h| per run (expected ~0): P107_AP3_c100_views4_800ep_seed0: 3.34e-06, P91_c100_simclr_views4_800ep_seed0: 4.29e-06
