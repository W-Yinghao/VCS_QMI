# P155 — paired contrasts on the official CIFAR-10 test (pre-stated rule; script output)

| family | contrast | metric | n | mean | 95 % t interval | per seed | reading |
|---|---|---|---|---|---|---|---|
| primary | A-P3 - SimCLR | test_linear | 5 | +0.63 | [+0.36, +0.90] | +0.76 / +0.86 / +0.54 / +0.68 / +0.30 | ahead |
| secondary | A-P3 - SimCLR | test_knn | 5 | -0.32 | [-0.94, +0.30] | +0.40 / -0.44 / -0.94 / -0.10 / -0.50 | no detectable difference |
| secondary | A-P3 - JS-AP3 | test_linear | 3 | -0.07 | [-0.57, +0.43] | +0.12 / -0.28 / -0.04 | no detectable difference |
| secondary | A-P3 - JS-AP3 | test_knn | 3 | -0.38 | [-0.78, +0.02] | -0.20 / -0.50 / -0.45 | no detectable difference |

## Test minus selection, per run

| run | linear | kNN |
|---|---|---|
| P107_AP3_views4_800ep_seed0 | -0.22 | +0.52 |
| P107_AP3_views4_800ep_seed1 | +0.01 | +0.12 |
| P107_AP3_views4_800ep_seed2 | -0.31 | +0.18 |
| P107_AP3_views4_800ep_seed3 | -0.11 | +0.66 |
| P107_AP3_views4_800ep_seed4 | -0.33 | +0.24 |
| P41_simclr_views4_800ep_seed0 | -0.12 | +0.22 |
| P41_simclr_views4_800ep_seed1 | +0.01 | +0.28 |
| P41_simclr_views4_800ep_seed2 | -0.45 | +0.50 |
| P111_simclr_views4_800ep_seed3 | -0.07 | -0.14 |
| P111_simclr_views4_800ep_seed4 | +0.11 | +0.18 |
| P114_JSAP3_views4_800ep_seed0 | +0.00 | +0.52 |
| P114_JSAP3_views4_800ep_seed1 | -0.05 | +0.12 |
| P114_JSAP3_views4_800ep_seed2 | +0.37 | +0.45 |
