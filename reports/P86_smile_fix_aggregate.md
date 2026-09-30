# P85 aggregate — 85 cells loaded from /home/infres/yinwang/CS_QMI/outputs/P85_estim_benchmark_smile_fix

## Axis `pad` — values [2, 10, 20, 50, 100]

### Same-target error (S): posterior MSE E_M(T − η)² (mean ± sd over seeds)

| method | 2 | 10 | 20 | 50 | 100 |
|---|---|---|---|---|---|

### Native error |value − own truth| on each estimator's own scale (not comparable across targets)

| method | estimand | 2 | 10 | 20 | 50 | 100 |
|---|---|---|---|---|---|---|
| neural:smile:inbatch | MI | 0.0326 ± 0.0118 (n=3) | 0.1167 ± 0.0064 (n=3) | 0.1984 ± 0.0041 (n=3) | 0.5715 ± 0.0388 (n=3) | 1.4071 ± 0.0170 (n=3) |
| neural:smile:product | MI | 0.1001 ± 0.0748 (n=3) | 1.6785 ± 0.3193 (n=3) | 2.6847 ± 0.3089 (n=3) | 2.3091 ± 0.5579 (n=3) | 1.5029 ± 0.0001 (n=3) |

### Cost: fit seconds / peak memory MB (mean over seeds)

| method | 2 | 10 | 20 | 50 | 100 |
|---|---|---|---|---|---|
| neural:smile:inbatch | 3.4 s / 2156 MB | 3.4 s / 2284 MB | 3.5 s / 2444 MB | 3.6 s / 2924 MB | 3.9 s / 3726 MB |
| neural:smile:product | 2.3 s / 131 MB | 2.3 s / 136 MB | 2.3 s / 141 MB | 2.3 s / 156 MB | 2.3 s / 180 MB |

## Axis `nsize` — values [256, 1024, 4096, 16384]

### Same-target error (S): posterior MSE E_M(T − η)² (mean ± sd over seeds)

| method | 256 | 1024 | 4096 | 16384 |
|---|---|---|---|---|

### Native error |value − own truth| on each estimator's own scale (not comparable across targets)

| method | estimand | 256 | 1024 | 4096 | 16384 |
|---|---|---|---|---|---|
| neural:smile:inbatch | MI | 1.8459 ± 0.0903 (n=3) | 0.8261 ± 0.0497 (n=3) | 0.2717 ± 0.0567 (n=3) | 0.0884 ± 0.0572 (n=3) |
| neural:smile:product | MI | 0.6870 ± 0.5458 (n=3) | 5.9998 ± 0.7152 (n=3) | 11.1292 ± 0.4277 (n=3) | 4.2074 ± 0.3297 (n=3) |

### Cost: fit seconds / peak memory MB (mean over seeds)

| method | 256 | 1024 | 4096 | 16384 |
|---|---|---|---|---|
| neural:smile:inbatch | 3.3 s / 2444 MB | 3.3 s / 2444 MB | 3.5 s / 2444 MB | 4.0 s / 2451 MB |
| neural:smile:product | 2.3 s / 141 MB | 2.2 s / 141 MB | 2.3 s / 141 MB | 2.3 s / 141 MB |

## Axis `batch_equal_updates` — values [64, 256, 1024]

### Same-target error (S): posterior MSE E_M(T − η)² (mean ± sd over seeds)

| method | 64 | 256 | 1024 |
|---|---|---|---|

### Native error |value − own truth| on each estimator's own scale (not comparable across targets)

| method | estimand | 64 | 256 | 1024 |
|---|---|---|---|---|
| neural:smile:inbatch | MI | 0.2138 ± 0.1219 (n=3) | 0.2717 ± 0.0567 (n=3) | 0.2363 ± 0.0124 (n=3) |
| neural:smile:product | MI | 5.9671 ± 0.1763 (n=3) | 11.1292 ± 0.4277 (n=3) | 13.5516 ± 1.2607 (n=3) |

### Cost: fit seconds / peak memory MB (mean over seeds)

| method | 64 | 256 | 1024 |
|---|---|---|---|
| neural:smile:inbatch | 2.2 s / 2444 MB | 3.5 s / 2444 MB | 40.9 s / 4338 MB |
| neural:smile:product | 2.2 s / 141 MB | 2.3 s / 141 MB | 2.3 s / 141 MB |

## Axis `batch_equal_exposure` — values [64, 256, 1024]

### Same-target error (S): posterior MSE E_M(T − η)² (mean ± sd over seeds)

| method | 64 | 256 | 1024 |
|---|---|---|---|

### Native error |value − own truth| on each estimator's own scale (not comparable across targets)

| method | estimand | 64 | 256 | 1024 |
|---|---|---|---|---|
| neural:smile:inbatch | MI | 0.2673 ± 0.1281 (n=3) | 0.2717 ± 0.0567 (n=3) | 0.2101 ± 0.0575 (n=3) |
| neural:smile:product | MI | 13.6838 ± 0.6693 (n=3) | 11.1292 ± 0.4277 (n=3) | 7.8104 ± 0.5340 (n=3) |

### Cost: fit seconds / peak memory MB (mean over seeds)

| method | 64 | 256 | 1024 |
|---|---|---|---|
| neural:smile:inbatch | 8.6 s / 2444 MB | 3.5 s / 2444 MB | 10.2 s / 4338 MB |
| neural:smile:product | 9.0 s / 141 MB | 2.3 s / 141 MB | 0.6 s / 141 MB |

## Axis `dependence` — values [('cubic', 2.0), ('cubic', 4.0), ('cubic', 6.0), ('cubic', 8.0), ('cubic', 10.0), ('gaussian', 2.0), ('gaussian', 4.0), ('gaussian', 6.0), ('gaussian', 8.0), ('gaussian', 10.0), ('xor_mixture', 2.0), ('xor_mixture', 4.0), ('xor_mixture', 6.0), ('xor_mixture', 8.0), ('xor_mixture', 10.0)]

### Same-target error (S): posterior MSE E_M(T − η)² (mean ± sd over seeds)

| method | ('cubic', 2.0) | ('cubic', 4.0) | ('cubic', 6.0) | ('cubic', 8.0) | ('cubic', 10.0) | ('gaussian', 2.0) | ('gaussian', 4.0) | ('gaussian', 6.0) | ('gaussian', 8.0) | ('gaussian', 10.0) | ('xor_mixture', 2.0) | ('xor_mixture', 4.0) | ('xor_mixture', 6.0) | ('xor_mixture', 8.0) | ('xor_mixture', 10.0) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|

### Native error |value − own truth| on each estimator's own scale (not comparable across targets)

| method | estimand | ('cubic', 2.0) | ('cubic', 4.0) | ('cubic', 6.0) | ('cubic', 8.0) | ('cubic', 10.0) | ('gaussian', 2.0) | ('gaussian', 4.0) | ('gaussian', 6.0) | ('gaussian', 8.0) | ('gaussian', 10.0) | ('xor_mixture', 2.0) | ('xor_mixture', 4.0) | ('xor_mixture', 6.0) | ('xor_mixture', 8.0) | ('xor_mixture', 10.0) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| neural:smile:inbatch | MI | 1.0044 ± 0.0085 (n=3) | 1.7170 ± 0.0842 (n=3) | 1.7936 ± 0.3324 (n=3) | 1.6924 ± 0.0411 (n=3) | 2.1247 ± 0.0576 (n=3) | 0.3297 ± 0.0268 (n=3) | 0.2717 ± 0.0567 (n=3) | 1.7967 ± 0.1200 (n=3) | 3.3515 ± 0.4325 (n=3) | 3.3856 ± 0.2916 (n=3) | 1.4617 ± 0.0304 (n=3) | 2.1497 ± 0.1398 (n=3) | 2.3458 ± 0.4768 (n=3) | 2.5296 ± 0.5118 (n=3) | 2.9640 ± 0.7471 (n=3) |
| neural:smile:product | MI | 1.3924 ± 0.2113 (n=3) | 5.1743 ± 0.2934 (n=3) | 7.1595 ± 0.4951 (n=3) | 7.7207 ± 0.5627 (n=3) | 6.9025 ± 0.6597 (n=3) | 4.7498 ± 0.2204 (n=3) | 11.1292 ± 0.4277 (n=3) | 12.2407 ± 0.8517 (n=3) | 12.0905 ± 2.2724 (n=3) | 9.3161 ± 1.3624 (n=3) | 2.0056 ± 0.0037 (n=3) | 1.3222 ± 0.3522 (n=3) | 1.2824 ± 0.3712 (n=3) | 1.7847 ± 0.4794 (n=3) | 3.3087 ± 0.4644 (n=3) |

### Cost: fit seconds / peak memory MB (mean over seeds)

| method | ('cubic', 2.0) | ('cubic', 4.0) | ('cubic', 6.0) | ('cubic', 8.0) | ('cubic', 10.0) | ('gaussian', 2.0) | ('gaussian', 4.0) | ('gaussian', 6.0) | ('gaussian', 8.0) | ('gaussian', 10.0) | ('xor_mixture', 2.0) | ('xor_mixture', 4.0) | ('xor_mixture', 6.0) | ('xor_mixture', 8.0) | ('xor_mixture', 10.0) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| neural:smile:inbatch | 3.5 s / 2444 MB | 3.5 s / 2444 MB | 3.5 s / 2444 MB | 3.5 s / 2445 MB | 3.5 s / 2445 MB | 3.5 s / 2444 MB | 3.5 s / 2444 MB | 3.5 s / 2444 MB | 3.5 s / 2444 MB | 3.5 s / 2445 MB | 3.4 s / 2284 MB | 3.4 s / 2284 MB | 3.4 s / 2284 MB | 3.4 s / 2284 MB | 3.4 s / 2284 MB |
| neural:smile:product | 2.3 s / 141 MB | 2.3 s / 141 MB | 2.3 s / 141 MB | 2.3 s / 141 MB | 2.2 s / 141 MB | 2.3 s / 141 MB | 2.3 s / 141 MB | 2.2 s / 141 MB | 2.3 s / 141 MB | 2.3 s / 141 MB | 2.2 s / 135 MB | 2.2 s / 136 MB | 2.2 s / 136 MB | 2.2 s / 136 MB | 2.2 s / 136 MB |

### Ordering probability P(V_{l+1} > V_l) between adjacent levels (unit bootstrap pooled over seeds; 0.5 = no resolution) and |Δmean| / pooled boot sd

**gaussian**

| method | 2→4 | 4→6 | 6→8 | 8→10 |
|---|---|---|---|---|
| neural:smile:inbatch | 1.00 / 61.0 | 1.00 / 42.5 | 1.00 / 13.6 | 1.00 / 6.7 |
| neural:smile:product | 1.00 / 29.3 | 1.00 / 5.7 | 0.78 / 1.3 | 0.43 / 0.5 |

**cubic**

| method | 2→4 | 4→6 | 6→8 | 8→10 |
|---|---|---|---|---|
| neural:smile:inbatch | 1.00 / 25.9 | 1.00 / 9.7 | 1.00 / 10.8 | 1.00 / 30.8 |
| neural:smile:product | 1.00 / 26.2 | 1.00 / 11.8 | 1.00 / 5.9 | 0.94 / 2.3 |

**xor_mixture**

| method | 2→4 | 4→6 | 6→8 | 8→10 |
|---|---|---|---|---|
| neural:smile:inbatch | 1.00 / 15.9 | 1.00 / 6.3 | 1.00 / 4.5 | 1.00 / 3.0 |
| neural:smile:product | 1.00 / 13.1 | 1.00 / 6.9 | 1.00 / 4.2 | 0.72 / 1.2 |


## Cells outside the axes (pilot / smoke)

### P85_gaussian_ds2_dt100_I1.5_N16384_B256_U2000_s0 — device cuda:0, wall 23 s, S = 0.5411, methods smile

| method | estimand | native value | own truth | abs err | J_eval − S | posterior MSE | fit s | eval s | peak MB |
|---|---|---|---|---|---|---|---|---|---|
| neural:smile:product:lr0.002 | MI | 3.6701 | 1.5000 | 2.1701 | — | — | 2.3 | 0.1 | 182 |
| neural:smile:inbatch:lr0.002 | MI | 1.1396 | 1.5000 | 0.3604 | — | — | 4.7 | 0.4 | 3754 |

channel derivative (rho = 0.8814022009568447): oracle FD 2.110086176453913 ± 0.012925251001503636; envelope {'value': 2.1415374791682456, 'se': 0.026395173304397633}; learned: 

### P85_gaussian_ds2_dt100_I1.5_N256_B256_U2000_s0 — device cuda:0, wall 20 s, S = 0.5402, methods smile

| method | estimand | native value | own truth | abs err | J_eval − S | posterior MSE | fit s | eval s | peak MB |
|---|---|---|---|---|---|---|---|---|---|
| neural:smile:product:lr0.0001 | MI | -0.0037 | 1.5000 | 1.5037 | — | — | 2.3 | 0.1 | 180 |
| neural:smile:inbatch:lr0.0001 | MI | -0.0087 | 1.5000 | 1.5087 | — | — | 3.7 | 0.4 | 3725 |

channel derivative (rho = 0.8814022009568447): oracle FD 2.1271673224076824 ± 0.012944048454980704; envelope {'value': 2.0914197607712937, 'se': 0.025899794160290052}; learned: 

### P85_gaussian_ds2_dt2_I1.5_N16384_B256_U2000_s0 — device cuda:0, wall 20 s, S = 0.5394, methods smile

| method | estimand | native value | own truth | abs err | J_eval − S | posterior MSE | fit s | eval s | peak MB |
|---|---|---|---|---|---|---|---|---|---|
| neural:smile:product:lr0.002 | MI | 1.4335 | 1.5000 | 0.0665 | — | — | 2.3 | 0.1 | 132 |
| neural:smile:inbatch:lr0.002 | MI | 1.4804 | 1.5000 | 0.0196 | — | — | 3.9 | 0.3 | 2158 |

channel derivative (rho = 0.8814022009568447): oracle FD 2.1224963694079175 ± 0.012928339470005159; envelope {'value': 2.1156112106729936, 'se': 0.025875750875112537}; learned: 

### P85_gaussian_ds2_dt2_I1.5_N256_B256_U2000_s0 — device cuda:0, wall 18 s, S = 0.5405, methods smile

| method | estimand | native value | own truth | abs err | J_eval − S | posterior MSE | fit s | eval s | peak MB |
|---|---|---|---|---|---|---|---|---|---|
| neural:smile:product:lr0.002 | MI | 4.1410 | 1.5000 | 2.6410 | — | — | 2.3 | 0.1 | 132 |
| neural:smile:inbatch:lr0.0001 | MI | 1.3884 | 1.5000 | 0.1116 | — | — | 3.2 | 0.3 | 2155 |

channel derivative (rho = 0.8814022009568447): oracle FD 2.1303528536808627 ± 0.012960270831748085; envelope {'value': 2.0921922620274085, 'se': 0.025933159054311684}; learned: 

## QC

- oracle J on TRUTH within 3 se of S: 85 / 85 cells
- non-finite training steps (sum over rows): 0
- missing cells per axis: {'pad': 0, 'nsize': 0, 'batch_equal_updates': 0, 'batch_equal_exposure': 0, 'dependence': 0}

