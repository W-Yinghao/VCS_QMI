# r5 final evaluation — results (test mode)

plan sha256 `f6d3ef59a0503ba45831118fac553a293abf03c35835e451d4eb78d89a252c68`; prediction manifest sha256 `2ed80e2cdd794070377f9bd2137fc100a5546f8ed1cc48af923522c9aca978a1`.

Evaluation sets (counts; ceiling = queries with >= 1 proposal at IoU >= 0.5):

| set | images | queries | images >= 2 referred | queries >= 2 | no proposal | ceiling all / >= 2 |
|---|---|---|---|---|---|---|
| refcoco_testA | 750 | 5657 | 715 | 5564 | 0 | 98.87 / 98.90 |
| refcoco_testB | 750 | 5095 | 691 | 4939 | 0 | 97.00 / 97.10 |
| refcocog_test | 2600 | 9602 | 1580 | 7622 | 0 | 97.28 / 97.18 |
| refcocoplus_testA | 750 | 5726 | 715 | 5626 | 0 | 98.85 / 98.88 |
| refcocoplus_testB | 750 | 4889 | 689 | 4716 | 0 | 97.44 / 97.56 |

## Primary confirmation — VQ − VCS, query-level Acc@0.5 over ALL official expressions (primary; uncovered / no proposal = error); Holm over the evaluation sets

| set | VQ − VCS | seed-paired 95 % t | image bootstrap 95 % | p (two-sided) | Holm p | decision |
|---|---|---|---|---|---|---|
| refcoco_testA | +5.57 | [+4.22, +6.92] | [+4.52, +6.69] | 0.0032 | 0.0127 | VQ improves VCS |
| refcoco_testB | +8.73 | [+7.90, +9.55] | [+7.20, +10.12] | 0.0005 | 0.0024 | VQ improves VCS |
| refcocog_test | +0.83 | [+0.11, +1.54] | [+0.23, +1.42] | 0.0381 | 0.1143 | no detectable difference (Holm) |
| refcocoplus_testA | +1.36 | [-0.09, +2.80] | [+0.65, +2.08] | 0.0561 | 0.1143 | no detectable difference (Holm) |
| refcocoplus_testB | +2.65 | [-0.95, +6.24] | [+1.68, +3.59] | 0.0871 | 0.1143 | no detectable difference (Holm) |

## Table 1 — query-level Acc@0.5 over ALL official expressions (primary; uncovered / no proposal = error)

| set | method | seed 0 | seed 1 | seed 2 | mean ± sd |
|---|---|---|---|---|---|
| refcoco_testA | D | 79.35 | 80.01 | 79.85 | 79.74 ± 0.34 |
| refcoco_testA | E | 80.45 | 79.95 | 79.85 | 80.08 ± 0.32 |
| refcoco_testA | JQ | 77.78 | 78.91 | 77.59 | 78.09 ± 0.72 |
| refcoco_testA | JS | 72.26 | 72.67 | 72.81 | 72.58 ± 0.28 |
| refcoco_testA | VCS | 73.24 | 73.61 | 72.37 | 73.07 ± 0.63 |
| refcoco_testA | VQ | 78.59 | 78.77 | 78.56 | 78.64 ± 0.11 |
| refcoco_testB | D | 67.79 | 66.36 | 68.03 | 67.39 ± 0.90 |
| refcoco_testB | E | 66.93 | 67.30 | 67.63 | 67.29 ± 0.35 |
| refcoco_testB | JQ | 64.63 | 66.14 | 65.69 | 65.49 ± 0.78 |
| refcoco_testB | JS | 56.68 | 56.60 | 55.74 | 56.34 ± 0.52 |
| refcoco_testB | VCS | 56.31 | 58.02 | 56.00 | 56.77 ± 1.09 |
| refcoco_testB | VQ | 65.32 | 66.38 | 64.81 | 65.50 ± 0.80 |
| refcocog_test | D | 75.23 | 74.53 | 74.71 | 74.82 ± 0.37 |
| refcocog_test | E | 74.32 | 74.99 | 74.66 | 74.66 ± 0.34 |
| refcocog_test | E+F | 75.95 | 75.91 | 75.52 | 75.79 ± 0.24 |
| refcocog_test | JNU | 74.60 | 75.47 | 74.76 | 74.94 ± 0.47 |
| refcocog_test | JNU+F | 74.65 | 75.47 | 75.15 | 75.09 ± 0.41 |
| refcocog_test | JQ | 73.79 | 73.75 | 73.45 | 73.66 ± 0.18 |
| refcocog_test | JS | 72.30 | 72.57 | 72.29 | 72.38 ± 0.16 |
| refcocog_test | JS+F | 74.51 | 74.83 | 74.42 | 74.59 ± 0.21 |
| refcocog_test | VCS | 72.57 | 73.15 | 72.59 | 72.77 ± 0.33 |
| refcocog_test | VCS+F | 74.64 | 75.60 | 74.70 | 74.98 ± 0.54 |
| refcocog_test | VNU | 75.36 | 74.99 | 75.46 | 75.27 ± 0.25 |
| refcocog_test | VNU+F | 75.76 | 75.27 | 75.45 | 75.49 ± 0.25 |
| refcocog_test | VQ | 73.08 | 74.05 | 73.66 | 73.60 ± 0.49 |
| refcocoplus_testA | D | 81.07 | 81.09 | 81.61 | 81.26 ± 0.31 |
| refcocoplus_testA | E | 81.77 | 80.77 | 81.35 | 81.30 ± 0.50 |
| refcocoplus_testA | E+F | 82.62 | 81.49 | 82.31 | 82.14 ± 0.59 |
| refcocoplus_testA | JNU | 81.73 | 82.08 | 81.77 | 81.86 ± 0.19 |
| refcocoplus_testA | JNU+F | 81.73 | 82.08 | 82.47 | 82.09 ± 0.37 |
| refcocoplus_testA | JQ | 81.26 | 80.49 | 80.07 | 80.61 ± 0.60 |
| refcocoplus_testA | JS | 79.11 | 78.34 | 79.39 | 78.95 ± 0.54 |
| refcocoplus_testA | JS+F | 81.37 | 80.75 | 82.22 | 81.45 ± 0.74 |
| refcocoplus_testA | VCS | 79.51 | 78.87 | 79.24 | 79.21 ± 0.32 |
| refcocoplus_testA | VCS+F | 81.65 | 81.44 | 81.59 | 81.56 ± 0.11 |
| refcocoplus_testA | VNU | 81.96 | 82.29 | 81.87 | 82.04 ± 0.22 |
| refcocoplus_testA | VNU+F | 82.87 | 82.61 | 81.87 | 82.45 ± 0.52 |
| refcocoplus_testA | VQ | 80.58 | 80.89 | 80.21 | 80.56 ± 0.34 |
| refcocoplus_testB | D | 64.10 | 63.26 | 63.82 | 63.73 ± 0.43 |
| refcocoplus_testB | E | 63.22 | 63.69 | 62.79 | 63.24 ± 0.45 |
| refcocoplus_testB | E+F | 64.00 | 64.00 | 63.69 | 63.90 ± 0.18 |
| refcocoplus_testB | JNU | 63.88 | 62.38 | 62.90 | 63.05 ± 0.76 |
| refcocoplus_testB | JNU+F | 63.88 | 62.38 | 63.02 | 63.09 ± 0.75 |
| refcocoplus_testB | JQ | 63.53 | 63.04 | 62.51 | 63.03 ± 0.51 |
| refcocoplus_testB | JS | 60.42 | 59.83 | 60.38 | 60.21 ± 0.33 |
| refcocoplus_testB | JS+F | 61.67 | 61.48 | 62.34 | 61.83 ± 0.45 |
| refcocoplus_testB | VCS | 59.40 | 61.01 | 61.71 | 60.71 ± 1.19 |
| refcocoplus_testB | VCS+F | 60.83 | 62.32 | 62.59 | 61.91 ± 0.95 |
| refcocoplus_testB | VNU | 63.59 | 63.18 | 63.22 | 63.33 ± 0.23 |
| refcocoplus_testB | VNU+F | 63.55 | 62.53 | 63.22 | 63.10 ± 0.52 |
| refcocoplus_testB | VQ | 63.41 | 63.82 | 62.83 | 63.35 ± 0.49 |

Contrasts on this metric (paired by seed; t = 95 % seed-paired t interval, df = n - 1; boot = 95 % image bootstrap, B = 1000)

| family | contrast | set | mean | t interval | boot interval | p (two-sided) | Holm p | reading (t / boot) |
|---|---|---|---|---|---|---|---|---|
| primary | VQ - VCS | refcoco_testA | +5.57 | [+4.22, +6.92] | [+4.52, +6.69] | 0.0032 | 0.0127 * | above 0 / above 0 |
| estimation | VQ - D | refcoco_testA | -1.10 | [-1.82, -0.37] | [-1.72, -0.37] | 0.0229 |  | below 0 / below 0 |
| estimation | VQ - E | refcoco_testA | -1.44 | [-2.34, -0.55] | [-2.32, -0.57] | 0.0202 |  | below 0 / below 0 |
| estimation | VQ - JQ | refcoco_testA | +0.55 | [-0.95, +2.04] | [-0.04, +1.12] | 0.2558 |  | contains 0 / contains 0 |
| secondary | VCS - E | refcoco_testA | -7.01 | [-8.48, -5.54] | [-8.08, -5.98] | 0.0024 |  | below 0 / below 0 |
| secondary | D - E | refcoco_testA | -0.35 | [-1.96, +1.26] | [-1.16, +0.43] | 0.4512 |  | contains 0 / contains 0 |
| primary | VQ - VCS | refcoco_testB | +8.73 | [+7.90, +9.55] | [+7.20, +10.12] | 0.0005 | 0.0024 * | above 0 / above 0 |
| estimation | VQ - D | refcoco_testB | -1.89 | [-6.10, +2.32] | [-2.78, -0.96] | 0.1932 |  | contains 0 / below 0 |
| estimation | VQ - E | refcoco_testB | -1.79 | [-4.18, +0.61] | [-2.89, -0.67] | 0.0849 |  | contains 0 / below 0 |
| estimation | VQ - JQ | refcoco_testB | +0.01 | [-2.00, +2.02] | [-0.71, +0.68] | 0.9802 |  | contains 0 / contains 0 |
| secondary | VCS - E | refcoco_testB | -10.51 | [-13.45, -7.58] | [-11.87, -8.99] | 0.0042 |  | below 0 / below 0 |
| secondary | D - E | refcoco_testB | +0.10 | [-2.22, +2.43] | [-0.81, +1.10] | 0.8644 |  | contains 0 / contains 0 |
| primary | VQ - VCS | refcocog_test | +0.83 | [+0.11, +1.54] | [+0.23, +1.42] | 0.0381 | 0.1143 | above 0 / above 0 |
| estimation | VQ - D | refcocog_test | -1.23 | [-3.35, +0.89] | [-1.71, -0.74] | 0.1298 |  | contains 0 / below 0 |
| estimation | VQ - E | refcocog_test | -1.06 | [-1.45, -0.68] | [-1.53, -0.54] | 0.0071 |  | below 0 / below 0 |
| estimation | VQ - JQ | refcocog_test | -0.07 | [-1.45, +1.32] | [-0.42, +0.31] | 0.8568 |  | contains 0 / contains 0 |
| secondary | VNU - E | refcocog_test | +0.61 | [-0.74, +1.97] | [-0.05, +1.30] | 0.1903 |  | contains 0 / contains 0 |
| secondary | VCS+F - E+F | refcocog_test | -0.81 | [-2.05, +0.43] | [-1.33, -0.30] | 0.1065 |  | contains 0 / below 0 |
| secondary | VNU - JNU | refcocog_test | +0.33 | [-1.41, +2.07] | [-0.03, +0.69] | 0.5007 |  | contains 0 / contains 0 |
| secondary | VCS - E | refcocog_test | -1.89 | [-2.30, -1.48] | [-2.46, -1.28] | 0.0026 |  | below 0 / below 0 |
| secondary | D - E | refcocog_test | +0.17 | [-1.57, +1.90] | [-0.26, +0.65] | 0.7200 |  | contains 0 / contains 0 |
| primary | VQ - VCS | refcocoplus_testA | +1.36 | [-0.09, +2.80] | [+0.65, +2.08] | 0.0561 | 0.1143 | contains 0 / above 0 |
| estimation | VQ - D | refcocoplus_testA | -0.69 | [-2.25, +0.87] | [-1.25, -0.13] | 0.1961 |  | contains 0 / below 0 |
| estimation | VQ - E | refcocoplus_testA | -0.73 | [-2.58, +1.11] | [-1.53, +0.05] | 0.2288 |  | contains 0 / contains 0 |
| estimation | VQ - JQ | refcocoplus_testA | -0.05 | [-1.45, +1.36] | [-0.51, +0.44] | 0.8995 |  | contains 0 / contains 0 |
| secondary | VNU - E | refcocoplus_testA | +0.75 | [-0.97, +2.46] | [-0.19, +1.61] | 0.2026 |  | contains 0 / contains 0 |
| secondary | VCS+F - E+F | refcocoplus_testA | -0.58 | [-1.77, +0.60] | [-1.35, +0.18] | 0.1689 |  | contains 0 / contains 0 |
| secondary | VNU - JNU | refcocoplus_testA | +0.18 | [+0.02, +0.34] | [-0.32, +0.63] | 0.0419 |  | above 0 / contains 0 |
| secondary | VCS - E | refcocoplus_testA | -2.09 | [-2.53, -1.65] | [-2.88, -1.29] | 0.0024 |  | below 0 / below 0 |
| secondary | D - E | refcocoplus_testA | -0.04 | [-1.46, +1.38] | [-0.75, +0.65] | 0.9128 |  | contains 0 / contains 0 |
| primary | VQ - VCS | refcocoplus_testB | +2.65 | [-0.95, +6.24] | [+1.68, +3.59] | 0.0871 | 0.1143 | contains 0 / above 0 |
| estimation | VQ - D | refcocoplus_testB | -0.37 | [-2.40, +1.65] | [-1.17, +0.38] | 0.5094 |  | contains 0 / contains 0 |
| estimation | VQ - E | refcocoplus_testB | +0.12 | [-0.06, +0.29] | [-0.73, +1.01] | 0.1077 |  | contains 0 / contains 0 |
| estimation | VQ - JQ | refcocoplus_testB | +0.33 | [-0.79, +1.45] | [-0.32, +0.99] | 0.3349 |  | contains 0 / contains 0 |
| secondary | VNU - E | refcocoplus_testB | +0.10 | [-1.21, +1.40] | [-1.02, +1.19] | 0.7832 |  | contains 0 / contains 0 |
| secondary | VCS+F - E+F | refcocoplus_testB | -1.98 | [-4.63, +0.67] | [-3.01, -0.99] | 0.0843 |  | contains 0 / below 0 |
| secondary | VNU - JNU | refcocoplus_testB | +0.28 | [-1.07, +1.63] | [-0.32, +0.87] | 0.4671 |  | contains 0 / contains 0 |
| secondary | VCS - E | refcocoplus_testB | -2.53 | [-5.95, +0.89] | [-3.45, -1.61] | 0.0862 |  | contains 0 / below 0 |
| secondary | D - E | refcocoplus_testB | +0.49 | [-1.50, +2.48] | [-0.32, +1.29] | 0.3994 |  | contains 0 / contains 0 |

## Table 2 — development image-macro Acc@0.5 on images with >= 2 referred objects (the DEV definition)

| set | method | seed 0 | seed 1 | seed 2 | mean ± sd |
|---|---|---|---|---|---|
| refcoco_testA | D | 81.48 | 81.81 | 81.73 | 81.67 ± 0.17 |
| refcoco_testA | E | 82.35 | 81.89 | 81.55 | 81.93 ± 0.40 |
| refcoco_testA | JQ | 80.03 | 81.18 | 79.69 | 80.30 ± 0.78 |
| refcoco_testA | JS | 74.54 | 74.76 | 75.03 | 74.78 ± 0.25 |
| refcoco_testA | VCS | 75.52 | 75.77 | 74.78 | 75.36 ± 0.51 |
| refcoco_testA | VQ | 80.83 | 80.85 | 80.90 | 80.86 ± 0.04 |
| refcoco_testB | D | 69.86 | 68.01 | 69.89 | 69.25 ± 1.08 |
| refcoco_testB | E | 69.12 | 69.11 | 69.99 | 69.41 ± 0.50 |
| refcoco_testB | JQ | 66.37 | 68.17 | 67.36 | 67.30 ± 0.90 |
| refcoco_testB | JS | 58.14 | 58.32 | 57.08 | 57.85 ± 0.67 |
| refcoco_testB | VCS | 57.64 | 59.85 | 57.41 | 58.30 ± 1.35 |
| refcoco_testB | VQ | 66.93 | 68.43 | 66.28 | 67.21 ± 1.10 |
| refcocog_test | D | 73.04 | 72.65 | 72.77 | 72.82 ± 0.20 |
| refcocog_test | E | 72.16 | 72.87 | 72.63 | 72.55 ± 0.37 |
| refcocog_test | E+F | 73.49 | 73.47 | 73.26 | 73.40 ± 0.12 |
| refcocog_test | JNU | 72.24 | 73.03 | 72.25 | 72.50 ± 0.46 |
| refcocog_test | JNU+F | 72.10 | 73.03 | 72.70 | 72.61 ± 0.47 |
| refcocog_test | JQ | 71.79 | 71.82 | 71.36 | 71.65 ± 0.26 |
| refcocog_test | JS | 70.11 | 70.42 | 70.17 | 70.23 ± 0.17 |
| refcocog_test | JS+F | 72.02 | 72.32 | 71.97 | 72.10 ± 0.19 |
| refcocog_test | VCS | 70.57 | 70.91 | 70.49 | 70.66 ± 0.22 |
| refcocog_test | VCS+F | 72.30 | 73.05 | 72.13 | 72.49 ± 0.49 |
| refcocog_test | VNU | 72.76 | 72.59 | 73.00 | 72.78 ± 0.21 |
| refcocog_test | VNU+F | 73.14 | 72.88 | 72.90 | 72.97 ± 0.14 |
| refcocog_test | VQ | 70.84 | 72.19 | 71.52 | 71.52 ± 0.67 |
| refcocoplus_testA | D | 82.48 | 82.75 | 83.24 | 82.83 ± 0.38 |
| refcocoplus_testA | E | 83.59 | 82.68 | 83.21 | 83.16 ± 0.46 |
| refcocoplus_testA | E+F | 84.45 | 83.42 | 84.19 | 84.02 ± 0.53 |
| refcocoplus_testA | JNU | 83.32 | 83.60 | 83.38 | 83.43 ± 0.15 |
| refcocoplus_testA | JNU+F | 83.32 | 83.60 | 84.18 | 83.70 ± 0.44 |
| refcocoplus_testA | JQ | 82.99 | 82.27 | 81.90 | 82.39 ± 0.56 |
| refcocoplus_testA | JS | 80.78 | 80.04 | 80.58 | 80.46 ± 0.38 |
| refcocoplus_testA | JS+F | 83.16 | 82.57 | 83.63 | 83.12 ± 0.53 |
| refcocoplus_testA | VCS | 81.21 | 80.31 | 80.90 | 80.80 ± 0.46 |
| refcocoplus_testA | VCS+F | 83.24 | 83.19 | 83.39 | 83.27 ± 0.10 |
| refcocoplus_testA | VNU | 83.37 | 83.88 | 83.25 | 83.50 ± 0.33 |
| refcocoplus_testA | VNU+F | 84.55 | 84.32 | 83.25 | 84.04 ± 0.69 |
| refcocoplus_testA | VQ | 82.35 | 82.73 | 81.79 | 82.29 ± 0.47 |
| refcocoplus_testB | D | 65.35 | 64.45 | 65.00 | 64.93 ± 0.45 |
| refcocoplus_testB | E | 64.28 | 65.05 | 63.73 | 64.35 ± 0.66 |
| refcocoplus_testB | E+F | 65.42 | 65.31 | 64.84 | 65.19 ± 0.31 |
| refcocoplus_testB | JNU | 65.23 | 63.41 | 63.95 | 64.20 ± 0.93 |
| refcocoplus_testB | JNU+F | 65.23 | 63.41 | 64.13 | 64.26 ± 0.92 |
| refcocoplus_testB | JQ | 64.60 | 64.25 | 63.65 | 64.17 ± 0.48 |
| refcocoplus_testB | JS | 61.11 | 61.00 | 61.30 | 61.14 ± 0.15 |
| refcocoplus_testB | JS+F | 62.84 | 62.91 | 63.61 | 63.12 ± 0.43 |
| refcocoplus_testB | VCS | 60.24 | 61.96 | 62.50 | 61.57 ± 1.18 |
| refcocoplus_testB | VCS+F | 61.67 | 63.40 | 63.81 | 62.96 ± 1.14 |
| refcocoplus_testB | VNU | 64.94 | 64.70 | 64.57 | 64.73 ± 0.19 |
| refcocoplus_testB | VNU+F | 64.91 | 64.00 | 64.57 | 64.49 ± 0.46 |
| refcocoplus_testB | VQ | 64.56 | 65.02 | 64.07 | 64.55 ± 0.47 |

Contrasts on this metric (paired by seed; t = 95 % seed-paired t interval, df = n - 1; boot = 95 % image bootstrap, B = 1000)

| family | contrast | set | mean | t interval | boot interval | p (two-sided) | Holm p | reading (t / boot) |
|---|---|---|---|---|---|---|---|---|
| primary | VQ - VCS | refcoco_testA | +5.50 | [+4.15, +6.86] | [+4.46, +6.61] | 0.0033 |  | above 0 / above 0 |
| estimation | VQ - D | refcoco_testA | -0.81 | [-1.20, -0.42] | [-1.46, -0.10] | 0.0125 |  | below 0 / below 0 |
| estimation | VQ - E | refcoco_testA | -1.07 | [-2.15, +0.02] | [-2.03, -0.15] | 0.0513 |  | contains 0 / below 0 |
| estimation | VQ - JQ | refcoco_testA | +0.56 | [-1.43, +2.55] | [-0.06, +1.16] | 0.3493 |  | contains 0 / contains 0 |
| secondary | VCS - E | refcoco_testA | -6.57 | [-7.55, -5.60] | [-7.67, -5.57] | 0.0012 |  | below 0 / below 0 |
| secondary | D - E | refcoco_testA | -0.26 | [-1.62, +1.11] | [-1.11, +0.55] | 0.5017 |  | contains 0 / contains 0 |
| primary | VQ - VCS | refcoco_testB | +8.92 | [+8.03, +9.80] | [+7.24, +10.36] | 0.0005 |  | above 0 / above 0 |
| estimation | VQ - D | refcoco_testB | -2.04 | [-7.39, +3.31] | [-2.99, -1.14] | 0.2431 |  | contains 0 / below 0 |
| estimation | VQ - E | refcoco_testB | -2.19 | [-5.95, +1.56] | [-3.38, -1.07] | 0.1285 |  | contains 0 / below 0 |
| estimation | VQ - JQ | refcoco_testB | -0.09 | [-2.25, +2.07] | [-0.84, +0.64] | 0.8752 |  | contains 0 / contains 0 |
| secondary | VCS - E | refcoco_testB | -11.11 | [-15.31, -6.91] | [-12.58, -9.46] | 0.0076 |  | below 0 / below 0 |
| secondary | D - E | refcoco_testB | -0.16 | [-2.43, +2.12] | [-1.14, +0.91] | 0.7971 |  | contains 0 / contains 0 |
| primary | VQ - VCS | refcocog_test | +0.86 | [-0.44, +2.17] | [+0.17, +1.59] | 0.1043 |  | contains 0 / above 0 |
| estimation | VQ - D | refcocog_test | -1.30 | [-3.46, +0.86] | [-1.87, -0.76] | 0.1220 |  | contains 0 / below 0 |
| estimation | VQ - E | refcocog_test | -1.04 | [-1.83, -0.24] | [-1.62, -0.44] | 0.0306 |  | below 0 / below 0 |
| estimation | VQ - JQ | refcocog_test | -0.14 | [-1.90, +1.63] | [-0.58, +0.31] | 0.7718 |  | contains 0 / contains 0 |
| secondary | VNU - E | refcocog_test | +0.23 | [-0.92, +1.37] | [-0.57, +1.08] | 0.4862 |  | contains 0 / contains 0 |
| secondary | VCS+F - E+F | refcocog_test | -0.91 | [-1.97, +0.15] | [-1.54, -0.32] | 0.0661 |  | contains 0 / below 0 |
| secondary | VNU - JNU | refcocog_test | +0.28 | [-1.30, +1.85] | [-0.19, +0.75] | 0.5310 |  | contains 0 / contains 0 |
| secondary | VCS - E | refcocog_test | -1.90 | [-2.60, -1.20] | [-2.59, -1.19] | 0.0073 |  | below 0 / below 0 |
| secondary | D - E | refcocog_test | +0.27 | [-1.14, +1.67] | [-0.25, +0.80] | 0.4994 |  | contains 0 / contains 0 |
| primary | VQ - VCS | refcocoplus_testA | +1.48 | [-0.55, +3.52] | [+0.72, +2.28] | 0.0887 |  | contains 0 / above 0 |
| estimation | VQ - D | refcocoplus_testA | -0.54 | [-2.51, +1.43] | [-1.14, +0.08] | 0.3600 |  | contains 0 / contains 0 |
| estimation | VQ - E | refcocoplus_testA | -0.87 | [-2.86, +1.12] | [-1.66, -0.05] | 0.2008 |  | contains 0 / below 0 |
| estimation | VQ - JQ | refcocoplus_testA | -0.10 | [-1.47, +1.27] | [-0.56, +0.38] | 0.7795 |  | contains 0 / contains 0 |
| secondary | VNU - E | refcocoplus_testA | +0.34 | [-1.53, +2.22] | [-0.63, +1.25] | 0.5131 |  | contains 0 / contains 0 |
| secondary | VCS+F - E+F | refcocoplus_testA | -0.75 | [-1.96, +0.47] | [-1.52, +0.02] | 0.1184 |  | contains 0 / contains 0 |
| secondary | VNU - JNU | refcocoplus_testA | +0.07 | [-0.43, +0.57] | [-0.42, +0.52] | 0.6099 |  | contains 0 / contains 0 |
| secondary | VCS - E | refcocoplus_testA | -2.35 | [-2.45, -2.25] | [-3.21, -1.50] | 0.0001 |  | below 0 / below 0 |
| secondary | D - E | refcocoplus_testA | -0.33 | [-2.00, +1.34] | [-1.10, +0.43] | 0.4823 |  | contains 0 / contains 0 |
| primary | VQ - VCS | refcocoplus_testB | +2.98 | [-0.43, +6.40] | [+1.93, +3.99] | 0.0640 |  | contains 0 / above 0 |
| estimation | VQ - D | refcocoplus_testB | -0.38 | [-2.44, +1.67] | [-1.26, +0.46] | 0.5073 |  | contains 0 / contains 0 |
| estimation | VQ - E | refcocoplus_testB | +0.19 | [-0.31, +0.69] | [-0.80, +1.18] | 0.2361 |  | contains 0 / contains 0 |
| estimation | VQ - JQ | refcocoplus_testB | +0.38 | [-0.64, +1.40] | [-0.35, +1.04] | 0.2499 |  | contains 0 / contains 0 |
| secondary | VNU - E | refcocoplus_testB | +0.38 | [-1.21, +1.98] | [-0.85, +1.59] | 0.4127 |  | contains 0 / contains 0 |
| secondary | VCS+F - E+F | refcocoplus_testB | -2.23 | [-5.69, +1.22] | [-3.39, -1.19] | 0.1087 |  | contains 0 / below 0 |
| secondary | VNU - JNU | refcocoplus_testB | +0.54 | [-1.43, +2.51] | [-0.11, +1.18] | 0.3601 |  | contains 0 / contains 0 |
| secondary | VCS - E | refcocoplus_testB | -2.79 | [-6.34, +0.76] | [-3.91, -1.74] | 0.0775 |  | contains 0 / below 0 |
| secondary | D - E | refcocoplus_testB | +0.58 | [-1.98, +3.13] | [-0.33, +1.44] | 0.4336 |  | contains 0 / contains 0 |

## Table 3 — query-level Acc@0.5 on the >= 2-referred-objects subset

| set | method | seed 0 | seed 1 | seed 2 | mean ± sd |
|---|---|---|---|---|---|
| refcoco_testA | D | 79.58 | 80.14 | 79.94 | 79.89 ± 0.28 |
| refcoco_testA | E | 80.52 | 80.09 | 79.98 | 80.19 ± 0.29 |
| refcoco_testA | JQ | 77.93 | 79.15 | 77.71 | 78.27 ± 0.78 |
| refcoco_testA | JS | 72.41 | 72.84 | 73.01 | 72.75 ± 0.31 |
| refcoco_testA | VCS | 73.45 | 73.80 | 72.54 | 73.26 ± 0.65 |
| refcoco_testA | VQ | 78.67 | 78.86 | 78.83 | 78.79 ± 0.11 |
| refcoco_testB | D | 67.71 | 66.07 | 67.89 | 67.22 ± 1.00 |
| refcoco_testB | E | 66.94 | 67.06 | 67.71 | 67.23 ± 0.41 |
| refcoco_testB | JQ | 64.35 | 66.03 | 65.60 | 65.32 ± 0.87 |
| refcoco_testB | JS | 56.33 | 56.39 | 55.48 | 56.06 ± 0.51 |
| refcoco_testB | VCS | 56.06 | 57.60 | 55.80 | 56.49 ± 0.97 |
| refcoco_testB | VQ | 64.97 | 66.33 | 64.53 | 65.28 ± 0.94 |
| refcocog_test | D | 72.91 | 72.24 | 72.37 | 72.51 ± 0.35 |
| refcocog_test | E | 71.94 | 72.58 | 72.36 | 72.29 ± 0.33 |
| refcocog_test | E+F | 73.25 | 73.10 | 72.92 | 73.09 ± 0.16 |
| refcocog_test | JNU | 72.07 | 72.89 | 72.19 | 72.38 ± 0.45 |
| refcocog_test | JNU+F | 71.92 | 72.89 | 72.53 | 72.45 ± 0.49 |
| refcocog_test | JQ | 71.44 | 71.29 | 71.06 | 71.26 ± 0.19 |
| refcocog_test | JS | 69.81 | 70.10 | 69.89 | 69.93 ± 0.15 |
| refcocog_test | JS+F | 71.82 | 71.98 | 71.78 | 71.86 ± 0.10 |
| refcocog_test | VCS | 70.28 | 70.80 | 70.22 | 70.43 ± 0.32 |
| refcocog_test | VCS+F | 72.08 | 72.91 | 72.02 | 72.33 ± 0.50 |
| refcocog_test | VNU | 72.79 | 72.59 | 72.99 | 72.79 ± 0.20 |
| refcocog_test | VNU+F | 73.10 | 72.76 | 72.79 | 72.89 ± 0.19 |
| refcocog_test | VQ | 70.48 | 71.78 | 71.18 | 71.14 ± 0.65 |
| refcocoplus_testA | D | 81.14 | 81.16 | 81.71 | 81.34 ± 0.32 |
| refcocoplus_testA | E | 81.85 | 80.89 | 81.57 | 81.44 ± 0.49 |
| refcocoplus_testA | E+F | 82.69 | 81.60 | 82.46 | 82.25 ± 0.57 |
| refcocoplus_testA | JNU | 81.78 | 82.14 | 81.83 | 81.92 ± 0.19 |
| refcocoplus_testA | JNU+F | 81.78 | 82.14 | 82.53 | 82.15 ± 0.37 |
| refcocoplus_testA | JQ | 81.43 | 80.66 | 80.27 | 80.79 ± 0.59 |
| refcocoplus_testA | JS | 79.29 | 78.51 | 79.52 | 79.11 ± 0.53 |
| refcocoplus_testA | JS+F | 81.48 | 80.87 | 82.33 | 81.56 ± 0.73 |
| refcocoplus_testA | VCS | 79.70 | 79.01 | 79.43 | 79.38 ± 0.35 |
| refcocoplus_testA | VCS+F | 81.76 | 81.57 | 81.75 | 81.69 ± 0.11 |
| refcocoplus_testA | VNU | 82.05 | 82.35 | 81.92 | 82.11 ± 0.22 |
| refcocoplus_testA | VNU+F | 82.97 | 82.65 | 81.92 | 82.52 ± 0.54 |
| refcocoplus_testA | VQ | 80.80 | 81.09 | 80.36 | 80.75 ± 0.37 |
| refcocoplus_testB | D | 63.87 | 63.13 | 63.66 | 63.55 ± 0.38 |
| refcocoplus_testB | E | 62.93 | 63.53 | 62.68 | 63.05 ± 0.44 |
| refcocoplus_testB | E+F | 63.76 | 63.80 | 63.61 | 63.73 ± 0.10 |
| refcocoplus_testB | JNU | 63.63 | 62.07 | 62.66 | 62.79 ± 0.79 |
| refcocoplus_testB | JNU+F | 63.63 | 62.07 | 62.66 | 62.79 ± 0.79 |
| refcocoplus_testB | JQ | 63.15 | 62.68 | 62.21 | 62.68 ± 0.47 |
| refcocoplus_testB | JS | 60.20 | 59.67 | 60.16 | 60.01 ± 0.29 |
| refcocoplus_testB | JS+F | 61.37 | 61.24 | 62.13 | 61.58 ± 0.48 |
| refcocoplus_testB | VCS | 59.12 | 60.90 | 61.39 | 60.47 ± 1.19 |
| refcocoplus_testB | VCS+F | 60.52 | 62.09 | 62.26 | 61.62 ± 0.96 |
| refcocoplus_testB | VNU | 63.40 | 63.04 | 62.93 | 63.13 ± 0.24 |
| refcocoplus_testB | VNU+F | 63.32 | 62.28 | 62.93 | 62.84 ± 0.53 |
| refcocoplus_testB | VQ | 63.15 | 63.66 | 62.55 | 63.12 ± 0.55 |

Contrasts on this metric (paired by seed; t = 95 % seed-paired t interval, df = n - 1; boot = 95 % image bootstrap, B = 1000)

| family | contrast | set | mean | t interval | boot interval | p (two-sided) | Holm p | reading (t / boot) |
|---|---|---|---|---|---|---|---|---|
| primary | VQ - VCS | refcoco_testA | +5.52 | [+3.86, +7.18] | [+4.51, +6.68] | 0.0048 |  | above 0 / above 0 |
| estimation | VQ - D | refcoco_testA | -1.10 | [-1.55, -0.66] | [-1.73, -0.38] | 0.0088 |  | below 0 / below 0 |
| estimation | VQ - E | refcoco_testA | -1.41 | [-2.37, -0.45] | [-2.28, -0.51] | 0.0241 |  | below 0 / below 0 |
| estimation | VQ - JQ | refcoco_testA | +0.52 | [-1.28, +2.32] | [-0.07, +1.10] | 0.3394 |  | contains 0 / contains 0 |
| secondary | VCS - E | refcoco_testA | -6.93 | [-8.39, -5.47] | [-8.04, -5.87] | 0.0024 |  | below 0 / below 0 |
| secondary | D - E | refcoco_testA | -0.31 | [-1.66, +1.05] | [-1.10, +0.47] | 0.4351 |  | contains 0 / contains 0 |
| primary | VQ - VCS | refcoco_testB | +8.79 | [+8.53, +9.05] | [+7.22, +10.18] | 0.0000 |  | above 0 / above 0 |
| estimation | VQ - D | refcoco_testB | -1.94 | [-6.76, +2.87] | [-2.83, -0.99] | 0.2243 |  | contains 0 / below 0 |
| estimation | VQ - E | refcoco_testB | -1.96 | [-5.00, +1.09] | [-3.06, -0.79] | 0.1095 |  | contains 0 / below 0 |
| estimation | VQ - JQ | refcoco_testB | -0.05 | [-2.29, +2.20] | [-0.77, +0.64] | 0.9361 |  | contains 0 / contains 0 |
| secondary | VCS - E | refcoco_testB | -10.74 | [-13.80, -7.69] | [-12.14, -9.21] | 0.0043 |  | below 0 / below 0 |
| secondary | D - E | refcoco_testB | -0.01 | [-2.24, +2.21] | [-0.93, +1.02] | 0.9816 |  | contains 0 / contains 0 |
| primary | VQ - VCS | refcocog_test | +0.71 | [-0.40, +1.82] | [+0.01, +1.42] | 0.1099 |  | contains 0 / above 0 |
| estimation | VQ - D | refcocog_test | -1.36 | [-3.83, +1.11] | [-1.92, -0.78] | 0.1414 |  | contains 0 / below 0 |
| estimation | VQ - E | refcocog_test | -1.15 | [-1.96, -0.33] | [-1.72, -0.58] | 0.0265 |  | below 0 / below 0 |
| estimation | VQ - JQ | refcocog_test | -0.12 | [-1.98, +1.75] | [-0.54, +0.33] | 0.8107 |  | contains 0 / contains 0 |
| secondary | VNU - E | refcocog_test | +0.50 | [-0.58, +1.58] | [-0.24, +1.31] | 0.1855 |  | contains 0 / contains 0 |
| secondary | VCS+F - E+F | refcocog_test | -0.76 | [-2.00, +0.49] | [-1.36, -0.18] | 0.1208 |  | contains 0 / below 0 |
| secondary | VNU - JNU | refcocog_test | +0.41 | [-1.12, +1.93] | [-0.04, +0.84] | 0.3705 |  | contains 0 / contains 0 |
| secondary | VCS - E | refcocog_test | -1.86 | [-2.48, -1.23] | [-2.51, -1.18] | 0.0060 |  | below 0 / below 0 |
| secondary | D - E | refcocog_test | +0.21 | [-1.47, +1.90] | [-0.27, +0.77] | 0.6393 |  | contains 0 / contains 0 |
| primary | VQ - VCS | refcocoplus_testA | +1.37 | [-0.18, +2.91] | [+0.62, +2.11] | 0.0625 |  | contains 0 / above 0 |
| estimation | VQ - D | refcocoplus_testA | -0.59 | [-2.26, +1.09] | [-1.16, -0.05] | 0.2713 |  | contains 0 / below 0 |
| estimation | VQ - E | refcocoplus_testA | -0.69 | [-2.60, +1.22] | [-1.51, +0.11] | 0.2616 |  | contains 0 / contains 0 |
| estimation | VQ - JQ | refcocoplus_testA | -0.04 | [-1.37, +1.29] | [-0.52, +0.43] | 0.9189 |  | contains 0 / contains 0 |
| secondary | VNU - E | refcocoplus_testA | +0.67 | [-1.04, +2.38] | [-0.29, +1.55] | 0.2335 |  | contains 0 / contains 0 |
| secondary | VCS+F - E+F | refcocoplus_testA | -0.56 | [-1.71, +0.60] | [-1.31, +0.21] | 0.1731 |  | contains 0 / contains 0 |
| secondary | VNU - JNU | refcocoplus_testA | +0.19 | [-0.04, +0.42] | [-0.33, +0.65] | 0.0692 |  | contains 0 / contains 0 |
| secondary | VCS - E | refcocoplus_testA | -2.06 | [-2.43, -1.69] | [-2.84, -1.23] | 0.0017 |  | below 0 / below 0 |
| secondary | D - E | refcocoplus_testA | -0.10 | [-1.42, +1.22] | [-0.83, +0.62] | 0.7742 |  | contains 0 / contains 0 |
| primary | VQ - VCS | refcocoplus_testB | +2.65 | [-0.91, +6.21] | [+1.66, +3.67] | 0.0853 |  | contains 0 / above 0 |
| estimation | VQ - D | refcocoplus_testB | -0.43 | [-2.55, +1.69] | [-1.27, +0.38] | 0.4741 |  | contains 0 / contains 0 |
| estimation | VQ - E | refcocoplus_testB | +0.07 | [-0.37, +0.51] | [-0.80, +1.01] | 0.5598 |  | contains 0 / contains 0 |
| estimation | VQ - JQ | refcocoplus_testB | +0.44 | [-0.79, +1.67] | [-0.23, +1.11] | 0.2650 |  | contains 0 / contains 0 |
| secondary | VNU - E | refcocoplus_testB | +0.08 | [-1.17, +1.32] | [-1.07, +1.19] | 0.8133 |  | contains 0 / contains 0 |
| secondary | VCS+F - E+F | refcocoplus_testB | -2.11 | [-4.60, +0.38] | [-3.16, -1.12] | 0.0678 |  | contains 0 / below 0 |
| secondary | VNU - JNU | refcocoplus_testB | +0.34 | [-1.17, +1.85] | [-0.27, +0.95] | 0.4350 |  | contains 0 / contains 0 |
| secondary | VCS - E | refcocoplus_testB | -2.58 | [-5.72, +0.56] | [-3.57, -1.63] | 0.0714 |  | contains 0 / below 0 |
| secondary | D - E | refcocoplus_testB | +0.50 | [-1.45, +2.45] | [-0.31, +1.32] | 0.3829 |  | contains 0 / contains 0 |

Primary family: primary contrast VQ - VCS on query_all over the evaluation sets; Holm at alpha = 0.05 over m = 5 evaluation sets (* = rejected).

Rows of the proposal without a checkpoint / lambda_CAL (listed, never filled):

- JNU refcoco seed 0: never trained (no checkpoint)
- JNU refcoco seed 1: never trained (no checkpoint)
- JNU refcoco seed 2: never trained (no checkpoint)
- VNU refcoco seed 0: never trained (no checkpoint)
- VNU refcoco seed 1: never trained (no checkpoint)
- VNU refcoco seed 2: never trained (no checkpoint)
- D+F refcoco seed 0: no lambda_CAL (not part of R5-10 F1)
- D+F refcoco seed 1: no lambda_CAL (not part of R5-10 F1)
- D+F refcoco seed 2: no lambda_CAL (not part of R5-10 F1)
- E+F refcoco seed 0: no lambda_CAL (R5-10 F1 covered RefCOCOg / RefCOCO+ only)
- E+F refcoco seed 1: no lambda_CAL (R5-10 F1 covered RefCOCOg / RefCOCO+ only)
- E+F refcoco seed 2: no lambda_CAL (R5-10 F1 covered RefCOCOg / RefCOCO+ only)
- JQ+F refcoco seed 0: no lambda_CAL (not part of R5-10 F1)
- JQ+F refcoco seed 1: no lambda_CAL (not part of R5-10 F1)
- JQ+F refcoco seed 2: no lambda_CAL (not part of R5-10 F1)
- JS+F refcoco seed 0: no lambda_CAL (R5-10 F1 covered RefCOCOg / RefCOCO+ only)
- JS+F refcoco seed 1: no lambda_CAL (R5-10 F1 covered RefCOCOg / RefCOCO+ only)
- JS+F refcoco seed 2: no lambda_CAL (R5-10 F1 covered RefCOCOg / RefCOCO+ only)
- VCS+F refcoco seed 0: no lambda_CAL (R5-10 F1 covered RefCOCOg / RefCOCO+ only)
- VCS+F refcoco seed 1: no lambda_CAL (R5-10 F1 covered RefCOCOg / RefCOCO+ only)
- VCS+F refcoco seed 2: no lambda_CAL (R5-10 F1 covered RefCOCOg / RefCOCO+ only)
- VQ+F refcoco seed 0: no lambda_CAL (not part of R5-10 F1)
- VQ+F refcoco seed 1: no lambda_CAL (not part of R5-10 F1)
- VQ+F refcoco seed 2: no lambda_CAL (not part of R5-10 F1)
- D+F refcocog seed 0: no lambda_CAL (not part of R5-10 F1)
- D+F refcocog seed 1: no lambda_CAL (not part of R5-10 F1)
- D+F refcocog seed 2: no lambda_CAL (not part of R5-10 F1)
- JQ+F refcocog seed 0: no lambda_CAL (not part of R5-10 F1)
- JQ+F refcocog seed 1: no lambda_CAL (not part of R5-10 F1)
- JQ+F refcocog seed 2: no lambda_CAL (not part of R5-10 F1)
- VQ+F refcocog seed 0: no lambda_CAL (not part of R5-10 F1)
- VQ+F refcocog seed 1: no lambda_CAL (not part of R5-10 F1)
- VQ+F refcocog seed 2: no lambda_CAL (not part of R5-10 F1)
- D+F refcocoplus seed 0: no lambda_CAL (not part of R5-10 F1)
- D+F refcocoplus seed 1: no lambda_CAL (not part of R5-10 F1)
- D+F refcocoplus seed 2: no lambda_CAL (not part of R5-10 F1)
- JQ+F refcocoplus seed 0: no lambda_CAL (not part of R5-10 F1)
- JQ+F refcocoplus seed 1: no lambda_CAL (not part of R5-10 F1)
- JQ+F refcocoplus seed 2: no lambda_CAL (not part of R5-10 F1)
- VQ+F refcocoplus seed 0: no lambda_CAL (not part of R5-10 F1)
- VQ+F refcocoplus seed 1: no lambda_CAL (not part of R5-10 F1)
- VQ+F refcocoplus seed 2: no lambda_CAL (not part of R5-10 F1)

