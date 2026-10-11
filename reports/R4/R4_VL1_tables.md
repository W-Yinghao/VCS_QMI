# R4-VL1 — proposal-aware full fine-tuning (option A): results

Gates (step-0 given-box DEV within 0.3 of the frozen raw value): 18 / 18 pass.  Missing runs: none.

## refcocog

| quantity | vcs | js | softmax |
|---|---|---|---|
| DEV Acc@0.5, proposal-trained (primary) | 71.73 ± 0.21 | 71.36 ± 0.27 | 73.58 ± 0.36 |
| Table C (given-box-trained VL3, same proposals) | 56.22 ± 0.79 | 57.59 ± 0.46 | 56.86 ± 0.21 |
| adaptation gain (paired) | +15.51 [+13.71, +17.31] | +13.77 [+12.43, +15.10] | +16.72 [+16.01, +17.42] |
| zero-shot (step 0) | 42.65 ± 0.11 | 42.59 ± 0.00 | 42.59 ± 0.00 |
| Acc@0.75 | 55.13 ± 0.26 | 55.12 ± 0.12 | 58.87 ± 0.30 |
| query-level Acc@0.5 | 71.44 ± 0.17 | 71.10 ± 0.17 | 73.24 ± 0.34 |
| given-box Top-1 of the new encoder | 81.47 ± 0.70 | 80.72 ± 0.13 | 82.16 ± 0.09 |
| DEV J_recal (same p / q) | 60.64 ± 0.80 | 60.03 ± 0.43 | 56.39 ± 0.44 |
| margin (×100 cos) | 7.33 ± 0.57 | 7.83 ± 0.24 | 9.34 ± 0.76 |
| top-K, new encoder (K on CAL) | 72.20 ± 0.29 (K [32, 32, 32]) | 71.75 ± 0.20 (K [32, 32, 32]) | 73.41 ± 0.37 (K [32, 32, 32]) |
| top-K, VL3 checkpoint (K on CAL) | 63.27 ± 0.38 (K [8, 8, 8]) | 63.91 ± 0.07 (K [8, 8, 8]) | 63.23 ± 0.31 (K [8, 8, 8]) |
| selected epoch | [5, 6, 8] | [8, 7, 7] | [2, 3, 2] |

**vcs_minus_softmax** -1.85 [-2.41, -1.29] **vcs_minus_js** +0.38 [-0.60, +1.36]

## refcocoplus

| quantity | vcs | js | softmax |
|---|---|---|---|
| DEV Acc@0.5, proposal-trained (primary) | 73.03 ± 0.34 | 72.86 ± 0.16 | 76.25 ± 0.45 |
| Table C (given-box-trained VL3, same proposals) | 53.46 ± 0.66 | 53.43 ± 0.28 | 55.66 ± 0.37 |
| adaptation gain (paired) | +19.56 [+18.20, +20.93] | +19.43 [+18.99, +19.87] | +20.59 [+19.24, +21.93] |
| zero-shot (step 0) | 32.73 ± 0.00 | 32.72 ± 0.03 | 32.70 ± 0.03 |
| Acc@0.75 | 58.61 ± 0.30 | 58.42 ± 0.31 | 62.91 ± 0.35 |
| query-level Acc@0.5 | 72.16 ± 0.33 | 71.88 ± 0.20 | 75.34 ± 0.44 |
| given-box Top-1 of the new encoder | 81.66 ± 0.52 | 81.22 ± 0.65 | 83.23 ± 0.70 |
| DEV J_recal (same p / q) | 64.03 ± 0.72 | 64.13 ± 0.89 | 59.32 ± 0.79 |
| margin (×100 cos) | 7.29 ± 0.72 | 6.92 ± 0.79 | 12.86 ± 0.16 |
| top-K, new encoder (K on CAL) | 73.89 ± 0.40 (K [16, 16, 16]) | 73.88 ± 0.31 (K [16, 16, 16]) | 76.30 ± 0.50 (K [32, 32, 32]) |
| top-K, VL3 checkpoint (K on CAL) | 66.45 ± 0.60 (K [8, 8, 8]) | 66.22 ± 0.21 (K [8, 8, 8]) | 65.59 ± 0.35 (K [8, 8, 8]) |
| selected epoch | [4, 6, 7] | [4, 7, 5] | [4, 4, 4] |

**vcs_minus_softmax** -3.22 [-5.08, -1.37] **vcs_minus_js** +0.17 [-0.43, +0.77]

Pooled over the two datasets (descriptive): vcs_minus_softmax -2.54; vcs_minus_js +0.27
