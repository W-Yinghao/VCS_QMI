# R4-VL1 — proposal-aware full fine-tuning (option A): results

Gates (step-0 given-box DEV within 0.3 of the frozen raw value): 9 / 9 pass.  Missing runs: none.

## refcoco

| quantity | vcs | js | softmax |
|---|---|---|---|
| DEV Acc@0.5, proposal-trained (primary) | 66.60 ± 0.63 | 66.54 ± 0.45 | 75.45 ± 0.30 |
| Table C (given-box-trained VL3, same proposals) | 56.56 ± 0.81 | 55.82 ± 0.17 | 57.76 ± 0.54 |
| adaptation gain (paired) | +10.04 [+7.45, +12.63] | +10.72 [+9.88, +11.56] | +17.69 [+16.63, +18.75] |
| zero-shot (step 0) | 28.69 ± 0.00 | 28.69 ± 0.00 | 28.67 ± 0.02 |
| Acc@0.75 | 53.69 ± 0.48 | 53.51 ± 0.08 | 62.53 ± 0.35 |
| query-level Acc@0.5 | 65.15 ± 0.43 | 65.13 ± 0.51 | 74.00 ± 0.13 |
| given-box Top-1 of the new encoder | 75.42 ± 0.43 | 74.51 ± 0.52 | 83.33 ± 0.21 |
| DEV J_recal (same p / q) | 61.00 ± 0.90 | 61.73 ± 1.65 | 58.20 ± 0.34 |
| margin (×100 cos) | 5.89 ± 0.42 | 5.35 ± 0.84 | 13.75 ± 0.96 |
| top-K, new encoder (K on CAL) | 67.07 ± 0.61 (K [32, 32, 32]) | 66.87 ± 0.49 (K [32, 32, 32]) | 75.59 ± 0.21 (K [32, 32, 32]) |
| top-K, VL3 checkpoint (K on CAL) | — | — | — |
| selected epoch | [8, 10, 7] | [8, 8, 4] | [5, 7, 5] |

**vcs_minus_softmax** -8.84 [-10.57, -7.12] **vcs_minus_js** +0.06 [-1.36, +1.49]
