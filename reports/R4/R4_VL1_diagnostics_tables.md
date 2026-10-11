# R4_VL1_diagnostics — candidate diagnostics, r4vl1 checkpoints (exploratory; items 3–4 oracle-assisted)

`oracle_nd_<n>_acc` is the frozen reading (queries with fewer than n distractors are scored against all they have); `_frac_full` = fraction of covered queries with >= n distractors; `_acc_strict` = restricted to those queries.  Contrasts: VCS − other, per-seed differences of per-image means, image bootstrap 95 %.

## refcocog (1356 images, 6442 expressions); gates pass 9 / 9; missing score files: none

| quantity | vcs (n=3) | js (n=3) | softmax (n=3) |
|---|---|---|---|
| acc_all_image_macro | 71.75 ± 0.17 | 71.34 ± 0.28 | 73.61 ± 0.35 |
| coverage_image_macro | 97.28 ± 0.00 | 97.28 ± 0.00 | 97.28 ± 0.00 |
| acc_covered_image_macro | 73.59 ± 0.12 | 73.10 ± 0.29 | 75.60 ± 0.40 |
| topK_4_acc | 63.72 ± 0.11 | 63.47 ± 0.29 | 63.95 ± 0.17 |
| topK_4_coverage | 75.76 ± 0.00 | 75.76 ± 0.00 | 75.76 ± 0.00 |
| topK_8_acc | 69.69 ± 0.22 | 69.28 ± 0.18 | 70.16 ± 0.09 |
| topK_8_coverage | 86.71 ± 0.00 | 86.71 ± 0.00 | 86.71 ± 0.00 |
| topK_16_acc | 71.86 ± 0.23 | 71.47 ± 0.27 | 72.64 ± 0.24 |
| topK_16_coverage | 92.70 ± 0.00 | 92.70 ± 0.00 | 92.70 ± 0.00 |
| topK_32_acc | 72.20 ± 0.29 | 71.75 ± 0.24 | 73.50 ± 0.42 |
| topK_32_coverage | 95.82 ± 0.00 | 95.82 ± 0.00 | 95.82 ± 0.00 |
| topK_None_acc | 71.75 ± 0.17 | 71.34 ± 0.28 | 73.61 ± 0.35 |
| topK_None_coverage | 97.28 ± 0.00 | 97.28 ± 0.00 | 97.28 ± 0.00 |
| oracle_nd_1_acc | 96.79 ± 0.09 | 96.66 ± 0.19 | 97.03 ± 0.08 |
| oracle_nd_1_acc_strict | 96.79 ± 0.09 | 96.66 ± 0.19 | 97.03 ± 0.08 |
| oracle_nd_1_frac_full | 100.00 ± 0.00 | 100.00 ± 0.00 | 100.00 ± 0.00 |
| oracle_nd_3_acc | 91.70 ± 0.30 | 91.53 ± 0.13 | 92.62 ± 0.25 |
| oracle_nd_3_acc_strict | 91.88 ± 0.28 | 91.69 ± 0.15 | 92.80 ± 0.26 |
| oracle_nd_3_frac_full | 98.59 ± 0.00 | 98.59 ± 0.00 | 98.59 ± 0.00 |
| oracle_nd_7_acc | 85.60 ± 0.37 | 85.35 ± 0.15 | 86.64 ± 0.24 |
| oracle_nd_7_acc_strict | 86.52 ± 0.39 | 86.23 ± 0.25 | 87.53 ± 0.14 |
| oracle_nd_7_frac_full | 93.03 ± 0.00 | 93.03 ± 0.00 | 93.03 ± 0.00 |
| oracle_nd_15_acc | 79.60 ± 0.37 | 79.59 ± 0.08 | 80.85 ± 0.33 |
| oracle_nd_15_acc_strict | 80.99 ± 0.62 | 81.10 ± 0.18 | 82.52 ± 0.27 |
| oracle_nd_15_frac_full | 79.54 ± 0.00 | 79.54 ± 0.00 | 79.54 ± 0.00 |
| oracle_nd_31_acc | 74.12 ± 0.21 | 73.80 ± 0.20 | 75.95 ± 0.40 |
| oracle_nd_31_acc_strict | 75.61 ± 0.49 | 75.41 ± 0.49 | 77.76 ± 0.31 |
| oracle_nd_31_frac_full | 56.75 ± 0.00 | 56.75 ± 0.00 | 56.75 ± 0.00 |
| oracle_nd_None_acc | 69.56 ± 0.08 | 69.22 ± 0.37 | 72.15 ± 0.73 |
| oracle_nd_None_acc_strict | 69.56 ± 0.08 | 69.22 ± 0.37 | 72.15 ± 0.73 |
| oracle_nd_None_frac_full | 100.00 ± 0.00 | 100.00 ± 0.00 | 100.00 ± 0.00 |
| 2x2_A+C | 82.07 ± 0.67 | 81.52 ± 0.24 | 82.85 ± 0.17 |
| 2x2_B+C | 80.69 ± 0.15 | 80.17 ± 0.35 | 80.68 ± 0.32 |
| 2x2_A+D | 88.25 ± 0.52 | 87.60 ± 0.22 | 89.02 ± 0.14 |
| 2x2_B+D | 87.11 ± 0.28 | 86.46 ± 0.36 | 87.24 ± 0.26 |
| margin | 7.52 ± 0.57 | 8.05 ± 0.25 | 9.54 ± 0.84 |
| err_other_same_cat | 15.17 ± 0.31 | 15.67 ± 0.08 | 15.31 ± 0.23 |
| err_other_diff_cat | 1.52 ± 0.20 | 1.64 ± 0.17 | 1.47 ± 0.31 |
| err_part_or_duplicate | 8.71 ± 0.59 | 8.62 ± 0.15 | 7.17 ± 0.57 |
| err_no_matching_annotation | 2.85 ± 0.03 | 2.73 ± 0.22 | 2.44 ± 0.07 |
| acc_all_query | 71.46 ± 0.13 | 71.09 ± 0.05 | 73.26 ± 0.36 |

| contrast | mean (×100) | image bootstrap 95 % | per seed |
|---|---|---|---|
| vcs_minus_softmax/acc_all | -1.86 | [-2.64, -1.15] | -1.66, -1.80, -2.12 |
| vcs_minus_softmax/coverage | +0.00 | [+0.00, +0.00] | +0.00, +0.00, +0.00 |
| vcs_minus_softmax/acc_covered | -2.02 | [-2.79, -1.30] | -1.80, -1.87, -2.38 |
| vcs_minus_softmax/topK_4_acc | -0.23 | [-0.77, +0.26] | -0.19, -0.20, -0.30 |
| vcs_minus_softmax/topK_4_coverage | +0.00 | [+0.00, +0.00] | +0.00, +0.00, +0.00 |
| vcs_minus_softmax/topK_8_acc | -0.47 | [-1.11, +0.17] | -0.63, -0.38, -0.39 |
| vcs_minus_softmax/topK_8_coverage | +0.00 | [+0.00, +0.00] | +0.00, +0.00, +0.00 |
| vcs_minus_softmax/topK_16_acc | -0.78 | [-1.42, -0.11] | -0.75, -0.79, -0.80 |
| vcs_minus_softmax/topK_16_coverage | +0.00 | [+0.00, +0.00] | +0.00, +0.00, +0.00 |
| vcs_minus_softmax/topK_32_acc | -1.31 | [-2.06, -0.62] | -1.07, -1.29, -1.55 |
| vcs_minus_softmax/topK_32_coverage | +0.00 | [+0.00, +0.00] | +0.00, +0.00, +0.00 |
| vcs_minus_softmax/topK_None_acc | -1.86 | [-2.66, -1.16] | -1.66, -1.80, -2.12 |
| vcs_minus_softmax/topK_None_coverage | +0.00 | [+0.00, +0.00] | +0.00, +0.00, +0.00 |
| vcs_minus_softmax/oracle_nd_1_acc | -0.24 | [-0.54, +0.09] | -0.22, -0.26, -0.23 |
| vcs_minus_softmax/oracle_nd_1_acc_strict | -0.24 | [-0.53, +0.06] | -0.22, -0.26, -0.23 |
| vcs_minus_softmax/oracle_nd_1_frac_full | +0.00 | [+0.00, +0.00] | +0.00, +0.00, +0.00 |
| vcs_minus_softmax/oracle_nd_3_acc | -0.92 | [-1.44, -0.46] | -1.12, -0.40, -1.25 |
| vcs_minus_softmax/oracle_nd_3_acc_strict | -0.92 | [-1.41, -0.45] | -1.12, -0.37, -1.26 |
| vcs_minus_softmax/oracle_nd_3_frac_full | +0.00 | [+0.00, +0.00] | +0.00, +0.00, +0.00 |
| vcs_minus_softmax/oracle_nd_7_acc | -1.04 | [-1.64, -0.42] | -1.41, -0.34, -1.36 |
| vcs_minus_softmax/oracle_nd_7_acc_strict | -1.01 | [-1.66, -0.45] | -1.36, -0.40, -1.26 |
| vcs_minus_softmax/oracle_nd_7_frac_full | +0.00 | [+0.00, +0.00] | +0.00, +0.00, +0.00 |
| vcs_minus_softmax/oracle_nd_15_acc | -1.25 | [-1.99, -0.57] | -1.10, -0.73, -1.93 |
| vcs_minus_softmax/oracle_nd_15_acc_strict | -1.53 | [-2.43, -0.70] | -1.57, -0.70, -2.33 |
| vcs_minus_softmax/oracle_nd_15_frac_full | +0.00 | [+0.00, +0.00] | +0.00, +0.00, +0.00 |
| vcs_minus_softmax/oracle_nd_31_acc | -1.83 | [-2.60, -1.11] | -1.76, -1.28, -2.44 |
| vcs_minus_softmax/oracle_nd_31_acc_strict | -2.15 | [-3.16, -1.22] | -2.15, -1.39, -2.92 |
| vcs_minus_softmax/oracle_nd_31_frac_full | +0.00 | [+0.00, +0.00] | +0.00, +0.00, +0.00 |
| vcs_minus_softmax/oracle_nd_None_acc | -2.60 | [-3.43, -1.89] | -2.07, -2.22, -3.50 |
| vcs_minus_softmax/oracle_nd_None_acc_strict | -2.60 | [-3.43, -1.84] | -2.07, -2.22, -3.50 |
| vcs_minus_softmax/oracle_nd_None_frac_full | +0.00 | [+0.00, +0.00] | +0.00, +0.00, +0.00 |
| vcs_minus_softmax/2x2_A+C | -0.78 | [-1.37, -0.21] | -0.58, -0.21, -1.56 |
| vcs_minus_softmax/2x2_B+C | +0.00 | [-0.66, +0.70] | -0.07, +0.45, -0.36 |
| vcs_minus_softmax/2x2_A+D | -0.77 | [-1.30, -0.25] | -0.66, -0.45, -1.20 |
| vcs_minus_softmax/2x2_B+D | -0.13 | [-0.72, +0.42] | -0.12, +0.20, -0.48 |
| vcs_minus_softmax/margin | -2.02 | [-2.20, -1.84] | -1.85, -3.03, -1.19 |
| vcs_minus_softmax/err_other_same_cat | -0.14 | [-0.70, +0.45] | -0.56, +0.03, +0.12 |
| vcs_minus_softmax/err_other_diff_cat | +0.05 | [-0.16, +0.29] | +0.22, -0.25, +0.19 |
| vcs_minus_softmax/err_part_or_duplicate | +1.54 | [+0.97, +2.12] | +1.54, +1.57, +1.51 |
| vcs_minus_softmax/err_no_matching_annotation | +0.41 | [+0.10, +0.73] | +0.47, +0.45, +0.30 |
| vcs_minus_js/acc_all | +0.41 | [-0.15, +0.89] | +0.67, -0.09, +0.65 |
| vcs_minus_js/coverage | +0.00 | [+0.00, +0.00] | +0.00, +0.00, +0.00 |
| vcs_minus_js/acc_covered | +0.49 | [-0.12, +1.04] | +0.77, +0.03, +0.66 |
| vcs_minus_js/topK_4_acc | +0.26 | [-0.08, +0.60] | +0.32, -0.11, +0.56 |
| vcs_minus_js/topK_4_coverage | +0.00 | [+0.00, +0.00] | +0.00, +0.00, +0.00 |
| vcs_minus_js/topK_8_acc | +0.40 | [-0.01, +0.83] | +0.16, +0.30, +0.75 |
| vcs_minus_js/topK_8_coverage | +0.00 | [+0.00, +0.00] | +0.00, +0.00, +0.00 |
| vcs_minus_js/topK_16_acc | +0.39 | [-0.07, +0.84] | +0.67, -0.12, +0.63 |
| vcs_minus_js/topK_16_coverage | +0.00 | [+0.00, +0.00] | +0.00, +0.00, +0.00 |
| vcs_minus_js/topK_32_acc | +0.45 | [-0.07, +0.93] | +0.67, -0.16, +0.82 |
| vcs_minus_js/topK_32_coverage | +0.00 | [+0.00, +0.00] | +0.00, +0.00, +0.00 |
| vcs_minus_js/topK_None_acc | +0.41 | [-0.17, +0.96] | +0.67, -0.09, +0.65 |
| vcs_minus_js/topK_None_coverage | +0.00 | [+0.00, +0.00] | +0.00, +0.00, +0.00 |
| vcs_minus_js/oracle_nd_1_acc | +0.13 | [-0.09, +0.33] | +0.43, -0.10, +0.07 |
| vcs_minus_js/oracle_nd_1_acc_strict | +0.13 | [-0.06, +0.35] | +0.43, -0.10, +0.07 |
| vcs_minus_js/oracle_nd_1_frac_full | +0.00 | [+0.00, +0.00] | +0.00, +0.00, +0.00 |
| vcs_minus_js/oracle_nd_3_acc | +0.16 | [-0.14, +0.49] | +0.37, +0.34, -0.22 |
| vcs_minus_js/oracle_nd_3_acc_strict | +0.19 | [-0.14, +0.49] | +0.40, +0.38, -0.20 |
| vcs_minus_js/oracle_nd_3_frac_full | +0.00 | [+0.00, +0.00] | +0.00, +0.00, +0.00 |
| vcs_minus_js/oracle_nd_7_acc | +0.25 | [-0.19, +0.69] | +0.20, +0.59, -0.04 |
| vcs_minus_js/oracle_nd_7_acc_strict | +0.29 | [-0.12, +0.71] | +0.32, +0.63, -0.09 |
| vcs_minus_js/oracle_nd_7_frac_full | +0.00 | [+0.00, +0.00] | +0.00, +0.00, +0.00 |
| vcs_minus_js/oracle_nd_15_acc | +0.01 | [-0.49, +0.51] | -0.03, +0.33, -0.27 |
| vcs_minus_js/oracle_nd_15_acc_strict | -0.11 | [-0.68, +0.43] | -0.20, +0.68, -0.82 |
| vcs_minus_js/oracle_nd_15_frac_full | +0.00 | [+0.00, +0.00] | +0.00, +0.00, +0.00 |
| vcs_minus_js/oracle_nd_31_acc | +0.32 | [-0.23, +0.86] | +0.38, +0.64, -0.06 |
| vcs_minus_js/oracle_nd_31_acc_strict | +0.19 | [-0.50, +0.90] | +0.67, +0.80, -0.90 |
| vcs_minus_js/oracle_nd_31_frac_full | +0.00 | [+0.00, +0.00] | +0.00, +0.00, +0.00 |
| vcs_minus_js/oracle_nd_None_acc | +0.34 | [-0.23, +0.98] | +0.66, +0.02, +0.33 |
| vcs_minus_js/oracle_nd_None_acc_strict | +0.34 | [-0.21, +0.86] | +0.66, +0.02, +0.33 |
| vcs_minus_js/oracle_nd_None_frac_full | +0.00 | [+0.00, +0.00] | +0.00, +0.00, +0.00 |
| vcs_minus_js/2x2_A+C | +0.55 | [+0.13, +1.02] | +0.83, +1.05, -0.23 |
| vcs_minus_js/2x2_B+C | +0.51 | [+0.08, +0.95] | +0.77, +0.43, +0.34 |
| vcs_minus_js/2x2_A+D | +0.65 | [+0.28, +1.00] | +1.01, +1.02, -0.09 |
| vcs_minus_js/2x2_B+D | +0.65 | [+0.27, +1.03] | +1.24, +0.33, +0.38 |
| vcs_minus_js/margin | -0.53 | [-0.64, -0.42] | -1.33, -0.39, +0.14 |
| vcs_minus_js/err_other_same_cat | -0.49 | [-0.91, -0.10] | -0.67, -0.67, -0.13 |
| vcs_minus_js/err_other_diff_cat | -0.12 | [-0.33, +0.07] | -0.48, -0.01, +0.11 |
| vcs_minus_js/err_part_or_duplicate | +0.09 | [-0.36, +0.53] | +0.31, +0.44, -0.49 |
| vcs_minus_js/err_no_matching_annotation | +0.12 | [-0.11, +0.37] | +0.17, +0.34, -0.14 |

## refcocoplus (1587 images, 11666 expressions); gates pass 9 / 9; missing score files: none

| quantity | vcs (n=3) | js (n=3) | softmax (n=3) |
|---|---|---|---|
| acc_all_image_macro | 73.09 ± 0.33 | 72.88 ± 0.18 | 76.31 ± 0.45 |
| coverage_image_macro | 98.80 ± 0.00 | 98.80 ± 0.00 | 98.80 ± 0.00 |
| acc_covered_image_macro | 73.82 ± 0.32 | 73.59 ± 0.18 | 77.01 ± 0.43 |
| topK_4_acc | 66.11 ± 0.41 | 66.00 ± 0.34 | 67.13 ± 0.31 |
| topK_4_coverage | 77.83 ± 0.00 | 77.83 ± 0.00 | 77.83 ± 0.00 |
| topK_8_acc | 72.81 ± 0.45 | 72.65 ± 0.45 | 74.46 ± 0.38 |
| topK_8_coverage | 90.40 ± 0.00 | 90.40 ± 0.00 | 90.40 ± 0.00 |
| topK_16_acc | 73.95 ± 0.37 | 73.87 ± 0.25 | 76.10 ± 0.43 |
| topK_16_coverage | 95.66 ± 0.00 | 95.66 ± 0.00 | 95.66 ± 0.00 |
| topK_32_acc | 73.75 ± 0.28 | 73.51 ± 0.07 | 76.36 ± 0.48 |
| topK_32_coverage | 97.68 ± 0.00 | 97.68 ± 0.00 | 97.68 ± 0.00 |
| topK_None_acc | 73.09 ± 0.33 | 72.88 ± 0.18 | 76.31 ± 0.45 |
| topK_None_coverage | 98.80 ± 0.00 | 98.80 ± 0.00 | 98.80 ± 0.00 |
| oracle_nd_1_acc | 96.73 ± 0.19 | 96.77 ± 0.12 | 97.15 ± 0.13 |
| oracle_nd_1_acc_strict | 96.72 ± 0.19 | 96.77 ± 0.12 | 97.15 ± 0.13 |
| oracle_nd_1_frac_full | 99.94 ± 0.00 | 99.94 ± 0.00 | 99.94 ± 0.00 |
| oracle_nd_3_acc | 92.68 ± 0.15 | 92.58 ± 0.13 | 93.43 ± 0.15 |
| oracle_nd_3_acc_strict | 92.70 ± 0.17 | 92.61 ± 0.15 | 93.46 ± 0.17 |
| oracle_nd_3_frac_full | 99.18 ± 0.00 | 99.18 ± 0.00 | 99.18 ± 0.00 |
| oracle_nd_7_acc | 87.00 ± 0.12 | 86.93 ± 0.34 | 88.37 ± 0.21 |
| oracle_nd_7_acc_strict | 87.39 ± 0.21 | 87.28 ± 0.42 | 88.71 ± 0.19 |
| oracle_nd_7_frac_full | 95.87 ± 0.00 | 95.87 ± 0.00 | 95.87 ± 0.00 |
| oracle_nd_15_acc | 81.33 ± 0.16 | 81.10 ± 0.39 | 83.18 ± 0.25 |
| oracle_nd_15_acc_strict | 82.33 ± 0.28 | 82.17 ± 0.48 | 84.22 ± 0.34 |
| oracle_nd_15_frac_full | 85.60 ± 0.00 | 85.60 ± 0.00 | 85.60 ± 0.00 |
| oracle_nd_31_acc | 75.94 ± 0.14 | 75.77 ± 0.15 | 78.62 ± 0.29 |
| oracle_nd_31_acc_strict | 77.13 ± 0.37 | 77.10 ± 0.30 | 79.73 ± 0.27 |
| oracle_nd_31_frac_full | 68.03 ± 0.00 | 68.03 ± 0.00 | 68.03 ± 0.00 |
| oracle_nd_None_acc | 70.46 ± 0.42 | 70.15 ± 0.16 | 73.83 ± 0.42 |
| oracle_nd_None_acc_strict | 70.46 ± 0.42 | 70.15 ± 0.16 | 73.83 ± 0.42 |
| oracle_nd_None_frac_full | 100.00 ± 0.00 | 100.00 ± 0.00 | 100.00 ± 0.00 |
| 2x2_A+C | 82.17 ± 0.52 | 81.72 ± 0.60 | 83.68 ± 0.64 |
| 2x2_B+C | 79.72 ± 0.44 | 79.66 ± 0.69 | 81.07 ± 0.77 |
| 2x2_A+D | 87.84 ± 0.26 | 87.86 ± 0.45 | 88.30 ± 0.39 |
| 2x2_B+D | 86.41 ± 0.25 | 86.16 ± 0.46 | 86.60 ± 0.44 |
| margin | 7.59 ± 0.75 | 7.21 ± 0.84 | 13.32 ± 0.15 |
| err_other_same_cat | 15.62 ± 0.47 | 15.74 ± 0.45 | 14.36 ± 0.66 |
| err_other_diff_cat | 1.44 ± 0.18 | 1.36 ± 0.10 | 1.70 ± 0.11 |
| err_part_or_duplicate | 7.16 ± 0.25 | 7.32 ± 0.51 | 5.47 ± 0.24 |
| err_no_matching_annotation | 2.69 ± 0.08 | 2.70 ± 0.14 | 2.16 ± 0.07 |
| acc_all_query | 72.21 ± 0.34 | 71.90 ± 0.25 | 75.40 ± 0.43 |

| contrast | mean (×100) | image bootstrap 95 % | per seed |
|---|---|---|---|
| vcs_minus_softmax/acc_all | -3.22 | [-3.89, -2.50] | -3.18, -2.51, -3.97 |
| vcs_minus_softmax/coverage | +0.00 | [+0.00, +0.00] | +0.00, +0.00, +0.00 |
| vcs_minus_softmax/acc_covered | -3.19 | [-3.84, -2.50] | -3.16, -2.50, -3.92 |
| vcs_minus_softmax/topK_4_acc | -1.01 | [-1.46, -0.57] | -1.38, -0.22, -1.44 |
| vcs_minus_softmax/topK_4_coverage | +0.00 | [+0.00, +0.00] | +0.00, +0.00, +0.00 |
| vcs_minus_softmax/topK_8_acc | -1.64 | [-2.19, -1.06] | -2.00, -0.79, -2.14 |
| vcs_minus_softmax/topK_8_coverage | +0.00 | [+0.00, +0.00] | +0.00, +0.00, +0.00 |
| vcs_minus_softmax/topK_16_acc | -2.16 | [-2.76, -1.53] | -2.21, -1.38, -2.89 |
| vcs_minus_softmax/topK_16_coverage | +0.00 | [+0.00, +0.00] | +0.00, +0.00, +0.00 |
| vcs_minus_softmax/topK_32_acc | -2.61 | [-3.16, -2.02] | -2.78, -1.79, -3.25 |
| vcs_minus_softmax/topK_32_coverage | +0.00 | [+0.00, +0.00] | +0.00, +0.00, +0.00 |
| vcs_minus_softmax/topK_None_acc | -3.22 | [-3.90, -2.58] | -3.18, -2.51, -3.97 |
| vcs_minus_softmax/topK_None_coverage | +0.00 | [+0.00, +0.00] | +0.00, +0.00, +0.00 |
| vcs_minus_softmax/oracle_nd_1_acc | -0.42 | [-0.72, -0.12] | -0.35, -0.38, -0.54 |
| vcs_minus_softmax/oracle_nd_1_acc_strict | -0.42 | [-0.70, -0.12] | -0.35, -0.38, -0.54 |
| vcs_minus_softmax/oracle_nd_1_frac_full | +0.00 | [+0.00, +0.00] | +0.00, +0.00, +0.00 |
| vcs_minus_softmax/oracle_nd_3_acc | -0.75 | [-1.16, -0.34] | -0.76, -0.74, -0.77 |
| vcs_minus_softmax/oracle_nd_3_acc_strict | -0.76 | [-1.22, -0.38] | -0.77, -0.75, -0.76 |
| vcs_minus_softmax/oracle_nd_3_frac_full | +0.00 | [+0.00, +0.00] | +0.00, +0.00, +0.00 |
| vcs_minus_softmax/oracle_nd_7_acc | -1.37 | [-1.87, -0.83] | -1.46, -1.28, -1.37 |
| vcs_minus_softmax/oracle_nd_7_acc_strict | -1.32 | [-1.86, -0.79] | -1.31, -1.22, -1.42 |
| vcs_minus_softmax/oracle_nd_7_frac_full | +0.00 | [+0.00, +0.00] | +0.00, +0.00, +0.00 |
| vcs_minus_softmax/oracle_nd_15_acc | -1.86 | [-2.47, -1.24] | -1.91, -1.66, -2.00 |
| vcs_minus_softmax/oracle_nd_15_acc_strict | -1.89 | [-2.52, -1.23] | -1.94, -1.63, -2.11 |
| vcs_minus_softmax/oracle_nd_15_frac_full | +0.00 | [+0.00, +0.00] | +0.00, +0.00, +0.00 |
| vcs_minus_softmax/oracle_nd_31_acc | -2.68 | [-3.33, -2.09] | -2.86, -2.19, -3.00 |
| vcs_minus_softmax/oracle_nd_31_acc_strict | -2.60 | [-3.39, -1.81] | -2.92, -1.88, -3.00 |
| vcs_minus_softmax/oracle_nd_31_frac_full | +0.00 | [+0.00, +0.00] | +0.00, +0.00, +0.00 |
| vcs_minus_softmax/oracle_nd_None_acc | -3.37 | [-4.07, -2.61] | -3.38, -2.55, -4.18 |
| vcs_minus_softmax/oracle_nd_None_acc_strict | -3.37 | [-4.07, -2.66] | -3.38, -2.55, -4.18 |
| vcs_minus_softmax/oracle_nd_None_frac_full | +0.00 | [+0.00, +0.00] | +0.00, +0.00, +0.00 |
| vcs_minus_softmax/2x2_A+C | -1.51 | [-2.05, -0.99] | -2.45, -0.26, -1.81 |
| vcs_minus_softmax/2x2_B+C | -1.34 | [-1.93, -0.81] | -2.15, -0.14, -1.74 |
| vcs_minus_softmax/2x2_A+D | -0.46 | [-0.96, +0.08] | -0.87, +0.29, -0.79 |
| vcs_minus_softmax/2x2_B+D | -0.18 | [-0.70, +0.37] | -0.54, +0.60, -0.61 |
| vcs_minus_softmax/margin | -5.73 | [-6.02, -5.44] | -6.42, -5.49, -5.29 |
| vcs_minus_softmax/err_other_same_cat | +1.26 | [+0.73, +1.81] | +1.98, +0.15, +1.64 |
| vcs_minus_softmax/err_other_diff_cat | -0.26 | [-0.47, -0.04] | -0.52, -0.09, -0.16 |
| vcs_minus_softmax/err_part_or_duplicate | +1.69 | [+1.20, +2.12] | +1.15, +1.92, +1.98 |
| vcs_minus_softmax/err_no_matching_annotation | +0.53 | [+0.26, +0.80] | +0.56, +0.52, +0.52 |
| vcs_minus_js/acc_all | +0.21 | [-0.22, +0.63] | -0.03, +0.50, +0.18 |
| vcs_minus_js/coverage | +0.00 | [+0.00, +0.00] | +0.00, +0.00, +0.00 |
| vcs_minus_js/acc_covered | +0.23 | [-0.22, +0.68] | -0.04, +0.53, +0.19 |
| vcs_minus_js/topK_4_acc | +0.11 | [-0.19, +0.41] | -0.05, +0.80, -0.42 |
| vcs_minus_js/topK_4_coverage | +0.00 | [+0.00, +0.00] | +0.00, +0.00, +0.00 |
| vcs_minus_js/topK_8_acc | +0.16 | [-0.20, +0.55] | +0.09, +0.85, -0.45 |
| vcs_minus_js/topK_8_coverage | +0.00 | [+0.00, +0.00] | +0.00, +0.00, +0.00 |
| vcs_minus_js/topK_16_acc | +0.08 | [-0.28, +0.46] | +0.19, +0.37, -0.32 |
| vcs_minus_js/topK_16_coverage | +0.00 | [+0.00, +0.00] | +0.00, +0.00, +0.00 |
| vcs_minus_js/topK_32_acc | +0.24 | [-0.17, +0.63] | +0.07, +0.52, +0.15 |
| vcs_minus_js/topK_32_coverage | +0.00 | [+0.00, +0.00] | +0.00, +0.00, +0.00 |
| vcs_minus_js/topK_None_acc | +0.21 | [-0.23, +0.63] | -0.03, +0.50, +0.18 |
| vcs_minus_js/topK_None_coverage | +0.00 | [+0.00, +0.00] | +0.00, +0.00, +0.00 |
| vcs_minus_js/oracle_nd_1_acc | -0.04 | [-0.22, +0.13] | +0.05, -0.10, -0.07 |
| vcs_minus_js/oracle_nd_1_acc_strict | -0.04 | [-0.24, +0.14] | +0.05, -0.10, -0.07 |
| vcs_minus_js/oracle_nd_1_frac_full | +0.00 | [+0.00, +0.00] | +0.00, +0.00, +0.00 |
| vcs_minus_js/oracle_nd_3_acc | +0.10 | [-0.16, +0.36] | +0.25, +0.22, -0.17 |
| vcs_minus_js/oracle_nd_3_acc_strict | +0.09 | [-0.19, +0.33] | +0.22, +0.23, -0.19 |
| vcs_minus_js/oracle_nd_3_frac_full | +0.00 | [+0.00, +0.00] | +0.00, +0.00, +0.00 |
| vcs_minus_js/oracle_nd_7_acc | +0.07 | [-0.27, +0.41] | -0.17, +0.26, +0.11 |
| vcs_minus_js/oracle_nd_7_acc_strict | +0.11 | [-0.22, +0.43] | -0.12, +0.46, -0.00 |
| vcs_minus_js/oracle_nd_7_frac_full | +0.00 | [+0.00, +0.00] | +0.00, +0.00, +0.00 |
| vcs_minus_js/oracle_nd_15_acc | +0.23 | [-0.17, +0.64] | -0.03, +0.48, +0.24 |
| vcs_minus_js/oracle_nd_15_acc_strict | +0.16 | [-0.23, +0.52] | -0.07, +0.52, +0.02 |
| vcs_minus_js/oracle_nd_15_frac_full | +0.00 | [+0.00, +0.00] | +0.00, +0.00, +0.00 |
| vcs_minus_js/oracle_nd_31_acc | +0.17 | [-0.22, +0.59] | -0.05, +0.44, +0.11 |
| vcs_minus_js/oracle_nd_31_acc_strict | +0.03 | [-0.43, +0.49] | -0.44, +0.61, -0.10 |
| vcs_minus_js/oracle_nd_31_frac_full | +0.00 | [+0.00, +0.00] | +0.00, +0.00, +0.00 |
| vcs_minus_js/oracle_nd_None_acc | +0.31 | [-0.15, +0.75] | +0.00, +0.85, +0.07 |
| vcs_minus_js/oracle_nd_None_acc_strict | +0.31 | [-0.14, +0.75] | +0.00, +0.85, +0.07 |
| vcs_minus_js/oracle_nd_None_frac_full | +0.00 | [+0.00, +0.00] | +0.00, +0.00, +0.00 |
| vcs_minus_js/2x2_A+C | +0.45 | [+0.11, +0.76] | +0.59, +0.72, +0.04 |
| vcs_minus_js/2x2_B+C | +0.06 | [-0.30, +0.43] | +0.27, +0.33, -0.43 |
| vcs_minus_js/2x2_A+D | -0.01 | [-0.30, +0.25] | -0.09, +0.65, -0.60 |
| vcs_minus_js/2x2_B+D | +0.26 | [-0.07, +0.57] | +0.26, +0.90, -0.38 |
| vcs_minus_js/margin | +0.38 | [+0.30, +0.45] | +0.20, -0.32, +1.25 |
| vcs_minus_js/err_other_same_cat | -0.12 | [-0.48, +0.26] | -0.05, -0.47, +0.16 |
| vcs_minus_js/err_other_diff_cat | +0.08 | [-0.04, +0.21] | -0.07, +0.01, +0.30 |
| vcs_minus_js/err_part_or_duplicate | -0.16 | [-0.49, +0.14] | +0.02, -0.01, -0.50 |
| vcs_minus_js/err_no_matching_annotation | -0.01 | [-0.20, +0.18] | +0.13, -0.03, -0.14 |
