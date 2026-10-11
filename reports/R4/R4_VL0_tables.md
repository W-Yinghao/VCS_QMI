# R4-VL0 — candidate diagnostics (exploratory; items 3–4 oracle-assisted)

`oracle_nd_<n>_acc` is the frozen reading (queries with fewer than n distractors are scored against all they have); `_frac_full` = fraction of covered queries with >= n distractors; `_acc_strict` = restricted to those queries.  Contrasts: VCS − other, per-seed differences of per-image means, image bootstrap 95 %.

## refcocog (1356 images, 6442 expressions); gates pass 9 / 9; missing score files: none

| quantity | vcs (n=3) | js (n=3) | softmax (n=3) |
|---|---|---|---|
| acc_all_image_macro | 56.22 ± 0.79 | 57.58 ± 0.46 | 56.85 ± 0.19 |
| coverage_image_macro | 97.28 ± 0.00 | 97.28 ± 0.00 | 97.28 ± 0.00 |
| acc_covered_image_macro | 57.60 ± 0.82 | 59.04 ± 0.41 | 58.29 ± 0.21 |
| topK_4_acc | 61.64 ± 0.23 | 61.56 ± 0.18 | 61.38 ± 0.57 |
| topK_4_coverage | 75.76 ± 0.00 | 75.76 ± 0.00 | 75.76 ± 0.00 |
| topK_8_acc | 63.27 ± 0.38 | 63.91 ± 0.07 | 63.23 ± 0.31 |
| topK_8_coverage | 86.71 ± 0.00 | 86.71 ± 0.00 | 86.71 ± 0.00 |
| topK_16_acc | 61.86 ± 0.52 | 62.66 ± 0.16 | 62.13 ± 0.35 |
| topK_16_coverage | 92.70 ± 0.00 | 92.70 ± 0.00 | 92.70 ± 0.00 |
| topK_32_acc | 58.93 ± 0.51 | 60.13 ± 0.51 | 59.29 ± 0.43 |
| topK_32_coverage | 95.82 ± 0.00 | 95.82 ± 0.00 | 95.82 ± 0.00 |
| topK_None_acc | 56.22 ± 0.79 | 57.58 ± 0.46 | 56.85 ± 0.19 |
| topK_None_coverage | 97.28 ± 0.00 | 97.28 ± 0.00 | 97.28 ± 0.00 |
| oracle_nd_1_acc | 91.96 ± 0.27 | 92.20 ± 0.06 | 91.55 ± 0.30 |
| oracle_nd_1_acc_strict | 91.96 ± 0.27 | 92.20 ± 0.06 | 91.55 ± 0.30 |
| oracle_nd_1_frac_full | 100.00 ± 0.00 | 100.00 ± 0.00 | 100.00 ± 0.00 |
| oracle_nd_3_acc | 82.73 ± 0.41 | 83.51 ± 0.12 | 82.54 ± 0.26 |
| oracle_nd_3_acc_strict | 82.93 ± 0.45 | 83.69 ± 0.08 | 82.75 ± 0.33 |
| oracle_nd_3_frac_full | 98.59 ± 0.00 | 98.59 ± 0.00 | 98.59 ± 0.00 |
| oracle_nd_7_acc | 73.66 ± 0.52 | 74.74 ± 0.17 | 73.60 ± 0.19 |
| oracle_nd_7_acc_strict | 74.54 ± 0.58 | 75.66 ± 0.18 | 74.40 ± 0.14 |
| oracle_nd_7_frac_full | 93.03 ± 0.00 | 93.03 ± 0.00 | 93.03 ± 0.00 |
| oracle_nd_15_acc | 64.95 ± 0.68 | 65.95 ± 0.26 | 65.27 ± 0.34 |
| oracle_nd_15_acc_strict | 66.48 ± 0.82 | 67.69 ± 0.24 | 66.93 ± 0.20 |
| oracle_nd_15_frac_full | 79.54 ± 0.00 | 79.54 ± 0.00 | 79.54 ± 0.00 |
| oracle_nd_31_acc | 57.22 ± 0.71 | 58.61 ± 0.54 | 58.30 ± 0.46 |
| oracle_nd_31_acc_strict | 58.42 ± 1.10 | 60.15 ± 0.35 | 60.09 ± 0.33 |
| oracle_nd_31_frac_full | 56.75 ± 0.00 | 56.75 ± 0.00 | 56.75 ± 0.00 |
| oracle_nd_None_acc | 51.90 ± 0.68 | 53.42 ± 0.75 | 53.38 ± 0.38 |
| oracle_nd_None_acc_strict | 51.90 ± 0.68 | 53.42 ± 0.75 | 53.38 ± 0.38 |
| oracle_nd_None_frac_full | 100.00 ± 0.00 | 100.00 ± 0.00 | 100.00 ± 0.00 |
| 2x2_A+C | 84.94 ± 0.35 | 84.83 ± 0.05 | 85.18 ± 0.30 |
| 2x2_B+C | 84.01 ± 0.16 | 83.93 ± 0.24 | 84.04 ± 0.25 |
| 2x2_A+D | 89.80 ± 0.08 | 89.60 ± 0.03 | 88.76 ± 0.29 |
| 2x2_B+D | 88.86 ± 0.09 | 88.66 ± 0.24 | 88.13 ± 0.26 |
| margin | 1.54 ± 0.14 | 2.06 ± 0.12 | 2.72 ± 0.24 |
| err_other_same_cat | 9.41 ± 0.35 | 9.79 ± 0.21 | 8.76 ± 0.14 |
| err_other_diff_cat | 1.65 ± 0.01 | 1.59 ± 0.04 | 1.90 ± 0.03 |
| err_part_or_duplicate | 23.65 ± 0.33 | 22.91 ± 0.26 | 23.07 ± 0.62 |
| err_no_matching_annotation | 9.08 ± 0.49 | 8.12 ± 0.37 | 9.42 ± 0.63 |
| acc_all_query | 56.02 ± 0.86 | 57.41 ± 0.29 | 56.65 ± 0.21 |

| contrast | mean (×100) | image bootstrap 95 % | per seed |
|---|---|---|---|
| vcs_minus_softmax/acc_all | -0.63 | [-1.45, +0.29] | -1.08, -0.91, +0.09 |
| vcs_minus_softmax/coverage | +0.00 | [+0.00, +0.00] | +0.00, +0.00, +0.00 |
| vcs_minus_softmax/acc_covered | -0.69 | [-1.60, +0.23] | -1.09, -1.03, +0.05 |
| vcs_minus_softmax/topK_4_acc | +0.26 | [-0.28, +0.82] | +0.78, -0.23, +0.23 |
| vcs_minus_softmax/topK_4_coverage | +0.00 | [+0.00, +0.00] | +0.00, +0.00, +0.00 |
| vcs_minus_softmax/topK_8_acc | +0.05 | [-0.66, +0.72] | +0.26, -0.62, +0.49 |
| vcs_minus_softmax/topK_8_coverage | +0.00 | [+0.00, +0.00] | +0.00, +0.00, +0.00 |
| vcs_minus_softmax/topK_16_acc | -0.26 | [-1.08, +0.59] | -0.27, -0.68, +0.16 |
| vcs_minus_softmax/topK_16_coverage | +0.00 | [+0.00, +0.00] | +0.00, +0.00, +0.00 |
| vcs_minus_softmax/topK_32_acc | -0.36 | [-1.20, +0.52] | -0.27, -0.59, -0.23 |
| vcs_minus_softmax/topK_32_coverage | +0.00 | [+0.00, +0.00] | +0.00, +0.00, +0.00 |
| vcs_minus_softmax/topK_None_acc | -0.63 | [-1.59, +0.22] | -1.08, -0.91, +0.09 |
| vcs_minus_softmax/topK_None_coverage | +0.00 | [+0.00, +0.00] | +0.00, +0.00, +0.00 |
| vcs_minus_softmax/oracle_nd_1_acc | +0.41 | [-0.10, +0.94] | +0.40, -0.15, +0.97 |
| vcs_minus_softmax/oracle_nd_1_acc_strict | +0.41 | [-0.08, +0.85] | +0.40, -0.15, +0.97 |
| vcs_minus_softmax/oracle_nd_1_frac_full | +0.00 | [+0.00, +0.00] | +0.00, +0.00, +0.00 |
| vcs_minus_softmax/oracle_nd_3_acc | +0.19 | [-0.50, +0.91] | -0.18, -0.21, +0.95 |
| vcs_minus_softmax/oracle_nd_3_acc_strict | +0.18 | [-0.51, +0.87] | -0.19, -0.35, +1.07 |
| vcs_minus_softmax/oracle_nd_3_frac_full | +0.00 | [+0.00, +0.00] | +0.00, +0.00, +0.00 |
| vcs_minus_softmax/oracle_nd_7_acc | +0.06 | [-0.71, +0.85] | -0.25, -0.00, +0.43 |
| vcs_minus_softmax/oracle_nd_7_acc_strict | +0.14 | [-0.72, +0.91] | -0.20, -0.14, +0.76 |
| vcs_minus_softmax/oracle_nd_7_frac_full | +0.00 | [+0.00, +0.00] | +0.00, +0.00, +0.00 |
| vcs_minus_softmax/oracle_nd_15_acc | -0.32 | [-1.22, +0.52] | -0.53, -0.60, +0.17 |
| vcs_minus_softmax/oracle_nd_15_acc_strict | -0.44 | [-1.41, +0.39] | -0.84, -1.18, +0.68 |
| vcs_minus_softmax/oracle_nd_15_frac_full | +0.00 | [+0.00, +0.00] | +0.00, +0.00, +0.00 |
| vcs_minus_softmax/oracle_nd_31_acc | -1.08 | [-1.95, -0.22] | -1.17, -1.38, -0.69 |
| vcs_minus_softmax/oracle_nd_31_acc_strict | -1.67 | [-2.87, -0.55] | -2.37, -2.54, -0.11 |
| vcs_minus_softmax/oracle_nd_31_frac_full | +0.00 | [+0.00, +0.00] | +0.00, +0.00, +0.00 |
| vcs_minus_softmax/oracle_nd_None_acc | -1.48 | [-2.38, -0.65] | -1.74, -1.59, -1.11 |
| vcs_minus_softmax/oracle_nd_None_acc_strict | -1.48 | [-2.35, -0.61] | -1.74, -1.59, -1.11 |
| vcs_minus_softmax/oracle_nd_None_frac_full | +0.00 | [+0.00, +0.00] | +0.00, +0.00, +0.00 |
| vcs_minus_softmax/2x2_A+C | -0.24 | [-0.82, +0.32] | -0.83, -0.34, +0.44 |
| vcs_minus_softmax/2x2_B+C | -0.03 | [-0.58, +0.56] | -0.14, -0.06, +0.12 |
| vcs_minus_softmax/2x2_A+D | +1.04 | [+0.51, +1.52] | +0.73, +1.08, +1.31 |
| vcs_minus_softmax/2x2_B+D | +0.74 | [+0.20, +1.30] | +1.00, +0.34, +0.87 |
| vcs_minus_softmax/margin | -1.18 | [-1.37, -0.95] | -1.16, -1.07, -1.31 |
| vcs_minus_softmax/err_other_same_cat | +0.64 | [+0.11, +1.22] | +0.84, +0.31, +0.78 |
| vcs_minus_softmax/err_other_diff_cat | -0.26 | [-0.52, -0.01] | -0.24, -0.28, -0.24 |
| vcs_minus_softmax/err_part_or_duplicate | +0.59 | [-0.17, +1.39] | +0.64, +0.20, +0.92 |
| vcs_minus_softmax/err_no_matching_annotation | -0.34 | [-0.91, +0.23] | -0.16, +0.68, -1.54 |
| vcs_minus_js/acc_all | -1.36 | [-1.93, -0.77] | -2.12, -1.98, +0.01 |
| vcs_minus_js/coverage | +0.00 | [+0.00, +0.00] | +0.00, +0.00, +0.00 |
| vcs_minus_js/acc_covered | -1.44 | [-2.07, -0.81] | -2.19, -2.09, -0.05 |
| vcs_minus_js/topK_4_acc | +0.08 | [-0.31, +0.44] | +0.00, +0.25, -0.01 |
| vcs_minus_js/topK_4_coverage | +0.00 | [+0.00, +0.00] | +0.00, +0.00, +0.00 |
| vcs_minus_js/topK_8_acc | -0.64 | [-1.08, -0.22] | -0.64, -0.98, -0.29 |
| vcs_minus_js/topK_8_coverage | +0.00 | [+0.00, +0.00] | +0.00, +0.00, +0.00 |
| vcs_minus_js/topK_16_acc | -0.80 | [-1.30, -0.25] | -1.27, -1.08, -0.04 |
| vcs_minus_js/topK_16_coverage | +0.00 | [+0.00, +0.00] | +0.00, +0.00, +0.00 |
| vcs_minus_js/topK_32_acc | -1.20 | [-1.82, -0.63] | -1.84, -1.73, -0.03 |
| vcs_minus_js/topK_32_coverage | +0.00 | [+0.00, +0.00] | +0.00, +0.00, +0.00 |
| vcs_minus_js/topK_None_acc | -1.36 | [-1.94, -0.79] | -2.12, -1.98, +0.01 |
| vcs_minus_js/topK_None_coverage | +0.00 | [+0.00, +0.00] | +0.00, +0.00, +0.00 |
| vcs_minus_js/oracle_nd_1_acc | -0.24 | [-0.56, +0.07] | -0.25, -0.53, +0.05 |
| vcs_minus_js/oracle_nd_1_acc_strict | -0.24 | [-0.57, +0.06] | -0.25, -0.53, +0.05 |
| vcs_minus_js/oracle_nd_1_frac_full | +0.00 | [+0.00, +0.00] | +0.00, +0.00, +0.00 |
| vcs_minus_js/oracle_nd_3_acc | -0.78 | [-1.22, -0.36] | -1.07, -1.08, -0.19 |
| vcs_minus_js/oracle_nd_3_acc_strict | -0.76 | [-1.17, -0.32] | -1.03, -1.09, -0.18 |
| vcs_minus_js/oracle_nd_3_frac_full | +0.00 | [+0.00, +0.00] | +0.00, +0.00, +0.00 |
| vcs_minus_js/oracle_nd_7_acc | -1.08 | [-1.58, -0.59] | -1.43, -1.25, -0.56 |
| vcs_minus_js/oracle_nd_7_acc_strict | -1.11 | [-1.65, -0.61] | -1.39, -1.41, -0.54 |
| vcs_minus_js/oracle_nd_7_frac_full | +0.00 | [+0.00, +0.00] | +0.00, +0.00, +0.00 |
| vcs_minus_js/oracle_nd_15_acc | -1.00 | [-1.55, -0.45] | -1.60, -1.41, -0.00 |
| vcs_minus_js/oracle_nd_15_acc_strict | -1.21 | [-1.82, -0.60] | -1.48, -2.00, -0.16 |
| vcs_minus_js/oracle_nd_15_frac_full | +0.00 | [+0.00, +0.00] | +0.00, +0.00, +0.00 |
| vcs_minus_js/oracle_nd_31_acc | -1.39 | [-1.98, -0.75] | -1.91, -2.15, -0.10 |
| vcs_minus_js/oracle_nd_31_acc_strict | -1.73 | [-2.49, -0.94] | -2.48, -2.59, -0.13 |
| vcs_minus_js/oracle_nd_31_frac_full | +0.00 | [+0.00, +0.00] | +0.00, +0.00, +0.00 |
| vcs_minus_js/oracle_nd_None_acc | -1.52 | [-2.08, -0.92] | -2.10, -2.29, -0.16 |
| vcs_minus_js/oracle_nd_None_acc_strict | -1.52 | [-2.11, -0.94] | -2.10, -2.29, -0.16 |
| vcs_minus_js/oracle_nd_None_frac_full | +0.00 | [+0.00, +0.00] | +0.00, +0.00, +0.00 |
| vcs_minus_js/2x2_A+C | +0.11 | [-0.29, +0.48] | -0.24, +0.08, +0.49 |
| vcs_minus_js/2x2_B+C | +0.08 | [-0.36, +0.49] | +0.30, +0.16, -0.24 |
| vcs_minus_js/2x2_A+D | +0.20 | [-0.15, +0.54] | +0.16, +0.26, +0.18 |
| vcs_minus_js/2x2_B+D | +0.21 | [-0.12, +0.55] | +0.36, +0.29, -0.02 |
| vcs_minus_js/margin | -0.52 | [-0.60, -0.46] | -0.73, -0.60, -0.24 |
| vcs_minus_js/err_other_same_cat | -0.38 | [-0.76, -0.01] | -0.28, -0.64, -0.22 |
| vcs_minus_js/err_other_diff_cat | +0.05 | [-0.11, +0.23] | +0.00, +0.08, +0.07 |
| vcs_minus_js/err_part_or_duplicate | +0.74 | [+0.19, +1.30] | +1.05, +1.07, +0.09 |
| vcs_minus_js/err_no_matching_annotation | +0.95 | [+0.55, +1.33] | +1.35, +1.46, +0.05 |

## refcocoplus (1587 images, 11666 expressions); gates pass 9 / 9; missing score files: none

| quantity | vcs (n=3) | js (n=3) | softmax (n=3) |
|---|---|---|---|
| acc_all_image_macro | 53.45 ± 0.66 | 53.44 ± 0.26 | 55.66 ± 0.37 |
| coverage_image_macro | 98.80 ± 0.00 | 98.80 ± 0.00 | 98.80 ± 0.00 |
| acc_covered_image_macro | 53.99 ± 0.66 | 53.98 ± 0.32 | 56.21 ± 0.37 |
| topK_4_acc | 63.93 ± 0.56 | 63.75 ± 0.17 | 63.23 ± 0.14 |
| topK_4_coverage | 77.83 ± 0.00 | 77.83 ± 0.00 | 77.83 ± 0.00 |
| topK_8_acc | 66.45 ± 0.60 | 66.22 ± 0.21 | 65.59 ± 0.35 |
| topK_8_coverage | 90.40 ± 0.00 | 90.40 ± 0.00 | 90.40 ± 0.00 |
| topK_16_acc | 62.46 ± 0.86 | 62.26 ± 0.04 | 62.39 ± 0.25 |
| topK_16_coverage | 95.66 ± 0.00 | 95.66 ± 0.00 | 95.66 ± 0.00 |
| topK_32_acc | 57.92 ± 0.77 | 57.84 ± 0.27 | 59.25 ± 0.14 |
| topK_32_coverage | 97.68 ± 0.00 | 97.68 ± 0.00 | 97.68 ± 0.00 |
| topK_None_acc | 53.45 ± 0.66 | 53.44 ± 0.26 | 55.66 ± 0.37 |
| topK_None_coverage | 98.80 ± 0.00 | 98.80 ± 0.00 | 98.80 ± 0.00 |
| oracle_nd_1_acc | 91.17 ± 0.21 | 90.73 ± 0.26 | 89.37 ± 0.76 |
| oracle_nd_1_acc_strict | 91.17 ± 0.21 | 90.73 ± 0.26 | 89.37 ± 0.76 |
| oracle_nd_1_frac_full | 99.94 ± 0.00 | 99.94 ± 0.00 | 99.94 ± 0.00 |
| oracle_nd_3_acc | 82.66 ± 0.37 | 82.34 ± 0.34 | 80.58 ± 0.66 |
| oracle_nd_3_acc_strict | 82.72 ± 0.36 | 82.38 ± 0.33 | 80.59 ± 0.63 |
| oracle_nd_3_frac_full | 99.18 ± 0.00 | 99.18 ± 0.00 | 99.18 ± 0.00 |
| oracle_nd_7_acc | 73.44 ± 0.50 | 73.27 ± 0.09 | 72.66 ± 0.56 |
| oracle_nd_7_acc_strict | 73.83 ± 0.46 | 73.73 ± 0.14 | 73.17 ± 0.59 |
| oracle_nd_7_frac_full | 95.87 ± 0.00 | 95.87 ± 0.00 | 95.87 ± 0.00 |
| oracle_nd_15_acc | 64.25 ± 0.52 | 64.22 ± 0.16 | 64.95 ± 0.58 |
| oracle_nd_15_acc_strict | 64.90 ± 0.52 | 65.04 ± 0.18 | 65.92 ± 0.54 |
| oracle_nd_15_frac_full | 85.60 ± 0.00 | 85.60 ± 0.00 | 85.60 ± 0.00 |
| oracle_nd_31_acc | 56.38 ± 0.69 | 56.23 ± 0.42 | 58.11 ± 0.65 |
| oracle_nd_31_acc_strict | 56.91 ± 0.52 | 56.93 ± 0.40 | 59.40 ± 0.48 |
| oracle_nd_31_frac_full | 68.03 ± 0.00 | 68.03 ± 0.00 | 68.03 ± 0.00 |
| oracle_nd_None_acc | 49.22 ± 0.53 | 49.04 ± 0.31 | 52.09 ± 0.70 |
| oracle_nd_None_acc_strict | 49.22 ± 0.53 | 49.04 ± 0.31 | 52.09 ± 0.70 |
| oracle_nd_None_frac_full | 100.00 ± 0.00 | 100.00 ± 0.00 | 100.00 ± 0.00 |
| 2x2_A+C | 86.92 ± 0.22 | 86.79 ± 0.10 | 87.47 ± 0.21 |
| 2x2_B+C | 86.21 ± 0.24 | 86.17 ± 0.17 | 86.81 ± 0.17 |
| 2x2_A+D | 89.14 ± 0.34 | 88.95 ± 0.16 | 88.54 ± 0.31 |
| 2x2_B+D | 88.55 ± 0.32 | 88.16 ± 0.26 | 87.99 ± 0.31 |
| margin | 0.69 ± 0.08 | 0.82 ± 0.01 | 1.81 ± 0.01 |
| err_other_same_cat | 8.81 ± 0.26 | 8.83 ± 0.16 | 7.01 ± 0.28 |
| err_other_diff_cat | 3.07 ± 0.12 | 3.13 ± 0.14 | 3.40 ± 0.48 |
| err_part_or_duplicate | 25.08 ± 0.37 | 25.01 ± 0.18 | 22.42 ± 0.21 |
| err_no_matching_annotation | 9.59 ± 0.53 | 9.60 ± 0.11 | 11.51 ± 0.32 |
| acc_all_query | 53.14 ± 0.65 | 52.99 ± 0.36 | 55.47 ± 0.43 |

| contrast | mean (×100) | image bootstrap 95 % | per seed |
|---|---|---|---|
| vcs_minus_softmax/acc_all | -2.20 | [-2.99, -1.39] | -2.15, -1.55, -2.91 |
| vcs_minus_softmax/coverage | +0.00 | [+0.00, +0.00] | +0.00, +0.00, +0.00 |
| vcs_minus_softmax/acc_covered | -2.22 | [-3.09, -1.36] | -2.17, -1.61, -2.89 |
| vcs_minus_softmax/topK_4_acc | +0.71 | [+0.23, +1.19] | +0.95, +0.96, +0.22 |
| vcs_minus_softmax/topK_4_coverage | +0.00 | [+0.00, +0.00] | +0.00, +0.00, +0.00 |
| vcs_minus_softmax/topK_8_acc | +0.86 | [+0.22, +1.51] | +1.11, +1.49, -0.03 |
| vcs_minus_softmax/topK_8_coverage | +0.00 | [+0.00, +0.00] | +0.00, +0.00, +0.00 |
| vcs_minus_softmax/topK_16_acc | +0.07 | [-0.64, +0.77] | +0.77, +0.43, -0.98 |
| vcs_minus_softmax/topK_16_coverage | +0.00 | [+0.00, +0.00] | +0.00, +0.00, +0.00 |
| vcs_minus_softmax/topK_32_acc | -1.33 | [-2.13, -0.51] | -0.87, -0.99, -2.14 |
| vcs_minus_softmax/topK_32_coverage | +0.00 | [+0.00, +0.00] | +0.00, +0.00, +0.00 |
| vcs_minus_softmax/topK_None_acc | -2.20 | [-3.09, -1.41] | -2.15, -1.55, -2.91 |
| vcs_minus_softmax/topK_None_coverage | +0.00 | [+0.00, +0.00] | +0.00, +0.00, +0.00 |
| vcs_minus_softmax/oracle_nd_1_acc | +1.80 | [+1.34, +2.24] | +1.38, +2.54, +1.47 |
| vcs_minus_softmax/oracle_nd_1_acc_strict | +1.80 | [+1.34, +2.23] | +1.38, +2.55, +1.47 |
| vcs_minus_softmax/oracle_nd_1_frac_full | +0.00 | [+0.00, +0.00] | +0.00, +0.00, +0.00 |
| vcs_minus_softmax/oracle_nd_3_acc | +2.08 | [+1.48, +2.67] | +1.70, +2.85, +1.69 |
| vcs_minus_softmax/oracle_nd_3_acc_strict | +2.13 | [+1.56, +2.71] | +1.79, +2.87, +1.74 |
| vcs_minus_softmax/oracle_nd_3_frac_full | +0.00 | [+0.00, +0.00] | +0.00, +0.00, +0.00 |
| vcs_minus_softmax/oracle_nd_7_acc | +0.78 | [+0.03, +1.47] | +0.68, +1.00, +0.65 |
| vcs_minus_softmax/oracle_nd_7_acc_strict | +0.66 | [-0.03, +1.37] | +0.49, +0.83, +0.66 |
| vcs_minus_softmax/oracle_nd_7_frac_full | +0.00 | [+0.00, +0.00] | +0.00, +0.00, +0.00 |
| vcs_minus_softmax/oracle_nd_15_acc | -0.70 | [-1.42, +0.03] | -1.06, -0.03, -1.02 |
| vcs_minus_softmax/oracle_nd_15_acc_strict | -1.02 | [-1.77, -0.27] | -1.44, -0.35, -1.26 |
| vcs_minus_softmax/oracle_nd_15_frac_full | +0.00 | [+0.00, +0.00] | +0.00, +0.00, +0.00 |
| vcs_minus_softmax/oracle_nd_31_acc | -1.73 | [-2.58, -0.91] | -2.09, -0.98, -2.13 |
| vcs_minus_softmax/oracle_nd_31_acc_strict | -2.49 | [-3.46, -1.58] | -2.75, -2.00, -2.73 |
| vcs_minus_softmax/oracle_nd_31_frac_full | +0.00 | [+0.00, +0.00] | +0.00, +0.00, +0.00 |
| vcs_minus_softmax/oracle_nd_None_acc | -2.87 | [-3.68, -2.04] | -3.34, -2.31, -2.96 |
| vcs_minus_softmax/oracle_nd_None_acc_strict | -2.87 | [-3.67, -2.03] | -3.34, -2.31, -2.96 |
| vcs_minus_softmax/oracle_nd_None_frac_full | +0.00 | [+0.00, +0.00] | +0.00, +0.00, +0.00 |
| vcs_minus_softmax/2x2_A+C | -0.55 | [-0.99, -0.12] | -0.64, -0.65, -0.36 |
| vcs_minus_softmax/2x2_B+C | -0.60 | [-1.06, -0.17] | -0.75, -0.82, -0.24 |
| vcs_minus_softmax/2x2_A+D | +0.59 | [+0.17, +1.07] | +0.62, +0.70, +0.47 |
| vcs_minus_softmax/2x2_B+D | +0.57 | [+0.08, +1.05] | +0.65, +0.45, +0.61 |
| vcs_minus_softmax/margin | -1.12 | [-1.37, -0.85] | -1.10, -1.06, -1.22 |
| vcs_minus_softmax/err_other_same_cat | +1.80 | [+1.37, +2.28] | +1.40, +2.10, +1.89 |
| vcs_minus_softmax/err_other_diff_cat | -0.33 | [-0.64, -0.02] | -0.00, -0.26, -0.73 |
| vcs_minus_softmax/err_part_or_duplicate | +2.66 | [+1.91, +3.33] | +3.06, +2.17, +2.74 |
| vcs_minus_softmax/err_no_matching_annotation | -1.92 | [-2.38, -1.42] | -2.31, -2.46, -0.99 |
| vcs_minus_js/acc_all | +0.01 | [-0.48, +0.52] | +0.20, +0.32, -0.48 |
| vcs_minus_js/coverage | +0.00 | [+0.00, +0.00] | +0.00, +0.00, +0.00 |
| vcs_minus_js/acc_covered | +0.01 | [-0.48, +0.54] | +0.15, +0.31, -0.43 |
| vcs_minus_js/topK_4_acc | +0.19 | [-0.10, +0.48] | +0.62, +0.32, -0.37 |
| vcs_minus_js/topK_4_coverage | +0.00 | [+0.00, +0.00] | +0.00, +0.00, +0.00 |
| vcs_minus_js/topK_8_acc | +0.23 | [-0.14, +0.60] | +0.44, +0.57, -0.32 |
| vcs_minus_js/topK_8_coverage | +0.00 | [+0.00, +0.00] | +0.00, +0.00, +0.00 |
| vcs_minus_js/topK_16_acc | +0.20 | [-0.25, +0.66] | +0.98, +0.28, -0.66 |
| vcs_minus_js/topK_16_coverage | +0.00 | [+0.00, +0.00] | +0.00, +0.00, +0.00 |
| vcs_minus_js/topK_32_acc | +0.07 | [-0.39, +0.52] | +0.40, +0.36, -0.55 |
| vcs_minus_js/topK_32_coverage | +0.00 | [+0.00, +0.00] | +0.00, +0.00, +0.00 |
| vcs_minus_js/topK_None_acc | +0.01 | [-0.47, +0.51] | +0.20, +0.32, -0.48 |
| vcs_minus_js/topK_None_coverage | +0.00 | [+0.00, +0.00] | +0.00, +0.00, +0.00 |
| vcs_minus_js/oracle_nd_1_acc | +0.44 | [+0.16, +0.71] | +0.77, +0.56, -0.00 |
| vcs_minus_js/oracle_nd_1_acc_strict | +0.44 | [+0.16, +0.71] | +0.77, +0.56, -0.00 |
| vcs_minus_js/oracle_nd_1_frac_full | +0.00 | [+0.00, +0.00] | +0.00, +0.00, +0.00 |
| vcs_minus_js/oracle_nd_3_acc | +0.32 | [-0.05, +0.65] | +0.76, +0.68, -0.49 |
| vcs_minus_js/oracle_nd_3_acc_strict | +0.34 | [-0.01, +0.71] | +0.82, +0.66, -0.45 |
| vcs_minus_js/oracle_nd_3_frac_full | +0.00 | [+0.00, +0.00] | +0.00, +0.00, +0.00 |
| vcs_minus_js/oracle_nd_7_acc | +0.17 | [-0.22, +0.56] | +0.73, +0.12, -0.35 |
| vcs_minus_js/oracle_nd_7_acc_strict | +0.10 | [-0.34, +0.51] | +0.74, -0.02, -0.41 |
| vcs_minus_js/oracle_nd_7_frac_full | +0.00 | [+0.00, +0.00] | +0.00, +0.00, +0.00 |
| vcs_minus_js/oracle_nd_15_acc | +0.03 | [-0.43, +0.46] | +0.42, +0.14, -0.47 |
| vcs_minus_js/oracle_nd_15_acc_strict | -0.13 | [-0.63, +0.32] | +0.22, +0.05, -0.67 |
| vcs_minus_js/oracle_nd_15_frac_full | +0.00 | [+0.00, +0.00] | +0.00, +0.00, +0.00 |
| vcs_minus_js/oracle_nd_31_acc | +0.15 | [-0.31, +0.62] | +0.65, +0.09, -0.28 |
| vcs_minus_js/oracle_nd_31_acc_strict | -0.02 | [-0.57, +0.52] | +0.51, -0.17, -0.39 |
| vcs_minus_js/oracle_nd_31_frac_full | +0.00 | [+0.00, +0.00] | +0.00, +0.00, +0.00 |
| vcs_minus_js/oracle_nd_None_acc | +0.18 | [-0.31, +0.71] | +0.53, +0.14, -0.13 |
| vcs_minus_js/oracle_nd_None_acc_strict | +0.18 | [-0.35, +0.65] | +0.53, +0.14, -0.13 |
| vcs_minus_js/oracle_nd_None_frac_full | +0.00 | [+0.00, +0.00] | +0.00, +0.00, +0.00 |
| vcs_minus_js/2x2_A+C | +0.13 | [-0.14, +0.43] | -0.17, +0.19, +0.38 |
| vcs_minus_js/2x2_B+C | +0.03 | [-0.26, +0.32] | -0.00, -0.13, +0.23 |
| vcs_minus_js/2x2_A+D | +0.19 | [-0.09, +0.45] | +0.53, +0.27, -0.24 |
| vcs_minus_js/2x2_B+D | +0.40 | [+0.13, +0.68] | +0.89, +0.49, -0.20 |
| vcs_minus_js/margin | -0.14 | [-0.20, -0.08] | -0.10, -0.10, -0.22 |
| vcs_minus_js/err_other_same_cat | -0.01 | [-0.31, +0.27] | +0.01, +0.33, -0.38 |
| vcs_minus_js/err_other_diff_cat | -0.06 | [-0.27, +0.15] | -0.11, -0.16, +0.09 |
| vcs_minus_js/err_part_or_duplicate | +0.07 | [-0.41, +0.52] | +0.15, -0.14, +0.21 |
| vcs_minus_js/err_no_matching_annotation | -0.01 | [-0.31, +0.30] | -0.25, -0.34, +0.55 |
