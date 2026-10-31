Publication copy (sample names aliased; numbers and conclusions unchanged).

# R27-S / R27-D / R27-O 数字表（自动生成，full DEV；未读 TEST；只列数字，不含解读）

Flickr30K × seed1001。A0 = R25 原生 H0；A1/A3 为 R27 已完成单元；R27-O 为显式使用真对应的诊断（O1/O2：t2i 候选救援；O3：t2i 编码器目标直接取当前 peer 重编码的真图），不进入主结果表。bn_cross_t2i / cn_mix_t2i：A1 宿主，t2i 候选改为直接跨模态检索 / 固定 3+2 混合（唯一 ID，整体接口改动）。

缺失单元（未完成，记 NA，不以其他臂替代）：f40_s1001/d3_snce_i2t, f40_s1001/bn_cross_t2i, f40_s1001/cn_mix_t2i, f60_s1001/d3_snce_i2t, f60_s1001/bn_cross_t2i, f60_s1001/cn_mix_t2i

## f40_s1001：各臂终点

| 臂 | best rSum | best epoch | last | last5 | i2t R@1 | i2t R@5 | i2t R@10 | t2i R@1 | t2i R@5 | t2i R@10 | attempts |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A0_native | 512.62 | 18 | 511.48 | 511.61 | 81.95 | 95.66 | 98.22 | 61.16 | 84.85 | 90.77 |  |
| a1_iface | 514.97 | 27 | 514.14 | 514.07 | 82.84 | 96.35 | 98.13 | 61.50 | 85.15 | 91.01 | 1 |
| a3_vcs | 513.02 | 31 | 512.41 | 512.56 | 81.85 | 95.96 | 98.42 | 60.95 | 85.01 | 90.83 | 1 |
| s1_balanced_dual | 513.61 | 18 | 512.84 | 512.88 | 83.14 | 95.36 | 97.93 | 61.58 | 84.87 | 90.73 | 1 |
| s2_single_free_vcs | 513.35 | 21 | 512.21 | 512.58 | 81.95 | 96.35 | 98.42 | 61.30 | 84.77 | 90.55 | 1 |
| s3_single_cap_delta | 513.04 | 27 | 512.60 | 512.53 | 81.95 | 95.96 | 98.72 | 60.99 | 84.71 | 90.71 | 1 |
| s4_single_cap_vcs | 512.50 | 33 | 511.64 | 512.09 | 81.56 | 96.55 | 98.22 | 60.65 | 84.95 | 90.57 | 1 |
| s5_single_cap_bce | 512.76 | 17 | 511.10 | 511.10 | 81.56 | 96.06 | 98.52 | 61.24 | 84.95 | 90.43 | 1 |
| s6_single_cap_snce | 514.46 | 21 | 513.10 | 513.22 | 82.54 | 96.25 | 98.42 | 61.28 | 85.19 | 90.77 | 3 |
| d1_vcs_i2t | 513.16 | 25 | 512.45 | 512.52 | 82.05 | 95.36 | 98.52 | 61.52 | 85.11 | 90.59 | 1 |
| d2_vcs_t2i | 511.95 | 15 | 510.95 | 511.14 | 81.46 | 95.96 | 98.42 | 60.61 | 84.69 | 90.81 | 1 |
| d3_snce_i2t | NA | | | | | | | | | | | |
| d4_snce_t2i | 513.53 | 22 | 512.49 | 512.70 | 81.95 | 95.66 | 98.42 | 61.58 | 85.09 | 90.83 | 2 |
| o1_rescue_native_t2i | 513.75 | 21 | 512.86 | 512.99 | 81.46 | 96.25 | 98.42 | 61.40 | 85.33 | 90.89 | 1 |
| o2_rescue_vcs_t2i | 514.77 | 26 | 514.38 | 514.37 | 83.04 | 96.15 | 98.42 | 61.22 | 85.15 | 90.79 | 1 |
| o3_true_target_t2i | 515.96 | 23 | 514.99 | 515.17 | 83.53 | 96.35 | 98.22 | 61.38 | 85.27 | 91.20 | 1 |
| bn_cross_t2i | NA | | | | | | | | | | | |
| cn_mix_t2i | NA | | | | | | | | | | | |

## f40_s1001：配对差（单 seed）

| 比较 | best rSum | last5 | i2t R@1 | t2i R@1 |
|---|---|---|---|---|
| s1_balanced_dual-a1_iface | -1.36 | -1.19 | +0.30 | +0.08 |
| s2_single_free_vcs-a1_iface | -1.62 | -1.50 | -0.89 | -0.20 |
| s3_single_cap_delta-a1_iface | -1.93 | -1.54 | -0.89 | -0.51 |
| s4_single_cap_vcs-a1_iface | -2.47 | -1.98 | -1.28 | -0.85 |
| s5_single_cap_bce-a1_iface | -2.21 | -2.97 | -1.28 | -0.26 |
| s6_single_cap_snce-a1_iface | -0.51 | -0.85 | -0.30 | -0.22 |
| s2_single_free_vcs-a3_vcs | +0.34 | +0.02 | +0.10 | +0.36 |
| s4_single_cap_vcs-a3_vcs | -0.51 | -0.47 | -0.30 | -0.30 |
| s4_single_cap_vcs-s5_single_cap_bce | -0.26 | +0.99 | +0.00 | -0.59 |
| s4_single_cap_vcs-s6_single_cap_snce | -1.95 | -1.13 | -0.99 | -0.63 |
| s4_single_cap_vcs-s3_single_cap_delta | -0.53 | -0.44 | -0.39 | -0.34 |
| d1_vcs_i2t-a1_iface | -1.81 | -1.55 | -0.79 | +0.02 |
| d2_vcs_t2i-a1_iface | -3.02 | -2.93 | -1.38 | -0.89 |
| d3_snce_i2t-a1_iface | NA | NA | NA | NA |
| d4_snce_t2i-a1_iface | -1.44 | -1.37 | -0.89 | +0.08 |
| d1_vcs_i2t-d3_snce_i2t | NA | NA | NA | NA |
| d2_vcs_t2i-d4_snce_t2i | -1.58 | -1.56 | -0.49 | -0.97 |
| s4_single_cap_vcs-d1_vcs_i2t | -0.65 | -0.43 | -0.49 | -0.87 |
| s4_single_cap_vcs-d2_vcs_t2i | +0.55 | +0.95 | +0.10 | +0.04 |
| s6_single_cap_snce-d3_snce_i2t | NA | NA | NA | NA |
| s6_single_cap_snce-d4_snce_t2i | +0.93 | +0.52 | +0.59 | -0.30 |
| o1_rescue_native_t2i-a1_iface | -1.22 | -1.08 | -1.38 | -0.10 |
| o2_rescue_vcs_t2i-d2_vcs_t2i | +2.82 | +3.23 | +1.58 | +0.61 |
| o2_rescue_vcs_t2i-o1_rescue_native_t2i | +1.03 | +1.38 | +1.58 | -0.18 |
| o3_true_target_t2i-o1_rescue_native_t2i | +2.21 | +2.19 | +2.07 | -0.02 |
| o3_true_target_t2i-a1_iface | +0.99 | +1.10 | +0.69 | -0.12 |
| o3_true_target_t2i-o2_rescue_vcs_t2i | +1.18 | +0.80 | +0.49 | +0.16 |
| bn_cross_t2i-a1_iface | NA | NA | NA | NA |
| cn_mix_t2i-a1_iface | NA | NA | NA | NA |
| bn_cross_t2i-cn_mix_t2i | NA | NA | NA | NA |

## f40_s1001：2×2 交互（包 tools/analyze_results.py；I_V = S4−D1−D2+A1，I_N = S6−D3−D4+A1）

| 交互 | best rSum | last5 | i2t R@1 | t2i R@1 |
|---|---|---|---|---|
| I_vcs | +2.37 | +2.50 | +0.89 | +0.02 |
| I_snce | NA | NA | NA | NA |

## f60_s1001：各臂终点

| 臂 | best rSum | best epoch | last | last5 | i2t R@1 | i2t R@5 | i2t R@10 | t2i R@1 | t2i R@5 | t2i R@10 | attempts |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A0_native | 502.15 | 25 | 501.48 | 501.62 | 80.18 | 94.28 | 97.73 | 58.22 | 83.00 | 88.74 |  |
| a1_iface | 504.48 | 12 | 502.17 | 502.34 | 81.36 | 94.67 | 97.93 | 58.78 | 82.92 | 88.82 | 1 |
| a3_vcs | 503.04 | 35 | 502.70 | 502.82 | 80.37 | 94.97 | 97.34 | 58.64 | 82.86 | 88.86 | 1 |
| s1_balanced_dual | 503.98 | 16 | 503.81 | 503.58 | 81.26 | 94.77 | 97.14 | 58.66 | 83.31 | 88.84 | 1 |
| s2_single_free_vcs | 502.11 | 12 | 501.46 | 501.42 | 78.70 | 94.77 | 97.83 | 59.07 | 82.68 | 89.05 | 1 |
| s3_single_cap_delta | 504.30 | 15 | 502.78 | 502.82 | 80.67 | 95.07 | 97.93 | 58.58 | 82.96 | 89.09 | 1 |
| s4_single_cap_vcs | 504.89 | 14 | 503.61 | 503.70 | 81.26 | 95.27 | 97.44 | 58.54 | 83.08 | 89.31 | 1 |
| s5_single_cap_bce | 503.93 | 16 | 502.19 | 501.98 | 80.18 | 95.46 | 97.73 | 58.62 | 82.96 | 88.97 | 1 |
| s6_single_cap_snce | 504.26 | 12 | 501.52 | 501.85 | 80.57 | 95.17 | 97.24 | 58.99 | 83.25 | 89.03 | 2 |
| d1_vcs_i2t | 503.63 | 39 | 503.63 | 503.12 | 80.97 | 95.27 | 97.34 | 58.28 | 83.10 | 88.68 | 1 |
| d2_vcs_t2i | 503.75 | 22 | 502.88 | 502.88 | 80.28 | 95.17 | 97.63 | 58.21 | 83.25 | 89.21 | 1 |
| d3_snce_i2t | NA | | | | | | | | | | | |
| d4_snce_t2i | 504.22 | 17 | 502.49 | 502.45 | 79.88 | 95.27 | 98.13 | 58.78 | 83.06 | 89.11 | 1 |
| o1_rescue_native_t2i | 505.48 | 26 | 503.59 | 503.55 | 80.87 | 95.17 | 98.42 | 58.93 | 83.21 | 88.88 | 1 |
| o2_rescue_vcs_t2i | 503.53 | 37 | 503.39 | 503.42 | 79.59 | 95.36 | 98.13 | 58.80 | 82.86 | 88.80 | 1 |
| o3_true_target_t2i | 507.91 | 18 | 507.38 | 507.20 | 80.97 | 95.56 | 97.73 | 60.00 | 84.08 | 89.57 | 1 |
| bn_cross_t2i | NA | | | | | | | | | | | |
| cn_mix_t2i | NA | | | | | | | | | | | |

## f60_s1001：配对差（单 seed）

| 比较 | best rSum | last5 | i2t R@1 | t2i R@1 |
|---|---|---|---|---|
| s1_balanced_dual-a1_iface | -0.49 | +1.24 | -0.10 | -0.12 |
| s2_single_free_vcs-a1_iface | -2.37 | -0.92 | -2.66 | +0.30 |
| s3_single_cap_delta-a1_iface | -0.18 | +0.47 | -0.69 | -0.20 |
| s4_single_cap_vcs-a1_iface | +0.41 | +1.36 | -0.10 | -0.24 |
| s5_single_cap_bce-a1_iface | -0.55 | -0.36 | -1.18 | -0.16 |
| s6_single_cap_snce-a1_iface | -0.22 | -0.49 | -0.79 | +0.22 |
| s2_single_free_vcs-a3_vcs | -0.93 | -1.40 | -1.68 | +0.43 |
| s4_single_cap_vcs-a3_vcs | +1.85 | +0.88 | +0.89 | -0.10 |
| s4_single_cap_vcs-s5_single_cap_bce | +0.97 | +1.72 | +1.08 | -0.08 |
| s4_single_cap_vcs-s6_single_cap_snce | +0.63 | +1.85 | +0.69 | -0.45 |
| s4_single_cap_vcs-s3_single_cap_delta | +0.59 | +0.88 | +0.59 | -0.04 |
| d1_vcs_i2t-a1_iface | -0.85 | +0.77 | -0.39 | -0.49 |
| d2_vcs_t2i-a1_iface | -0.73 | +0.54 | -1.08 | -0.57 |
| d3_snce_i2t-a1_iface | NA | NA | NA | NA |
| d4_snce_t2i-a1_iface | -0.26 | +0.10 | -1.48 | +0.00 |
| d1_vcs_i2t-d3_snce_i2t | NA | NA | NA | NA |
| d2_vcs_t2i-d4_snce_t2i | -0.47 | +0.43 | +0.39 | -0.57 |
| s4_single_cap_vcs-d1_vcs_i2t | +1.26 | +0.58 | +0.30 | +0.26 |
| s4_single_cap_vcs-d2_vcs_t2i | +1.14 | +0.82 | +0.99 | +0.34 |
| s6_single_cap_snce-d3_snce_i2t | NA | NA | NA | NA |
| s6_single_cap_snce-d4_snce_t2i | +0.04 | -0.59 | +0.69 | +0.22 |
| o1_rescue_native_t2i-a1_iface | +1.01 | +1.20 | -0.49 | +0.16 |
| o2_rescue_vcs_t2i-d2_vcs_t2i | -0.22 | +0.54 | -0.69 | +0.59 |
| o2_rescue_vcs_t2i-o1_rescue_native_t2i | -1.95 | -0.13 | -1.28 | -0.14 |
| o3_true_target_t2i-o1_rescue_native_t2i | +2.43 | +3.66 | +0.10 | +1.07 |
| o3_true_target_t2i-a1_iface | +3.43 | +4.86 | -0.39 | +1.22 |
| o3_true_target_t2i-o2_rescue_vcs_t2i | +4.38 | +3.78 | +1.38 | +1.20 |
| bn_cross_t2i-a1_iface | NA | NA | NA | NA |
| cn_mix_t2i-a1_iface | NA | NA | NA | NA |
| bn_cross_t2i-cn_mix_t2i | NA | NA | NA | NA |

## f60_s1001：2×2 交互（包 tools/analyze_results.py；I_V = S4−D1−D2+A1，I_N = S6−D3−D4+A1）

| 交互 | best rSum | last5 | i2t R@1 | t2i R@1 |
|---|---|---|---|---|
| I_vcs | +1.99 | +0.05 | +1.38 | +0.83 |
| I_snce | NA | NA | NA | NA |

