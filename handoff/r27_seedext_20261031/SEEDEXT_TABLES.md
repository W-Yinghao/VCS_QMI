Publication copy (sample names aliased; numbers and conclusions unchanged).

# Flickr 多 seed 数字表（R27-D / O / O3 / CS；自动生成；full DEV；未读 TEST；只列数字，不含解读）

seed1001 为原单 seed 单元；seed1002/1003 为种子扩展（训练代码除种子门槛外逐字节相同）。A1 参照取同格同 seed 的 R27 a1_iface。未完成单元记 NA，不以其他臂替代；均值只在已完成的 seed 上计算并标注 n。R27-O / O3 为显式使用真对应的诊断，不进入主结果表。

未完成单元：f40_s1002/d3_snce_i2t, f40_s1002/d4_snce_t2i, f40_s1002/o1_rescue_native_t2i, f40_s1002/o2_rescue_vcs_t2i, f40_s1002/o3_true_target_t2i, f40_s1002/bn_cross_t2i, f40_s1002/cn_mix_t2i, f40_s1003/d3_snce_i2t, f40_s1003/d4_snce_t2i, f40_s1003/o1_rescue_native_t2i, f40_s1003/o2_rescue_vcs_t2i, f40_s1003/o3_true_target_t2i, f40_s1003/bn_cross_t2i, f40_s1003/cn_mix_t2i, f60_s1002/d3_snce_i2t, f60_s1002/o1_rescue_native_t2i, f60_s1002/o2_rescue_vcs_t2i, f60_s1002/o3_true_target_t2i, f60_s1002/bn_cross_t2i, f60_s1002/cn_mix_t2i, f60_s1003/d2_vcs_t2i, f60_s1003/d3_snce_i2t, f60_s1003/d4_snce_t2i, f60_s1003/o1_rescue_native_t2i, f60_s1003/o2_rescue_vcs_t2i, f60_s1003/o3_true_target_t2i, f60_s1003/bn_cross_t2i, f60_s1003/cn_mix_t2i

## Flickr 40%：逐 seed 最佳 full-DEV rSum

| 臂 | s1001 | s1002 | s1003 | n | 均值 best | 均值 last5 | −A1 逐 seed | −A1 均值 | >0 |
|---|---|---|---|---|---|---|---|---|---|
| a1_iface | 514.97 | 510.49 | 515.05 | 3 | 513.50 | 512.67 | — | — | — |
| d1_vcs_i2t | 513.16 | 511.44 | 514.40 | 3 | 513.00 | 512.43 | -1.81, +0.95, -0.65 | -0.51 | 1/3 |
| d2_vcs_t2i | 511.95 | 509.86 | 513.83 | 3 | 511.88 | 511.06 | -3.02, -0.63, -1.22 | -1.62 | 0/3 |
| d3_snce_i2t | 513.02 | NA | NA | 1 | 513.02 | 512.27 | -1.95 | -1.95 | 0/1 |
| d4_snce_t2i | 513.53 | NA | NA | 1 | 513.53 | 512.70 | -1.44 | -1.44 | 0/1 |
| o1_rescue_native_t2i | 513.75 | NA | NA | 1 | 513.75 | 512.99 | -1.22 | -1.22 | 0/1 |
| o2_rescue_vcs_t2i | 514.77 | NA | NA | 1 | 514.77 | 514.37 | -0.20 | -0.20 | 0/1 |
| o3_true_target_t2i | 515.96 | NA | NA | 1 | 515.96 | 515.17 | +0.99 | +0.99 | 1/1 |
| bn_cross_t2i | 515.38 | NA | NA | 1 | 515.38 | 514.20 | +0.41 | +0.41 | 1/1 |
| cn_mix_t2i | 514.60 | NA | NA | 1 | 514.60 | 512.77 | -0.37 | -0.37 | 0/1 |

## Flickr 60%：逐 seed 最佳 full-DEV rSum

| 臂 | s1001 | s1002 | s1003 | n | 均值 best | 均值 last5 | −A1 逐 seed | −A1 均值 | >0 |
|---|---|---|---|---|---|---|---|---|---|
| a1_iface | 504.48 | 482.68 | 495.96 | 3 | 494.37 | 492.28 | — | — | — |
| d1_vcs_i2t | 503.63 | 482.54 | 494.81 | 3 | 493.66 | 492.26 | -0.85, -0.14, -1.14 | -0.71 | 0/3 |
| d2_vcs_t2i | 503.75 | 483.96 | NA | 2 | 493.86 | 493.05 | -0.73, +1.28 | +0.28 | 1/2 |
| d3_snce_i2t | 504.42 | NA | NA | 1 | 504.42 | 503.75 | -0.06 | -0.06 | 0/1 |
| d4_snce_t2i | 504.22 | 482.72 | NA | 2 | 493.47 | 491.40 | -0.26, +0.04 | -0.11 | 1/2 |
| o1_rescue_native_t2i | 505.48 | NA | NA | 1 | 505.48 | 503.55 | +1.01 | +1.01 | 1/1 |
| o2_rescue_vcs_t2i | 503.53 | NA | NA | 1 | 503.53 | 503.42 | -0.95 | -0.95 | 0/1 |
| o3_true_target_t2i | 507.91 | NA | NA | 1 | 507.91 | 507.20 | +3.43 | +3.43 | 1/1 |
| bn_cross_t2i | 502.17 | NA | NA | 1 | 502.17 | 500.70 | -2.31 | -2.31 | 0/1 |
| cn_mix_t2i | 503.25 | NA | NA | 1 | 503.25 | 501.27 | -1.22 | -1.22 | 0/1 |

