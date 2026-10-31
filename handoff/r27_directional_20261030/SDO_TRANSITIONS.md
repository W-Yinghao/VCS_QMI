Publication copy (sample names aliased; numbers and conclusions unchanged).

# 逐 query 排名迁移（自动生成；只列计数，不含解读）

来源：审计 3 保存的 rank（各单元 best full-DEV checkpoint，同一固定 DEV 顺序）。“新/参照”= 参照 → 新 的 query 数；分桶 1 / 2–10 / >10（1 起计）。i2t 每格 1,014 个图像 query，t2i 5,070 个 caption query。单 seed。

缺失比较（任一单元未完成或未导出）：f40_s1001:d3_snce_i2t/a1_iface, f40_s1001:d1_vcs_i2t/d3_snce_i2t, f40_s1001:s6_single_cap_snce/d3_snce_i2t, f60_s1001:d3_snce_i2t/a1_iface, f60_s1001:d1_vcs_i2t/d3_snce_i2t, f60_s1001:s6_single_cap_snce/d3_snce_i2t

## f40_s1001

| 新/参照 | 方向 | 1->2-10 | 1->>10 | 2-10->1 | >10->1 | >10->2-10 | 救回 R@1 | 损失 R@1 | 净 | R@1 计数 参照 | R@1 计数 新 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| a1_iface/H0_A0 | i2t | 32 | 0 | 41 | 0 | 4 | 41 | 32 | +9 | 831 | 840 |
| a1_iface/H0_A0 | t2i | 182 | 0 | 198 | 1 | 70 | 199 | 182 | +17 | 3101 | 3118 |
| s1_balanced_dual/a1_iface | i2t | 33 | 0 | 36 | 0 | 4 | 36 | 33 | +3 | 840 | 843 |
| s1_balanced_dual/a1_iface | t2i | 172 | 0 | 176 | 0 | 48 | 176 | 172 | +4 | 3118 | 3122 |
| s2_single_free_vcs/a1_iface | i2t | 42 | 0 | 33 | 0 | 8 | 33 | 42 | -9 | 840 | 831 |
| s2_single_free_vcs/a1_iface | t2i | 181 | 0 | 171 | 0 | 46 | 171 | 181 | -10 | 3118 | 3108 |
| s3_single_cap_delta/a1_iface | i2t | 43 | 0 | 34 | 0 | 9 | 34 | 43 | -9 | 840 | 831 |
| s3_single_cap_delta/a1_iface | t2i | 181 | 0 | 155 | 0 | 53 | 155 | 181 | -26 | 3118 | 3092 |
| s4_single_cap_vcs/a1_iface | i2t | 45 | 0 | 32 | 0 | 4 | 32 | 45 | -13 | 840 | 827 |
| s4_single_cap_vcs/a1_iface | t2i | 198 | 0 | 155 | 0 | 50 | 155 | 198 | -43 | 3118 | 3075 |
| s5_single_cap_bce/a1_iface | i2t | 42 | 0 | 29 | 0 | 6 | 29 | 42 | -13 | 840 | 827 |
| s5_single_cap_bce/a1_iface | t2i | 175 | 1 | 163 | 0 | 46 | 163 | 176 | -13 | 3118 | 3105 |
| s6_single_cap_snce/a1_iface | i2t | 39 | 0 | 36 | 0 | 5 | 36 | 39 | -3 | 840 | 837 |
| s6_single_cap_snce/a1_iface | t2i | 177 | 0 | 166 | 0 | 56 | 166 | 177 | -11 | 3118 | 3107 |
| s4_single_cap_vcs/s6_single_cap_snce | i2t | 42 | 0 | 32 | 0 | 4 | 32 | 42 | -10 | 837 | 827 |
| s4_single_cap_vcs/s6_single_cap_snce | t2i | 185 | 0 | 153 | 0 | 54 | 153 | 185 | -32 | 3107 | 3075 |
| s2_single_free_vcs/a3_vcs | i2t | 33 | 0 | 34 | 0 | 4 | 34 | 33 | +1 | 830 | 831 |
| s2_single_free_vcs/a3_vcs | t2i | 151 | 0 | 169 | 0 | 58 | 169 | 151 | +18 | 3090 | 3108 |
| d1_vcs_i2t/a1_iface | i2t | 35 | 1 | 28 | 0 | 7 | 28 | 36 | -8 | 840 | 832 |
| d1_vcs_i2t/a1_iface | t2i | 153 | 0 | 154 | 0 | 51 | 154 | 153 | +1 | 3118 | 3119 |
| d2_vcs_t2i/a1_iface | i2t | 46 | 0 | 32 | 0 | 6 | 32 | 46 | -14 | 840 | 826 |
| d2_vcs_t2i/a1_iface | t2i | 217 | 1 | 173 | 0 | 54 | 173 | 218 | -45 | 3118 | 3073 |
| d3_snce_i2t/a1_iface | — | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA |
| d4_snce_t2i/a1_iface | i2t | 38 | 0 | 29 | 0 | 5 | 29 | 38 | -9 | 840 | 831 |
| d4_snce_t2i/a1_iface | t2i | 169 | 0 | 173 | 0 | 54 | 173 | 169 | +4 | 3118 | 3122 |
| d1_vcs_i2t/d3_snce_i2t | — | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA |
| d2_vcs_t2i/d4_snce_t2i | i2t | 38 | 0 | 33 | 0 | 4 | 33 | 38 | -5 | 831 | 826 |
| d2_vcs_t2i/d4_snce_t2i | t2i | 223 | 0 | 174 | 0 | 56 | 174 | 223 | -49 | 3122 | 3073 |
| s4_single_cap_vcs/d1_vcs_i2t | i2t | 40 | 0 | 35 | 0 | 2 | 35 | 40 | -5 | 832 | 827 |
| s4_single_cap_vcs/d1_vcs_i2t | t2i | 182 | 0 | 138 | 0 | 58 | 138 | 182 | -44 | 3119 | 3075 |
| s4_single_cap_vcs/d2_vcs_t2i | i2t | 43 | 0 | 44 | 0 | 3 | 44 | 43 | +1 | 826 | 827 |
| s4_single_cap_vcs/d2_vcs_t2i | t2i | 183 | 0 | 185 | 0 | 61 | 185 | 183 | +2 | 3073 | 3075 |
| s6_single_cap_snce/d3_snce_i2t | — | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA |
| s6_single_cap_snce/d4_snce_t2i | i2t | 30 | 0 | 36 | 0 | 3 | 36 | 30 | +6 | 831 | 837 |
| s6_single_cap_snce/d4_snce_t2i | t2i | 172 | 0 | 157 | 0 | 57 | 157 | 172 | -15 | 3122 | 3107 |
| o1_rescue_native_t2i/a1_iface | i2t | 45 | 0 | 31 | 0 | 6 | 31 | 45 | -14 | 840 | 826 |
| o1_rescue_native_t2i/a1_iface | t2i | 172 | 0 | 167 | 0 | 56 | 167 | 172 | -5 | 3118 | 3113 |
| o2_rescue_vcs_t2i/d2_vcs_t2i | i2t | 32 | 0 | 48 | 0 | 3 | 48 | 32 | +16 | 826 | 842 |
| o2_rescue_vcs_t2i/d2_vcs_t2i | t2i | 166 | 0 | 197 | 0 | 63 | 197 | 166 | +31 | 3073 | 3104 |
| o2_rescue_vcs_t2i/o1_rescue_native_t2i | i2t | 29 | 0 | 44 | 1 | 4 | 45 | 29 | +16 | 826 | 842 |
| o2_rescue_vcs_t2i/o1_rescue_native_t2i | t2i | 165 | 2 | 158 | 0 | 62 | 158 | 167 | -9 | 3113 | 3104 |
| o3_true_target_t2i/o1_rescue_native_t2i | i2t | 27 | 0 | 48 | 0 | 3 | 48 | 27 | +21 | 826 | 847 |
| o3_true_target_t2i/o1_rescue_native_t2i | t2i | 186 | 0 | 184 | 1 | 68 | 185 | 186 | -1 | 3113 | 3112 |
| o3_true_target_t2i/a1_iface | i2t | 36 | 0 | 43 | 0 | 6 | 43 | 36 | +7 | 840 | 847 |
| o3_true_target_t2i/a1_iface | t2i | 168 | 0 | 162 | 0 | 66 | 162 | 168 | -6 | 3118 | 3112 |
| bn_cross_t2i/a1_iface | i2t | 40 | 0 | 35 | 0 | 5 | 35 | 40 | -5 | 840 | 835 |
| bn_cross_t2i/a1_iface | t2i | 172 | 0 | 179 | 0 | 54 | 179 | 172 | +7 | 3118 | 3125 |
| cn_mix_t2i/a1_iface | i2t | 34 | 0 | 35 | 0 | 5 | 35 | 34 | +1 | 840 | 841 |
| cn_mix_t2i/a1_iface | t2i | 169 | 0 | 167 | 0 | 50 | 167 | 169 | -2 | 3118 | 3116 |
| bn_cross_t2i/cn_mix_t2i | i2t | 34 | 0 | 28 | 0 | 4 | 28 | 34 | -6 | 841 | 835 |
| bn_cross_t2i/cn_mix_t2i | t2i | 169 | 0 | 178 | 0 | 64 | 178 | 169 | +9 | 3116 | 3125 |

## f60_s1001

| 新/参照 | 方向 | 1->2-10 | 1->>10 | 2-10->1 | >10->1 | >10->2-10 | 救回 R@1 | 损失 R@1 | 净 | R@1 计数 参照 | R@1 计数 新 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| a1_iface/H0_A0 | i2t | 34 | 0 | 46 | 0 | 9 | 46 | 34 | +12 | 813 | 825 |
| a1_iface/H0_A0 | t2i | 203 | 0 | 231 | 0 | 79 | 231 | 203 | +28 | 2952 | 2980 |
| s1_balanced_dual/a1_iface | i2t | 38 | 0 | 37 | 0 | 4 | 37 | 38 | -1 | 825 | 824 |
| s1_balanced_dual/a1_iface | t2i | 212 | 1 | 206 | 1 | 80 | 207 | 213 | -6 | 2980 | 2974 |
| s2_single_free_vcs/a1_iface | i2t | 51 | 2 | 26 | 0 | 9 | 26 | 53 | -27 | 825 | 798 |
| s2_single_free_vcs/a1_iface | t2i | 206 | 2 | 223 | 0 | 82 | 223 | 208 | +15 | 2980 | 2995 |
| s3_single_cap_delta/a1_iface | i2t | 46 | 0 | 39 | 0 | 7 | 39 | 46 | -7 | 825 | 818 |
| s3_single_cap_delta/a1_iface | t2i | 227 | 0 | 217 | 0 | 86 | 217 | 227 | -10 | 2980 | 2970 |
| s4_single_cap_vcs/a1_iface | i2t | 45 | 0 | 44 | 0 | 7 | 44 | 45 | -1 | 825 | 824 |
| s4_single_cap_vcs/a1_iface | t2i | 230 | 0 | 218 | 0 | 89 | 218 | 230 | -12 | 2980 | 2968 |
| s5_single_cap_bce/a1_iface | i2t | 38 | 0 | 26 | 0 | 7 | 26 | 38 | -12 | 825 | 813 |
| s5_single_cap_bce/a1_iface | t2i | 219 | 1 | 212 | 0 | 74 | 212 | 220 | -8 | 2980 | 2972 |
| s6_single_cap_snce/a1_iface | i2t | 40 | 0 | 32 | 0 | 6 | 32 | 40 | -8 | 825 | 817 |
| s6_single_cap_snce/a1_iface | t2i | 225 | 0 | 236 | 0 | 79 | 236 | 225 | +11 | 2980 | 2991 |
| s4_single_cap_vcs/s6_single_cap_snce | i2t | 39 | 1 | 47 | 0 | 10 | 47 | 40 | +7 | 817 | 824 |
| s4_single_cap_vcs/s6_single_cap_snce | t2i | 218 | 0 | 195 | 0 | 89 | 195 | 218 | -23 | 2991 | 2968 |
| s2_single_free_vcs/a3_vcs | i2t | 48 | 2 | 33 | 0 | 10 | 33 | 50 | -17 | 815 | 798 |
| s2_single_free_vcs/a3_vcs | t2i | 215 | 0 | 237 | 0 | 86 | 237 | 215 | +22 | 2973 | 2995 |
| d1_vcs_i2t/a1_iface | i2t | 38 | 1 | 35 | 0 | 6 | 35 | 39 | -4 | 825 | 821 |
| d1_vcs_i2t/a1_iface | t2i | 223 | 0 | 198 | 0 | 63 | 198 | 223 | -25 | 2980 | 2955 |
| d2_vcs_t2i/a1_iface | i2t | 42 | 0 | 31 | 0 | 8 | 31 | 42 | -11 | 825 | 814 |
| d2_vcs_t2i/a1_iface | t2i | 234 | 1 | 206 | 0 | 87 | 206 | 235 | -29 | 2980 | 2951 |
| d3_snce_i2t/a1_iface | — | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA |
| d4_snce_t2i/a1_iface | i2t | 39 | 0 | 24 | 0 | 11 | 24 | 39 | -15 | 825 | 810 |
| d4_snce_t2i/a1_iface | t2i | 213 | 0 | 213 | 0 | 86 | 213 | 213 | +0 | 2980 | 2980 |
| d1_vcs_i2t/d3_snce_i2t | — | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA |
| d2_vcs_t2i/d4_snce_t2i | i2t | 36 | 0 | 40 | 0 | 2 | 40 | 36 | +4 | 810 | 814 |
| d2_vcs_t2i/d4_snce_t2i | t2i | 198 | 0 | 169 | 0 | 64 | 169 | 198 | -29 | 2980 | 2951 |
| s4_single_cap_vcs/d1_vcs_i2t | i2t | 37 | 0 | 40 | 0 | 9 | 40 | 37 | +3 | 821 | 824 |
| s4_single_cap_vcs/d1_vcs_i2t | t2i | 210 | 1 | 224 | 0 | 85 | 224 | 211 | +13 | 2955 | 2968 |
| s4_single_cap_vcs/d2_vcs_t2i | i2t | 40 | 0 | 50 | 0 | 8 | 50 | 40 | +10 | 814 | 824 |
| s4_single_cap_vcs/d2_vcs_t2i | t2i | 205 | 0 | 222 | 0 | 70 | 222 | 205 | +17 | 2951 | 2968 |
| s6_single_cap_snce/d3_snce_i2t | — | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA |
| s6_single_cap_snce/d4_snce_t2i | i2t | 34 | 0 | 41 | 0 | 4 | 41 | 34 | +7 | 810 | 817 |
| s6_single_cap_snce/d4_snce_t2i | t2i | 203 | 0 | 212 | 2 | 71 | 214 | 203 | +11 | 2980 | 2991 |
| o1_rescue_native_t2i/a1_iface | i2t | 41 | 0 | 36 | 0 | 11 | 36 | 41 | -5 | 825 | 820 |
| o1_rescue_native_t2i/a1_iface | t2i | 218 | 1 | 227 | 0 | 77 | 227 | 219 | +8 | 2980 | 2988 |
| o2_rescue_vcs_t2i/d2_vcs_t2i | i2t | 39 | 0 | 32 | 0 | 8 | 32 | 39 | -7 | 814 | 807 |
| o2_rescue_vcs_t2i/d2_vcs_t2i | t2i | 161 | 1 | 192 | 0 | 61 | 192 | 162 | +30 | 2951 | 2981 |
| o2_rescue_vcs_t2i/o1_rescue_native_t2i | i2t | 41 | 0 | 28 | 0 | 4 | 28 | 41 | -13 | 820 | 807 |
| o2_rescue_vcs_t2i/o1_rescue_native_t2i | t2i | 184 | 0 | 177 | 0 | 71 | 177 | 184 | -7 | 2988 | 2981 |
| o3_true_target_t2i/o1_rescue_native_t2i | i2t | 38 | 0 | 39 | 0 | 4 | 39 | 38 | +1 | 820 | 821 |
| o3_true_target_t2i/o1_rescue_native_t2i | t2i | 155 | 1 | 210 | 0 | 87 | 210 | 156 | +54 | 2988 | 3042 |
| o3_true_target_t2i/a1_iface | i2t | 43 | 0 | 39 | 0 | 4 | 39 | 43 | -4 | 825 | 821 |
| o3_true_target_t2i/a1_iface | t2i | 195 | 0 | 257 | 0 | 91 | 257 | 195 | +62 | 2980 | 3042 |
| bn_cross_t2i/a1_iface | i2t | 43 | 1 | 34 | 0 | 6 | 34 | 44 | -10 | 825 | 815 |
| bn_cross_t2i/a1_iface | t2i | 254 | 2 | 199 | 0 | 85 | 199 | 256 | -57 | 2980 | 2923 |
| cn_mix_t2i/a1_iface | i2t | 45 | 0 | 33 | 0 | 9 | 33 | 45 | -12 | 825 | 813 |
| cn_mix_t2i/a1_iface | t2i | 222 | 2 | 213 | 0 | 72 | 213 | 224 | -11 | 2980 | 2969 |
| bn_cross_t2i/cn_mix_t2i | i2t | 42 | 0 | 44 | 0 | 3 | 44 | 42 | +2 | 813 | 815 |
| bn_cross_t2i/cn_mix_t2i | t2i | 251 | 0 | 202 | 3 | 84 | 205 | 251 | -46 | 2969 | 2923 |

