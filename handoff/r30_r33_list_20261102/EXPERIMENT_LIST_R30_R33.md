Publication copy (sample names aliased; numbers and conclusions unchanged).

# R30–R33 实验清单（自动生成；只列设计、状态与数字，不含解读）

状态来自各单元 DONE 文件与当时的 squeue；DEV 为各线自身登记的终点；TEST 为常设 TEST 工具对 DEV（或登记的固定 epoch）选出 checkpoint 的一次性评测 rSum（COCO 为 5-fold 1K），未评测记 —。TEST 不用于任何选择。

## R30 v1.1 + 修订 A1：共同原生 warmup 后的直接竞争学习（阶段交叉与六臂完整比较）

单元数 36；DONE 36

| 单元 | 条件 | seed | 臂 | 轮/步 | 状态 | DEV（E10 rSum / best-40 rSum） | TEST rSum |
|---|---|---|---|---|---|---|---|
| r30_f00_s1001_H0 | Flickr 0% | 1001 | H→H | 40 | DONE | 519.11 / 522.07 | 525.74 |
| r30_f00_s1001_L0 | Flickr 0% | 1001 | H→L | 40 | DONE | 503.21 / 508.78 | 513.46 |
| r30_f00_s1001_Ls | Flickr 0% | 1001 | H→L (static scale) | 40 | DONE | 505.01 / 511.20 | 514.44 |
| r30_f00_s1001_PN | Flickr 0% | 1001 | H→PN (static scale) | 40 | DONE | 517.06 / 522.88 | 527.88 |
| r30_f00_s1001_PV | Flickr 0% | 1001 | H→PV (static scale) | 40 | DONE | 520.06 / 522.84 | 524.72 |
| r30_f00_s1001_PB | Flickr 0% | 1001 | H→PB (static scale) | 40 | DONE | 516.96 / 522.37 | 525.24 |
| r30_f60_s1001_H0 | Flickr 60% | 1001 | H→H | 40 | DONE | 501.58 / 505.21 | 503.52 |
| r30_f60_s1001_L0 | Flickr 60% | 1001 | H→L | 40 | DONE | 478.46 / 482.92 | 480.38 |
| r30_f60_s1001_Ls | Flickr 60% | 1001 | H→L (static scale) | 40 | DONE | 478.07 / 483.29 | 483.16 |
| r30_f60_s1001_PN | Flickr 60% | 1001 | H→PN (static scale) | 40 | DONE | 496.21 / 500.59 | 502.48 |
| r30_f60_s1001_PV | Flickr 60% | 1001 | H→PV (static scale) | 40 | DONE | 496.55 / 501.89 | 501.80 |
| r30_f60_s1001_PB | Flickr 60% | 1001 | H→PB (static scale) | 40 | DONE | 497.06 / 501.38 | 503.34 |
| r30v11_f00_s1001_HN | Flickr 0% | 1001 | H→N | 40 | DONE | 514.06 / 518.21 | 522.64 |
| r30v11_f00_s1001_HB | Flickr 0% | 1001 | H→B | 40 | DONE | 504.22 / 509.90 | 513.70 |
| r30v11_f60_s1001_HN | Flickr 60% | 1001 | H→N | 40 | DONE | 493.23 / 495.88 | 496.78 |
| r30v11_f60_s1001_HB | Flickr 60% | 1001 | H→B | 40 | DONE | 481.28 / 481.28 | 481.08 |
| r30v11_f60_s1001_LN | Flickr 60% | 1001 | L→N | 10 | DONE | 479.70 / — | 484.28 |
| r30v11_f60_s1001_LL | Flickr 60% | 1001 | L→L | 10 | DONE | 444.87 / — | 449.32 |
| r30v11_f60_s1002_HN | Flickr 60% | 1002 | H→N | 10 | DONE | 469.88 / — | 474.30 |
| r30v11_f60_s1002_HB | Flickr 60% | 1002 | H→B | 10 | DONE | 457.32 / — | 455.68 |
| r30v11_f60_s1002_HL | Flickr 60% | 1002 | H→L | 10 | DONE | 453.02 / — | 457.10 |
| r30v11_f60_s1002_HH | Flickr 60% | 1002 | H→H | 40 | DONE | 479.35 / 482.94 | 482.28 |
| r30v11_f60_s1002_LN | Flickr 60% | 1002 | L→N | 10 | DONE | 460.12 / — | 460.38 |
| r30v11_f60_s1002_LL | Flickr 60% | 1002 | L→L | 10 | DONE | 409.01 / — | 413.98 |
| r30v11_f60_s1003_HN | Flickr 60% | 1003 | H→N | 10 | DONE | 485.84 / — | 489.78 |
| r30v11_f60_s1003_HB | Flickr 60% | 1003 | H→B | 10 | DONE | 469.41 / — | 470.68 |
| r30v11_f60_s1003_HL | Flickr 60% | 1003 | H→L | 10 | DONE | 470.00 / — | 472.46 |
| r30v11_f60_s1003_HH | Flickr 60% | 1003 | H→H | 40 | DONE | 494.40 / 496.07 | 496.44 |
| r30v11_f60_s1003_LN | Flickr 60% | 1003 | L→N | 10 | DONE | 483.73 / — | 485.36 |
| r30v11_f60_s1003_LL | Flickr 60% | 1003 | L→L | 10 | DONE | 448.99 / — | 452.12 |
| r30a1_f60_s1002_PN | Flickr 60% | 1002 | H→PN (static scale) | 40 | DONE | 474.48 / 480.34 | 482.32 |
| r30a1_f60_s1002_PV | Flickr 60% | 1002 | H→PV (static scale) | 40 | DONE | 473.63 / 481.20 | 480.78 |
| r30a1_f60_s1002_PB | Flickr 60% | 1002 | H→PB (static scale) | 40 | DONE | 473.33 / 482.50 | 481.48 |
| r30a1_f60_s1003_PN | Flickr 60% | 1003 | H→PN (static scale) | 40 | DONE | 488.19 / 495.74 | 496.92 |
| r30a1_f60_s1003_PV | Flickr 60% | 1003 | H→PV (static scale) | 40 | DONE | 488.28 / 493.10 | 495.60 |
| r30a1_f60_s1003_PB | Flickr 60% | 1003 | H→PB (static scale) | 40 | DONE | 488.07 / 494.24 | 496.14 |

## R31：IN²R 复现忠实度（(a) 官方 NCR 20% 噪声；(b) 作者 run.py 默认；(b′) 默认 + i2t/t2i）

单元数 15；DONE 15

| 单元 | 条件 | seed | 臂 | 轮/步 | 状态 | DEV（best full-DEV rSum） | TEST rSum |
|---|---|---|---|---|---|---|---|
| f20_s1001/H0_codecfg | Flickr 20% (official NCR index) | 1001 | set a: H0_codecfg | 40 | DONE | 517.71 | 519.60 |
| f20_s1002/H0_codecfg | Flickr 20% (official NCR index) | 1002 | set a: H0_codecfg | 40 | DONE | 516.59 | 518.92 |
| f20_s1003/H0_codecfg | Flickr 20% (official NCR index) | 1003 | set a: H0_codecfg | 40 | DONE | 516.27 | 518.56 |
| f40_s1001/H0_default | Flickr 40% | 1001 | set b: H0_default | 40 | DONE | 514.18 | 516.24 |
| f40_s1002/H0_default | Flickr 40% | 1002 | set b: H0_default | 40 | DONE | 509.45 | 512.04 |
| f40_s1003/H0_default | Flickr 40% | 1003 | set b: H0_default | 40 | DONE | 510.79 | 515.04 |
| cc_s1001/H0_default | CC152K | 1001 | set b: H0_default | 40 | DONE | 389.70 | 401.90 |
| cc_s1002/H0_default | CC152K | 1002 | set b: H0_default | 40 | DONE | 394.60 | 402.80 |
| cc_s1003/H0_default | CC152K | 1003 | set b: H0_default | 40 | DONE | 392.60 | 406.40 |
| f40_s1001/H0_default_i2t_t2i | Flickr 40% | 1001 | set b_prime: H0_default_i2t_t2i | 40 | DONE | 514.28 | 518.32 |
| f40_s1002/H0_default_i2t_t2i | Flickr 40% | 1002 | set b_prime: H0_default_i2t_t2i | 40 | DONE | 510.47 | 510.40 |
| f40_s1003/H0_default_i2t_t2i | Flickr 40% | 1003 | set b_prime: H0_default_i2t_t2i | 40 | DONE | 512.88 | 515.60 |
| cc_s1001/H0_default_i2t_t2i | CC152K | 1001 | set b_prime: H0_default_i2t_t2i | 40 | DONE | 391.00 | 405.20 |
| cc_s1002/H0_default_i2t_t2i | CC152K | 1002 | set b_prime: H0_default_i2t_t2i | 40 | DONE | 392.50 | 404.50 |
| cc_s1003/H0_default_i2t_t2i | CC152K | 1003 | set b_prime: H0_default_i2t_t2i | 40 | DONE | 391.50 | 404.20 |

## R32：VCS_NATIVE_RATIO_v1（GPO 双塔 + 有边缘约束的 ratio critic；S0/W2/F1/E2/C2）

单元数 95；DONE 95

| 单元 | 条件 | seed | 臂 | 轮/步 | 状态 | DEV（epoch-20 DEV rSum (E2/C2) / J (S0)） | TEST rSum |
|---|---|---|---|---|---|---|---|
| nrcs_v1_s0_n00_s1001_cosine_J_b0_i0 | latent_mix | 1001 | S0: cosine_J | 0 | DONE | J 0.1094 | — |
| nrcs_v1_s0_n00_s1001_ratio_J_b1_i0 | latent_mix | 1001 | S0: ratio_J | 0 | DONE | J 0.1097 | — |
| nrcs_v1_s0_n00_s1001_ratio_logistic_b1_i0 | latent_mix | 1001 | S0: ratio_logistic | 0 | DONE | J 0.1097 | — |
| nrcs_v1_s0_n00_s1002_cosine_J_b0_i0 | latent_mix | 1002 | S0: cosine_J | 0 | DONE | J 0.1093 | — |
| nrcs_v1_s0_n00_s1002_ratio_J_b1_i0 | latent_mix | 1002 | S0: ratio_J | 0 | DONE | J 0.1097 | — |
| nrcs_v1_s0_n00_s1002_ratio_logistic_b1_i0 | latent_mix | 1002 | S0: ratio_logistic | 0 | DONE | J 0.1097 | — |
| nrcs_v1_s0_n00_s1003_cosine_J_b0_i0 | latent_mix | 1003 | S0: cosine_J | 0 | DONE | J 0.1094 | — |
| nrcs_v1_s0_n00_s1003_ratio_J_b1_i0 | latent_mix | 1003 | S0: ratio_J | 0 | DONE | J 0.1097 | — |
| nrcs_v1_s0_n00_s1003_ratio_logistic_b1_i0 | latent_mix | 1003 | S0: ratio_logistic | 0 | DONE | J 0.1097 | — |
| nrcs_v1_s0_n60_s1001_cosine_J_b0_i0 | latent_mix | 1001 | S0: cosine_J | 0 | DONE | J 0.0164 | — |
| nrcs_v1_s0_n60_s1001_ratio_J_b1_i0 | latent_mix | 1001 | S0: ratio_J | 0 | DONE | J 0.0164 | — |
| nrcs_v1_s0_n60_s1001_ratio_logistic_b1_i0 | latent_mix | 1001 | S0: ratio_logistic | 0 | DONE | J 0.0164 | — |
| nrcs_v1_s0_n60_s1002_cosine_J_b0_i0 | latent_mix | 1002 | S0: cosine_J | 0 | DONE | J 0.0164 | — |
| nrcs_v1_s0_n60_s1002_ratio_J_b1_i0 | latent_mix | 1002 | S0: ratio_J | 0 | DONE | J 0.0164 | — |
| nrcs_v1_s0_n60_s1002_ratio_logistic_b1_i0 | latent_mix | 1002 | S0: ratio_logistic | 0 | DONE | J 0.0164 | — |
| nrcs_v1_s0_n60_s1003_cosine_J_b0_i0 | latent_mix | 1003 | S0: cosine_J | 0 | DONE | J 0.0164 | — |
| nrcs_v1_s0_n60_s1003_ratio_J_b1_i0 | latent_mix | 1003 | S0: ratio_J | 0 | DONE | J 0.0164 | — |
| nrcs_v1_s0_n60_s1003_ratio_logistic_b1_i0 | latent_mix | 1003 | S0: ratio_logistic | 0 | DONE | J 0.0164 | — |
| nrcs_v1_s0_n90_s1001_cosine_J_b0_i0 | latent_mix | 1001 | S0: cosine_J | 0 | DONE | J 0.0011 | — |
| nrcs_v1_s0_n90_s1001_ratio_J_b1_i0 | latent_mix | 1001 | S0: ratio_J | 0 | DONE | J 0.0011 | — |
| nrcs_v1_s0_n90_s1001_ratio_logistic_b1_i0 | latent_mix | 1001 | S0: ratio_logistic | 0 | DONE | J 0.0011 | — |
| nrcs_v1_s0_n90_s1002_cosine_J_b0_i0 | latent_mix | 1002 | S0: cosine_J | 0 | DONE | J 0.0011 | — |
| nrcs_v1_s0_n90_s1002_ratio_J_b1_i0 | latent_mix | 1002 | S0: ratio_J | 0 | DONE | J 0.0011 | — |
| nrcs_v1_s0_n90_s1002_ratio_logistic_b1_i0 | latent_mix | 1002 | S0: ratio_logistic | 0 | DONE | J 0.0011 | — |
| nrcs_v1_s0_n90_s1003_cosine_J_b0_i0 | latent_mix | 1003 | S0: cosine_J | 0 | DONE | J 0.0011 | — |
| nrcs_v1_s0_n90_s1003_ratio_J_b1_i0 | latent_mix | 1003 | S0: ratio_J | 0 | DONE | J 0.0011 | — |
| nrcs_v1_s0_n90_s1003_ratio_logistic_b1_i0 | latent_mix | 1003 | S0: ratio_logistic | 0 | DONE | J 0.0011 | — |
| nrcs_v1_w2_n00_s1001_cosine_hinge_b0_i0p1 | flickr_fit | 1001 | W2: cosine_hinge | 5 | DONE | NA | — |
| nrcs_v1_w2_n00_s1002_cosine_hinge_b0_i0p1 | flickr_fit | 1002 | W2: cosine_hinge | 5 | DONE | NA | — |
| nrcs_v1_w2_n00_s1003_cosine_hinge_b0_i0p1 | flickr_fit | 1003 | W2: cosine_hinge | 5 | DONE | NA | — |
| nrcs_v1_w2_n60_s1001_cosine_hinge_b0_i0p1 | flickr_fit | 1001 | W2: cosine_hinge | 5 | DONE | NA | — |
| nrcs_v1_w2_n60_s1002_cosine_hinge_b0_i0p1 | flickr_fit | 1002 | W2: cosine_hinge | 5 | DONE | NA | — |
| nrcs_v1_w2_n60_s1003_cosine_hinge_b0_i0p1 | flickr_fit | 1003 | W2: cosine_hinge | 5 | DONE | NA | — |
| nrcs_v1_f1_n00_s1001_cosine_J_b0_i0 | flickr_frozen_W2 | 1001 | F1: cosine_J | 0 | DONE | 281.07 | — |
| nrcs_v1_f1_n00_s1001_ratio_J_b1_i0 | flickr_frozen_W2 | 1001 | F1: ratio_J | 0 | DONE | 96.86 | — |
| nrcs_v1_f1_n00_s1001_ratio_logistic_b1_i0 | flickr_frozen_W2 | 1001 | F1: ratio_logistic | 0 | DONE | 105.82 | — |
| nrcs_v1_f1_n00_s1001_ratio_J_b0_i0 | flickr_frozen_W2 | 1001 | F1: ratio_J | 0 | DONE | 122.43 | — |
| nrcs_v1_e2_n00_s1001_cosine_J_b0_i0 | flickr_trainable_GPO | 1001 | E2: cosine_J | 20 | DONE | 374.58 | — |
| nrcs_v1_e2_n00_s1001_cosine_J_b0_i0p1 | flickr_trainable_GPO | 1001 | E2: cosine_J | 20 | DONE | 383.91 | — |
| nrcs_v1_e2_n00_s1001_ratio_J_b1_i0 | flickr_trainable_GPO | 1001 | E2: ratio_J | 20 | DONE | 121.52 | — |
| nrcs_v1_e2_n00_s1001_ratio_J_b1_i0p1 | flickr_trainable_GPO | 1001 | E2: ratio_J | 20 | DONE | 134.60 | — |
| nrcs_v1_e2_n00_s1001_ratio_logistic_b1_i0p1 | flickr_trainable_GPO | 1001 | E2: ratio_logistic | 20 | DONE | 152.45 | — |
| nrcs_v1_e2_n00_s1001_cosine_hinge_b0_i0p1 | flickr_trainable_GPO | 1001 | E2: cosine_hinge | 20 | DONE | 475.23 | — |
| nrcs_v1_f1_n00_s1002_cosine_J_b0_i0 | flickr_frozen_W2 | 1002 | F1: cosine_J | 0 | DONE | 331.16 | — |
| nrcs_v1_f1_n00_s1002_ratio_J_b1_i0 | flickr_frozen_W2 | 1002 | F1: ratio_J | 0 | DONE | 107.55 | — |
| nrcs_v1_f1_n00_s1002_ratio_logistic_b1_i0 | flickr_frozen_W2 | 1002 | F1: ratio_logistic | 0 | DONE | 122.09 | — |
| nrcs_v1_f1_n00_s1002_ratio_J_b0_i0 | flickr_frozen_W2 | 1002 | F1: ratio_J | 0 | DONE | 142.68 | — |
| nrcs_v1_e2_n00_s1002_cosine_J_b0_i0 | flickr_trainable_GPO | 1002 | E2: cosine_J | 20 | DONE | 408.15 | — |
| nrcs_v1_e2_n00_s1002_cosine_J_b0_i0p1 | flickr_trainable_GPO | 1002 | E2: cosine_J | 20 | DONE | 408.78 | — |
| nrcs_v1_e2_n00_s1002_ratio_J_b1_i0 | flickr_trainable_GPO | 1002 | E2: ratio_J | 20 | DONE | 183.65 | — |
| nrcs_v1_e2_n00_s1002_ratio_J_b1_i0p1 | flickr_trainable_GPO | 1002 | E2: ratio_J | 20 | DONE | 184.20 | — |
| nrcs_v1_e2_n00_s1002_ratio_logistic_b1_i0p1 | flickr_trainable_GPO | 1002 | E2: ratio_logistic | 20 | DONE | 196.25 | — |
| nrcs_v1_e2_n00_s1002_cosine_hinge_b0_i0p1 | flickr_trainable_GPO | 1002 | E2: cosine_hinge | 20 | DONE | 473.67 | — |
| nrcs_v1_f1_n00_s1003_cosine_J_b0_i0 | flickr_frozen_W2 | 1003 | F1: cosine_J | 0 | DONE | 298.66 | — |
| nrcs_v1_f1_n00_s1003_ratio_J_b1_i0 | flickr_frozen_W2 | 1003 | F1: ratio_J | 0 | DONE | 97.32 | — |
| nrcs_v1_f1_n00_s1003_ratio_logistic_b1_i0 | flickr_frozen_W2 | 1003 | F1: ratio_logistic | 0 | DONE | 110.61 | — |
| nrcs_v1_f1_n00_s1003_ratio_J_b0_i0 | flickr_frozen_W2 | 1003 | F1: ratio_J | 0 | DONE | 127.73 | — |
| nrcs_v1_e2_n00_s1003_cosine_J_b0_i0 | flickr_trainable_GPO | 1003 | E2: cosine_J | 20 | DONE | 399.05 | — |
| nrcs_v1_e2_n00_s1003_cosine_J_b0_i0p1 | flickr_trainable_GPO | 1003 | E2: cosine_J | 20 | DONE | 399.19 | — |
| nrcs_v1_e2_n00_s1003_ratio_J_b1_i0 | flickr_trainable_GPO | 1003 | E2: ratio_J | 20 | DONE | 177.22 | — |
| nrcs_v1_e2_n00_s1003_ratio_J_b1_i0p1 | flickr_trainable_GPO | 1003 | E2: ratio_J | 20 | DONE | 168.42 | — |
| nrcs_v1_e2_n00_s1003_ratio_logistic_b1_i0p1 | flickr_trainable_GPO | 1003 | E2: ratio_logistic | 20 | DONE | 192.11 | — |
| nrcs_v1_e2_n00_s1003_cosine_hinge_b0_i0p1 | flickr_trainable_GPO | 1003 | E2: cosine_hinge | 20 | DONE | 473.47 | — |
| nrcs_v1_f1_n60_s1001_cosine_J_b0_i0 | flickr_frozen_W2 | 1001 | F1: cosine_J | 0 | DONE | 185.21 | — |
| nrcs_v1_f1_n60_s1001_ratio_J_b1_i0 | flickr_frozen_W2 | 1001 | F1: ratio_J | 0 | DONE | 38.84 | — |
| nrcs_v1_f1_n60_s1001_ratio_logistic_b1_i0 | flickr_frozen_W2 | 1001 | F1: ratio_logistic | 0 | DONE | 35.74 | — |
| nrcs_v1_f1_n60_s1001_ratio_J_b0_i0 | flickr_frozen_W2 | 1001 | F1: ratio_J | 0 | DONE | 49.47 | — |
| nrcs_v1_e2_n60_s1001_cosine_J_b0_i0 | flickr_trainable_GPO | 1001 | E2: cosine_J | 20 | DONE | 143.91 | — |
| nrcs_v1_e2_n60_s1001_cosine_J_b0_i0p1 | flickr_trainable_GPO | 1001 | E2: cosine_J | 20 | DONE | 179.23 | — |
| nrcs_v1_e2_n60_s1001_ratio_J_b1_i0 | flickr_trainable_GPO | 1001 | E2: ratio_J | 20 | DONE | 116.23 | — |
| nrcs_v1_e2_n60_s1001_ratio_J_b1_i0p1 | flickr_trainable_GPO | 1001 | E2: ratio_J | 20 | DONE | 102.92 | — |
| nrcs_v1_e2_n60_s1001_ratio_logistic_b1_i0p1 | flickr_trainable_GPO | 1001 | E2: ratio_logistic | 20 | DONE | 108.68 | — |
| nrcs_v1_e2_n60_s1001_cosine_hinge_b0_i0p1 | flickr_trainable_GPO | 1001 | E2: cosine_hinge | 20 | DONE | 12.37 | — |
| nrcs_v1_f1_n60_s1002_cosine_J_b0_i0 | flickr_frozen_W2 | 1002 | F1: cosine_J | 0 | DONE | 180.20 | — |
| nrcs_v1_f1_n60_s1002_ratio_J_b1_i0 | flickr_frozen_W2 | 1002 | F1: ratio_J | 0 | DONE | 37.30 | — |
| nrcs_v1_f1_n60_s1002_ratio_logistic_b1_i0 | flickr_frozen_W2 | 1002 | F1: ratio_logistic | 0 | DONE | 31.52 | — |
| nrcs_v1_f1_n60_s1002_ratio_J_b0_i0 | flickr_frozen_W2 | 1002 | F1: ratio_J | 0 | DONE | 44.52 | — |
| nrcs_v1_e2_n60_s1002_cosine_J_b0_i0 | flickr_trainable_GPO | 1002 | E2: cosine_J | 20 | DONE | 136.55 | — |
| nrcs_v1_e2_n60_s1002_cosine_J_b0_i0p1 | flickr_trainable_GPO | 1002 | E2: cosine_J | 20 | DONE | 171.56 | — |
| nrcs_v1_e2_n60_s1002_ratio_J_b1_i0 | flickr_trainable_GPO | 1002 | E2: ratio_J | 20 | DONE | 105.66 | — |
| nrcs_v1_e2_n60_s1002_ratio_J_b1_i0p1 | flickr_trainable_GPO | 1002 | E2: ratio_J | 20 | DONE | 95.42 | — |
| nrcs_v1_e2_n60_s1002_ratio_logistic_b1_i0p1 | flickr_trainable_GPO | 1002 | E2: ratio_logistic | 20 | DONE | 105.09 | — |
| nrcs_v1_e2_n60_s1002_cosine_hinge_b0_i0p1 | flickr_trainable_GPO | 1002 | E2: cosine_hinge | 20 | DONE | 12.19 | — |
| nrcs_v1_f1_n60_s1003_cosine_J_b0_i0 | flickr_frozen_W2 | 1003 | F1: cosine_J | 0 | DONE | 190.28 | — |
| nrcs_v1_f1_n60_s1003_ratio_J_b1_i0 | flickr_frozen_W2 | 1003 | F1: ratio_J | 0 | DONE | 35.80 | — |
| nrcs_v1_f1_n60_s1003_ratio_logistic_b1_i0 | flickr_frozen_W2 | 1003 | F1: ratio_logistic | 0 | DONE | 33.20 | — |
| nrcs_v1_f1_n60_s1003_ratio_J_b0_i0 | flickr_frozen_W2 | 1003 | F1: ratio_J | 0 | DONE | 54.30 | — |
| nrcs_v1_e2_n60_s1003_cosine_J_b0_i0 | flickr_trainable_GPO | 1003 | E2: cosine_J | 20 | DONE | 135.66 | — |
| nrcs_v1_e2_n60_s1003_cosine_J_b0_i0p1 | flickr_trainable_GPO | 1003 | E2: cosine_J | 20 | DONE | 174.69 | — |
| nrcs_v1_e2_n60_s1003_ratio_J_b1_i0 | flickr_trainable_GPO | 1003 | E2: ratio_J | 20 | DONE | 119.51 | — |
| nrcs_v1_e2_n60_s1003_ratio_J_b1_i0p1 | flickr_trainable_GPO | 1003 | E2: ratio_J | 20 | DONE | 107.89 | — |
| nrcs_v1_e2_n60_s1003_ratio_logistic_b1_i0p1 | flickr_trainable_GPO | 1003 | E2: ratio_logistic | 20 | DONE | 108.13 | — |
| nrcs_v1_e2_n60_s1003_cosine_hinge_b0_i0p1 | flickr_trainable_GPO | 1003 | E2: cosine_hinge | 20 | DONE | 12.82 | — |
| nrcs_v1_c2_n00_s1001_cosine_J_b0_i0p1 | flickr_cold_diagnostic | 1001 | C2: cosine_J | 20 | DONE | 391.34 | — |
| nrcs_v1_c2_n00_s1001_ratio_J_b1_i0p1 | flickr_cold_diagnostic | 1001 | C2: ratio_J | 20 | DONE | 185.94 | — |

## R33：基准基线行（原生 IN²R 与 IN²R + A1；COCO 60%、Flickr 80%、COCO 20%/80%、Flickr 20% A1）

单元数 27；DONE 24，RUNNING 3

| 单元 | 条件 | seed | 臂 | 轮/步 | 状态 | DEV（best full-DEV rSum） | TEST rSum |
|---|---|---|---|---|---|---|---|
| c60_s1001/H0 | COCO 60% | 1001 | native IN²R | 40 | DONE | 538.94 | — |
| c60_s1001/a1_iface | COCO 60% | 1001 | IN²R + A1 | 40 | DONE | 538.42 | — |
| c60_s1002/H0 | COCO 60% | 1002 | native IN²R | 40 | DONE | 538.44 | — |
| c60_s1002/a1_iface | COCO 60% | 1002 | IN²R + A1 | 40 | DONE | 539.24 | — |
| c60_s1003/H0 | COCO 60% | 1003 | native IN²R | 40 | DONE | 539.32 | — |
| c60_s1003/a1_iface | COCO 60% | 1003 | IN²R + A1 | 40 | DONE | 538.50 | — |
| f80_s1001/H0 | Flickr 80% | 1001 | native IN²R | 40 | DONE | 464.93 | 467.08 |
| f80_s1001/a1_iface | Flickr 80% | 1001 | IN²R + A1 | 40 | DONE | 470.24 | 467.32 |
| f80_s1002/H0 | Flickr 80% | 1002 | native IN²R | 40 | DONE | 437.51 | 431.90 |
| f80_s1002/a1_iface | Flickr 80% | 1002 | IN²R + A1 | 40 | DONE | 440.97 | 437.52 |
| f80_s1003/H0 | Flickr 80% | 1003 | native IN²R | 40 | DONE | 454.12 | 450.30 |
| f80_s1003/a1_iface | Flickr 80% | 1003 | IN²R + A1 | 40 | DONE | 461.03 | 454.44 |
| c20_s1001/H0 | COCO 20% | 1001 | native IN²R | 40 | DONE | 546.06 | — |
| c20_s1001/a1_iface | COCO 20% | 1001 | IN²R + A1 | 40 | RUNNING | NA | — |
| c20_s1002/H0 | COCO 20% | 1002 | native IN²R | 40 | DONE | 546.10 | — |
| c20_s1002/a1_iface | COCO 20% | 1002 | IN²R + A1 | 40 | RUNNING | NA | — |
| c20_s1003/H0 | COCO 20% | 1003 | native IN²R | 40 | DONE | 542.66 | — |
| c20_s1003/a1_iface | COCO 20% | 1003 | IN²R + A1 | 40 | RUNNING | NA | — |
| c80_s1001/H0 | COCO 80% | 1001 | native IN²R | 40 | DONE | 526.52 | — |
| c80_s1001/a1_iface | COCO 80% | 1001 | IN²R + A1 | 40 | DONE | 526.14 | — |
| c80_s1002/H0 | COCO 80% | 1002 | native IN²R | 40 | DONE | 511.66 | — |
| c80_s1002/a1_iface | COCO 80% | 1002 | IN²R + A1 | 40 | DONE | 512.66 | — |
| c80_s1003/H0 | COCO 80% | 1003 | native IN²R | 40 | DONE | 512.18 | — |
| c80_s1003/a1_iface | COCO 80% | 1003 | IN²R + A1 | 40 | DONE | 512.68 | — |
| f20_s1001/a1_iface | Flickr 20% | 1001 | IN²R + A1 | 40 | DONE | 518.99 | 520.72 |
| f20_s1002/a1_iface | Flickr 20% | 1002 | IN²R + A1 | 40 | DONE | 516.65 | 522.58 |
| f20_s1003/a1_iface | Flickr 20% | 1003 | IN²R + A1 | 40 | DONE | 516.98 | 522.08 |

