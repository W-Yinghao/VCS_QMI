Publication copy (sample names aliased; numbers and conclusions unchanged).

# R37S 结果（仅数字；开发实验，本线未读 TEST）

生成 2026-11-02T19:50:32Z。父模型：CC152K 上按 R30 方案训练的 H（r37s_cc_s1001_H0；warmup 5 + H0 co-train 40，A1，K5，i2t/t2i 开，seed 1001，DEV-best）。与 R37C/R38 的 R31 H0_default 父模型属不同系列，不在同一表中混合；所有差值只相对本父模型。冻结 encoder 的 critic fits；固定主终点 8000 步（曲线最大值只作次要读数）。

未完成单元：无。完成 18/18（批准 6 项 shared/constant + 用户补充 12 项 R38 对应格）。失败记录：0 条。

## 父模型训练
|item|value|seconds|
|---|---|---|
|warmup (5 epochs, both peers)|DEV rSum by epoch 306.3 / 327.4 / 347.3 / 353.2 / 360.6|611|
|co-train H0 (40 epochs)|epochs done 40; best epoch 22; best DEV rSum 393.2|11734|
|best-epoch GPU / GPU block|NVIDIA RTX PRO 6000 Blackwell Server Edition / NVIDIA RTX PRO 6000 Blackwell Server Edition (attempts: 1082196 NVIDIA RTX PRO 6000 Blackwell Server Edition)|NA|
|parents/wp_cc_s1001_H_e5/PEAK_1082196.json|peak alloc 1.36 GiB, reserved 1.67 GiB, NVIDIA RTX PRO 6000 Blackwell Server Edition|634|
|runs/r37s_cc_s1001_H0/PEAK_1082196.json|peak alloc 45.15 GiB, reserved 49.32 GiB, NVIDIA RTX PRO 6000 Blackwell Server Edition|11787|

co-train 每轮 DEV rSum（epoch 0..39）：367.1 / 371.3 / 372.4 / 375.0 / 376.4 / 381.1 / 383.9 / 385.8 / 386.0 / 389.6 / 388.6 / 388.9 / 388.9 / 389.5 / 385.7 / 389.8 / 390.3 / 390.1 / 390.6 / 392.0 / 391.1 / 391.5 / 393.2 / 391.6 / 392.7 / 392.4 / 391.5 / 392.3 / 393.0 / 392.2 / 393.2 / 393.2 / 392.0 / 392.0 / 392.6 / 392.6 / 392.4 / 392.3 / 392.6 / 392.9

cache：原生 DEV 复算 rSum 393.2（与 checkpoint best_rsum 完全一致=True，GPU NVIDIA RTX PRO 6000 Blackwell Server Edition）；prep：a0 20.3624、b0 -4.5880；trust 子集 139787（保留率 0.9319）；候选：t2i 观测图在 top-32 的比例 0.7206，i2t 0.7543。

## 父模型 H（native DEV，按 cache 复算）
|i2t_r1|i2t_r5|i2t_r10|t2i_r1|t2i_r5|t2i_r10|rsum|
|---|---|---|---|---|---|---|
|45.40|72.60|80.50|42.70|71.70|80.30|393.20|

## 全部单元固定终点（8000 步；Δ 相对本父模型 H）
|unit|route|architecture|schedule|objective|i2t_r1|i2t_r5|i2t_r10|t2i_r1|t2i_r5|t2i_r10|rsum|d_rsum|d_i2t_r1|d_t2i_r1|change_2000_to_8000|curve_max_rsum_secondary|step0_passed|graph_check_passed|gpu|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|r37s_cc_r30h_s1001_N_directional_BCE_constant|N|directional|constant|BCE|44.70|72.00|79.50|43.20|72.30|80.80|392.50|-0.70|-0.70|0.50|0.50|394.50|True|True|NVIDIA L40S|
|r37s_cc_r30h_s1001_N_directional_J_constant|N|directional|constant|J|44.60|71.90|79.70|43.60|72.00|80.50|392.30|-0.90|-0.80|0.90|-0.30|394.00|True|True|NVIDIA L40S|
|r37s_cc_r30h_s1001_N_directional_NCE_constant|N|directional|constant|NCE|44.90|71.60|79.20|43.30|72.00|79.50|390.50|-2.70|-0.50|0.60|0.60|393.20|True|True|NVIDIA L40S|
|r37s_cc_r30h_s1001_N_directional_BCE_cosine|N|directional|cosine|BCE|45.30|71.80|79.30|44.00|72.30|80.50|393.20|0.00|-0.10|1.30|0.70|395.00|True|True|NVIDIA L40S|
|r37s_cc_r30h_s1001_N_directional_J_cosine|N|directional|cosine|J|45.40|72.30|79.70|43.60|72.00|80.40|393.40|0.20|0.00|0.90|0.50|394.50|True|True|NVIDIA L40S|
|r37s_cc_r30h_s1001_N_directional_NCE_cosine|N|directional|cosine|NCE|45.20|70.70|78.80|44.20|71.70|79.50|390.10|-3.10|-0.20|1.50|-0.30|393.20|True|True|NVIDIA L40S|
|r37s_cc_r30h_s1001_N_shared_BCE_constant|N|shared|constant|BCE|44.80|71.90|79.50|43.20|72.20|80.80|392.40|-0.80|-0.60|0.50|-0.00|394.50|True|True|NVIDIA L40S|
|r37s_cc_r30h_s1001_N_shared_J_constant|N|shared|constant|J|44.80|71.90|79.80|43.40|72.00|80.50|392.40|-0.80|-0.60|0.70|0.20|393.70|True|True|NVIDIA L40S|
|r37s_cc_r30h_s1001_N_shared_NCE_constant|N|shared|constant|NCE|44.90|71.20|79.30|43.40|71.90|79.50|390.20|-3.00|-0.50|0.70|0.40|393.20|True|True|NVIDIA L40S|
|r37s_cc_r30h_s1001_N_shared_BCE_cosine|N|shared|cosine|BCE|45.30|71.90|79.50|43.70|72.30|80.50|393.20|0.00|-0.10|1.00|1.30|394.80|True|True|NVIDIA L40S|
|r37s_cc_r30h_s1001_N_shared_J_cosine|N|shared|cosine|J|45.10|72.30|79.70|43.90|72.00|80.40|393.40|0.20|-0.30|1.20|1.10|394.40|True|True|NVIDIA L40S|
|r37s_cc_r30h_s1001_N_shared_NCE_cosine|N|shared|cosine|NCE|45.20|70.50|78.90|44.30|71.80|79.60|390.30|-2.90|-0.20|1.60|0.50|393.20|True|True|NVIDIA L40S|
|r37s_cc_r30h_s1001_T_shared_BCE_constant|T|shared|constant|BCE|0.90|2.90|5.30|0.50|3.00|5.80|18.40|-374.80|-44.50|-42.20|-373.20|393.40|True|True|NVIDIA RTX PRO 6000 Blackwell Server Edition|
|r37s_cc_r30h_s1001_T_shared_J_constant|T|shared|constant|J|41.00|66.30|76.90|40.70|67.80|76.10|368.80|-24.40|-4.40|-2.00|-26.10|394.90|True|True|NVIDIA RTX PRO 6000 Blackwell Server Edition|
|r37s_cc_r30h_s1001_T_shared_NCE_constant|T|shared|constant|NCE|0.00|0.60|0.80|0.00|0.20|0.50|2.10|-391.10|-45.40|-42.70|-364.40|393.90|True|True|NVIDIA RTX PRO 6000 Blackwell Server Edition|
|r37s_cc_r30h_s1001_T_shared_BCE_cosine|T|shared|cosine|BCE|37.40|65.50|75.90|39.80|66.00|76.20|360.80|-32.40|-8.00|-2.90|-30.50|393.40|True|True|NVIDIA RTX PRO 6000 Blackwell Server Edition|
|r37s_cc_r30h_s1001_T_shared_J_cosine|T|shared|cosine|J|45.10|71.80|80.30|43.30|71.30|80.40|392.20|-1.00|-0.30|0.60|-2.50|394.70|True|True|NVIDIA RTX PRO 6000 Blackwell Server Edition|
|r37s_cc_r30h_s1001_T_shared_NCE_cosine|T|shared|cosine|NCE|0.20|1.10|2.60|0.10|0.70|1.60|6.30|-386.90|-45.20|-42.60|-362.70|393.90|True|True|NVIDIA RTX PRO 6000 Blackwell Server Edition|

## N：结构 × 日程（rSum 与双向 R@1）
|objective|H_rsum|rsum_shared_constant|rsum_shared_cosine|rsum_directional_constant|rsum_directional_cosine|rsum_dir_minus_shared_constant|rsum_dir_minus_shared_cosine|rsum_cos_minus_const_shared|rsum_cos_minus_const_dir|rsum_interaction|
|---|---|---|---|---|---|---|---|---|---|---|
|J|393.20|392.40|393.40|392.30|393.40|-0.10|0.00|1.00|1.10|0.10|
|BCE|393.20|392.40|393.20|392.50|393.20|0.10|0.00|0.80|0.70|-0.10|
|NCE|393.20|390.20|390.30|390.50|390.10|0.30|-0.20|0.10|-0.40|-0.50|

|objective|H_i2t_r1|i2t_r1_shared_constant|i2t_r1_shared_cosine|i2t_r1_directional_constant|i2t_r1_directional_cosine|i2t_r1_dir_minus_shared_constant|i2t_r1_dir_minus_shared_cosine|
|---|---|---|---|---|---|---|---|
|J|45.40|44.80|45.10|44.60|45.40|-0.20|0.30|
|BCE|45.40|44.80|45.30|44.70|45.30|-0.10|0.00|
|NCE|45.40|44.90|45.20|44.90|45.20|0.00|0.00|

|objective|H_t2i_r1|t2i_r1_shared_constant|t2i_r1_shared_cosine|t2i_r1_directional_constant|t2i_r1_directional_cosine|t2i_r1_dir_minus_shared_constant|t2i_r1_dir_minus_shared_cosine|
|---|---|---|---|---|---|---|---|
|J|42.70|43.40|43.90|43.60|43.60|0.20|-0.30|
|BCE|42.70|43.20|43.70|43.20|44.00|0.00|0.30|
|NCE|42.70|43.40|44.30|43.30|44.20|-0.10|-0.10|

## T 日程（shared）
|objective|H_rsum|T_constant_rsum|T_cosine_rsum|T_cos_minus_const|N_constant_rsum|N_cosine_rsum|T_minus_N_constant|T_minus_N_cosine|T_constant_curve|T_cosine_curve|
|---|---|---|---|---|---|---|---|---|---|---|
|J|393.20|368.80|392.20|23.40|392.40|393.40|-23.60|-1.20|393.2 / 391.6 / 392.9 / 393.5 / 394.9 / 393.2 / 368.8|393.2 / 391.5 / 392.9 / 393.4 / 394.7 / 393.4 / 392.2|
|BCE|393.20|18.40|360.80|342.40|392.40|393.20|-374.00|-32.40|393.2 / 392.0 / 392.5 / 393.4 / 391.6 / 378.9 / 18.4|393.2 / 391.9 / 392.5 / 393.4 / 391.3 / 387.6 / 360.8|
|NCE|393.20|2.10|6.30|4.20|390.20|390.30|-388.10|-384.00|393.2 / 393.9 / 391.0 / 385.4 / 366.5 / 15.7 / 2.1|393.2 / 393.9 / 391.1 / 385.5 / 369.0 / 78.5 / 6.3|

## 目标对比（同路线、同结构、同日程）
|route|architecture|schedule|rsum_J|rsum_BCE|rsum_NCE|rsum_J_minus_BCE|rsum_J_minus_NCE|i2t_r1_J_minus_BCE|t2i_r1_J_minus_BCE|i2t_r1_J_minus_NCE|t2i_r1_J_minus_NCE|
|---|---|---|---|---|---|---|---|---|---|---|---|
|N|directional|constant|392.30|392.50|390.50|-0.20|1.80|-0.10|0.40|-0.30|0.30|
|N|directional|cosine|393.40|393.20|390.10|0.20|3.30|0.10|-0.40|0.20|-0.60|
|N|shared|constant|392.40|392.40|390.20|0.00|2.20|0.00|0.20|-0.10|0.00|
|N|shared|cosine|393.40|393.20|390.30|0.20|3.10|-0.20|0.20|-0.10|-0.40|
|T|shared|constant|368.80|18.40|2.10|350.40|366.70|40.10|40.20|41.00|40.70|
|T|shared|cosine|392.20|360.80|6.30|31.40|385.90|7.70|3.50|44.90|43.20|

## 曲线（DEV rSum，步 0/250/500/1000/2000/4000/8000）
|unit|s0|s250|s500|s1000|s2000|s4000|s8000|
|---|---|---|---|---|---|---|---|
|N_directional_BCE_constant|393.20|391.60|391.80|392.80|392.00|394.50|392.50|
|N_directional_J_constant|393.20|389.80|391.60|392.70|392.60|394.00|392.30|
|N_directional_NCE_constant|393.20|389.70|390.40|390.90|389.90|390.20|390.50|
|N_directional_BCE_cosine|393.20|391.60|391.80|392.90|392.50|395.00|393.20|
|N_directional_J_cosine|393.20|389.70|391.60|392.70|392.90|394.50|393.40|
|N_directional_NCE_cosine|393.20|389.70|390.50|391.00|390.40|390.20|390.10|
|N_shared_BCE_constant|393.20|391.50|391.80|392.60|392.40|394.50|392.40|
|N_shared_J_constant|393.20|389.80|391.40|392.90|392.20|393.70|392.40|
|N_shared_NCE_constant|393.20|389.60|390.50|390.80|389.80|390.10|390.20|
|N_shared_BCE_cosine|393.20|391.50|391.80|392.60|391.90|394.80|393.20|
|N_shared_J_cosine|393.20|389.80|391.50|393.00|392.30|394.40|393.40|
|N_shared_NCE_cosine|393.20|389.60|390.50|391.00|389.80|389.90|390.30|
|T_shared_BCE_constant|393.20|392.00|392.50|393.40|391.60|378.90|18.40|
|T_shared_J_constant|393.20|391.60|392.90|393.50|394.90|393.20|368.80|
|T_shared_NCE_constant|393.20|393.90|391.00|385.40|366.50|15.70|2.10|
|T_shared_BCE_cosine|393.20|391.90|392.50|393.40|391.30|387.60|360.80|
|T_shared_J_cosine|393.20|391.50|392.90|393.40|394.70|393.40|392.20|
|T_shared_NCE_cosine|393.20|393.90|391.10|385.50|369.00|78.50|6.30|

## 同路线单元的初始化 / 抽样一致性（共享键 init 哈希与批索引哈希链）
|route|n_units|n_distinct_init_shared_keys|n_distinct_batch_chains|identical|
|---|---|---|---|---|
|N|12|1|1|True|
|T|6|1|1|True|

## R@1 转移（a 对 b：仅 a 对 / 仅 b 对 / 净；top1 改变数）
|objective|contrast|a|i2t_a_only_r1|i2t_b_only_r1|i2t_net_r1|i2t_top1_changed|t2i_a_only_r1|t2i_b_only_r1|t2i_net_r1|t2i_top1_changed|
|---|---|---|---|---|---|---|---|---|---|---|
|NA|unit vs native H|r37s_cc_r30h_s1001_N_directional_BCE_constant|28|35|-7|177|48|43|5|203|
|NA|unit vs native H|r37s_cc_r30h_s1001_N_directional_J_constant|27|35|-8|174|46|37|9|182|
|NA|unit vs native H|r37s_cc_r30h_s1001_N_directional_NCE_constant|27|32|-5|186|53|47|6|218|
|NA|unit vs native H|r37s_cc_r30h_s1001_N_directional_BCE_cosine|29|30|-1|179|45|32|13|186|
|NA|unit vs native H|r37s_cc_r30h_s1001_N_directional_J_cosine|28|28|0|166|42|33|9|175|
|NA|unit vs native H|r37s_cc_r30h_s1001_N_directional_NCE_cosine|31|33|-2|194|52|37|15|206|
|NA|unit vs native H|r37s_cc_r30h_s1001_N_shared_BCE_constant|30|36|-6|182|48|43|5|204|
|NA|unit vs native H|r37s_cc_r30h_s1001_N_shared_J_constant|28|34|-6|171|46|39|7|183|
|NA|unit vs native H|r37s_cc_r30h_s1001_N_shared_NCE_constant|28|33|-5|195|52|45|7|209|
|NA|unit vs native H|r37s_cc_r30h_s1001_N_shared_BCE_cosine|29|30|-1|178|47|37|10|194|
|NA|unit vs native H|r37s_cc_r30h_s1001_N_shared_J_cosine|27|30|-3|168|45|33|12|179|
|NA|unit vs native H|r37s_cc_r30h_s1001_N_shared_NCE_cosine|31|33|-2|195|51|35|16|202|
|NA|unit vs native H|r37s_cc_r30h_s1001_T_shared_BCE_constant|1|446|-445|992|1|423|-422|995|
|NA|unit vs native H|r37s_cc_r30h_s1001_T_shared_J_constant|51|95|-44|389|67|87|-20|378|
|NA|unit vs native H|r37s_cc_r30h_s1001_T_shared_NCE_constant|0|454|-454|1000|0|427|-427|1000|
|NA|unit vs native H|r37s_cc_r30h_s1001_T_shared_BCE_cosine|50|130|-80|459|65|94|-29|421|
|NA|unit vs native H|r37s_cc_r30h_s1001_T_shared_J_cosine|51|54|-3|238|57|51|6|260|
|NA|unit vs native H|r37s_cc_r30h_s1001_T_shared_NCE_cosine|2|454|-452|1000|1|427|-426|1000|
|J|N dir-shared|constant|r37s_cc_r30h_s1001_N_directional_J_constant|0|2|-2|6|2|0|2|3|
|J|N dir-shared|cosine|r37s_cc_r30h_s1001_N_directional_J_cosine|3|0|3|4|0|3|-3|7|
|J|N cos-const|shared|r37s_cc_r30h_s1001_N_shared_J_cosine|10|7|3|46|10|5|5|47|
|J|N cos-const|dir|r37s_cc_r30h_s1001_N_directional_J_cosine|14|6|8|44|7|7|0|47|
|BCE|N dir-shared|constant|r37s_cc_r30h_s1001_N_directional_BCE_constant|1|2|-1|8|1|1|0|4|
|BCE|N dir-shared|cosine|r37s_cc_r30h_s1001_N_directional_BCE_cosine|0|0|0|1|5|2|3|13|
|BCE|N cos-const|shared|r37s_cc_r30h_s1001_N_shared_BCE_cosine|11|6|5|40|8|3|5|38|
|BCE|N cos-const|dir|r37s_cc_r30h_s1001_N_directional_BCE_cosine|10|4|6|35|14|6|8|51|
|NCE|N dir-shared|constant|r37s_cc_r30h_s1001_N_directional_NCE_constant|1|1|0|10|1|2|-1|15|
|NCE|N dir-shared|cosine|r37s_cc_r30h_s1001_N_directional_NCE_cosine|0|0|0|5|1|2|-1|4|
|NCE|N cos-const|shared|r37s_cc_r30h_s1001_N_shared_NCE_cosine|9|6|3|48|17|8|9|65|
|NCE|N cos-const|dir|r37s_cc_r30h_s1001_N_directional_NCE_cosine|8|5|3|46|19|10|9|74|
|J-BCE|J-BCE|N,directional,constant|r37s_cc_r30h_s1001_N_directional_J_constant|4|5|-1|21|8|4|4|34|
|J-NCE|J-NCE|N,directional,constant|r37s_cc_r30h_s1001_N_directional_J_constant|14|17|-3|86|18|15|3|86|
|J-BCE|J-BCE|N,directional,cosine|r37s_cc_r30h_s1001_N_directional_J_cosine|4|3|1|21|2|6|-4|26|
|J-NCE|J-NCE|N,directional,cosine|r37s_cc_r30h_s1001_N_directional_J_cosine|17|15|2|97|13|19|-6|84|
|J-BCE|J-BCE|N,shared,constant|r37s_cc_r30h_s1001_N_shared_J_constant|6|6|0|31|6|4|2|33|
|J-NCE|J-NCE|N,shared,constant|r37s_cc_r30h_s1001_N_shared_J_constant|15|16|-1|89|16|16|0|83|
|J-BCE|J-BCE|N,shared,cosine|r37s_cc_r30h_s1001_N_shared_J_cosine|3|5|-2|22|6|4|2|28|
|J-NCE|J-NCE|N,shared,cosine|r37s_cc_r30h_s1001_N_shared_J_cosine|16|17|-1|100|13|17|-4|83|
|J-BCE|J-BCE|T,shared,constant|r37s_cc_r30h_s1001_T_shared_J_constant|402|1|401|983|403|1|402|992|
|J-NCE|J-NCE|T,shared,constant|r37s_cc_r30h_s1001_T_shared_J_constant|410|0|410|995|407|0|407|1000|
|J-BCE|J-BCE|T,shared,cosine|r37s_cc_r30h_s1001_T_shared_J_cosine|91|14|77|303|53|18|35|232|
|J-NCE|J-NCE|T,shared,cosine|r37s_cc_r30h_s1001_T_shared_J_cosine|451|2|449|1000|433|1|432|1000|
|J|T cos-const|r37s_cc_r30h_s1001_T_shared_J_cosine|57|16|41|245|51|25|26|212|
|BCE|T cos-const|r37s_cc_r30h_s1001_T_shared_BCE_cosine|366|1|365|983|394|1|393|994|
|NCE|T cos-const|r37s_cc_r30h_s1001_T_shared_NCE_cosine|2|0|2|455|1|0|1|525|

## 按图像分组配对 bootstrap（rSum 差，B=2000，描述性）
|objective|contrast|a|diff|ci2_5|ci97_5|frac_boot_gt0|
|---|---|---|---|---|---|---|
|NA|unit - native H|r37s_cc_r30h_s1001_N_directional_BCE_constant|-0.70|-4.00|2.70|0.33|
|NA|unit - native H|r37s_cc_r30h_s1001_N_directional_J_constant|-0.90|-4.00|2.20|0.29|
|NA|unit - native H|r37s_cc_r30h_s1001_N_directional_NCE_constant|-2.70|-6.30|1.00|0.07|
|NA|unit - native H|r37s_cc_r30h_s1001_N_directional_BCE_cosine|-0.00|-3.20|3.10|0.48|
|NA|unit - native H|r37s_cc_r30h_s1001_N_directional_J_cosine|0.20|-2.90|3.20|0.54|
|NA|unit - native H|r37s_cc_r30h_s1001_N_directional_NCE_cosine|-3.10|-6.50|0.30|0.04|
|NA|unit - native H|r37s_cc_r30h_s1001_N_shared_BCE_constant|-0.80|-4.00|2.60|0.31|
|NA|unit - native H|r37s_cc_r30h_s1001_N_shared_J_constant|-0.80|-3.90|2.30|0.31|
|NA|unit - native H|r37s_cc_r30h_s1001_N_shared_NCE_constant|-3.00|-6.50|0.70|0.05|
|NA|unit - native H|r37s_cc_r30h_s1001_N_shared_BCE_cosine|-0.00|-3.20|3.20|0.47|
|NA|unit - native H|r37s_cc_r30h_s1001_N_shared_J_cosine|0.20|-2.90|3.20|0.53|
|NA|unit - native H|r37s_cc_r30h_s1001_N_shared_NCE_cosine|-2.90|-6.40|0.50|0.04|
|NA|unit - native H|r37s_cc_r30h_s1001_T_shared_BCE_constant|-374.80|-389.11|-360.40|0.00|
|NA|unit - native H|r37s_cc_r30h_s1001_T_shared_J_constant|-24.40|-30.40|-18.60|0.00|
|NA|unit - native H|r37s_cc_r30h_s1001_T_shared_NCE_constant|-391.10|-405.10|-377.00|0.00|
|NA|unit - native H|r37s_cc_r30h_s1001_T_shared_BCE_cosine|-32.40|-38.80|-25.80|0.00|
|NA|unit - native H|r37s_cc_r30h_s1001_T_shared_J_cosine|-1.00|-5.00|2.90|0.29|
|NA|unit - native H|r37s_cc_r30h_s1001_T_shared_NCE_cosine|-386.90|-401.10|-372.50|0.00|
|J|N dir-shared|constant|r37s_cc_r30h_s1001_N_directional_J_constant|-0.10|-0.50|0.40|0.29|
|J|N dir-shared|cosine|r37s_cc_r30h_s1001_N_directional_J_cosine|-0.00|-0.50|0.50|0.43|
|J|N cos-const|shared|r37s_cc_r30h_s1001_N_shared_J_cosine|1.00|-0.60|2.60|0.88|
|J|N cos-const|dir|r37s_cc_r30h_s1001_N_directional_J_cosine|1.10|-0.40|2.60|0.90|
|BCE|N dir-shared|constant|r37s_cc_r30h_s1001_N_directional_BCE_constant|0.10|-0.50|0.70|0.57|
|BCE|N dir-shared|cosine|r37s_cc_r30h_s1001_N_directional_BCE_cosine|0.00|-0.60|0.70|0.46|
|BCE|N cos-const|shared|r37s_cc_r30h_s1001_N_shared_BCE_cosine|0.80|-0.80|2.40|0.81|
|BCE|N cos-const|dir|r37s_cc_r30h_s1001_N_directional_BCE_cosine|0.70|-0.90|2.30|0.78|
|NCE|N dir-shared|constant|r37s_cc_r30h_s1001_N_directional_NCE_constant|0.30|-0.40|1.10|0.79|
|NCE|N dir-shared|cosine|r37s_cc_r30h_s1001_N_directional_NCE_cosine|-0.20|-0.80|0.40|0.22|
|NCE|N cos-const|shared|r37s_cc_r30h_s1001_N_shared_NCE_cosine|0.10|-1.80|1.90|0.52|
|NCE|N cos-const|dir|r37s_cc_r30h_s1001_N_directional_NCE_cosine|-0.40|-2.30|1.40|0.33|
|J-BCE|J-BCE|N,directional,constant|r37s_cc_r30h_s1001_N_directional_J_constant|-0.20|-1.60|1.20|0.38|
|J-NCE|J-NCE|N,directional,constant|r37s_cc_r30h_s1001_N_directional_J_constant|1.80|-0.50|4.10|0.93|
|J-BCE|J-BCE|N,directional,cosine|r37s_cc_r30h_s1001_N_directional_J_cosine|0.20|-0.90|1.40|0.63|
|J-NCE|J-NCE|N,directional,cosine|r37s_cc_r30h_s1001_N_directional_J_cosine|3.30|1.00|5.60|1.00|
|J-BCE|J-BCE|N,shared,constant|r37s_cc_r30h_s1001_N_shared_J_constant|0.00|-1.40|1.50|0.48|
|J-NCE|J-NCE|N,shared,constant|r37s_cc_r30h_s1001_N_shared_J_constant|2.20|-0.00|4.50|0.97|
|J-BCE|J-BCE|N,shared,cosine|r37s_cc_r30h_s1001_N_shared_J_cosine|0.20|-1.10|1.50|0.62|
|J-NCE|J-NCE|N,shared,cosine|r37s_cc_r30h_s1001_N_shared_J_cosine|3.10|0.80|5.40|1.00|
|J-BCE|J-BCE|T,shared,constant|r37s_cc_r30h_s1001_T_shared_J_constant|350.40|336.09|364.80|1.00|
|J-NCE|J-NCE|T,shared,constant|r37s_cc_r30h_s1001_T_shared_J_constant|366.70|352.40|381.10|1.00|
|J-BCE|J-BCE|T,shared,cosine|r37s_cc_r30h_s1001_T_shared_J_cosine|31.40|26.50|36.40|1.00|
|J-NCE|J-NCE|T,shared,cosine|r37s_cc_r30h_s1001_T_shared_J_cosine|385.90|371.80|400.10|1.00|
|J|T cos-const|r37s_cc_r30h_s1001_T_shared_J_cosine|23.40|18.90|28.10|1.00|
|BCE|T cos-const|r37s_cc_r30h_s1001_T_shared_BCE_cosine|342.40|328.40|356.80|1.00|
|NCE|T cos-const|r37s_cc_r30h_s1001_T_shared_NCE_cosine|4.20|2.50|6.10|1.00|

## 方向梯度夹角（训练批，trunk 上 i2t 与 t2i 梯度 cos，两 peer 平均）
|route|architecture|schedule|objective|cos_trunk_step1|cos_trunk_step250|cos_trunk_step1000|cos_trunk_step4000|
|---|---|---|---|---|---|---|---|
|N|directional|constant|BCE|0.976|0.639|0.215|-0.001|
|N|directional|constant|J|0.980|-0.308|0.616|-0.043|
|N|directional|constant|NCE|0.939|0.074|-0.120|-0.344|
|N|directional|cosine|BCE|0.976|0.637|0.215|0.009|
|N|directional|cosine|J|0.980|-0.308|0.614|-0.071|
|N|directional|cosine|NCE|0.939|0.075|-0.115|-0.428|
|N|shared|constant|BCE|0.992|0.590|0.493|-0.225|
|N|shared|constant|J|0.988|-0.420|0.759|-0.240|
|N|shared|constant|NCE|0.968|-0.010|0.161|-0.459|
|N|shared|cosine|BCE|0.992|0.587|0.486|-0.233|
|N|shared|cosine|J|0.988|-0.420|0.754|-0.265|
|N|shared|cosine|NCE|0.968|-0.009|0.164|-0.536|
|T|shared|constant|BCE|0.992|0.922|0.470|-0.051|
|T|shared|constant|J|0.981|0.498|-0.210|0.309|
|T|shared|constant|NCE|-0.013|-0.008|0.013|0.137|
|T|shared|cosine|BCE|0.992|0.922|0.474|0.023|
|T|shared|cosine|J|0.981|0.497|-0.216|0.203|
|T|shared|cosine|NCE|-0.013|-0.008|0.013|0.064|

## 成本
critic 单元 GPU 墙钟合计 974 s（unit_wall_seconds，不含作业启动与 cache 载入）；父模型训练成本见上（PEAK / EPOCHS 秒数）。
|unit|item|gpu|job|train_seconds|eval_seconds|unit_wall_seconds|seconds|peak_gpu_gb|
|---|---|---|---|---|---|---|---|---|
|r37s_cc_r30h_s1001_N_directional_BCE_constant|unit|NVIDIA L40S|1082774|80.2|3.6|84.2|NA|3.2|
|r37s_cc_r30h_s1001_N_directional_J_constant|unit|NVIDIA L40S|1082774|85.9|3.7|91.0|NA|3.2|
|r37s_cc_r30h_s1001_N_directional_NCE_constant|unit|NVIDIA L40S|1082774|77.5|3.7|81.6|NA|3.2|
|r37s_cc_r30h_s1001_N_directional_BCE_cosine|unit|NVIDIA L40S|1082774|81.7|3.7|85.9|NA|3.2|
|r37s_cc_r30h_s1001_N_directional_J_cosine|unit|NVIDIA L40S|1082774|88.5|3.6|92.7|NA|3.2|
|r37s_cc_r30h_s1001_N_directional_NCE_cosine|unit|NVIDIA L40S|1082774|79.1|3.8|83.4|NA|3.2|
|r37s_cc_r30h_s1001_N_shared_BCE_constant|unit|NVIDIA L40S|1082773|46.1|2.8|49.1|NA|3.2|
|r37s_cc_r30h_s1001_N_shared_J_constant|unit|NVIDIA L40S|1082773|50.4|3.4|55.1|NA|3.2|
|r37s_cc_r30h_s1001_N_shared_NCE_constant|unit|NVIDIA L40S|1082773|44.5|2.8|47.5|NA|3.2|
|r37s_cc_r30h_s1001_N_shared_BCE_cosine|unit|NVIDIA L40S|1082773|46.1|2.8|49.1|NA|3.2|
|r37s_cc_r30h_s1001_N_shared_J_cosine|unit|NVIDIA L40S|1082773|49.8|2.8|52.9|NA|3.2|
|r37s_cc_r30h_s1001_N_shared_NCE_cosine|unit|NVIDIA L40S|1082773|44.5|2.8|47.8|NA|3.2|
|r37s_cc_r30h_s1001_T_shared_BCE_constant|unit|NVIDIA RTX PRO 6000 Blackwell Server Edition|1082775|22.9|1.3|24.3|NA|3.2|
|r37s_cc_r30h_s1001_T_shared_J_constant|unit|NVIDIA RTX PRO 6000 Blackwell Server Edition|1082775|25.5|1.4|27.8|NA|3.2|
|r37s_cc_r30h_s1001_T_shared_NCE_constant|unit|NVIDIA RTX PRO 6000 Blackwell Server Edition|1082775|22.2|1.4|23.7|NA|3.2|
|r37s_cc_r30h_s1001_T_shared_BCE_cosine|unit|NVIDIA RTX PRO 6000 Blackwell Server Edition|1082775|24.3|1.3|25.8|NA|3.2|
|r37s_cc_r30h_s1001_T_shared_J_cosine|unit|NVIDIA RTX PRO 6000 Blackwell Server Edition|1082775|25.3|1.3|26.7|NA|3.2|
|r37s_cc_r30h_s1001_T_shared_NCE_cosine|unit|NVIDIA RTX PRO 6000 Blackwell Server Edition|1082775|23.8|1.4|25.4|NA|3.2|
|NA|cache (build_cache)|NVIDIA RTX PRO 6000 Blackwell Server Edition|1082742|NA|NA|NA|62.0|NA|
|NA|prep (locks/calibration/top-32/trust)|NVIDIA RTX PRO 6000 Blackwell Server Edition|1082742|NA|NA|NA|48.9|NA|
|NA|CRITIC_CPU_TEST|cpu|1082766|NA|NA|NA|175.8|NA|

## 失败
无。

文件：critic_summary/ 下 RESULTS_FIXED.csv、CURVES.csv、DIRECTIONAL_RECALL.csv、RANK_TRANSITIONS.csv、ARCH_X_SCHEDULE_CONTRASTS.csv、OBJECTIVE_CONTRASTS.csv、T_SCHEDULE.csv、BOOTSTRAP.csv、GRAD_DIRECTION_DIAG.csv、STEP0_CHECK.csv、NULL_READOUTS.csv、COST_LEDGER.csv、IDENTITY_CROSSCHECK.csv、FAILURES.csv。
