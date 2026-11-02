Publication copy (sample names aliased; numbers and conclusions unchanged).

# R37C 结果（CC152K，仅数字；开发实验，本线未读 TEST）

生成 2026-11-02T16:17:34Z；DEV 1000 图 × 1000 文（每图 1 文），全查询，两阶段（同父模型 H-top100 重排）；固定终点 8000 步。与 Flickr 分开报告：只报相对同一父模型的增益，不跨数据集比较绝对 rSum。
未完成单元：无。

## 父模型 H（R31 IN²R 原生，作者默认 warmup 1 / K 1 / i2t,t2i 关；与 Flickr R30-H 不同系列）
|unit|best_rsum_logged|best_epoch|native_rsum_reproduced|exact|gpu|i2t_r1|i2t_r5|i2t_r10|t2i_r1|t2i_r5|t2i_r10|
|---|---|---|---|---|---|---|---|---|---|---|---|
|r31_cc_s1001_H0_default|389.70|38|389.70|True|Tesla V100S-PCIE-32GB|45.00|70.80|79.60|44.50|70.90|78.90|

## 重建的 CC152K 监督构件（观测关系 = 原生恒等，记录 k ↔ 图 k ↔ 文 k）
|a0|b0|trust_n|trust_retention|i2t_obs_in_top32|t2i_obs_in_top32|i2t_obs_rank1|t2i_obs_rank1|
|---|---|---|---|---|---|---|---|
|21.5641|-4.3227|137675|0.9178|0.7191|0.6867|0.2612|0.2101|

## 固定终点（8000 步），相对同父模型 H
|unit|route|objective|i2t_r1|i2t_r5|i2t_r10|t2i_r1|t2i_r5|t2i_r10|rsum|H_rsum|d_rsum|rsum_steps_0_2000_4000_8000|step0_passed|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|r37c_pilot_cc_h1001_T_G_J|T_G|J|0.10|0.50|1.80|0.00|0.90|1.90|5.20|389.70|-384.50|389.70 / 393.00 / 386.50 / 5.20|True|
|r37c_pilot_cc_h1001_T_G_BCE|T_G|BCE|0.10|0.40|0.90|0.00|0.70|1.00|3.10|389.70|-386.60|389.70 / 386.90 / 324.40 / 3.10|True|
|r37c_pilot_cc_h1001_T_G_NCE|T_G|NCE|0.10|0.30|0.90|0.20|0.30|0.70|2.50|389.70|-387.20|389.70 / 90.20 / 2.40 / 2.50|True|
|r37c_pilot_cc_h1001_N_G_J|N_G|J|45.70|71.70|79.20|45.40|71.40|78.90|392.30|389.70|2.60|389.70 / 391.20 / 392.00 / 392.30|True|
|r37c_pilot_cc_h1001_N_G_BCE|N_G|BCE|45.90|71.90|79.20|45.60|71.20|79.30|393.10|389.70|3.40|389.70 / 391.30 / 391.30 / 393.10|True|
|r37c_pilot_cc_h1001_N_G_NCE|N_G|NCE|44.70|70.10|78.00|45.00|70.40|79.30|387.50|389.70|-2.20|389.70 / 388.20 / 387.90 / 387.50|True|

## 逐方向 R@K 差（单元 − H）
|unit|d_i2t_r1|d_i2t_r5|d_i2t_r10|d_t2i_r1|d_t2i_r5|d_t2i_r10|d_rsum|
|---|---|---|---|---|---|---|---|
|r37c_pilot_cc_h1001_T_G_J|-44.90|-70.30|-77.80|-44.50|-70.00|-77.00|-384.50|
|r37c_pilot_cc_h1001_T_G_BCE|-44.90|-70.40|-78.70|-44.50|-70.20|-77.90|-386.60|
|r37c_pilot_cc_h1001_T_G_NCE|-44.90|-70.50|-78.70|-44.30|-70.60|-78.20|-387.20|
|r37c_pilot_cc_h1001_N_G_J|0.70|0.90|-0.40|0.90|0.50|0.00|2.60|
|r37c_pilot_cc_h1001_N_G_BCE|0.90|1.10|-0.40|1.10|0.30|0.40|3.40|
|r37c_pilot_cc_h1001_N_G_NCE|-0.30|-0.70|-1.60|0.50|-0.50|0.40|-2.20|

## 救回 / 丢失（相对 H，R@1/5/10）
|unit|i2t_rescued_r1|i2t_lost_r1|i2t_rescued_r5|i2t_lost_r5|i2t_rescued_r10|i2t_lost_r10|t2i_rescued_r1|t2i_lost_r1|t2i_rescued_r5|t2i_lost_r5|t2i_rescued_r10|t2i_lost_r10|
|---|---|---|---|---|---|---|---|---|---|---|---|---|
|r37c_pilot_cc_h1001_T_G_J|1|450|4|707|15|793|0|445|7|707|13|783|
|r37c_pilot_cc_h1001_T_G_BCE|1|450|4|708|9|796|0|445|6|708|9|788|
|r37c_pilot_cc_h1001_T_G_NCE|1|450|3|708|9|796|2|445|3|709|7|789|
|r37c_pilot_cc_h1001_N_G_J|37|30|27|18|12|16|47|38|26|21|20|20|
|r37c_pilot_cc_h1001_N_G_BCE|40|31|32|21|15|19|49|38|27|24|24|20|
|r37c_pilot_cc_h1001_N_G_NCE|34|37|27|34|10|26|44|39|26|31|26|22|

## 按图像分组 bootstrap（rSum 差，B=2000，描述性）
|unit|diff|boot_mean|ci2_5|ci97_5|frac_boot_gt0|
|---|---|---|---|---|---|
|r37c_pilot_cc_h1001_T_G_J|-384.50|-384.40|-398.70|-369.70|0.00|
|r37c_pilot_cc_h1001_T_G_BCE|-386.60|-386.49|-400.70|-372.00|0.00|
|r37c_pilot_cc_h1001_T_G_NCE|-387.20|-387.11|-401.40|-372.80|0.00|
|r37c_pilot_cc_h1001_N_G_J|2.60|2.56|-0.90|6.20|0.91|
|r37c_pilot_cc_h1001_N_G_BCE|3.40|3.38|-0.30|7.10|0.96|
|r37c_pilot_cc_h1001_N_G_NCE|-2.20|-2.20|-5.90|1.60|0.12|

## N 类：J_real / J_null（FIT 锁定列表，观测标签）
|unit|i2t_J_real_part|i2t_J_null_part|t2i_J_real_part|t2i_J_null_part|i2t_null_label_fraction|t2i_null_label_fraction|
|---|---|---|---|---|---|---|
|r37c_pilot_cc_h1001_N_G_J|0.2066|0.0693|0.2166|0.0752|0.2949|0.3145|
|r37c_pilot_cc_h1001_N_G_BCE|0.2038|0.0701|0.2140|0.0761|0.2949|0.3145|
|r37c_pilot_cc_h1001_N_G_NCE|-0.0144|-0.0130|0.0283|0.0184|0.2949|0.3145|

## 训练监督：参与比例、正质量、重复（采样重放，与训练批哈希核对；CC152K 无注入真值，无真值列）
|unit|replay_batch_hash_matches_training|i2t_rows_observed_partner_in_list|t2i_rows_observed_partner_in_list|i2t_positive_mass_on_real_slots|t2i_positive_mass_on_real_slots|i2t_positive_mass_on_null|t2i_positive_mass_on_null|unique_query_records_t2i|max_repeats_query_record_t2i|
|---|---|---|---|---|---|---|---|---|---|
|r37c_pilot_cc_h1001_T_G_J|True|1.0000|1.0000|1.0000|1.0000|0.0000|0.0000|83374|8|
|r37c_pilot_cc_h1001_T_G_BCE|True|1.0000|1.0000|1.0000|1.0000|0.0000|0.0000|83374|8|
|r37c_pilot_cc_h1001_T_G_NCE|True|1.0000|1.0000|1.0000|1.0000|0.0000|0.0000|83374|8|
|r37c_pilot_cc_h1001_N_G_J|True|0.7174|0.6871|0.7174|0.6871|0.2826|0.3129|86088|8|
|r37c_pilot_cc_h1001_N_G_BCE|True|0.7174|0.6871|0.7174|0.6871|0.2826|0.3129|86088|8|
|r37c_pilot_cc_h1001_N_G_NCE|True|0.7174|0.6871|0.7174|0.6871|0.2826|0.3129|86088|8|

## 成本
|unit|item|gpu|job|train_seconds|eval_seconds|unit_wall_seconds|seconds|peak_gpu_gb|
|---|---|---|---|---|---|---|---|---|
|r37c_pilot_cc_h1001_T_G_J||NVIDIA L40S|1082183|57.4|6.1|65.1||3.2|
|r37c_pilot_cc_h1001_T_G_BCE||NVIDIA L40S|1082183|52.0|7.9|60.3||3.2|
|r37c_pilot_cc_h1001_T_G_NCE||NVIDIA L40S|1082183|49.9|4.1|54.4||3.2|
|r37c_pilot_cc_h1001_N_G_J||NVIDIA L40S|1082184|70.4|18.6|90.6||3.2|
|r37c_pilot_cc_h1001_N_G_BCE||NVIDIA L40S|1082184|64.5|5.8|70.7||3.2|
|r37c_pilot_cc_h1001_N_G_NCE||NVIDIA L40S|1082184|61.8|9.1|71.3||3.2|
||cache cc (native DEV reproduction + FIT encoding)|Tesla V100S-PCIE-32GB|1082118||||189.3||
||prep cc (locks, calibration, top-32 tables, B-TRUST eval_train)|Tesla V100S-PCIE-32GB|1082118||||233.0||
||cpu test|cpu|1082182||||9.5||
||parent H training (not R37C cost; existing R31 run cc_s1001/H0_default, 1+40 epochs)||||||see R31 RUN/EPOCHS||

文件：RESULTS_FIXED.csv、CURVES.csv、NULL_READOUTS.csv、PARTICIPATION.csv、BOOTSTRAP.csv、COST_LEDGER.csv。
