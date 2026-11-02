Publication copy (sample names aliased; numbers and conclusions unchanged).

# R37 结果（仅数字）

生成 2026-11-02T14:30:59Z；DEV 全查询，两阶段（同父模型 H-top100 重排）；固定终点 8000 步；TEST 未读。seed1001 行为 R36 已完成单元（只读引用）。
未完成单元：无。CC152K 6 项未运行：无同系列 R30-H 父模型（见 PARENT_BINDING.json）。

## 父模型 H（同父模型基线）
|parent|unit|H_rsum|best_epoch|gpu|
|---|---|---|---|---|
|n60|r30_f60_s1001_H0|505.21|19|NVIDIA RTX PRO 6000 Blackwell Server Edition|
|n60s1002|r30v11_f60_s1002_HH|482.94|12|NVIDIA L40S|
|n60s1003|r30v11_f60_s1003_HH|496.07|12|NVIDIA L40S|

## A：相对同父模型 H 的 rSum 差（固定 8000 步）
|route|objective|d_rsum_vs_H_s1001|d_rsum_vs_H_s1002|d_rsum_vs_H_s1003|mean_3_parents|n_parents_positive|
|---|---|---|---|---|---|---|
|T_G|J|2.76|0.73|0.71|1.40|3|
|T_G|BCE|-7.69|-53.35|-16.88|-25.98|0|
|T_G|NCE|-2.39|-45.94|-13.23|-20.52|0|
|N_G|J|2.52|1.48|1.66|1.89|3|
|N_G|BCE|2.37|2.17|1.28|1.94|3|
|N_G|NCE|1.28|2.05|1.91|1.75|3|

## B：同一 H1001 的结构对照（rSum，8000 步）
|objective|H|T_G|N_G|TRUST_NATIVE_NULL_G|N_PRESENT_ONLY_G|TRUST_NATIVE_NULL_G-N_G|TRUST_NATIVE_NULL_G-T_G|N_PRESENT_ONLY_G-N_G|
|---|---|---|---|---|---|---|---|---|
|J|505.21|507.97|507.73|507.81|507.44|0.08|-0.16|-0.30|
|BCE|505.21|497.51|507.57|507.40|507.02|-0.18|9.88|-0.55|
|NCE|505.21|502.82|506.49|507.99|505.96|1.50|5.17|-0.53|

## 固定终点完整表
|unit|parent_seed|route|objective|i2t_r1|i2t_r5|i2t_r10|t2i_r1|t2i_r5|t2i_r10|rsum|H_rsum|d_rsum|change_2000_to_8000|rsum_steps_2000_4000_8000|i2t_rescued_r1|i2t_lost_r1|t2i_rescued_r1|t2i_lost_r1|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|r37_confirm_f60_h1002_T_G_J|1002|T_G|J|73.08|92.11|95.96|54.20|80.93|87.40|483.67|482.94|0.73|-0.91|484.58 / 484.69 / 483.67|43|60|221|197|
|r37_confirm_f60_h1002_T_G_BCE|1002|T_G|BCE|69.13|84.12|89.15|46.51|66.53|74.14|429.59|482.94|-53.35|-53.35|482.94 / 482.72 / 429.59|34|91|153|519|
|r37_confirm_f60_h1002_T_G_NCE|1002|T_G|NCE|72.68|88.17|91.52|48.52|65.50|70.61|437.00|482.94|-45.94|-47.04|484.04 / 484.28 / 437.00|40|61|174|438|
|r37_confirm_f60_h1002_N_G_J|1002|N_G|J|73.67|92.31|96.06|54.16|80.87|87.36|484.42|482.94|1.48|-1.64|486.06 / 486.09 / 484.42|42|53|198|176|
|r37_confirm_f60_h1002_N_G_BCE|1002|N_G|BCE|74.06|92.41|96.06|54.32|80.93|87.34|485.11|482.94|2.17|-0.97|486.07 / 487.00 / 485.11|40|47|208|178|
|r37_confirm_f60_h1002_N_G_NCE|1002|N_G|NCE|74.95|92.31|95.86|53.75|80.67|87.46|484.99|482.94|2.05|-1.10|486.09 / 485.66 / 484.99|36|34|195|194|
|r37_confirm_f60_h1003_T_G_J|1003|T_G|J|77.02|93.20|96.75|57.83|82.76|89.23|496.79|496.07|0.71|0.30|496.49 / 497.14 / 496.79|33|51|238|185|
|r37_confirm_f60_h1003_T_G_BCE|1003|T_G|BCE|76.73|91.72|95.56|54.99|76.79|83.41|479.19|496.07|-16.88|-14.79|493.98 / 494.77 / 479.19|28|49|133|224|
|r37_confirm_f60_h1003_T_G_NCE|1003|T_G|NCE|75.74|92.70|95.86|56.21|78.82|83.51|482.84|496.07|-13.23|-12.64|495.48 / 497.55 / 482.84|25|56|222|251|
|r37_confirm_f60_h1003_N_G_J|1003|N_G|J|78.50|93.39|96.55|57.57|82.86|88.86|497.73|496.07|1.66|-0.10|497.83 / 498.11 / 497.73|37|40|210|170|
|r37_confirm_f60_h1003_N_G_BCE|1003|N_G|BCE|78.11|93.29|96.55|57.61|82.82|88.97|497.36|496.07|1.28|-0.63|497.99 / 498.32 / 497.36|32|39|213|171|
|r37_confirm_f60_h1003_N_G_NCE|1003|N_G|NCE|79.09|93.79|96.45|56.90|82.74|89.01|497.99|496.07|1.91|-0.36|498.34 / 497.95 / 497.99|28|25|203|197|
|r37_factor_f60_h1001_TRUST_NATIVE_NULL_G_J|1001|TRUST_NATIVE_NULL_G|J|80.87|95.76|98.13|59.45|84.00|89.61|507.81|505.21|2.60|-0.36|508.17 / 508.30 / 507.81|33|31|203|165|
|r37_factor_f60_h1001_TRUST_NATIVE_NULL_G_BCE|1001|TRUST_NATIVE_NULL_G|BCE|80.97|95.46|98.13|59.23|83.94|89.66|507.40|505.21|2.19|-0.34|507.73 / 508.62 / 507.40|32|29|202|175|
|r37_factor_f60_h1001_TRUST_NATIVE_NULL_G_NCE|1001|TRUST_NATIVE_NULL_G|NCE|81.46|95.46|98.32|58.90|84.02|89.82|507.99|505.21|2.78|0.04|507.95 / 507.26 / 507.99|31|23|191|181|
|r37_factor_f60_h1001_N_PRESENT_ONLY_G_J|1001|N_PRESENT_ONLY_G|J|81.36|95.76|98.03|59.05|83.59|89.64|507.44|505.21|2.23|0.08|507.36 / 507.73 / 507.44|32|25|209|191|
|r37_factor_f60_h1001_N_PRESENT_ONLY_G_BCE|1001|N_PRESENT_ONLY_G|BCE|81.07|95.86|98.03|58.84|83.69|89.55|507.02|505.21|1.81|0.61|506.41 / 507.42 / 507.02|32|28|203|196|
|r37_factor_f60_h1001_N_PRESENT_ONLY_G_NCE|1001|N_PRESENT_ONLY_G|NCE|80.67|95.56|98.13|58.70|83.45|89.45|505.96|505.21|0.75|0.34|505.62 / 505.46 / 505.96|26|26|183|183|
|r36_T_n60_G_J|1001|T_G|J|80.87|96.15|98.13|59.43|83.69|89.70|507.97|505.21|2.76|0.10|507.87 / 507.95 / 507.97|37|35|202|165|
|r36_T_n60_G_BCE|1001|T_G|BCE|80.67|95.07|97.63|58.11|80.26|85.78|497.51|505.21|-7.69|-7.22|504.73 / 506.67 / 497.51|26|26|117|147|
|r36_T_n60_G_NCE|1001|T_G|NCE|79.88|95.07|98.13|58.19|83.02|88.54|502.82|505.21|-2.39|-0.43|503.25 / 502.74 / 502.82|31|39|202|228|
|r36_N_n60_G_J|1001|N_G|J|80.97|96.15|98.22|59.17|83.57|89.64|507.73|505.21|2.52|-0.89|508.62 / 507.79 / 507.73|30|27|200|176|
|r36_N_n60_G_BCE|1001|N_G|BCE|81.07|96.06|98.03|59.05|83.81|89.57|507.57|505.21|2.37|-0.53|508.11 / 507.79 / 507.57|32|28|205|187|
|r36_N_n60_G_NCE|1001|N_G|NCE|80.67|95.46|98.32|59.05|83.35|89.63|506.49|505.21|1.28|-1.72|508.21 / 506.79 / 506.49|27|27|209|191|

## 按图像分组 bootstrap（同父模型，rSum 差，B=2000，描述性）
|unit|diff|boot_mean|ci2_5|ci97_5|frac_boot_gt0|
|---|---|---|---|---|---|
|r37_confirm_f60_h1002_T_G_J|0.73|0.74|-2.29|3.71|0.68|
|r37_confirm_f60_h1002_T_G_BCE|-53.35|-53.33|-58.88|-47.81|0.00|
|r37_confirm_f60_h1002_T_G_NCE|-45.94|-45.96|-51.05|-41.14|0.00|
|r37_confirm_f60_h1002_N_G_J|1.48|1.48|-1.38|4.28|0.86|
|r37_confirm_f60_h1002_N_G_BCE|2.17|2.18|-0.59|4.99|0.94|
|r37_confirm_f60_h1002_N_G_NCE|2.05|2.06|-0.49|4.60|0.94|
|r37_confirm_f60_h1003_T_G_J|0.71|0.73|-1.91|3.23|0.71|
|r37_confirm_f60_h1003_T_G_BCE|-16.88|-16.93|-20.02|-13.90|0.00|
|r37_confirm_f60_h1003_T_G_NCE|-13.23|-13.25|-16.51|-10.08|0.00|
|r37_confirm_f60_h1003_N_G_J|1.66|1.67|-0.79|4.16|0.90|
|r37_confirm_f60_h1003_N_G_BCE|1.28|1.30|-1.16|3.77|0.86|
|r37_confirm_f60_h1003_N_G_NCE|1.91|1.93|-0.32|4.30|0.95|
|r37_factor_f60_h1001_TRUST_NATIVE_NULL_G_J|2.60|2.62|0.16|5.01|0.98|
|r37_factor_f60_h1001_TRUST_NATIVE_NULL_G_BCE|2.19|2.22|-0.24|4.66|0.96|
|r37_factor_f60_h1001_TRUST_NATIVE_NULL_G_NCE|2.78|2.78|0.45|5.17|0.99|
|r37_factor_f60_h1001_N_PRESENT_ONLY_G_J|2.23|2.24|-0.22|4.66|0.96|
|r37_factor_f60_h1001_N_PRESENT_ONLY_G_BCE|1.81|1.83|-0.65|4.32|0.93|
|r37_factor_f60_h1001_N_PRESENT_ONLY_G_NCE|0.75|0.75|-1.58|3.08|0.73|
|r36_T_n60_G_J|2.76|2.75|0.24|5.29|0.98|
|r36_T_n60_G_BCE|-7.69|-7.70|-10.16|-5.33|0.00|
|r36_T_n60_G_NCE|-2.39|-2.39|-5.23|0.38|0.04|
|r36_N_n60_G_J|2.52|2.54|0.16|4.93|0.98|
|r36_N_n60_G_BCE|2.37|2.38|-0.04|4.87|0.97|
|r36_N_n60_G_NCE|1.28|1.30|-1.10|3.83|0.85|

## N 类：J_real / J_null（FIT 锁定列表，观测标签）
|unit|i2t_J_real_part|i2t_J_null_part|t2i_J_real_part|t2i_J_null_part|i2t_null_label_fraction|t2i_null_label_fraction|
|---|---|---|---|---|---|---|
|r37_confirm_f60_h1002_N_G_J|0.35|0.02|0.44|0.16|0.11|0.59|
|r37_confirm_f60_h1002_N_G_BCE|0.34|0.02|0.44|0.16|0.11|0.59|
|r37_confirm_f60_h1002_N_G_NCE|0.10|-0.03|0.31|0.10|0.11|0.59|
|r37_confirm_f60_h1003_N_G_J|0.35|0.02|0.45|0.16|0.11|0.59|
|r37_confirm_f60_h1003_N_G_BCE|0.35|0.02|0.45|0.16|0.11|0.59|
|r37_confirm_f60_h1003_N_G_NCE|0.07|-0.02|0.30|0.10|0.11|0.59|
|r37_factor_f60_h1001_TRUST_NATIVE_NULL_G_J|0.38|-0.00|0.48|0.09|0.09|0.58|
|r37_factor_f60_h1001_TRUST_NATIVE_NULL_G_BCE|0.38|-0.00|0.47|0.09|0.09|0.58|
|r37_factor_f60_h1001_TRUST_NATIVE_NULL_G_NCE|0.28|-0.00|0.46|-0.00|0.09|0.58|
|r37_factor_f60_h1001_N_PRESENT_ONLY_G_J|0.39|-0.03|0.48|-0.23|0.09|0.58|
|r37_factor_f60_h1001_N_PRESENT_ONLY_G_BCE|0.38|-0.03|0.47|-0.23|0.09|0.58|
|r37_factor_f60_h1001_N_PRESENT_ONLY_G_NCE|-0.85|-0.00|-0.68|-0.04|0.09|0.58|
|r36_N_n60_G_J|0.40|0.01|0.47|0.16|0.09|0.58|
|r36_N_n60_G_BCE|0.39|0.01|0.47|0.16|0.09|0.58|
|r36_N_n60_G_NCE|0.18|-0.02|0.35|0.10|0.09|0.58|

## 训练监督：参与比例、正质量、重复（采样重放；EVAL_ 列仅评价器使用注入真值）
|unit|replay_batch_hash_matches_training|i2t_rows_observed_partner_in_list|t2i_rows_observed_partner_in_list|i2t_positive_mass_on_real_slots|t2i_positive_mass_on_real_slots|EVAL_i2t_true_share_of_real_positive_mass|EVAL_t2i_true_share_of_real_positive_mass|unique_query_records_t2i|max_repeats_query_record_t2i|empty_valid_batches_i2t|empty_valid_batches_t2i|
|---|---|---|---|---|---|---|---|---|---|---|---|
|r37_confirm_f60_h1002_T_G_J|True|1.0000|1.0000|1.0000|1.0000|0.9467|0.9075|53733|11|||
|r37_confirm_f60_h1002_T_G_BCE|True|1.0000|1.0000|1.0000|1.0000|0.9467|0.9075|53733|11|||
|r37_confirm_f60_h1002_T_G_NCE|True|1.0000|1.0000|1.0000|1.0000|0.9467|0.9075|53733|11|||
|r37_confirm_f60_h1002_N_G_J|True|0.8768|0.3902|0.8768|0.3902|0.9846|0.9666|84814|8|||
|r37_confirm_f60_h1002_N_G_BCE|True|0.8768|0.3902|0.8768|0.3902|0.9846|0.9666|84814|8|||
|r37_confirm_f60_h1002_N_G_NCE|True|0.8768|0.3902|0.8768|0.3902|0.9846|0.9666|84814|8|||
|r37_confirm_f60_h1003_T_G_J|True|1.0000|1.0000|1.0000|1.0000|0.9512|0.9138|53687|10|||
|r37_confirm_f60_h1003_T_G_BCE|True|1.0000|1.0000|1.0000|1.0000|0.9512|0.9138|53687|10|||
|r37_confirm_f60_h1003_T_G_NCE|True|1.0000|1.0000|1.0000|1.0000|0.9512|0.9138|53687|10|||
|r37_confirm_f60_h1003_N_G_J|True|0.8800|0.3899|0.8800|0.3899|0.9870|0.9700|84814|8|||
|r37_confirm_f60_h1003_N_G_BCE|True|0.8800|0.3899|0.8800|0.3899|0.9870|0.9700|84814|8|||
|r37_confirm_f60_h1003_N_G_NCE|True|0.8800|0.3899|0.8800|0.3899|0.9870|0.9700|84814|8|||
|r37_factor_f60_h1001_TRUST_NATIVE_NULL_G_J|True|0.9764|0.9239|0.9764|0.9239|0.9830|0.9668|53879|12|||
|r37_factor_f60_h1001_TRUST_NATIVE_NULL_G_BCE|True|0.9764|0.9239|0.9764|0.9239|0.9830|0.9668|53879|12|||
|r37_factor_f60_h1001_TRUST_NATIVE_NULL_G_NCE|True|0.9764|0.9239|0.9764|0.9239|0.9830|0.9668|53879|12|||
|r37_factor_f60_h1001_N_PRESENT_ONLY_G_J|True|0.8951|0.3993|0.8951|0.3993|0.9825|0.9631|84814|8|0|3|
|r37_factor_f60_h1001_N_PRESENT_ONLY_G_BCE|True|0.8951|0.3993|0.8951|0.3993|0.9825|0.9631|84814|8|0|3|
|r37_factor_f60_h1001_N_PRESENT_ONLY_G_NCE|True|0.8951|0.3993|0.8951|0.3993|0.9825|0.9631|84814|8|0|3|
|r36_T_n60_G_J|True|1.0000|1.0000|1.0000|1.0000|0.9492|0.9145|53960|11|||
|r36_T_n60_G_BCE|True|1.0000|1.0000|1.0000|1.0000|0.9492|0.9145|53960|11|||
|r36_T_n60_G_NCE|True|1.0000|1.0000|1.0000|1.0000|0.9492|0.9145|53960|11|||
|r36_N_n60_G_J|True|0.8951|0.3993|0.8951|0.3993|0.9825|0.9631|84814|8|||
|r36_N_n60_G_BCE|True|0.8951|0.3993|0.8951|0.3993|0.9825|0.9631|84814|8|||
|r36_N_n60_G_NCE|True|0.8951|0.3993|0.8951|0.3993|0.9825|0.9631|84814|8|||

## 成本
|unit|item|gpu|train_seconds|eval_seconds|seconds|
|---|---|---|---|---|---|
|r37_confirm_f60_h1002_T_G_J||NVIDIA RTX PRO 6000 Blackwell Server Edition|29.0|9.6||
|r37_confirm_f60_h1002_T_G_BCE||NVIDIA RTX PRO 6000 Blackwell Server Edition|22.7|6.5||
|r37_confirm_f60_h1002_T_G_NCE||NVIDIA RTX PRO 6000 Blackwell Server Edition|22.2|5.1||
|r37_confirm_f60_h1002_N_G_J||NVIDIA RTX PRO 6000 Blackwell Server Edition|44.6|10.6||
|r37_confirm_f60_h1002_N_G_BCE||NVIDIA RTX PRO 6000 Blackwell Server Edition|38.5|8.5||
|r37_confirm_f60_h1002_N_G_NCE||NVIDIA RTX PRO 6000 Blackwell Server Edition|37.1|7.7||
|r37_confirm_f60_h1003_T_G_J||NVIDIA L40S|57.0|33.1||
|r37_confirm_f60_h1003_T_G_BCE||NVIDIA L40S|51.9|40.6||
|r37_confirm_f60_h1003_T_G_NCE||NVIDIA L40S|49.6|11.6||
|r37_confirm_f60_h1003_N_G_J||NVIDIA L40S|71.6|16.7||
|r37_confirm_f60_h1003_N_G_BCE||NVIDIA L40S|64.6|69.3||
|r37_confirm_f60_h1003_N_G_NCE||NVIDIA L40S|61.9|148.1||
|r37_factor_f60_h1001_TRUST_NATIVE_NULL_G_J||NVIDIA L40S|72.6|62.0||
|r37_factor_f60_h1001_TRUST_NATIVE_NULL_G_BCE||NVIDIA L40S|72.0|14.6||
|r37_factor_f60_h1001_TRUST_NATIVE_NULL_G_NCE||NVIDIA L40S|69.2|22.4||
|r37_factor_f60_h1001_N_PRESENT_ONLY_G_J||NVIDIA RTX PRO 6000 Blackwell Server Edition|32.1|7.3||
|r37_factor_f60_h1001_N_PRESENT_ONLY_G_BCE||NVIDIA RTX PRO 6000 Blackwell Server Edition|29.2|13.0||
|r37_factor_f60_h1001_N_PRESENT_ONLY_G_NCE||NVIDIA RTX PRO 6000 Blackwell Server Edition|28.3|6.1||
|r36_T_n60_G_J||NVIDIA RTX PRO 6000 Blackwell Server Edition|26.0|9.1||
|r36_T_n60_G_BCE||NVIDIA RTX PRO 6000 Blackwell Server Edition|22.5|6.2||
|r36_T_n60_G_NCE||NVIDIA RTX PRO 6000 Blackwell Server Edition|21.8|4.9||
|r36_N_n60_G_J||NVIDIA RTX PRO 6000 Blackwell Server Edition|35.0|9.8||
|r36_N_n60_G_BCE||NVIDIA RTX PRO 6000 Blackwell Server Edition|28.3|5.9||
|r36_N_n60_G_NCE||NVIDIA RTX PRO 6000 Blackwell Server Edition|28.0|5.3||
||cache n60s1002|NVIDIA L40S|||191.2|
||cache n60s1003|NVIDIA L40S|||124.6|
||prep n60s1002+n60s1003 (calibration, top-32 tables, B-TRUST eval_train)|NVIDIA L40S|||253.1|
||cpu test|cpu|||3.5|
||parent H training (not R37 cost; existing R30 runs)||||R30 PEAK wall: s1001 10986, s1002 10979, s1003 13455 (co-train only; + 5-epoch warmup parents)|

文件：RESULTS_FIXED.csv、CURVES.csv、PAIRED_THREE_PARENT.csv、STRUCTURE_B.csv、NULL_READOUTS.csv、PARTICIPATION.csv、BOOTSTRAP.csv、COST_LEDGER.csv。
