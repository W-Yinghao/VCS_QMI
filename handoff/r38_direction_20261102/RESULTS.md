Publication copy (sample names aliased; numbers and conclusions unchanged).

# R38 结果（仅数字；开发实验，本线未读 TEST）

生成 2026-11-02T18:45:46Z。冻结 encoder 的 critic fits；固定主终点 8000 步（曲线最大值只作次要读数）。Flickr60 R30-H1001 与 CC152K R31 父模型不同系列，只报同父模型内的差。 复用（只读）的 shared/constant 参照：R36（Flickr s1001 N/T）、R37（Flickr s1002/s1003 N/T）、R37C（CC s1001 N/T）；CC s1002/s1003 的 N shared/constant 为 R38 A 块。54 项为用户补充（包内 B/C 格应用到全部父模型，冻结前登记、非结果驱动）。

未完成单元：无。新单元完成 84/84（包内 30 + 用户补充 54）。

## 父模型 H（native DEV，按各自 cache 复算）
|parent|i2t_r1|i2t_r5|i2t_r10|t2i_r1|t2i_r5|t2i_r10|rsum|
|---|---|---|---|---|---|---|---|
|f60_R30_s1001|80.67|95.07|98.22|58.70|83.31|89.23|505.21|
|f60_R30_s1002|74.75|91.42|95.86|53.73|80.06|87.12|482.94|
|f60_R30_s1003|78.80|93.69|96.75|56.79|81.89|88.17|496.07|
|cc_R31_s1001|45.00|70.80|79.60|44.50|70.90|78.90|389.70|
|cc_R31_s1002|45.30|72.40|80.60|44.80|70.80|80.70|394.60|
|cc_R31_s1003|46.00|72.00|79.70|42.80|71.10|81.00|392.60|

## B：结构 × 日程（N 路线；rSum 与双向 R@1；shared/constant = 复用参照）
|parent|objective|H_rsum|rsum_shared_constant|rsum_shared_cosine|rsum_directional_constant|rsum_directional_cosine|rsum_dir_minus_shared_constant|rsum_dir_minus_shared_cosine|rsum_cos_minus_const_shared|rsum_cos_minus_const_dir|rsum_interaction|
|---|---|---|---|---|---|---|---|---|---|---|---|
|f60_R30_s1001|J|505.21|507.73|509.15|507.51|508.95|-0.22|-0.20|1.42|1.44|0.02|
|f60_R30_s1001|BCE|505.21|507.57|508.40|507.73|508.76|0.16|0.36|0.83|1.03|0.20|
|f60_R30_s1001|NCE|505.21|506.49|506.69|505.82|506.15|-0.67|-0.53|0.20|0.34|0.14|
|f60_R30_s1002|J|482.94|484.42|485.94|484.26|486.33|-0.16|0.39|1.52|2.07|0.55|
|f60_R30_s1002|BCE|482.94|485.11|486.51|485.27|487.08|0.16|0.57|1.40|1.81|0.41|
|f60_R30_s1002|NCE|482.94|484.99|485.74|484.40|485.29|-0.59|-0.45|0.75|0.89|0.14|
|f60_R30_s1003|J|496.07|497.73|498.46|497.04|498.36|-0.69|-0.10|0.73|1.32|0.59|
|f60_R30_s1003|BCE|496.07|497.36|497.91|497.26|497.89|-0.10|-0.02|0.55|0.63|0.08|
|f60_R30_s1003|NCE|496.07|497.99|498.48|497.48|497.51|-0.51|-0.97|0.49|0.04|-0.45|
|cc_R31_s1001|J|389.70|392.30|392.70|392.50|392.60|0.20|-0.10|0.40|0.10|-0.30|
|cc_R31_s1001|BCE|389.70|393.10|392.80|393.00|392.10|-0.10|-0.70|-0.30|-0.90|-0.60|
|cc_R31_s1001|NCE|389.70|387.50|388.60|387.50|388.20|-0.00|-0.40|1.10|0.70|-0.40|
|cc_R31_s1002|J|394.60|394.20|394.10|393.70|394.50|-0.50|0.40|-0.10|0.80|0.90|
|cc_R31_s1002|BCE|394.60|392.40|392.90|392.70|393.50|0.30|0.60|0.50|0.80|0.30|
|cc_R31_s1002|NCE|394.60|391.10|390.30|390.40|390.40|-0.70|0.10|-0.80|0.00|0.80|
|cc_R31_s1003|J|392.60|394.60|393.50|394.40|393.80|-0.20|0.30|-1.10|-0.60|0.50|
|cc_R31_s1003|BCE|392.60|393.50|393.00|393.30|393.60|-0.20|0.60|-0.50|0.30|0.80|
|cc_R31_s1003|NCE|392.60|392.50|391.30|392.40|391.30|-0.10|0.00|-1.20|-1.10|0.10|

|parent|objective|H_i2t_r1|i2t_r1_shared_constant|i2t_r1_shared_cosine|i2t_r1_directional_constant|i2t_r1_directional_cosine|i2t_r1_dir_minus_shared_constant|i2t_r1_dir_minus_shared_cosine|
|---|---|---|---|---|---|---|---|---|
|f60_R30_s1001|J|80.67|80.97|81.76|80.97|81.66|0.00|-0.10|
|f60_R30_s1001|BCE|80.67|81.07|81.36|81.56|81.66|0.49|0.30|
|f60_R30_s1001|NCE|80.67|80.67|80.87|80.57|80.77|-0.10|-0.10|
|f60_R30_s1002|J|74.75|73.67|74.26|73.67|74.46|0.00|0.20|
|f60_R30_s1002|BCE|74.75|74.06|74.75|74.26|75.25|0.20|0.49|
|f60_R30_s1002|NCE|74.75|74.95|75.05|74.85|75.05|-0.10|0.00|
|f60_R30_s1003|J|78.80|78.50|78.30|78.11|78.30|-0.39|0.00|
|f60_R30_s1003|BCE|78.80|78.11|78.01|78.21|78.21|0.10|0.20|
|f60_R30_s1003|NCE|78.80|79.09|78.80|78.99|78.60|-0.10|-0.20|
|cc_R31_s1001|J|45.00|45.70|45.30|45.80|45.30|0.10|0.00|
|cc_R31_s1001|BCE|45.00|45.90|45.60|46.00|45.40|0.10|-0.20|
|cc_R31_s1001|NCE|45.00|44.70|44.90|44.50|44.80|-0.20|-0.10|
|cc_R31_s1002|J|45.30|45.10|44.70|45.00|44.90|-0.10|0.20|
|cc_R31_s1002|BCE|45.30|44.40|44.70|44.30|44.70|-0.10|0.00|
|cc_R31_s1002|NCE|45.30|44.30|44.90|44.00|45.00|-0.30|0.10|
|cc_R31_s1003|J|46.00|44.70|45.90|44.80|46.00|0.10|0.10|
|cc_R31_s1003|BCE|46.00|44.80|45.30|44.80|45.60|0.00|0.30|
|cc_R31_s1003|NCE|46.00|45.10|44.80|45.10|44.90|0.00|0.10|

|parent|objective|H_t2i_r1|t2i_r1_shared_constant|t2i_r1_shared_cosine|t2i_r1_directional_constant|t2i_r1_directional_cosine|t2i_r1_dir_minus_shared_constant|t2i_r1_dir_minus_shared_cosine|
|---|---|---|---|---|---|---|---|---|
|f60_R30_s1001|J|58.70|59.17|59.55|59.25|59.59|0.08|0.04|
|f60_R30_s1001|BCE|58.70|59.05|59.41|58.93|59.49|-0.12|0.08|
|f60_R30_s1001|NCE|58.70|59.05|59.23|58.86|58.97|-0.20|-0.26|
|f60_R30_s1002|J|53.73|54.16|54.65|54.12|54.46|-0.04|-0.20|
|f60_R30_s1002|BCE|53.73|54.32|54.58|54.22|54.44|-0.10|-0.14|
|f60_R30_s1002|NCE|53.73|53.75|54.42|53.81|54.12|0.06|-0.30|
|f60_R30_s1003|J|56.79|57.57|58.07|57.44|57.89|-0.14|-0.18|
|f60_R30_s1003|BCE|56.79|57.61|58.15|57.44|57.89|-0.18|-0.26|
|f60_R30_s1003|NCE|56.79|56.90|57.55|56.84|57.14|-0.06|-0.41|
|cc_R31_s1001|J|44.50|45.40|45.30|45.40|45.40|0.00|0.10|
|cc_R31_s1001|BCE|44.50|45.60|45.00|45.30|44.90|-0.30|-0.10|
|cc_R31_s1001|NCE|44.50|45.00|44.70|44.80|44.60|-0.20|-0.10|
|cc_R31_s1002|J|44.80|45.30|45.20|45.10|45.30|-0.20|0.10|
|cc_R31_s1002|BCE|44.80|45.30|44.90|45.40|45.00|0.10|0.10|
|cc_R31_s1002|NCE|44.80|44.20|44.30|44.00|44.30|-0.20|0.00|
|cc_R31_s1003|J|42.80|44.90|44.90|44.90|44.80|0.00|-0.10|
|cc_R31_s1003|BCE|42.80|44.80|44.70|44.80|44.80|0.00|0.10|
|cc_R31_s1003|NCE|42.80|45.10|45.20|45.20|45.10|0.10|-0.10|

## 三父状态合并（每数据集内；均值与正号数，不跨数据集）
|dataset|objective|rsum_dir_minus_shared_constant_mean3|rsum_dir_minus_shared_constant_npos|rsum_dir_minus_shared_cosine_mean3|rsum_dir_minus_shared_cosine_npos|rsum_cos_minus_const_shared_mean3|rsum_cos_minus_const_dir_mean3|i2t_r1_dir_minus_shared_constant_mean3|t2i_r1_dir_minus_shared_constant_mean3|T_cos_minus_const_mean3|T_cos_minus_const_npos|
|---|---|---|---|---|---|---|---|---|---|---|---|
|Flickr60 (R30-H s1001/1002/1003)|J|-0.36|0|0.03|1|1.22|1.61|-0.13|-0.03|0.78|3|
|Flickr60 (R30-H s1001/1002/1003)|BCE|0.07|2|0.30|2|0.93|1.16|0.26|-0.13|26.33|3|
|Flickr60 (R30-H s1001/1002/1003)|NCE|-0.59|0|-0.65|0|0.48|0.42|-0.10|-0.07|21.10|3|
|CC152K (R31 H0_default s1001/1002/1003)|J|-0.17|1|0.20|2|-0.27|0.10|0.03|-0.07|379.47|3|
|CC152K (R31 H0_default s1001/1002/1003)|BCE|0.00|1|0.17|2|-0.10|0.07|0.00|-0.07|237.70|3|
|CC152K (R31 H0_default s1001/1002/1003)|NCE|-0.27|0|-0.10|1|-0.30|-0.13|-0.17|-0.10|0.20|1|

|dataset|objective|rsum_J_minus_BCE_mean3|rsum_J_minus_BCE_npos|rsum_J_minus_NCE_mean3|rsum_J_minus_NCE_npos|i2t_r1_J_minus_BCE_mean3|t2i_r1_J_minus_BCE_mean3|
|---|---|---|---|---|---|---|---|
|Flickr60 (R30-H s1001/1002/1003)|J-BCE / J-NCE | N shared constant|-0.05|2|0.14|1|-0.03|-0.03|
|Flickr60 (R30-H s1001/1002/1003)|J-BCE / J-NCE | N shared cosine|0.24|2|0.88|2|0.07|0.05|
|Flickr60 (R30-H s1001/1002/1003)|J-BCE / J-NCE | N directional constant|-0.48|0|0.37|1|-0.43|0.07|
|Flickr60 (R30-H s1001/1002/1003)|J-BCE / J-NCE | N directional cosine|-0.03|2|1.56|3|-0.23|0.04|
|Flickr60 (R30-H s1001/1002/1003)|J-BCE / J-NCE | T shared constant|27.38|3|21.92|3|1.48|3.95|
|Flickr60 (R30-H s1001/1002/1003)|J-BCE / J-NCE | T shared cosine|1.83|3|1.60|2|0.59|0.34|
|CC152K (R31 H0_default s1001/1002/1003)|J-BCE / J-NCE | N shared constant|0.70|2|3.33|3|0.13|-0.03|
|CC152K (R31 H0_default s1001/1002/1003)|J-BCE / J-NCE | N shared cosine|0.53|2|3.37|3|0.10|0.27|
|CC152K (R31 H0_default s1001/1002/1003)|J-BCE / J-NCE | N directional constant|0.53|2|3.43|3|0.17|-0.03|
|CC152K (R31 H0_default s1001/1002/1003)|J-BCE / J-NCE | N directional cosine|0.57|3|3.67|3|0.17|0.27|
|CC152K (R31 H0_default s1001/1002/1003)|J-BCE / J-NCE | T shared constant|1.43|3|2.17|3|0.00|0.03|
|CC152K (R31 H0_default s1001/1002/1003)|J-BCE / J-NCE | T shared cosine|143.20|3|381.43|3|25.87|17.53|

## 同父模型同路线单元的初始化 / 抽样一致性（共享键 init 哈希与批索引哈希链）
|parent|route|n_units|n_distinct_init_shared_keys|n_distinct_batch_chains|identical|
|---|---|---|---|---|---|
|f60_R30_s1001|N|12|1|1|True|
|f60_R30_s1001|T|6|1|1|True|
|f60_R30_s1002|N|12|1|1|True|
|f60_R30_s1002|T|6|1|1|True|
|f60_R30_s1003|N|12|1|1|True|
|f60_R30_s1003|T|6|1|1|True|
|cc_R31_s1001|N|12|1|1|True|
|cc_R31_s1001|T|6|1|1|True|
|cc_R31_s1002|N|12|1|1|True|
|cc_R31_s1002|T|6|1|1|True|
|cc_R31_s1003|N|12|1|1|True|
|cc_R31_s1003|T|6|1|1|True|

## 目标对比（同父模型、同路线、同结构、同日程）
|parent|route|architecture|schedule|rsum_J|rsum_BCE|rsum_NCE|rsum_J_minus_BCE|rsum_J_minus_NCE|i2t_r1_J_minus_BCE|t2i_r1_J_minus_BCE|
|---|---|---|---|---|---|---|---|---|---|---|
|cc_R31_s1001|N|directional|constant|392.50|393.00|387.50|-0.50|5.00|-0.20|0.10|
|cc_R31_s1001|N|directional|cosine|392.60|392.10|388.20|0.50|4.40|-0.10|0.50|
|cc_R31_s1001|N|shared|constant|392.30|393.10|387.50|-0.80|4.80|-0.20|-0.20|
|cc_R31_s1001|N|shared|cosine|392.70|392.80|388.60|-0.10|4.10|-0.30|0.30|
|cc_R31_s1001|T|shared|constant|5.20|3.10|2.50|2.10|2.70|0.00|0.00|
|cc_R31_s1001|T|shared|cosine|381.70|240.20|2.40|141.50|379.30|26.10|17.90|
|cc_R31_s1002|N|directional|constant|393.70|392.70|390.40|1.00|3.30|0.70|-0.30|
|cc_R31_s1002|N|directional|cosine|394.50|393.50|390.40|1.00|4.10|0.20|0.30|
|cc_R31_s1002|N|shared|constant|394.20|392.40|391.10|1.80|3.10|0.70|0.00|
|cc_R31_s1002|N|shared|cosine|394.10|392.90|390.30|1.20|3.80|0.00|0.30|
|cc_R31_s1002|T|shared|constant|4.30|3.00|2.20|1.30|2.10|0.00|0.10|
|cc_R31_s1002|T|shared|cosine|386.00|219.30|3.00|166.70|383.00|28.40|20.60|
|cc_R31_s1003|N|directional|constant|394.40|393.30|392.40|1.10|2.00|0.00|0.10|
|cc_R31_s1003|N|directional|cosine|393.80|393.60|391.30|0.20|2.50|0.40|0.00|
|cc_R31_s1003|N|shared|constant|394.60|393.50|392.50|1.10|2.10|-0.10|0.10|
|cc_R31_s1003|N|shared|cosine|393.50|393.00|391.30|0.50|2.20|0.60|0.20|
|cc_R31_s1003|T|shared|constant|3.80|2.90|2.10|0.90|1.70|0.00|0.00|
|cc_R31_s1003|T|shared|cosine|384.00|262.60|2.00|121.40|382.00|23.10|14.10|
|f60_R30_s1001|N|directional|constant|507.51|507.73|505.82|-0.22|1.70|-0.59|0.32|
|f60_R30_s1001|N|directional|cosine|508.95|508.76|506.15|0.20|2.80|0.00|0.10|
|f60_R30_s1001|N|shared|constant|507.73|507.57|506.49|0.16|1.24|-0.10|0.12|
|f60_R30_s1001|N|shared|cosine|509.15|508.40|506.69|0.75|2.47|0.39|0.14|
|f60_R30_s1001|T|shared|constant|507.97|497.51|502.82|10.45|5.15|0.20|1.32|
|f60_R30_s1001|T|shared|cosine|508.01|506.59|503.53|1.42|4.48|0.30|0.41|
|f60_R30_s1002|N|directional|constant|484.26|485.27|484.40|-1.01|-0.14|-0.59|-0.10|
|f60_R30_s1002|N|directional|cosine|486.33|487.08|485.29|-0.75|1.05|-0.79|0.02|
|f60_R30_s1002|N|shared|constant|484.42|485.11|484.99|-0.69|-0.57|-0.39|-0.16|
|f60_R30_s1002|N|shared|cosine|485.94|486.51|485.74|-0.57|0.20|-0.49|0.08|
|f60_R30_s1002|T|shared|constant|483.67|429.59|437.00|54.08|46.67|3.94|7.69|
|f60_R30_s1002|T|shared|cosine|485.23|483.75|484.85|1.48|0.37|0.10|0.06|
|f60_R30_s1003|N|directional|constant|497.04|497.26|497.48|-0.22|-0.43|-0.10|0.00|
|f60_R30_s1003|N|directional|cosine|498.36|497.89|497.51|0.47|0.85|0.10|0.00|
|f60_R30_s1003|N|shared|constant|497.73|497.36|497.99|0.37|-0.26|0.39|-0.04|
|f60_R30_s1003|N|shared|cosine|498.46|497.91|498.48|0.55|-0.02|0.30|-0.08|
|f60_R30_s1003|T|shared|constant|496.79|479.19|482.84|17.59|13.94|0.30|2.84|
|f60_R30_s1003|T|shared|cosine|497.53|494.95|497.59|2.58|-0.06|1.38|0.53|

## C：T 日程压力（shared；constant = 复用参照）
|parent|objective|H_rsum|T_constant_rsum|T_cosine_rsum|T_cos_minus_const|N_cosine_rsum|T_minus_N_cosine|T_constant_curve|T_cosine_curve|
|---|---|---|---|---|---|---|---|---|---|
|f60_R30_s1001|J|505.21|507.97|508.01|0.04|509.15|-1.14|505.2 / 505.4 / 505.2 / 506.4 / 507.9 / 507.9 / 508.0|505.2 / 505.4 / 505.2 / 506.4 / 508.1 / 508.2 / 508.0|
|f60_R30_s1001|BCE|505.21|497.51|506.59|9.07|508.40|-1.81|505.2 / 505.6 / 505.2 / 505.6 / 504.7 / 506.7 / 497.5|505.2 / 505.6 / 505.2 / 505.5 / 504.9 / 506.1 / 506.6|
|f60_R30_s1001|NCE|505.21|502.82|503.53|0.71|506.69|-3.16|505.2 / 502.4 / 502.6 / 502.7 / 503.3 / 502.7 / 502.8|505.2 / 502.4 / 502.6 / 502.7 / 503.3 / 502.7 / 503.5|
|f60_R30_s1002|J|482.94|483.67|485.23|1.56|485.94|-0.71|482.9 / 483.6 / 482.4 / 483.8 / 484.6 / 484.7 / 483.7|482.9 / 483.6 / 482.4 / 483.7 / 484.6 / 485.8 / 485.2|
|f60_R30_s1002|BCE|482.94|429.59|483.75|54.16|486.51|-2.76|482.9 / 483.8 / 482.6 / 483.6 / 482.9 / 482.7 / 429.6|482.9 / 483.8 / 482.6 / 483.5 / 482.9 / 482.9 / 483.7|
|f60_R30_s1002|NCE|482.94|437.00|484.85|47.85|485.74|-0.89|482.9 / 481.3 / 483.5 / 483.3 / 484.0 / 484.3 / 437.0|482.9 / 481.3 / 483.5 / 483.3 / 484.1 / 483.8 / 484.9|
|f60_R30_s1003|J|496.07|496.79|497.53|0.75|498.46|-0.93|496.1 / 495.6 / 496.1 / 496.4 / 496.5 / 497.1 / 496.8|496.1 / 495.6 / 496.2 / 496.3 / 496.7 / 496.3 / 497.5|
|f60_R30_s1003|BCE|496.07|479.19|494.95|15.76|497.91|-2.96|496.1 / 495.7 / 495.8 / 496.0 / 494.0 / 494.8 / 479.2|496.1 / 495.7 / 495.8 / 495.9 / 494.4 / 494.8 / 495.0|
|f60_R30_s1003|NCE|496.07|482.84|497.59|14.75|498.48|-0.89|496.1 / 495.2 / 495.8 / 496.1 / 495.5 / 497.6 / 482.8|496.1 / 495.2 / 495.7 / 496.0 / 495.5 / 496.4 / 497.6|
|cc_R31_s1001|J|389.70|5.20|381.70|376.50|392.70|-11.00|389.7 / 388.5 / 389.3 / 389.7 / 393.0 / 386.5 / 5.2|389.7 / 388.5 / 389.3 / 389.7 / 392.7 / 389.3 / 381.7|
|cc_R31_s1001|BCE|389.70|3.10|240.20|237.10|392.80|-152.60|389.7 / 389.3 / 389.7 / 389.8 / 386.9 / 324.4 / 3.1|389.7 / 389.3 / 389.8 / 389.6 / 388.3 / 372.4 / 240.2|
|cc_R31_s1001|NCE|389.70|2.50|2.40|-0.10|388.60|-386.20|389.7 / 391.2 / 385.2 / 371.0 / 90.2 / 2.4 / 2.5|389.7 / 391.3 / 385.3 / 371.4 / 123.2 / 2.9 / 2.4|
|cc_R31_s1002|J|394.60|4.30|386.00|381.70|394.10|-8.10|394.6 / 393.2 / 394.0 / 394.4 / 395.0 / 389.4 / 4.3|394.6 / 393.2 / 394.1 / 394.3 / 394.7 / 391.9 / 386.0|
|cc_R31_s1002|BCE|394.60|3.00|219.30|216.30|392.90|-173.60|394.6 / 393.2 / 392.6 / 393.2 / 391.7 / 325.2 / 3.0|394.6 / 393.2 / 392.6 / 393.2 / 391.8 / 375.7 / 219.3|
|cc_R31_s1002|NCE|394.60|2.20|3.00|0.80|390.30|-387.30|394.6 / 394.7 / 385.7 / 376.1 / 185.2 / 3.1 / 2.2|394.6 / 394.8 / 385.9 / 376.7 / 214.1 / 3.3 / 3.0|
|cc_R31_s1003|J|392.60|3.80|384.00|380.20|393.50|-9.50|392.6 / 391.8 / 393.1 / 393.3 / 393.6 / 388.9 / 3.8|392.6 / 391.8 / 393.1 / 393.3 / 393.9 / 390.9 / 384.0|
|cc_R31_s1003|BCE|392.60|2.90|262.60|259.70|393.00|-130.40|392.6 / 391.9 / 392.5 / 392.0 / 389.0 / 343.2 / 2.9|392.6 / 391.9 / 392.5 / 392.1 / 388.6 / 378.6 / 262.6|
|cc_R31_s1003|NCE|392.60|2.10|2.00|-0.10|391.30|-389.30|392.6 / 392.5 / 383.9 / 373.2 / 192.0 / 2.4 / 2.1|392.6 / 392.4 / 383.8 / 374.0 / 222.3 / 4.2 / 2.0|

## A：CC152K 三个 R31 父状态（N shared constant；s1001 = R37C 复用）
|objective|cc_R31_s1001_H_rsum|cc_R31_s1001_rsum|cc_R31_s1001_d_rsum|cc_R31_s1002_H_rsum|cc_R31_s1002_rsum|cc_R31_s1002_d_rsum|cc_R31_s1003_H_rsum|cc_R31_s1003_rsum|cc_R31_s1003_d_rsum|mean_d_rsum_3parents|n_parents_positive|
|---|---|---|---|---|---|---|---|---|---|---|---|
|J|389.70|392.30|2.60|394.60|394.20|-0.40|392.60|394.60|2.00|1.40|2|
|BCE|389.70|393.10|3.40|394.60|392.40|-2.20|392.60|393.50|0.90|0.70|2|
|NCE|389.70|387.50|-2.20|394.60|391.10|-3.50|392.60|392.50|-0.10|-1.93|0|

|parent|J_minus_BCE|J_minus_NCE|
|---|---|---|
|cc_R31_s1001|-0.80|4.80|
|cc_R31_s1002|1.80|3.10|
|cc_R31_s1003|1.10|2.10|

## 全部单元固定终点（8000 步）
|unit|source|objective|i2t_r1|i2t_r5|i2t_r10|t2i_r1|t2i_r5|t2i_r10|rsum|d_rsum|change_2000_to_8000|curve_max_rsum_secondary|step0_passed|graph_check_passed|reference_init_equal|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|r38_B_cc_R31_s1001_N_directional_BCE_constant|R38 new|BCE|46.00|72.00|79.20|45.30|71.20|79.30|393.00|3.30|2.10|393.00|True|True|True|
|r38_B_cc_R31_s1001_N_directional_J_constant|R38 new|J|45.80|71.70|79.30|45.40|71.40|78.90|392.50|2.80|1.90|392.50|True|True|True|
|r38_B_cc_R31_s1001_N_directional_NCE_constant|R38 new|NCE|44.50|70.10|78.30|44.80|70.50|79.30|387.50|-2.20|-0.60|389.70|True|True|True|
|r38_B_cc_R31_s1001_N_directional_BCE_cosine|R38 new|BCE|45.40|71.30|79.90|44.90|71.60|79.00|392.10|2.40|1.20|392.10|True|True|True|
|r38_B_cc_R31_s1001_N_directional_J_cosine|R38 new|J|45.30|71.80|79.60|45.40|71.30|79.20|392.60|2.90|2.30|392.60|True|True|True|
|r38_B_cc_R31_s1001_N_directional_NCE_cosine|R38 new|NCE|44.80|70.10|79.00|44.60|70.80|78.90|388.20|-1.50|0.50|389.70|True|True|True|
|r37c_pilot_cc_h1001_N_G_BCE|reused read-only (R36/R37C)|BCE|45.90|71.90|79.20|45.60|71.20|79.30|393.10|3.40|1.80|393.10|True|NA|NA|
|r37c_pilot_cc_h1001_N_G_J|reused read-only (R36/R37C)|J|45.70|71.70|79.20|45.40|71.40|78.90|392.30|2.60|1.10|392.30|True|NA|NA|
|r37c_pilot_cc_h1001_N_G_NCE|reused read-only (R36/R37C)|NCE|44.70|70.10|78.00|45.00|70.40|79.30|387.50|-2.20|-0.70|389.70|True|NA|NA|
|r38_B_cc_R31_s1001_N_shared_BCE_cosine|R38 new|BCE|45.60|71.60|80.00|45.00|71.50|79.10|392.80|3.10|1.40|392.80|True|True|True|
|r38_B_cc_R31_s1001_N_shared_J_cosine|R38 new|J|45.30|71.90|79.60|45.30|71.40|79.20|392.70|3.00|2.30|392.70|True|True|True|
|r38_B_cc_R31_s1001_N_shared_NCE_cosine|R38 new|NCE|44.90|70.10|78.90|44.70|71.10|78.90|388.60|-1.10|0.60|389.70|True|True|True|
|r37c_pilot_cc_h1001_T_G_BCE|reused read-only (R36/R37C)|BCE|0.10|0.40|0.90|0.00|0.70|1.00|3.10|-386.60|-383.80|389.80|True|NA|NA|
|r37c_pilot_cc_h1001_T_G_J|reused read-only (R36/R37C)|J|0.10|0.50|1.80|0.00|0.90|1.90|5.20|-384.50|-387.80|393.00|True|NA|NA|
|r37c_pilot_cc_h1001_T_G_NCE|reused read-only (R36/R37C)|NCE|0.10|0.30|0.90|0.20|0.30|0.70|2.50|-387.20|-87.70|391.20|True|NA|NA|
|r38_C_cc_R31_s1001_T_shared_BCE_cosine|R38 new|BCE|16.70|37.80|51.60|25.50|48.10|60.50|240.20|-149.50|-148.10|389.80|True|True|True|
|r38_C_cc_R31_s1001_T_shared_J_cosine|R38 new|J|42.80|69.40|78.30|43.40|69.80|78.00|381.70|-8.00|-11.00|392.70|True|True|True|
|r38_C_cc_R31_s1001_T_shared_NCE_cosine|R38 new|NCE|0.10|0.40|1.00|0.10|0.30|0.50|2.40|-387.30|-120.80|391.30|True|True|True|
|r38_B_cc_R31_s1002_N_directional_BCE_constant|R38 new|BCE|44.30|71.80|80.40|45.40|71.20|79.60|392.70|-1.90|-1.60|394.60|True|True|NA|
|r38_B_cc_R31_s1002_N_directional_J_constant|R38 new|J|45.00|71.90|80.40|45.10|71.60|79.70|393.70|-0.90|-0.90|394.60|True|True|NA|
|r38_B_cc_R31_s1002_N_directional_NCE_constant|R38 new|NCE|44.00|71.50|80.20|44.00|71.00|79.70|390.40|-4.20|-1.20|394.60|True|True|NA|
|r38_B_cc_R31_s1002_N_directional_BCE_cosine|R38 new|BCE|44.70|71.80|80.00|45.00|71.70|80.30|393.50|-1.10|-1.00|394.60|True|True|NA|
|r38_B_cc_R31_s1002_N_directional_J_cosine|R38 new|J|44.90|72.00|80.20|45.30|71.70|80.40|394.50|-0.10|-0.10|395.10|True|True|NA|
|r38_B_cc_R31_s1002_N_directional_NCE_cosine|R38 new|NCE|45.00|70.90|79.40|44.30|70.70|80.10|390.40|-4.20|-1.40|394.60|True|True|NA|
|r38_A_cc_R31_s1002_N_shared_BCE_constant|R38 new|BCE|44.40|71.70|80.40|45.30|71.20|79.40|392.40|-2.20|-2.00|394.60|True|True|NA|
|r38_A_cc_R31_s1002_N_shared_J_constant|R38 new|J|45.10|72.00|80.40|45.30|71.60|79.80|394.20|-0.40|-0.20|394.70|True|True|NA|
|r38_A_cc_R31_s1002_N_shared_NCE_constant|R38 new|NCE|44.30|71.20|80.10|44.20|71.30|80.00|391.10|-3.50|-0.50|394.60|True|True|NA|
|r38_B_cc_R31_s1002_N_shared_BCE_cosine|R38 new|BCE|44.70|71.60|80.00|44.90|71.60|80.10|392.90|-1.70|-1.30|394.60|True|True|NA|
|r38_B_cc_R31_s1002_N_shared_J_cosine|R38 new|J|44.70|72.10|80.20|45.20|71.50|80.40|394.10|-0.50|-0.60|394.90|True|True|NA|
|r38_B_cc_R31_s1002_N_shared_NCE_cosine|R38 new|NCE|44.90|70.90|79.30|44.30|70.80|80.10|390.30|-4.30|-1.40|394.60|True|True|NA|
|r38_C_cc_R31_s1002_T_shared_BCE_constant|R38 new|BCE|0.10|0.40|1.40|0.00|0.40|0.70|3.00|-391.60|-388.70|394.60|True|True|NA|
|r38_C_cc_R31_s1002_T_shared_J_constant|R38 new|J|0.10|0.70|1.50|0.10|0.50|1.40|4.30|-390.30|-390.70|395.00|True|True|NA|
|r38_C_cc_R31_s1002_T_shared_NCE_constant|R38 new|NCE|0.00|0.40|0.80|0.10|0.30|0.60|2.20|-392.40|-183.00|394.70|True|True|NA|
|r38_C_cc_R31_s1002_T_shared_BCE_cosine|R38 new|BCE|13.80|33.30|46.10|23.60|45.70|56.80|219.30|-175.30|-172.50|394.60|True|True|NA|
|r38_C_cc_R31_s1002_T_shared_J_cosine|R38 new|J|42.20|70.20|79.50|44.20|70.50|79.40|386.00|-8.60|-8.70|394.70|True|True|NA|
|r38_C_cc_R31_s1002_T_shared_NCE_cosine|R38 new|NCE|0.10|0.60|1.10|0.10|0.40|0.70|3.00|-391.60|-211.10|394.80|True|True|NA|
|r38_B_cc_R31_s1003_N_directional_BCE_constant|R38 new|BCE|44.80|71.70|79.80|44.80|71.70|80.50|393.30|0.70|0.60|393.50|True|True|NA|
|r38_B_cc_R31_s1003_N_directional_J_constant|R38 new|J|44.80|72.30|80.10|44.90|71.70|80.60|394.40|1.80|1.10|394.40|True|True|NA|
|r38_B_cc_R31_s1003_N_directional_NCE_constant|R38 new|NCE|45.10|71.80|79.40|45.20|71.30|79.60|392.40|-0.20|0.30|392.60|True|True|NA|
|r38_B_cc_R31_s1003_N_directional_BCE_cosine|R38 new|BCE|45.60|71.80|80.10|44.80|71.10|80.20|393.60|1.00|1.00|393.70|True|True|NA|
|r38_B_cc_R31_s1003_N_directional_J_cosine|R38 new|J|46.00|71.50|80.00|44.80|71.00|80.50|393.80|1.20|1.10|393.80|True|True|NA|
|r38_B_cc_R31_s1003_N_directional_NCE_cosine|R38 new|NCE|44.90|71.60|79.80|45.10|70.20|79.70|391.30|-1.30|-0.20|392.60|True|True|NA|
|r38_A_cc_R31_s1003_N_shared_BCE_constant|R38 new|BCE|44.80|71.90|79.80|44.80|71.60|80.60|393.50|0.90|0.30|393.50|True|True|NA|
|r38_A_cc_R31_s1003_N_shared_J_constant|R38 new|J|44.70|72.40|80.20|44.90|71.70|80.70|394.60|2.00|1.60|394.60|True|True|NA|
|r38_A_cc_R31_s1003_N_shared_NCE_constant|R38 new|NCE|45.10|72.00|79.50|45.10|71.10|79.70|392.50|-0.10|0.20|392.60|True|True|NA|
|r38_B_cc_R31_s1003_N_shared_BCE_cosine|R38 new|BCE|45.30|71.70|80.20|44.70|70.90|80.20|393.00|0.40|0.30|393.70|True|True|NA|
|r38_B_cc_R31_s1003_N_shared_J_cosine|R38 new|J|45.90|71.40|80.20|44.90|70.90|80.20|393.50|0.90|0.90|393.50|True|True|NA|
|r38_B_cc_R31_s1003_N_shared_NCE_cosine|R38 new|NCE|44.80|71.70|79.60|45.20|70.30|79.70|391.30|-1.30|-0.20|392.60|True|True|NA|
|r38_C_cc_R31_s1003_T_shared_BCE_constant|R38 new|BCE|0.00|0.40|1.10|0.10|0.40|0.90|2.90|-389.70|-386.10|392.60|True|True|NA|
|r38_C_cc_R31_s1003_T_shared_J_constant|R38 new|J|0.00|0.30|1.20|0.10|0.80|1.40|3.80|-388.80|-389.80|393.60|True|True|NA|
|r38_C_cc_R31_s1003_T_shared_NCE_constant|R38 new|NCE|0.10|0.50|0.90|0.00|0.20|0.40|2.10|-390.50|-189.90|392.60|True|True|NA|
|r38_C_cc_R31_s1003_T_shared_BCE_cosine|R38 new|BCE|18.00|43.00|56.50|28.70|52.40|64.00|262.60|-130.00|-126.00|392.60|True|True|NA|
|r38_C_cc_R31_s1003_T_shared_J_cosine|R38 new|J|41.10|71.30|78.40|42.80|70.90|79.50|384.00|-8.60|-9.90|393.90|True|True|NA|
|r38_C_cc_R31_s1003_T_shared_NCE_cosine|R38 new|NCE|0.10|0.50|0.70|0.00|0.10|0.60|2.00|-390.60|-220.30|392.60|True|True|NA|
|r38_B_f60_R30_s1001_N_directional_BCE_constant|R38 new|BCE|81.56|95.96|98.03|58.93|83.61|89.64|507.73|2.52|-0.08|507.87|True|True|True|
|r38_B_f60_R30_s1001_N_directional_J_constant|R38 new|J|80.97|95.86|98.22|59.25|83.53|89.68|507.51|2.31|-1.30|508.82|True|True|True|
|r38_B_f60_R30_s1001_N_directional_NCE_constant|R38 new|NCE|80.57|95.46|98.13|58.86|83.25|89.55|505.82|0.61|-1.40|507.22|True|True|True|
|r38_B_f60_R30_s1001_N_directional_BCE_cosine|R38 new|BCE|81.66|95.86|98.22|59.49|83.67|89.86|508.76|3.55|0.89|508.76|True|True|True|
|r38_B_f60_R30_s1001_N_directional_J_cosine|R38 new|J|81.66|95.96|98.13|59.59|83.69|89.94|508.95|3.75|0.16|508.95|True|True|True|
|r38_B_f60_R30_s1001_N_directional_NCE_cosine|R38 new|NCE|80.77|95.17|98.22|58.97|83.47|89.55|506.15|0.95|-1.28|507.44|True|True|True|
|r36_N_n60_G_BCE|reused read-only (R36/R37C)|BCE|81.07|96.06|98.03|59.05|83.81|89.57|507.57|2.37|-0.53|508.11|NA|NA|NA|
|r36_N_n60_G_J|reused read-only (R36/R37C)|J|80.97|96.15|98.22|59.17|83.57|89.64|507.73|2.52|-0.89|508.62|NA|NA|NA|
|r36_N_n60_G_NCE|reused read-only (R36/R37C)|NCE|80.67|95.46|98.32|59.05|83.35|89.63|506.49|1.28|-1.72|508.21|NA|NA|NA|
|r38_B_f60_R30_s1001_N_shared_BCE_cosine|R38 new|BCE|81.36|95.76|98.22|59.41|83.87|89.78|508.40|3.20|0.20|508.40|True|True|True|
|r38_B_f60_R30_s1001_N_shared_J_cosine|R38 new|J|81.76|96.06|98.13|59.55|83.75|89.92|509.15|3.94|0.34|509.15|True|True|True|
|r38_B_f60_R30_s1001_N_shared_NCE_cosine|R38 new|NCE|80.87|95.07|98.32|59.23|83.57|89.63|506.69|1.48|-1.48|508.17|True|True|True|
|r36_T_n60_G_BCE|reused read-only (R36/R37C)|BCE|80.67|95.07|97.63|58.11|80.26|85.78|497.51|-7.69|-7.22|506.67|NA|NA|NA|
|r36_T_n60_G_J|reused read-only (R36/R37C)|J|80.87|96.15|98.13|59.43|83.69|89.70|507.97|2.76|0.10|507.97|NA|NA|NA|
|r36_T_n60_G_NCE|reused read-only (R36/R37C)|NCE|79.88|95.07|98.13|58.19|83.02|88.54|502.82|-2.39|-0.43|505.21|NA|NA|NA|
|r38_C_f60_R30_s1001_T_shared_BCE_cosine|R38 new|BCE|80.57|95.56|97.93|59.09|83.77|89.66|506.59|1.38|1.70|506.59|True|True|True|
|r38_C_f60_R30_s1001_T_shared_J_cosine|R38 new|J|80.87|96.06|98.13|59.51|83.81|89.64|508.01|2.80|-0.06|508.19|True|True|True|
|r38_C_f60_R30_s1001_T_shared_NCE_cosine|R38 new|NCE|79.88|95.17|98.13|58.44|83.00|88.92|503.53|-1.68|0.24|505.21|True|True|True|
|r38_B_f60_R30_s1002_N_directional_BCE_constant|R38 new|BCE|74.26|92.31|96.06|54.22|81.03|87.40|485.27|2.33|-0.73|486.73|True|True|True|
|r38_B_f60_R30_s1002_N_directional_J_constant|R38 new|J|73.67|92.31|95.96|54.12|80.89|87.32|484.26|1.32|-1.66|485.92|True|True|True|
|r38_B_f60_R30_s1002_N_directional_NCE_constant|R38 new|NCE|74.85|91.81|95.86|53.81|80.57|87.50|484.40|1.46|-2.01|486.41|True|True|True|
|r38_B_f60_R30_s1002_N_directional_BCE_cosine|R38 new|BCE|75.25|92.60|96.06|54.44|81.22|87.51|487.08|4.14|1.30|487.08|True|True|True|
|r38_B_f60_R30_s1002_N_directional_J_cosine|R38 new|J|74.46|92.41|96.15|54.46|81.32|87.53|486.33|3.39|0.34|486.47|True|True|True|
|r38_B_f60_R30_s1002_N_directional_NCE_cosine|R38 new|NCE|75.05|92.11|95.76|54.12|80.65|87.59|485.29|2.35|-0.97|486.25|True|True|True|
|r37_confirm_f60_h1002_N_G_BCE|reused read-only (R36/R37C)|BCE|74.06|92.41|96.06|54.32|80.93|87.34|485.11|2.17|-0.97|487.00|True|NA|NA|
|r37_confirm_f60_h1002_N_G_J|reused read-only (R36/R37C)|J|73.67|92.31|96.06|54.16|80.87|87.36|484.42|1.48|-1.64|486.09|True|NA|NA|
|r37_confirm_f60_h1002_N_G_NCE|reused read-only (R36/R37C)|NCE|74.95|92.31|95.86|53.75|80.67|87.46|484.99|2.05|-1.10|486.09|True|NA|NA|
|r38_B_f60_R30_s1002_N_shared_BCE_cosine|R38 new|BCE|74.75|92.31|95.96|54.58|81.28|87.63|486.51|3.57|0.57|486.51|True|True|True|
|r38_B_f60_R30_s1002_N_shared_J_cosine|R38 new|J|74.26|92.11|96.06|54.65|81.24|87.61|485.94|3.00|-0.10|486.37|True|True|True|
|r38_B_f60_R30_s1002_N_shared_NCE_cosine|R38 new|NCE|75.05|92.11|95.86|54.42|80.71|87.59|485.74|2.80|-0.83|486.86|True|True|True|
|r37_confirm_f60_h1002_T_G_BCE|reused read-only (R36/R37C)|BCE|69.13|84.12|89.15|46.51|66.53|74.14|429.59|-53.35|-53.35|483.85|True|NA|NA|
|r37_confirm_f60_h1002_T_G_J|reused read-only (R36/R37C)|J|73.08|92.11|95.96|54.20|80.93|87.40|483.67|0.73|-0.91|484.69|True|NA|NA|
|r37_confirm_f60_h1002_T_G_NCE|reused read-only (R36/R37C)|NCE|72.68|88.17|91.52|48.52|65.50|70.61|437.00|-45.94|-47.04|484.28|True|NA|NA|
|r38_C_f60_R30_s1002_T_shared_BCE_cosine|R38 new|BCE|73.67|91.72|96.06|54.44|80.59|87.28|483.75|0.81|0.89|483.85|True|True|True|
|r38_C_f60_R30_s1002_T_shared_J_cosine|R38 new|J|73.77|92.01|96.25|54.50|81.07|87.63|485.23|2.29|0.59|485.76|True|True|True|
|r38_C_f60_R30_s1002_T_shared_NCE_cosine|R38 new|NCE|74.75|91.72|95.66|54.38|80.71|87.63|484.85|1.91|0.71|484.85|True|True|True|
|r38_B_f60_R30_s1003_N_directional_BCE_constant|R38 new|BCE|78.21|93.39|96.55|57.44|82.70|88.97|497.26|1.18|-0.49|498.11|True|True|True|
|r38_B_f60_R30_s1003_N_directional_J_constant|R38 new|J|78.11|93.29|96.45|57.44|82.88|88.88|497.04|0.97|-0.95|498.07|True|True|True|
|r38_B_f60_R30_s1003_N_directional_NCE_constant|R38 new|NCE|78.99|93.89|96.35|56.84|82.56|88.84|497.48|1.40|-1.60|499.07|True|True|True|
|r38_B_f60_R30_s1003_N_directional_BCE_cosine|R38 new|BCE|78.21|93.29|96.45|57.89|82.78|89.27|497.89|1.81|0.51|497.89|True|True|True|
|r38_B_f60_R30_s1003_N_directional_J_cosine|R38 new|J|78.30|93.49|96.55|57.89|82.94|89.19|498.36|2.29|0.73|498.56|True|True|True|
|r38_B_f60_R30_s1003_N_directional_NCE_cosine|R38 new|NCE|78.60|93.69|96.35|57.14|82.80|88.93|497.51|1.44|-1.52|499.03|True|True|True|
|r37_confirm_f60_h1003_N_G_BCE|reused read-only (R36/R37C)|BCE|78.11|93.29|96.55|57.61|82.82|88.97|497.36|1.28|-0.63|498.32|True|NA|NA|
|r37_confirm_f60_h1003_N_G_J|reused read-only (R36/R37C)|J|78.50|93.39|96.55|57.57|82.86|88.86|497.73|1.66|-0.10|498.11|True|NA|NA|
|r37_confirm_f60_h1003_N_G_NCE|reused read-only (R36/R37C)|NCE|79.09|93.79|96.45|56.90|82.74|89.01|497.99|1.91|-0.36|498.34|True|NA|NA|
|r38_B_f60_R30_s1003_N_shared_BCE_cosine|R38 new|BCE|78.01|93.20|96.55|58.15|82.90|89.11|497.91|1.83|-0.49|498.40|True|True|True|
|r38_B_f60_R30_s1003_N_shared_J_cosine|R38 new|J|78.30|93.39|96.65|58.07|82.92|89.13|498.46|2.39|0.49|498.54|True|True|True|
|r38_B_f60_R30_s1003_N_shared_NCE_cosine|R38 new|NCE|78.80|93.79|96.75|57.55|82.68|88.92|498.48|2.41|-0.08|498.56|True|True|True|
|r37_confirm_f60_h1003_T_G_BCE|reused read-only (R36/R37C)|BCE|76.73|91.72|95.56|54.99|76.79|83.41|479.19|-16.88|-14.79|496.07|True|NA|NA|
|r37_confirm_f60_h1003_T_G_J|reused read-only (R36/R37C)|J|77.02|93.20|96.75|57.83|82.76|89.23|496.79|0.71|0.30|497.14|True|NA|NA|
|r37_confirm_f60_h1003_T_G_NCE|reused read-only (R36/R37C)|NCE|75.74|92.70|95.86|56.21|78.82|83.51|482.84|-13.23|-12.64|497.55|True|NA|NA|
|r38_C_f60_R30_s1003_T_shared_BCE_cosine|R38 new|BCE|76.43|93.20|96.45|57.38|82.50|88.99|494.95|-1.12|0.53|496.07|True|True|True|
|r38_C_f60_R30_s1003_T_shared_J_cosine|R38 new|J|77.81|93.29|96.55|57.91|82.74|89.23|497.53|1.46|0.81|497.53|True|True|True|
|r38_C_f60_R30_s1003_T_shared_NCE_cosine|R38 new|NCE|78.60|94.08|96.35|57.40|82.54|88.62|497.59|1.52|2.05|497.59|True|True|True|

## R@1 转移（a 对 b：仅 a 对 / 仅 b 对 / 净；top1 改变数）
|parent|objective|contrast|i2t_a_only_r1|i2t_b_only_r1|i2t_net_r1|i2t_top1_changed|t2i_a_only_r1|t2i_b_only_r1|t2i_net_r1|t2i_top1_changed|
|---|---|---|---|---|---|---|---|---|---|---|
|f60_R30_s1001|J|dir-shared|constant|1|1|0|12|17|13|4|64|
|f60_R30_s1001|J|dir-shared|cosine|0|1|-1|7|10|8|2|42|
|f60_R30_s1001|J|cos-const|shared|13|5|8|56|65|46|19|216|
|f60_R30_s1001|J|cos-const|dir|12|5|7|55|66|49|17|223|
|f60_R30_s1001|BCE|dir-shared|constant|5|0|5|16|10|16|-6|65|
|f60_R30_s1001|BCE|dir-shared|cosine|5|2|3|15|13|9|4|53|
|f60_R30_s1001|BCE|cos-const|shared|11|8|3|55|70|52|18|229|
|f60_R30_s1001|BCE|cos-const|dir|11|10|1|59|73|45|28|241|
|f60_R30_s1001|NCE|dir-shared|constant|3|4|-1|19|14|24|-10|78|
|f60_R30_s1001|NCE|dir-shared|cosine|1|2|-1|8|13|26|-13|90|
|f60_R30_s1001|NCE|cos-const|shared|12|10|2|59|54|45|9|209|
|f60_R30_s1001|NCE|cos-const|dir|11|9|2|59|48|42|6|200|
|f60_R30_s1002|J|dir-shared|constant|2|2|0|16|10|12|-2|59|
|f60_R30_s1002|J|dir-shared|cosine|3|1|2|11|9|19|-10|64|
|f60_R30_s1002|J|cos-const|shared|15|9|6|59|69|44|25|246|
|f60_R30_s1002|J|cos-const|dir|15|7|8|54|67|50|17|258|
|f60_R30_s1002|BCE|dir-shared|constant|8|6|2|33|10|15|-5|80|
|f60_R30_s1002|BCE|dir-shared|cosine|9|4|5|31|12|19|-7|69|
|f60_R30_s1002|BCE|cos-const|shared|17|10|7|66|55|42|13|227|
|f60_R30_s1002|BCE|cos-const|dir|16|6|10|61|59|48|11|252|
|f60_R30_s1002|NCE|dir-shared|constant|3|4|-1|17|20|17|3|92|
|f60_R30_s1002|NCE|dir-shared|cosine|2|2|0|18|18|33|-15|112|
|f60_R30_s1002|NCE|cos-const|shared|12|11|1|67|75|41|34|260|
|f60_R30_s1002|NCE|cos-const|dir|11|9|2|63|68|52|16|273|
|f60_R30_s1003|J|dir-shared|constant|1|5|-4|15|10|17|-7|59|
|f60_R30_s1003|J|dir-shared|cosine|1|1|0|6|5|14|-9|36|
|f60_R30_s1003|J|cos-const|shared|8|10|-2|62|55|30|25|199|
|f60_R30_s1003|J|cos-const|dir|8|6|2|54|57|34|23|223|
|f60_R30_s1003|BCE|dir-shared|constant|5|4|1|26|13|22|-9|77|
|f60_R30_s1003|BCE|dir-shared|cosine|4|2|2|23|6|19|-13|46|
|f60_R30_s1003|BCE|cos-const|shared|9|10|-1|69|59|32|27|200|
|f60_R30_s1003|BCE|cos-const|dir|8|8|0|60|70|47|23|236|
|f60_R30_s1003|NCE|dir-shared|constant|3|4|-1|19|16|19|-3|87|
|f60_R30_s1003|NCE|dir-shared|cosine|2|4|-2|20|13|34|-21|100|
|f60_R30_s1003|NCE|cos-const|shared|8|11|-3|60|70|37|33|231|
|f60_R30_s1003|NCE|cos-const|dir|7|11|-4|56|64|49|15|233|
|cc_R31_s1001|J|dir-shared|constant|1|0|1|3|1|1|0|2|
|cc_R31_s1001|J|dir-shared|cosine|0|0|0|5|1|0|1|6|
|cc_R31_s1001|J|cos-const|shared|11|15|-4|55|10|11|-1|67|
|cc_R31_s1001|J|cos-const|dir|10|15|-5|54|11|11|0|68|
|cc_R31_s1001|BCE|dir-shared|constant|1|0|1|4|0|3|-3|4|
|cc_R31_s1001|BCE|dir-shared|cosine|0|2|-2|5|2|3|-1|11|
|cc_R31_s1001|BCE|cos-const|shared|11|14|-3|56|6|12|-6|52|
|cc_R31_s1001|BCE|cos-const|dir|10|16|-6|59|8|12|-4|60|
|cc_R31_s1001|NCE|dir-shared|constant|0|2|-2|15|2|4|-2|17|
|cc_R31_s1001|NCE|dir-shared|cosine|0|1|-1|5|1|2|-1|7|
|cc_R31_s1001|NCE|cos-const|shared|8|6|2|47|13|16|-3|72|
|cc_R31_s1001|NCE|cos-const|dir|9|6|3|46|13|15|-2|76|
|cc_R31_s1002|J|dir-shared|constant|0|1|-1|5|1|3|-2|4|
|cc_R31_s1002|J|dir-shared|cosine|4|2|2|11|1|0|1|5|
|cc_R31_s1002|J|cos-const|shared|7|11|-4|52|12|13|-1|60|
|cc_R31_s1002|J|cos-const|dir|10|11|-1|57|12|10|2|61|
|cc_R31_s1002|BCE|dir-shared|constant|0|1|-1|6|1|0|1|5|
|cc_R31_s1002|BCE|dir-shared|cosine|2|2|0|9|1|0|1|6|
|cc_R31_s1002|BCE|cos-const|shared|14|11|3|62|8|12|-4|59|
|cc_R31_s1002|BCE|cos-const|dir|15|11|4|60|8|12|-4|58|
|cc_R31_s1002|NCE|dir-shared|constant|0|3|-3|9|2|4|-2|18|
|cc_R31_s1002|NCE|dir-shared|cosine|1|0|1|7|1|1|0|4|
|cc_R31_s1002|NCE|cos-const|shared|13|7|6|56|13|12|1|61|
|cc_R31_s1002|NCE|cos-const|dir|15|5|10|57|15|12|3|75|
|cc_R31_s1003|J|dir-shared|constant|2|1|1|7|0|0|0|5|
|cc_R31_s1003|J|dir-shared|cosine|1|0|1|5|1|2|-1|4|
|cc_R31_s1003|J|cos-const|shared|18|6|12|64|7|7|0|50|
|cc_R31_s1003|J|cos-const|dir|17|5|12|62|8|9|-1|49|
|cc_R31_s1003|BCE|dir-shared|constant|1|1|0|5|0|0|0|3|
|cc_R31_s1003|BCE|dir-shared|cosine|3|0|3|11|2|1|1|5|
|cc_R31_s1003|BCE|cos-const|shared|13|8|5|53|9|10|-1|48|
|cc_R31_s1003|BCE|cos-const|dir|14|6|8|57|11|11|0|52|
|cc_R31_s1003|NCE|dir-shared|constant|2|2|0|16|3|2|1|19|
|cc_R31_s1003|NCE|dir-shared|cosine|2|1|1|8|1|2|-1|7|
|cc_R31_s1003|NCE|cos-const|shared|4|7|-3|41|11|10|1|47|
|cc_R31_s1003|NCE|cos-const|dir|5|7|-2|42|12|13|-1|60|
|cc_R31_s1001|J-BCE|J-BCE|N,directional,constant|3|5|-2|29|6|5|1|25|
|cc_R31_s1001|J-NCE|J-NCE|N,directional,constant|26|13|13|96|14|8|6|71|
|cc_R31_s1001|J-BCE|J-BCE|N,directional,cosine|4|5|-1|28|7|2|5|27|
|cc_R31_s1001|J-NCE|J-NCE|N,directional,cosine|17|12|5|95|17|9|8|94|
|cc_R31_s1001|J-BCE|J-BCE|N,shared,constant|4|6|-2|30|5|7|-2|27|
|cc_R31_s1001|J-NCE|J-NCE|N,shared,constant|24|14|10|98|13|9|4|70|
|cc_R31_s1001|J-BCE|J-BCE|N,shared,cosine|3|6|-3|29|7|4|3|31|
|cc_R31_s1001|J-NCE|J-NCE|N,shared,cosine|17|13|4|96|16|10|6|92|
|cc_R31_s1001|J-BCE|J-BCE|T,shared,constant|0|0|0|309|0|0|0|296|
|cc_R31_s1001|J-NCE|J-NCE|T,shared,constant|1|1|0|628|0|2|-2|685|
|cc_R31_s1001|J-BCE|J-BCE|T,shared,cosine|270|9|261|716|199|20|179|570|
|cc_R31_s1001|J-NCE|J-NCE|T,shared,cosine|428|1|427|1000|434|1|433|1000|
|cc_R31_s1002|J-BCE|J-BCE|N,directional,constant|11|4|7|41|2|5|-3|25|
|cc_R31_s1002|J-NCE|J-NCE|N,directional,constant|23|13|10|94|26|15|11|98|
|cc_R31_s1002|J-BCE|J-BCE|N,directional,cosine|5|3|2|25|5|2|3|27|
|cc_R31_s1002|J-NCE|J-NCE|N,directional,cosine|20|21|-1|102|25|15|10|90|
|cc_R31_s1002|J-BCE|J-BCE|N,shared,constant|10|3|7|40|3|3|0|24|
|cc_R31_s1002|J-NCE|J-NCE|N,shared,constant|22|14|8|96|26|15|11|99|
|cc_R31_s1002|J-BCE|J-BCE|N,shared,cosine|6|6|0|31|5|2|3|24|
|cc_R31_s1002|J-NCE|J-NCE|N,shared,cosine|19|21|-2|103|24|15|9|91|
|cc_R31_s1002|J-BCE|J-BCE|T,shared,constant|0|0|0|271|1|0|1|238|
|cc_R31_s1002|J-NCE|J-NCE|T,shared,constant|1|0|1|591|0|0|0|586|
|cc_R31_s1002|J-BCE|J-BCE|T,shared,cosine|296|12|284|780|220|14|206|614|
|cc_R31_s1002|J-NCE|J-NCE|T,shared,cosine|422|1|421|1000|442|1|441|1000|
|cc_R31_s1003|J-BCE|J-BCE|N,directional,constant|6|6|0|31|5|4|1|28|
|cc_R31_s1003|J-NCE|J-NCE|N,directional,constant|14|17|-3|82|16|19|-3|78|
|cc_R31_s1003|J-BCE|J-BCE|N,directional,cosine|6|2|4|21|6|6|0|22|
|cc_R31_s1003|J-NCE|J-NCE|N,directional,cosine|22|11|11|79|14|17|-3|74|
|cc_R31_s1003|J-BCE|J-BCE|N,shared,constant|6|7|-1|29|5|4|1|29|
|cc_R31_s1003|J-NCE|J-NCE|N,shared,constant|15|19|-4|90|18|20|-2|83|
|cc_R31_s1003|J-BCE|J-BCE|N,shared,cosine|8|2|6|22|7|5|2|21|
|cc_R31_s1003|J-NCE|J-NCE|N,shared,cosine|22|11|11|80|14|17|-3|71|
|cc_R31_s1003|J-BCE|J-BCE|T,shared,constant|0|0|0|267|0|0|0|290|
|cc_R31_s1003|J-NCE|J-NCE|T,shared,constant|0|1|-1|584|1|0|1|637|
|cc_R31_s1003|J-BCE|J-BCE|T,shared,cosine|241|10|231|666|161|20|141|513|
|cc_R31_s1003|J-NCE|J-NCE|T,shared,cosine|411|1|410|1000|428|0|428|1000|
|f60_R30_s1001|J-BCE|J-BCE|N,directional,constant|2|8|-6|35|45|29|16|150|
|f60_R30_s1001|J-NCE|J-NCE|N,directional,constant|17|13|4|84|92|72|20|363|
|f60_R30_s1001|J-BCE|J-BCE|N,directional,cosine|5|5|0|28|28|23|5|103|
|f60_R30_s1001|J-NCE|J-NCE|N,directional,cosine|22|13|9|98|99|68|31|354|
|f60_R30_s1001|J-BCE|J-BCE|N,shared,constant|4|5|-1|37|39|33|6|144|
|f60_R30_s1001|J-NCE|J-NCE|N,shared,constant|16|13|3|81|74|68|6|318|
|f60_R30_s1001|J-BCE|J-BCE|N,shared,cosine|7|3|4|25|27|20|7|107|
|f60_R30_s1001|J-NCE|J-NCE|N,shared,cosine|21|12|9|92|74|58|16|282|
|f60_R30_s1001|J-BCE|J-BCE|T,shared,constant|20|18|2|94|172|105|67|724|
|f60_R30_s1001|J-NCE|J-NCE|T,shared,constant|33|23|10|156|148|85|63|500|
|f60_R30_s1001|J-BCE|J-BCE|T,shared,cosine|10|7|3|56|65|44|21|240|
|f60_R30_s1001|J-NCE|J-NCE|T,shared,cosine|32|22|10|152|138|84|54|459|
|f60_R30_s1002|J-BCE|J-BCE|N,directional,constant|6|12|-6|46|30|35|-5|169|
|f60_R30_s1002|J-NCE|J-NCE|N,directional,constant|15|27|-12|108|85|69|16|357|
|f60_R30_s1002|J-BCE|J-BCE|N,directional,cosine|2|10|-8|36|26|25|1|105|
|f60_R30_s1002|J-NCE|J-NCE|N,directional,cosine|14|20|-6|107|86|69|17|340|
|f60_R30_s1002|J-BCE|J-BCE|N,shared,constant|8|12|-4|51|24|32|-8|148|
|f60_R30_s1002|J-NCE|J-NCE|N,shared,constant|15|28|-13|109|72|51|21|311|
|f60_R30_s1002|J-BCE|J-BCE|N,shared,cosine|4|9|-5|38|21|17|4|98|
|f60_R30_s1002|J-NCE|J-NCE|N,shared,cosine|14|22|-8|110|65|53|12|250|
|f60_R30_s1002|J-BCE|J-BCE|T,shared,constant|66|26|40|238|524|134|390|1694|
|f60_R30_s1002|J-NCE|J-NCE|T,shared,constant|37|33|4|190|373|85|288|1247|
|f60_R30_s1002|J-BCE|J-BCE|T,shared,cosine|15|14|1|81|61|58|3|279|
|f60_R30_s1002|J-NCE|J-NCE|T,shared,cosine|24|34|-10|147|90|84|6|404|
|f60_R30_s1003|J-BCE|J-BCE|N,directional,constant|6|7|-1|45|27|27|0|140|
|f60_R30_s1003|J-NCE|J-NCE|N,directional,constant|10|19|-9|97|93|63|30|364|
|f60_R30_s1003|J-BCE|J-BCE|N,directional,cosine|5|4|1|41|15|15|0|85|
|f60_R30_s1003|J-NCE|J-NCE|N,directional,cosine|13|16|-3|99|93|55|38|331|
|f60_R30_s1003|J-BCE|J-BCE|N,shared,constant|10|6|4|52|26|28|-2|130|
|f60_R30_s1003|J-NCE|J-NCE|N,shared,constant|16|22|-6|108|87|53|34|303|
|f60_R30_s1003|J-BCE|J-BCE|N,shared,cosine|6|3|3|34|11|15|-4|74|
|f60_R30_s1003|J-NCE|J-NCE|N,shared,cosine|12|17|-5|94|68|42|26|262|
|f60_R30_s1003|J-BCE|J-BCE|T,shared,constant|31|28|3|164|280|136|144|1006|
|f60_R30_s1003|J-NCE|J-NCE|T,shared,constant|39|26|13|184|192|110|82|740|
|f60_R30_s1003|J-BCE|J-BCE|T,shared,cosine|17|3|14|74|74|47|27|239|
|f60_R30_s1003|J-NCE|J-NCE|T,shared,cosine|21|29|-8|156|104|78|26|393|

## 按图像分组配对 bootstrap（rSum 差，B=2000，描述性）
|parent|objective|contrast|diff|ci2_5|ci97_5|frac_boot_gt0|
|---|---|---|---|---|---|---|
|f60_R30_s1001|J|dir-shared|constant|-0.22|-0.75|0.28|0.20|
|f60_R30_s1001|J|dir-shared|cosine|-0.20|-0.65|0.24|0.19|
|f60_R30_s1001|J|cos-const|shared|1.42|0.24|2.56|0.99|
|f60_R30_s1001|J|cos-const|dir|1.44|0.32|2.56|0.99|
|f60_R30_s1001|BCE|dir-shared|constant|0.16|-0.37|0.73|0.70|
|f60_R30_s1001|BCE|dir-shared|cosine|0.36|-0.22|0.95|0.89|
|f60_R30_s1001|BCE|cos-const|shared|0.83|-0.38|2.01|0.91|
|f60_R30_s1001|BCE|cos-const|dir|1.03|-0.18|2.23|0.95|
|f60_R30_s1001|NCE|dir-shared|constant|-0.67|-1.34|0.02|0.03|
|f60_R30_s1001|NCE|dir-shared|cosine|-0.53|-1.28|0.20|0.07|
|f60_R30_s1001|NCE|cos-const|shared|0.20|-1.14|1.54|0.63|
|f60_R30_s1001|NCE|cos-const|dir|0.34|-0.95|1.52|0.71|
|f60_R30_s1002|J|dir-shared|constant|-0.16|-0.73|0.43|0.28|
|f60_R30_s1002|J|dir-shared|cosine|0.39|-0.20|1.07|0.88|
|f60_R30_s1002|J|cos-const|shared|1.52|0.20|2.88|0.99|
|f60_R30_s1002|J|cos-const|dir|2.07|0.85|3.39|1.00|
|f60_R30_s1002|BCE|dir-shared|constant|0.16|-0.77|1.12|0.62|
|f60_R30_s1002|BCE|dir-shared|cosine|0.57|-0.38|1.54|0.89|
|f60_R30_s1002|BCE|cos-const|shared|1.40|0.04|2.74|0.98|
|f60_R30_s1002|BCE|cos-const|dir|1.81|0.61|3.20|1.00|
|f60_R30_s1002|NCE|dir-shared|constant|-0.59|-1.36|0.18|0.07|
|f60_R30_s1002|NCE|dir-shared|cosine|-0.45|-1.10|0.20|0.08|
|f60_R30_s1002|NCE|cos-const|shared|0.75|-0.59|2.03|0.84|
|f60_R30_s1002|NCE|cos-const|dir|0.89|-0.41|2.13|0.91|
|f60_R30_s1003|J|dir-shared|constant|-0.69|-1.34|-0.10|0.01|
|f60_R30_s1003|J|dir-shared|cosine|-0.10|-0.63|0.43|0.35|
|f60_R30_s1003|J|cos-const|shared|0.73|-0.49|1.95|0.89|
|f60_R30_s1003|J|cos-const|dir|1.32|0.22|2.45|0.99|
|f60_R30_s1003|BCE|dir-shared|constant|-0.10|-0.81|0.63|0.40|
|f60_R30_s1003|BCE|dir-shared|cosine|-0.02|-0.69|0.67|0.47|
|f60_R30_s1003|BCE|cos-const|shared|0.55|-0.75|1.87|0.78|
|f60_R30_s1003|BCE|cos-const|dir|0.63|-0.65|1.89|0.82|
|f60_R30_s1003|NCE|dir-shared|constant|-0.51|-1.24|0.18|0.08|
|f60_R30_s1003|NCE|dir-shared|cosine|-0.97|-1.79|-0.22|0.01|
|f60_R30_s1003|NCE|cos-const|shared|0.49|-0.77|1.74|0.76|
|f60_R30_s1003|NCE|cos-const|dir|0.04|-1.24|1.28|0.53|
|cc_R31_s1001|J|dir-shared|constant|0.20|-0.40|0.80|0.68|
|cc_R31_s1001|J|dir-shared|cosine|-0.10|-0.70|0.50|0.32|
|cc_R31_s1001|J|cos-const|shared|0.40|-1.70|2.40|0.63|
|cc_R31_s1001|J|cos-const|dir|0.10|-2.00|2.20|0.51|
|cc_R31_s1001|BCE|dir-shared|constant|-0.10|-0.70|0.50|0.35|
|cc_R31_s1001|BCE|dir-shared|cosine|-0.70|-1.50|0.00|0.02|
|cc_R31_s1001|BCE|cos-const|shared|-0.30|-2.20|1.60|0.37|
|cc_R31_s1001|BCE|cos-const|dir|-0.90|-2.90|1.00|0.18|
|cc_R31_s1001|NCE|dir-shared|constant|0.00|-0.90|0.80|0.47|
|cc_R31_s1001|NCE|dir-shared|cosine|-0.40|-1.00|0.20|0.09|
|cc_R31_s1001|NCE|cos-const|shared|1.10|-0.90|3.10|0.85|
|cc_R31_s1001|NCE|cos-const|dir|0.70|-1.20|2.60|0.77|
|cc_R31_s1002|J|dir-shared|constant|-0.50|-1.10|-0.00|0.01|
|cc_R31_s1002|J|dir-shared|cosine|0.40|-0.30|1.20|0.84|
|cc_R31_s1002|J|cos-const|shared|-0.10|-2.00|1.80|0.44|
|cc_R31_s1002|J|cos-const|dir|0.80|-1.10|2.80|0.77|
|cc_R31_s1002|BCE|dir-shared|constant|0.30|-0.20|0.80|0.84|
|cc_R31_s1002|BCE|dir-shared|cosine|0.60|0.00|1.30|0.96|
|cc_R31_s1002|BCE|cos-const|shared|0.50|-1.50|2.40|0.68|
|cc_R31_s1002|BCE|cos-const|dir|0.80|-1.20|2.80|0.77|
|cc_R31_s1002|NCE|dir-shared|constant|-0.70|-1.70|0.30|0.07|
|cc_R31_s1002|NCE|dir-shared|cosine|0.10|-0.40|0.60|0.60|
|cc_R31_s1002|NCE|cos-const|shared|-0.80|-2.70|1.00|0.19|
|cc_R31_s1002|NCE|cos-const|dir|0.00|-2.00|1.90|0.47|
|cc_R31_s1003|J|dir-shared|constant|-0.20|-0.70|0.30|0.16|
|cc_R31_s1003|J|dir-shared|cosine|0.30|-0.40|1.00|0.76|
|cc_R31_s1003|J|cos-const|shared|-1.10|-2.90|0.90|0.13|
|cc_R31_s1003|J|cos-const|dir|-0.60|-2.40|1.30|0.27|
|cc_R31_s1003|BCE|dir-shared|constant|-0.20|-0.70|0.30|0.17|
|cc_R31_s1003|BCE|dir-shared|cosine|0.60|-0.20|1.40|0.93|
|cc_R31_s1003|BCE|cos-const|shared|-0.50|-2.60|1.50|0.30|
|cc_R31_s1003|BCE|cos-const|dir|0.30|-1.60|2.30|0.61|
|cc_R31_s1003|NCE|dir-shared|constant|-0.10|-0.90|0.70|0.39|
|cc_R31_s1003|NCE|dir-shared|cosine|0.00|-0.70|0.70|0.48|
|cc_R31_s1003|NCE|cos-const|shared|-1.20|-3.10|0.70|0.11|
|cc_R31_s1003|NCE|cos-const|dir|-1.10|-3.00|0.90|0.14|
|cc_R31_s1001|J-BCE|J-BCE|N,directional,constant|-0.50|-2.10|1.20|0.23|
|cc_R31_s1001|J-NCE|J-NCE|N,directional,constant|5.00|2.40|7.60|1.00|
|cc_R31_s1001|J-BCE|J-BCE|N,directional,cosine|0.50|-1.00|1.90|0.73|
|cc_R31_s1001|J-NCE|J-NCE|N,directional,cosine|4.40|2.00|6.80|1.00|
|cc_R31_s1001|J-BCE|J-BCE|N,shared,constant|-0.80|-2.50|0.90|0.15|
|cc_R31_s1001|J-NCE|J-NCE|N,shared,constant|4.80|2.40|7.30|1.00|
|cc_R31_s1001|J-BCE|J-BCE|N,shared,cosine|-0.10|-1.60|1.30|0.41|
|cc_R31_s1001|J-NCE|J-NCE|N,shared,cosine|4.10|1.60|6.50|1.00|
|cc_R31_s1001|J-BCE|J-BCE|T,shared,constant|2.10|1.00|3.30|1.00|
|cc_R31_s1001|J-NCE|J-NCE|T,shared,constant|2.70|1.10|4.50|1.00|
|cc_R31_s1001|J-BCE|J-BCE|T,shared,cosine|141.50|131.50|151.20|1.00|
|cc_R31_s1001|J-NCE|J-NCE|T,shared,cosine|379.30|365.20|393.20|1.00|
|cc_R31_s1002|J-BCE|J-BCE|N,directional,constant|1.00|-0.50|2.40|0.90|
|cc_R31_s1002|J-NCE|J-NCE|N,directional,constant|3.30|0.50|5.90|0.99|
|cc_R31_s1002|J-BCE|J-BCE|N,directional,cosine|1.00|-0.30|2.30|0.92|
|cc_R31_s1002|J-NCE|J-NCE|N,directional,cosine|4.10|1.40|6.80|1.00|
|cc_R31_s1002|J-BCE|J-BCE|N,shared,constant|1.80|0.40|3.30|0.99|
|cc_R31_s1002|J-NCE|J-NCE|N,shared,constant|3.10|0.50|5.60|0.99|
|cc_R31_s1002|J-BCE|J-BCE|N,shared,cosine|1.20|-0.20|2.70|0.95|
|cc_R31_s1002|J-NCE|J-NCE|N,shared,cosine|3.80|1.20|6.40|1.00|
|cc_R31_s1002|J-BCE|J-BCE|T,shared,constant|1.30|0.50|2.20|1.00|
|cc_R31_s1002|J-NCE|J-NCE|T,shared,constant|2.10|0.60|3.60|1.00|
|cc_R31_s1002|J-BCE|J-BCE|T,shared,cosine|166.70|156.50|177.50|1.00|
|cc_R31_s1002|J-NCE|J-NCE|T,shared,cosine|383.00|369.19|396.90|1.00|
|cc_R31_s1003|J-BCE|J-BCE|N,directional,constant|1.10|-0.30|2.60|0.93|
|cc_R31_s1003|J-NCE|J-NCE|N,directional,constant|2.00|-0.40|4.40|0.95|
|cc_R31_s1003|J-BCE|J-BCE|N,directional,cosine|0.20|-1.20|1.60|0.61|
|cc_R31_s1003|J-NCE|J-NCE|N,directional,cosine|2.50|-0.00|4.90|0.97|
|cc_R31_s1003|J-BCE|J-BCE|N,shared,constant|1.10|-0.40|2.60|0.92|
|cc_R31_s1003|J-NCE|J-NCE|N,shared,constant|2.10|-0.30|4.60|0.95|
|cc_R31_s1003|J-BCE|J-BCE|N,shared,cosine|0.50|-0.90|2.00|0.74|
|cc_R31_s1003|J-NCE|J-NCE|N,shared,cosine|2.20|-0.30|4.60|0.95|
|cc_R31_s1003|J-BCE|J-BCE|T,shared,constant|0.90|0.00|1.90|0.97|
|cc_R31_s1003|J-NCE|J-NCE|T,shared,constant|1.70|0.20|3.40|0.99|
|cc_R31_s1003|J-BCE|J-BCE|T,shared,cosine|121.40|112.50|129.90|1.00|
|cc_R31_s1003|J-NCE|J-NCE|T,shared,cosine|382.00|368.20|396.21|1.00|
|f60_R30_s1001|J-BCE|J-BCE|N,directional,constant|-0.22|-1.18|0.81|0.32|
|f60_R30_s1001|J-NCE|J-NCE|N,directional,constant|1.70|0.18|3.24|0.99|
|f60_R30_s1001|J-BCE|J-BCE|N,directional,cosine|0.20|-0.73|1.16|0.66|
|f60_R30_s1001|J-NCE|J-NCE|N,directional,cosine|2.80|1.16|4.46|1.00|
|f60_R30_s1001|J-BCE|J-BCE|N,shared,constant|0.16|-0.67|1.08|0.63|
|f60_R30_s1001|J-NCE|J-NCE|N,shared,constant|1.24|-0.20|2.66|0.95|
|f60_R30_s1001|J-BCE|J-BCE|N,shared,cosine|0.75|-0.12|1.58|0.95|
|f60_R30_s1001|J-NCE|J-NCE|N,shared,cosine|2.47|0.87|3.96|1.00|
|f60_R30_s1001|J-BCE|J-BCE|T,shared,constant|10.45|7.89|13.04|1.00|
|f60_R30_s1001|J-NCE|J-NCE|T,shared,constant|5.15|3.18|7.34|1.00|
|f60_R30_s1001|J-BCE|J-BCE|T,shared,cosine|1.42|0.18|2.64|0.99|
|f60_R30_s1001|J-NCE|J-NCE|T,shared,cosine|4.48|2.45|6.47|1.00|
|f60_R30_s1002|J-BCE|J-BCE|N,directional,constant|-1.01|-2.09|0.18|0.04|
|f60_R30_s1002|J-NCE|J-NCE|N,directional,constant|-0.14|-1.85|1.56|0.43|
|f60_R30_s1002|J-BCE|J-BCE|N,directional,cosine|-0.75|-1.70|0.18|0.05|
|f60_R30_s1002|J-NCE|J-NCE|N,directional,cosine|1.05|-0.65|2.82|0.89|
|f60_R30_s1002|J-BCE|J-BCE|N,shared,constant|-0.69|-1.89|0.53|0.13|
|f60_R30_s1002|J-NCE|J-NCE|N,shared,constant|-0.57|-2.31|1.11|0.26|
|f60_R30_s1002|J-BCE|J-BCE|N,shared,cosine|-0.57|-1.60|0.45|0.15|
|f60_R30_s1002|J-NCE|J-NCE|N,shared,cosine|0.20|-1.56|2.03|0.62|
|f60_R30_s1002|J-BCE|J-BCE|T,shared,constant|54.08|48.68|59.57|1.00|
|f60_R30_s1002|J-NCE|J-NCE|T,shared,constant|46.67|42.41|51.12|1.00|
|f60_R30_s1002|J-BCE|J-BCE|T,shared,cosine|1.48|-0.18|3.00|0.96|
|f60_R30_s1002|J-NCE|J-NCE|T,shared,cosine|0.37|-1.60|2.47|0.66|
|f60_R30_s1003|J-BCE|J-BCE|N,directional,constant|-0.22|-1.30|0.81|0.32|
|f60_R30_s1003|J-NCE|J-NCE|N,directional,constant|-0.43|-1.87|1.12|0.28|
|f60_R30_s1003|J-BCE|J-BCE|N,directional,cosine|0.47|-0.30|1.26|0.88|
|f60_R30_s1003|J-NCE|J-NCE|N,directional,cosine|0.85|-0.59|2.35|0.86|
|f60_R30_s1003|J-BCE|J-BCE|N,shared,constant|0.37|-0.75|1.46|0.74|
|f60_R30_s1003|J-NCE|J-NCE|N,shared,constant|-0.26|-1.85|1.42|0.36|
|f60_R30_s1003|J-BCE|J-BCE|N,shared,cosine|0.55|-0.22|1.38|0.91|
|f60_R30_s1003|J-NCE|J-NCE|N,shared,cosine|-0.02|-1.54|1.48|0.49|
|f60_R30_s1003|J-BCE|J-BCE|T,shared,constant|17.59|14.28|21.22|1.00|
|f60_R30_s1003|J-NCE|J-NCE|T,shared,constant|13.94|11.18|16.81|1.00|
|f60_R30_s1003|J-BCE|J-BCE|T,shared,cosine|2.58|1.34|3.89|1.00|
|f60_R30_s1003|J-NCE|J-NCE|T,shared,cosine|-0.06|-2.01|1.91|0.47|
|f60_R30_s1001|J|T cos-const|0.04|-0.97|1.08|0.53|
|f60_R30_s1001|BCE|T cos-const|9.07|6.76|11.52|1.00|
|f60_R30_s1001|NCE|T cos-const|0.71|-0.73|2.25|0.84|
|f60_R30_s1002|J|T cos-const|1.56|0.24|2.90|0.99|
|f60_R30_s1002|BCE|T cos-const|54.16|48.62|59.74|1.00|
|f60_R30_s1002|NCE|T cos-const|47.85|43.71|52.05|1.00|
|f60_R30_s1003|J|T cos-const|0.75|-0.51|1.97|0.88|
|f60_R30_s1003|BCE|T cos-const|15.76|12.60|18.99|1.00|
|f60_R30_s1003|NCE|T cos-const|14.75|12.33|17.44|1.00|
|cc_R31_s1001|J|T cos-const|376.50|362.20|390.40|1.00|
|cc_R31_s1001|BCE|T cos-const|237.10|225.00|250.40|1.00|
|cc_R31_s1001|NCE|T cos-const|-0.10|-0.80|0.70|0.37|
|cc_R31_s1002|J|T cos-const|381.70|367.90|395.50|1.00|
|cc_R31_s1002|BCE|T cos-const|216.30|203.60|229.40|1.00|
|cc_R31_s1002|NCE|T cos-const|0.80|0.10|1.60|0.99|
|cc_R31_s1003|J|T cos-const|380.20|366.20|394.40|1.00|
|cc_R31_s1003|BCE|T cos-const|259.70|246.40|272.90|1.00|
|cc_R31_s1003|NCE|T cos-const|-0.10|-1.20|0.90|0.42|
|cc_R31_s1001|J|unit-H (A block)|2.60|-0.90|6.20|0.91|
|cc_R31_s1002|J|unit-H (A block)|-0.40|-4.00|3.10|0.40|
|cc_R31_s1003|J|unit-H (A block)|2.00|-1.40|5.40|0.86|
|cc_R31_s1001|BCE|unit-H (A block)|3.40|-0.30|7.10|0.96|
|cc_R31_s1002|BCE|unit-H (A block)|-2.20|-6.20|1.50|0.12|
|cc_R31_s1003|BCE|unit-H (A block)|0.90|-2.90|4.50|0.66|
|cc_R31_s1001|NCE|unit-H (A block)|-2.20|-5.90|1.60|0.12|
|cc_R31_s1002|NCE|unit-H (A block)|-3.50|-7.20|0.30|0.04|
|cc_R31_s1003|NCE|unit-H (A block)|-0.10|-4.00|3.70|0.46|

## 候选支持（训练池，观测关系；trust = 父模型原生 GMM 子集）
|parent|n_records|trust_retention|i2t_frac_images_with_an_observed_caption_in_top32|t2i_frac_captions_observed_image_in_top32|i2t_frac_observed_partner_at_rank1|t2i_frac_observed_partner_at_rank1|trust_and_t2i_in_top32|trust_and_t2i_not_in_top32|notrust_and_t2i_in_top32|notrust_and_t2i_not_in_top32|t2i_observed_rank_hist_1_2to5_6to10_11to32_absent|
|---|---|---|---|---|---|---|---|---|---|---|---|
|f60_R30_s1001|145000|0.4247|0.8939|0.3985|NA|NA|56901|4687|888|82524|41356 10660 2706 3067 87211|
|f60_R30_s1002|145000|0.4232|0.8752|0.3896|NA|NA|55411|5957|1078|82554|37598 11738 3219 3934 88511|
|f60_R30_s1003|145000|0.4222|0.8786|0.3894|NA|NA|55487|5730|977|82806|38009 11556 3183 3716 88536|
|cc_R31_s1001|150000|0.9178|0.7191|0.6867|0.2612|0.2101|100650|37025|2351|9974|31515 31844 15068 24574 46999|
|cc_R31_s1002|150000|0.9173|0.7137|0.6820|0.2579|0.2079|99948|37644|2356|10052|31182 31514 15162 24446 47696|
|cc_R31_s1003|150000|0.9158|0.7151|0.6819|0.2589|0.2058|99823|37541|2465|10171|30870 31477 15357 24584 47712|

## 方向梯度夹角（训练批，trunk 上 i2t 与 t2i 梯度 cos，两 peer 平均；step 1 起 trunk 有梯度）
|parent|architecture|schedule|objective|cos_trunk_step1|cos_trunk_step250|cos_trunk_step1000|cos_trunk_step4000|
|---|---|---|---|---|---|---|---|
|cc_R31_s1001|directional|constant|BCE|0.976|0.756|0.474|-0.089|
|cc_R31_s1001|directional|constant|J|0.962|0.135|0.652|-0.220|
|cc_R31_s1001|directional|constant|NCE|0.914|-0.006|-0.265|-0.081|
|cc_R31_s1001|directional|cosine|BCE|0.976|0.755|0.483|-0.097|
|cc_R31_s1001|directional|cosine|J|0.962|0.135|0.654|-0.209|
|cc_R31_s1001|directional|cosine|NCE|0.914|-0.007|-0.252|-0.179|
|cc_R31_s1001|shared|cosine|BCE|0.992|0.099|0.155|-0.384|
|cc_R31_s1001|shared|cosine|J|0.688|-0.123|0.251|-0.496|
|cc_R31_s1001|shared|cosine|NCE|0.487|-0.070|0.060|-0.143|
|cc_R31_s1002|directional|constant|BCE|0.961|0.648|0.552|-0.229|
|cc_R31_s1002|directional|constant|J|0.962|-0.393|0.534|-0.331|
|cc_R31_s1002|directional|constant|NCE|0.908|0.006|0.126|-0.349|
|cc_R31_s1002|directional|cosine|BCE|0.961|0.646|0.556|-0.250|
|cc_R31_s1002|directional|cosine|J|0.962|-0.394|0.537|-0.400|
|cc_R31_s1002|directional|cosine|NCE|0.908|0.005|0.135|-0.417|
|cc_R31_s1002|shared|constant|BCE|0.992|-0.137|0.392|-0.314|
|cc_R31_s1002|shared|constant|J|0.979|-0.614|0.714|-0.284|
|cc_R31_s1002|shared|constant|NCE|0.496|-0.079|0.177|-0.265|
|cc_R31_s1002|shared|cosine|BCE|0.992|-0.138|0.390|-0.296|
|cc_R31_s1002|shared|cosine|J|0.979|-0.614|0.711|-0.310|
|cc_R31_s1002|shared|cosine|NCE|0.496|-0.079|0.179|-0.283|
|cc_R31_s1003|directional|constant|BCE|0.961|0.628|0.673|-0.260|
|cc_R31_s1003|directional|constant|J|0.960|-0.616|0.786|-0.349|
|cc_R31_s1003|directional|constant|NCE|0.935|-0.391|0.252|-0.441|
|cc_R31_s1003|directional|cosine|BCE|0.961|0.625|0.672|-0.118|
|cc_R31_s1003|directional|cosine|J|0.960|-0.617|0.787|-0.303|
|cc_R31_s1003|directional|cosine|NCE|0.935|-0.392|0.256|-0.458|
|cc_R31_s1003|shared|constant|BCE|0.991|0.310|0.788|-0.326|
|cc_R31_s1003|shared|constant|J|0.984|-0.554|0.790|-0.398|
|cc_R31_s1003|shared|constant|NCE|0.480|-0.247|0.214|-0.275|
|cc_R31_s1003|shared|cosine|BCE|0.991|0.307|0.786|-0.372|
|cc_R31_s1003|shared|cosine|J|0.984|-0.554|0.792|-0.362|
|cc_R31_s1003|shared|cosine|NCE|0.480|-0.247|0.214|-0.283|
|f60_R30_s1001|directional|constant|BCE|0.978|-0.174|0.132|-0.362|
|f60_R30_s1001|directional|constant|J|0.978|-0.065|0.076|-0.443|
|f60_R30_s1001|directional|constant|NCE|0.792|-0.238|0.070|-0.685|
|f60_R30_s1001|directional|cosine|BCE|0.978|-0.175|0.138|-0.412|
|f60_R30_s1001|directional|cosine|J|0.978|-0.065|0.093|-0.508|
|f60_R30_s1001|directional|cosine|NCE|0.792|-0.238|0.076|-0.709|
|f60_R30_s1001|shared|cosine|BCE|0.992|-0.207|-0.217|-0.617|
|f60_R30_s1001|shared|cosine|J|0.992|-0.084|-0.456|-0.720|
|f60_R30_s1001|shared|cosine|NCE|0.503|-0.404|-0.257|-0.241|
|f60_R30_s1002|directional|constant|BCE|0.970|-0.041|0.147|-0.252|
|f60_R30_s1002|directional|constant|J|0.978|0.170|-0.061|-0.339|
|f60_R30_s1002|directional|constant|NCE|0.813|-0.467|0.079|-0.638|
|f60_R30_s1002|directional|cosine|BCE|0.970|-0.041|0.157|-0.335|
|f60_R30_s1002|directional|cosine|J|0.978|0.170|-0.046|-0.516|
|f60_R30_s1002|directional|cosine|NCE|0.813|-0.467|0.080|-0.662|
|f60_R30_s1002|shared|cosine|BCE|0.993|-0.743|-0.418|-0.264|
|f60_R30_s1002|shared|cosine|J|0.993|-0.694|-0.638|-0.406|
|f60_R30_s1002|shared|cosine|NCE|0.495|-0.368|-0.318|-0.254|
|f60_R30_s1003|directional|constant|BCE|0.971|0.106|0.512|-0.335|
|f60_R30_s1003|directional|constant|J|0.971|0.239|0.261|-0.547|
|f60_R30_s1003|directional|constant|NCE|0.726|-0.286|0.145|-0.496|
|f60_R30_s1003|directional|cosine|BCE|0.971|0.106|0.521|-0.364|
|f60_R30_s1003|directional|cosine|J|0.971|0.239|0.277|-0.514|
|f60_R30_s1003|directional|cosine|NCE|0.726|-0.286|0.147|-0.585|
|f60_R30_s1003|shared|cosine|BCE|0.993|-0.843|-0.720|-0.158|
|f60_R30_s1003|shared|cosine|J|0.993|-0.786|-0.613|-0.462|
|f60_R30_s1003|shared|cosine|NCE|0.469|-0.408|-0.333|-0.080|

## 成本
R38 新单元 GPU 墙钟合计 6700 s（unit_wall_seconds）；cache/prep 与 CPU 检查见下。
|unit|item|gpu|job|train_seconds|eval_seconds|unit_wall_seconds|seconds|peak_gpu_gb|cost_scope|
|---|---|---|---|---|---|---|---|---|---|
|r38_B_cc_R31_s1001_N_directional_BCE_constant|unit|NVIDIA L40S|1082572|80.2|7.1|87.7|NA|3.2|R38 new|
|r38_B_cc_R31_s1001_N_directional_J_constant|unit|NVIDIA L40S|1082572|87.4|11.0|98.8|NA|3.2|R38 new|
|r38_B_cc_R31_s1001_N_directional_NCE_constant|unit|NVIDIA L40S|1082572|77.3|6.8|84.5|NA|3.2|R38 new|
|r38_B_cc_R31_s1001_N_directional_BCE_cosine|unit|NVIDIA L40S|1082572|78.7|17.9|97.0|NA|3.2|R38 new|
|r38_B_cc_R31_s1001_N_directional_J_cosine|unit|NVIDIA L40S|1082572|85.3|6.9|92.6|NA|3.2|R38 new|
|r38_B_cc_R31_s1001_N_directional_NCE_cosine|unit|NVIDIA L40S|1082572|75.3|7.2|82.9|NA|3.2|R38 new|
|r38_B_cc_R31_s1001_N_shared_BCE_cosine|unit|NVIDIA L40S|1082572|69.3|12.1|81.8|NA|3.2|R38 new|
|r38_B_cc_R31_s1001_N_shared_J_cosine|unit|NVIDIA L40S|1082572|75.8|6.6|83.7|NA|3.2|R38 new|
|r38_B_cc_R31_s1001_N_shared_NCE_cosine|unit|NVIDIA L40S|1082572|67.4|31.0|98.8|NA|3.2|R38 new|
|r38_C_cc_R31_s1001_T_shared_BCE_cosine|unit|NVIDIA L40S|1082573|53.8|7.1|61.3|NA|3.2|R38 new|
|r38_C_cc_R31_s1001_T_shared_J_cosine|unit|NVIDIA L40S|1082573|59.3|9.2|69.8|NA|3.2|R38 new|
|r38_C_cc_R31_s1001_T_shared_NCE_cosine|unit|NVIDIA L40S|1082573|52.0|3.9|56.2|NA|3.2|R38 new|
|r38_B_cc_R31_s1002_N_directional_BCE_constant|unit|NVIDIA RTX PRO 6000 Blackwell Server Edition|1082602|45.5|3.7|49.4|NA|3.2|R38 new|
|r38_B_cc_R31_s1002_N_directional_J_constant|unit|NVIDIA RTX PRO 6000 Blackwell Server Edition|1082602|48.1|5.8|54.1|NA|3.2|R38 new|
|r38_B_cc_R31_s1002_N_directional_NCE_constant|unit|NVIDIA RTX PRO 6000 Blackwell Server Edition|1082602|43.7|7.6|51.5|NA|3.2|R38 new|
|r38_B_cc_R31_s1002_N_directional_BCE_cosine|unit|NVIDIA RTX PRO 6000 Blackwell Server Edition|1082602|45.0|7.5|52.7|NA|3.2|R38 new|
|r38_B_cc_R31_s1002_N_directional_J_cosine|unit|NVIDIA RTX PRO 6000 Blackwell Server Edition|1082602|47.2|4.2|51.6|NA|3.2|R38 new|
|r38_B_cc_R31_s1002_N_directional_NCE_cosine|unit|NVIDIA RTX PRO 6000 Blackwell Server Edition|1082602|43.6|10.7|54.5|NA|3.2|R38 new|
|r38_A_cc_R31_s1002_N_shared_BCE_constant|unit|NVIDIA L40S|1082574|65.9|6.7|73.0|NA|3.2|R38 new|
|r38_A_cc_R31_s1002_N_shared_J_constant|unit|NVIDIA L40S|1082574|71.9|28.0|101.1|NA|3.2|R38 new|
|r38_A_cc_R31_s1002_N_shared_NCE_constant|unit|NVIDIA L40S|1082574|63.2|9.6|73.1|NA|3.2|R38 new|
|r38_B_cc_R31_s1002_N_shared_BCE_cosine|unit|NVIDIA RTX PRO 6000 Blackwell Server Edition|1082602|40.0|7.4|47.6|NA|3.2|R38 new|
|r38_B_cc_R31_s1002_N_shared_J_cosine|unit|NVIDIA RTX PRO 6000 Blackwell Server Edition|1082602|42.2|7.9|51.0|NA|3.2|R38 new|
|r38_B_cc_R31_s1002_N_shared_NCE_cosine|unit|NVIDIA RTX PRO 6000 Blackwell Server Edition|1082602|38.5|4.1|42.7|NA|3.2|R38 new|
|r38_C_cc_R31_s1002_T_shared_BCE_constant|unit|NVIDIA L40S|1082603|55.1|14.6|70.1|NA|3.2|R38 new|
|r38_C_cc_R31_s1002_T_shared_J_constant|unit|NVIDIA L40S|1082603|60.7|5.3|66.4|NA|3.2|R38 new|
|r38_C_cc_R31_s1002_T_shared_NCE_constant|unit|NVIDIA L40S|1082603|53.0|4.0|57.4|NA|3.2|R38 new|
|r38_C_cc_R31_s1002_T_shared_BCE_cosine|unit|NVIDIA L40S|1082603|55.0|18.0|73.3|NA|3.2|R38 new|
|r38_C_cc_R31_s1002_T_shared_J_cosine|unit|NVIDIA L40S|1082603|62.6|13.2|77.3|NA|3.2|R38 new|
|r38_C_cc_R31_s1002_T_shared_NCE_cosine|unit|NVIDIA L40S|1082603|53.2|4.3|57.8|NA|3.2|R38 new|
|r38_B_cc_R31_s1003_N_directional_BCE_constant|unit|NVIDIA L40S|1082604|76.8|6.3|83.6|NA|3.2|R38 new|
|r38_B_cc_R31_s1003_N_directional_J_constant|unit|NVIDIA L40S|1082604|82.2|46.8|129.4|NA|3.2|R38 new|
|r38_B_cc_R31_s1003_N_directional_NCE_constant|unit|NVIDIA L40S|1082604|74.0|8.7|83.1|NA|3.2|R38 new|
|r38_B_cc_R31_s1003_N_directional_BCE_cosine|unit|NVIDIA L40S|1082604|77.9|11.9|90.2|NA|3.2|R38 new|
|r38_B_cc_R31_s1003_N_directional_J_cosine|unit|NVIDIA L40S|1082604|82.5|9.5|92.4|NA|3.2|R38 new|
|r38_B_cc_R31_s1003_N_directional_NCE_cosine|unit|NVIDIA L40S|1082604|74.4|7.4|82.1|NA|3.2|R38 new|
|r38_A_cc_R31_s1003_N_shared_BCE_constant|unit|NVIDIA L40S|1082575|47.9|7.7|55.9|NA|3.2|R38 new|
|r38_A_cc_R31_s1003_N_shared_J_constant|unit|NVIDIA L40S|1082575|52.8|6.0|64.4|NA|3.2|R38 new|
|r38_A_cc_R31_s1003_N_shared_NCE_constant|unit|NVIDIA L40S|1082575|44.5|5.6|50.3|NA|3.2|R38 new|
|r38_B_cc_R31_s1003_N_shared_BCE_cosine|unit|NVIDIA L40S|1082604|64.9|9.5|74.7|NA|3.2|R38 new|
|r38_B_cc_R31_s1003_N_shared_J_cosine|unit|NVIDIA L40S|1082604|72.4|212.1|285.8|NA|3.2|R38 new|
|r38_B_cc_R31_s1003_N_shared_NCE_cosine|unit|NVIDIA L40S|1082604|62.1|8.1|70.7|NA|3.2|R38 new|
|r38_C_cc_R31_s1003_T_shared_BCE_constant|unit|NVIDIA L40S|1082605|52.6|17.7|70.7|NA|3.2|R38 new|
|r38_C_cc_R31_s1003_T_shared_J_constant|unit|NVIDIA L40S|1082605|58.1|8.5|66.9|NA|3.2|R38 new|
|r38_C_cc_R31_s1003_T_shared_NCE_constant|unit|NVIDIA L40S|1082605|50.6|4.0|55.0|NA|3.2|R38 new|
|r38_C_cc_R31_s1003_T_shared_BCE_cosine|unit|NVIDIA L40S|1082605|52.6|8.4|61.4|NA|3.2|R38 new|
|r38_C_cc_R31_s1003_T_shared_J_cosine|unit|NVIDIA L40S|1082605|58.3|10.6|70.1|NA|3.2|R38 new|
|r38_C_cc_R31_s1003_T_shared_NCE_cosine|unit|NVIDIA L40S|1082605|50.5|4.0|54.9|NA|3.2|R38 new|
|r38_B_f60_R30_s1001_N_directional_BCE_constant|unit|NVIDIA L40S|1082570|79.6|22.5|102.5|NA|2.4|R38 new|
|r38_B_f60_R30_s1001_N_directional_J_constant|unit|NVIDIA L40S|1082570|84.0|12.7|97.1|NA|2.4|R38 new|
|r38_B_f60_R30_s1001_N_directional_NCE_constant|unit|NVIDIA L40S|1082570|76.9|13.7|91.0|NA|2.4|R38 new|
|r38_B_f60_R30_s1001_N_directional_BCE_cosine|unit|NVIDIA L40S|1082570|82.5|16.8|99.7|NA|2.4|R38 new|
|r38_B_f60_R30_s1001_N_directional_J_cosine|unit|NVIDIA L40S|1082570|85.5|18.3|104.1|NA|2.4|R38 new|
|r38_B_f60_R30_s1001_N_directional_NCE_cosine|unit|NVIDIA L40S|1082570|81.4|23.5|105.3|NA|2.4|R38 new|
|r38_B_f60_R30_s1001_N_shared_BCE_cosine|unit|NVIDIA L40S|1082570|67.1|23.2|90.7|NA|2.4|R38 new|
|r38_B_f60_R30_s1001_N_shared_J_cosine|unit|NVIDIA L40S|1082570|76.4|13.9|92.8|NA|2.3|R38 new|
|r38_B_f60_R30_s1001_N_shared_NCE_cosine|unit|NVIDIA L40S|1082570|64.1|21.5|86.0|NA|2.4|R38 new|
|r38_C_f60_R30_s1001_T_shared_BCE_cosine|unit|NVIDIA RTX PRO 6000 Blackwell Server Edition|1082571|31.3|14.2|45.7|NA|2.4|R38 new|
|r38_C_f60_R30_s1001_T_shared_J_cosine|unit|NVIDIA RTX PRO 6000 Blackwell Server Edition|1082571|33.6|11.5|46.0|NA|2.3|R38 new|
|r38_C_f60_R30_s1001_T_shared_NCE_cosine|unit|NVIDIA RTX PRO 6000 Blackwell Server Edition|1082571|30.4|7.9|38.5|NA|2.4|R38 new|
|r38_B_f60_R30_s1002_N_directional_BCE_constant|unit|NVIDIA L40S|1082598|76.6|12.6|89.6|NA|2.4|R38 new|
|r38_B_f60_R30_s1002_N_directional_J_constant|unit|NVIDIA L40S|1082598|82.1|12.7|95.1|NA|2.4|R38 new|
|r38_B_f60_R30_s1002_N_directional_NCE_constant|unit|NVIDIA L40S|1082598|74.8|12.4|87.6|NA|2.4|R38 new|
|r38_B_f60_R30_s1002_N_directional_BCE_cosine|unit|NVIDIA L40S|1082598|76.9|18.6|95.8|NA|2.4|R38 new|
|r38_B_f60_R30_s1002_N_directional_J_cosine|unit|NVIDIA L40S|1082598|82.2|38.2|120.8|NA|2.4|R38 new|
|r38_B_f60_R30_s1002_N_directional_NCE_cosine|unit|NVIDIA L40S|1082598|74.6|12.1|87.2|NA|2.4|R38 new|
|r38_B_f60_R30_s1002_N_shared_BCE_cosine|unit|NVIDIA L40S|1082598|65.1|14.6|80.0|NA|2.4|R38 new|
|r38_B_f60_R30_s1002_N_shared_J_cosine|unit|NVIDIA L40S|1082598|71.1|54.9|127.7|NA|2.3|R38 new|
|r38_B_f60_R30_s1002_N_shared_NCE_cosine|unit|NVIDIA L40S|1082598|62.5|19.2|82.1|NA|2.4|R38 new|
|r38_C_f60_R30_s1002_T_shared_BCE_cosine|unit|NVIDIA L40S|1082599|38.6|12.1|51.0|NA|2.4|R38 new|
|r38_C_f60_R30_s1002_T_shared_J_cosine|unit|NVIDIA L40S|1082599|41.6|10.8|54.2|NA|2.3|R38 new|
|r38_C_f60_R30_s1002_T_shared_NCE_cosine|unit|NVIDIA L40S|1082599|36.9|10.8|48.1|NA|2.4|R38 new|
|r38_B_f60_R30_s1003_N_directional_BCE_constant|unit|NVIDIA L40S|1082600|78.3|12.5|91.2|NA|2.4|R38 new|
|r38_B_f60_R30_s1003_N_directional_J_constant|unit|NVIDIA L40S|1082600|83.8|12.4|96.6|NA|2.4|R38 new|
|r38_B_f60_R30_s1003_N_directional_NCE_constant|unit|NVIDIA L40S|1082600|75.7|29.5|105.6|NA|2.4|R38 new|
|r38_B_f60_R30_s1003_N_directional_BCE_cosine|unit|NVIDIA L40S|1082600|79.6|13.2|93.4|NA|2.4|R38 new|
|r38_B_f60_R30_s1003_N_directional_J_cosine|unit|NVIDIA L40S|1082600|84.2|42.6|127.3|NA|2.4|R38 new|
|r38_B_f60_R30_s1003_N_directional_NCE_cosine|unit|NVIDIA L40S|1082600|76.3|11.9|88.6|NA|2.4|R38 new|
|r38_B_f60_R30_s1003_N_shared_BCE_cosine|unit|NVIDIA L40S|1082600|66.6|31.4|98.4|NA|2.4|R38 new|
|r38_B_f60_R30_s1003_N_shared_J_cosine|unit|NVIDIA L40S|1082600|72.6|14.7|88.5|NA|2.3|R38 new|
|r38_B_f60_R30_s1003_N_shared_NCE_cosine|unit|NVIDIA L40S|1082600|63.5|12.4|76.3|NA|2.4|R38 new|
|r38_C_f60_R30_s1003_T_shared_BCE_cosine|unit|NVIDIA L40S|1082601|37.6|19.3|57.2|NA|2.4|R38 new|
|r38_C_f60_R30_s1003_T_shared_J_cosine|unit|NVIDIA L40S|1082601|43.5|20.7|69.5|NA|2.3|R38 new|
|r38_C_f60_R30_s1003_T_shared_NCE_cosine|unit|NVIDIA L40S|1082601|37.4|9.4|47.0|NA|2.4|R38 new|
|NA|cache cc_s1002|NVIDIA L40S|1082511|NA|NA|NA|94.1|NA|R38 new|
|NA|prep cc_s1002|NVIDIA L40S|1082511|NA|NA|NA|92.4|NA|R38 new|
|NA|cache cc_s1003|NVIDIA RTX PRO 6000 Blackwell Server Edition|1082512|NA|NA|NA|75.1|NA|R38 new|
|NA|prep cc_s1003|NVIDIA RTX PRO 6000 Blackwell Server Edition|1082512|NA|NA|NA|74.3|NA|R38 new|
|NA|CPU_TEST.json|cpu|1082521|NA|NA|NA|246.1|NA|R38 new|
|NA|PARENT_ARTIFACT_CHECK.json|cpu|1082569|NA|NA|NA|10.6|NA|R38 new|
|NA|staging of CC152K train/dev to node-local disk inside each prep job|NA|NA|NA|NA|NA|included in job wall time (logs/r38_prep_*)|NA|R38 new|
|NA|parent training (R30 f60 s1001; R31 cc s1001/s1002/s1003)|NA|NA|NA|NA|NA|existing runs, not R38 cost|NA|parent (separate)|

文件：RESULTS_FIXED.csv、CURVES.csv、DIRECTIONAL_RECALL.csv、RANK_TRANSITIONS.csv、ARCH_X_SCHEDULE_CONTRASTS.csv、OBJECTIVE_CONTRASTS.csv、T_SCHEDULE.csv、A_REAL_REPLICATION.csv、BOOTSTRAP.csv、CC_SUPPORT_CROSSTAB.csv、GRAD_DIRECTION_DIAG.csv、STEP0_CHECK.csv、NULL_READOUTS.csv、COST_LEDGER.csv。
