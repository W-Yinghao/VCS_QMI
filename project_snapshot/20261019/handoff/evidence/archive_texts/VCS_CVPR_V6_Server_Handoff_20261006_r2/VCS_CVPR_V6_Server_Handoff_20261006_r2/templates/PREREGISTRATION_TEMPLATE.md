# MV6 单元预注册 — DRAFT

## 标识与依据

Logical task / allocated P-number / owner / protocol version:
V6 paper section / source documents / current code commit:
Existing task overlap and reuse decision:
Authorization evidence / freeze UTC:

## 研究问题与估计对象

P、Q、M、输出统计、比较单位和下游任务各是什么？same-target / native-target / detection-only如何区分？此单元不是在证明什么？

## 固定设计

Dataset / base-image population / overlap with encoder pretraining:
Checkpoint and head provenance / hashes:
FIT / VAL / EVAL / POOL / PROBE / TRUTH roles and split hashes:
Attribute implementation / N|Y / null mechanisms / layer family / deterministic maps:
Methods / functions / score-coordinate conventions:
Seeds: encoder / development-selection / independent-confirmation / measurement-repeat:
Training schedule or measurement fitting budget / precision / resource cap:
Selection criterion / candidate grid / early stopping / EVAL access rules:

## 评价和统计

Primary endpoint and unit; secondary metrics; paired/unpaired comparison; uncertainty construction:
Null level / permutations / repeat count / family handling / level flag rule:
Cost timing scope / CPU threads / GPU / peak memory:
Technical failure handling, resume rule, missing-cell handling:

## 运行前检查

配对与梯度、split、null、actual map、constant-zero control和oracle仅列适用项。给实际命令、预计产物和退出条件；不复制未经检验的CLI。

## 读取规则

预定如何描述正/零/反向/不确定/无效测量；不要求VCS胜出，不结果驱动追加seeds。任何设计变化保留旧版并新增有日期的addendum。

## 交付位置

Raw run artifacts / aggregate / figures / protocol manifest / paper change log:


## r2：明确比较单位
comparison_axis = fixed_encoder_vary_estimator / fixed_measurement_vary_encoder / not_applicable
Encoder与estimator分别是什么？哪些因素固定、哪些变化？
检测与响应的matched_condition_id、各自EVAL样本量、head及parent-pool来源：
对应Table3还是Table4？禁止跨fixture合并的历史结果有哪些？
