Publication copy (sample names aliased; numbers and conclusions unchanged).

# 门禁说明修订：评估口径（2026-11-02，用户决定）

用户（2026-11-02）：R32 六行先作为“评估口径不一致”处理，不判为无效实验；保留原始日志和门禁失败记录，确定统一的论文评估规则，再对相关方法一致处理；若论文统一采用上游评估程序，可以保留该程序，补做 DEV 两种规则的对照核验，并正式修订门禁说明；不一定把 TEST 改成训练包规则；有保存的分数矩阵时优先直接重算排名。

## 1. 论文统一评估规则
所有方法、所有表（DEV 与 TEST）的 Recall/rSum 一律用上游 IN²R `evaluation.py` 的 `i2t`/`t2i`（909e5329，argsort 排名；COCO 5-fold 1K 与 5K、CC152K 同一程序）。这也是 TEST 工具一直使用的程序，不改。

## 2. DEV 复算门禁的含义（修订）
门禁用来核验“checkpoint 与评测器身份能复现该线训练时记录的 DEV”，不是规定排名规则。修订后：
- 训练时 DEV 已用上游程序记录的线（R25–R31、R33、R34 宿主 in2r/sps）：不变——上游复算与日志比较，单项 Recall 差 ≤0.1pp。
- 训练时 DEV 用其他排名规则记录的线（R32 NRCS-v1：package `retrieval_metrics`，降序、并列按候选索引递增）：在同一分数矩阵上用该线自身记录规则复算并与日志比较（≤0.1pp）→ 判 `PASS_LOGGING_RULE`；表中报告的 DEV/TEST 仍是上游程序的值，并逐单元并列给出 DEV 上 上游 − 训练包规则 之差与并列区间（全部并列计前 / 全部计后）。
- 原门禁记录（results/ 中 dev_gate=FAIL 的六行及其日志）原样保留；修订只在汇总层重新分类，不改原结果文件。

## 3. 核验依据
`diag/nrcs_rule_crosscheck.sbatch` → `diag/NRCS_RANK_RULE_CROSSCHECK.json`：38 个 NRCS view 的完整 DEV 分数矩阵（保存于 outputs_v9/r32_native_ratio_20261101/dev_scores_rulecheck/），同一矩阵上四种规则（上游 / 训练包 / 并列最优 / 并列最差）及两次计算的逐位一致性。先前 5 个 view 的初查：`diag/te_nrcs_tiecheck_1080922.out`。

## 4. 披露
本修订写于六行门禁失败及其 TEST 值已被看到之后；TEST 不重读，也不据此做任何选择。上游 argsort 对完全相等的分数没有确定的并列规则，并列较多的单元其上游值对分数的微小数值差异敏感（同一 checkpoint 两次评测 0% seed1001 ratio_J_b1_i0 的上游 DEV 为 121.716 与 121.815），这一敏感区间随表发布。

## 5. 核验结果（te_nrcs_rulecheck 1080969，L40S，DEV only）
- 38/38 个 NRCS view：两次计算的 DEV 分数矩阵逐位一致；训练包规则在同一矩阵上复算与日志完全一致（最大差 0.0pp）；上游程序复算与 TEST 工具的 DEV 复算完全一致。
- 六个原门禁失败 view 判 `PASS_LOGGING_RULE`；其 DEV 上游 − 训练包规则 = +0.20/+0.20/+0.20/+0.30/+1.18/+1.18 rSum，并列区间宽 9.7–13.0 rSum；其余 32 个 view 并列区间 ≤0.2 rSum。
- 汇总层实现：`tools/aggregate_test.py`（修订前备份 `aggregate_test.py.before_logging_rule`）；NRCS-v1 全部单元在表中的 DEV 列改为上游程序复算值，日志值保存在 TEST_RESULTS.json 的 `logged_dev_logging_rule`；TEST_TABLES.md 新增 “Units passing the DEV gate under their logging rank rule” 一节。
