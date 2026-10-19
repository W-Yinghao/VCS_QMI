# 来源、范围与复核深度

包截点 `54e1715e29dd2caaa6be7ed8e670829310951d39`。凡在父提交 `3a5e2f0b19d5d32a3913c3916413bae05753eab2` 读取的文件，最新单提交 diff 未修改它们，因此链接统一定位本包 pin；这不表示本轮重新逐行读取了所有旧报告。

本包不使用 10-10 历史批评的旧数字替代最新结果；不把历史文献新颖性判断重新审计一遍。公开文档只核对软件语义，推导和改进建议在正文单列。

## SRC-00 — main HEAD
- 范围：branch metadata
- 说明：本轮直接读取，2026-10-13 18:36:55 UTC
- 入口：[查看原始来源](https://api.github.com/repos/W-Yinghao/VCS_QMI/branches/main)

## SRC-01 — Latest commit / integrity gate
- 范围：commit diff
- 说明：本轮读取完整相关 diff
- 入口：[查看原始来源](https://github.com/W-Yinghao/VCS_QMI/commit/54e1715e29dd2caaa6be7ed8e670829310951d39)

## SRC-02 — LLaVA selector implementation
- 范围：L63–206; L289–441; functions Block/Selector/train/soft_hard_gap/spearman
- 说明：本轮重新读取；blob 706a464b494f8458b49026322030347a1d9aeaaf
- 文件：`scripts/r7_d1_selectors.py`
- 入口：[查看原始来源](https://github.com/W-Yinghao/VCS_QMI/blob/54e1715e29dd2caaa6be7ed8e670829310951d39/scripts/r7_d1_selectors.py)

## SRC-03 — Qwen selector implementation
- 范围：本轮 L245–265 与 L284–356；其余 encode/estimator_terms/gate_terms/soft_hard_gap40 在上一轮读取
- 说明：本轮重新核对 train40 定义和 B 模式路径；其余上一轮读取；blob 9a68d1c0926cdb46c06e84a38b8ab57251c1011c
- 文件：`scripts/r7_d1_40_selectors.py`
- 入口：[查看原始来源](https://github.com/W-Yinghao/VCS_QMI/blob/54e1715e29dd2caaa6be7ed8e670829310951d39/scripts/r7_d1_40_selectors.py)

## SRC-04 — D1-FULL training wrapper
- 范围：L1–180
- 说明：本轮重新读取，确认复用 frozen D1 trainer
- 文件：`scripts/r7_d1full_train.py`
- 入口：[查看原始来源](https://github.com/W-Yinghao/VCS_QMI/blob/54e1715e29dd2caaa6be7ed8e670829310951d39/scripts/r7_d1full_train.py)

## SRC-05 — r5 official test report
- 范围：§1–6, official score table
- 说明：上一轮读取；最新提交未改
- 文件：`reports/R5/R5_FINAL_EVAL_REPORT_20261012.md`
- 入口：[查看原始来源](https://github.com/W-Yinghao/VCS_QMI/blob/54e1715e29dd2caaa6be7ed8e670829310951d39/reports/R5/R5_FINAL_EVAL_REPORT_20261012.md)

## SRC-06 — LLaVA DEV/val result
- 范围：full report
- 说明：上一轮读取；最新提交未改
- 文件：`reports/R7/R7_D1_30_REPORT_20261013.md`
- 入口：[查看原始来源](https://github.com/W-Yinghao/VCS_QMI/blob/54e1715e29dd2caaa6be7ed8e670829310951d39/reports/R7/R7_D1_30_REPORT_20261013.md)

## SRC-07 — LLaVA official result
- 范围：full report
- 说明：上一轮读取；最新提交未改
- 文件：`reports/R7/R7_50_PHASE_L_REPORT_20261013.md`
- 入口：[查看原始来源](https://github.com/W-Yinghao/VCS_QMI/blob/54e1715e29dd2caaa6be7ed8e670829310951d39/reports/R7/R7_50_PHASE_L_REPORT_20261013.md)

## SRC-08 — Qwen DEV/val result
- 范围：full report
- 说明：上一轮读取；最新提交未改
- 文件：`reports/R7/R7_40C_REPORT_20261013.md`
- 入口：[查看原始来源](https://github.com/W-Yinghao/VCS_QMI/blob/54e1715e29dd2caaa6be7ed8e670829310951d39/reports/R7/R7_40C_REPORT_20261013.md)

## SRC-09 — Qwen official result
- 范围：full report
- 说明：上一轮读取；最新提交未改
- 文件：`reports/R7/R7_50_PHASE_Q_REPORT_20261013.md`
- 入口：[查看原始来源](https://github.com/W-Yinghao/VCS_QMI/blob/54e1715e29dd2caaa6be7ed8e670829310951d39/reports/R7/R7_50_PHASE_Q_REPORT_20261013.md)

## SRC-10 — LLaVA CAL readout
- 范围：full report
- 说明：上一轮读取；相关性数字为原代码读数，未在本包纠正
- 文件：`reports/R7/R7_D1_20_CAL_READOUT.md`
- 入口：[查看原始来源](https://github.com/W-Yinghao/VCS_QMI/blob/54e1715e29dd2caaa6be7ed8e670829310951d39/reports/R7/R7_D1_20_CAL_READOUT.md)

## SRC-11 — Qwen CAL readout
- 范围：full report
- 说明：上一轮读取；相关性数字为原代码读数
- 文件：`reports/R7/R7_40B_CAL_READOUT.md`
- 入口：[查看原始来源](https://github.com/W-Yinghao/VCS_QMI/blob/54e1715e29dd2caaa6be7ed8e670829310951d39/reports/R7/R7_40B_CAL_READOUT.md)

## SRC-12 — D1 route / task definitions
- 范围：§1–6 excerpt
- 说明：草稿只用于原始问题说明；执行以已冻结协议为准
- 文件：`reports/R7/R7_ROUTE_DECISION.md`
- 入口：[查看原始来源](https://github.com/W-Yinghao/VCS_QMI/blob/54e1715e29dd2caaa6be7ed8e670829310951d39/reports/R7/R7_ROUTE_DECISION.md)

## SRC-13 — LLaVA full-data freeze
- 范围：full file incl df_4
- 说明：上一轮读取；最新提交未改
- 文件：`reports/R7/R7_D1_FULL_STAGED_FREEZE_20261012.yaml`
- 入口：[查看原始来源](https://github.com/W-Yinghao/VCS_QMI/blob/54e1715e29dd2caaa6be7ed8e670829310951d39/reports/R7/R7_D1_FULL_STAGED_FREEZE_20261012.yaml)

## SRC-14 — REG second modality
- 范围：full report
- 说明：上一轮读取；最新提交未改
- 文件：`reports/R6/R6_REG_P3_REPORT_20261012.md`
- 入口：[查看原始来源](https://github.com/W-Yinghao/VCS_QMI/blob/54e1715e29dd2caaa6be7ed8e670829310951d39/reports/R6/R6_REG_P3_REPORT_20261012.md)

## SRC-15 — REG/GAIN synthesis
- 范围：full report
- 说明：上一轮读取；最新提交未改
- 文件：`reports/R6/R6_SYNTHESIS_20261012.md`
- 入口：[查看原始来源](https://github.com/W-Yinghao/VCS_QMI/blob/54e1715e29dd2caaa6be7ed8e670829310951d39/reports/R6/R6_SYNTHESIS_20261012.md)

## SRC-16 — Qwen full-data freeze
- 范围：full file
- 说明：上一轮读取；最新提交未改
- 文件：`reports/R7/R7_E2_STAGED_FREEZE_20261013.yaml`
- 入口：[查看原始来源](https://github.com/W-Yinghao/VCS_QMI/blob/54e1715e29dd2caaa6be7ed8e670829310951d39/reports/R7/R7_E2_STAGED_FREEZE_20261013.yaml)

## SRC-17 — E1/E2 decision
- 范围：full file
- 说明：上一轮读取；最新提交未改
- 文件：`reports/R7/R7_EXTENSIONS_DECISION_20261013.md`
- 入口：[查看原始来源](https://github.com/W-Yinghao/VCS_QMI/blob/54e1715e29dd2caaa6be7ed8e670829310951d39/reports/R7/R7_EXTENSIONS_DECISION_20261013.md)

## SRC-18 — Owner decisions
- 范围：decision history and updated actions
- 说明：上一轮读取相关内容；最新提交未改；接入时需再查实时授权
- 文件：`reports/OWNER_DECISIONS_20261012.md`
- 入口：[查看原始来源](https://github.com/W-Yinghao/VCS_QMI/blob/54e1715e29dd2caaa6be7ed8e670829310951d39/reports/OWNER_DECISIONS_20261012.md)

## EXT-01 — PyTorch Autograd mechanics: eval vs grad modes
- 范围：Evaluation Mode; requires_grad; no-grad
- 说明：本轮公开官方文档核对；不要求服务器升级依赖
- 入口：[查看原始来源](https://docs.pytorch.org/docs/stable/notes/autograd)

## EXT-02 — SciPy spearmanr
- 范围：ConstantInputWarning: constant coefficient undefined
- 说明：本轮官方文档核对
- 入口：[查看原始来源](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.spearmanr.html)

## EXT-03 — SciPy rankdata
- 范围：average/ordinal tie rules
- 说明：本轮官方文档核对
- 入口：[查看原始来源](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rankdata.html)

## H01 — Collaborator original September 2026 plan
- 范围：§4–7: S/J/gap and classification; §8: prior-art; §10–12: claims and experiments
- 说明：用户提供资料，完整内容已在对话中读取；本包不复制整个 PDF

## H02 — 10-10 consolidated audit with Appendix 4
- 范围：附录4优先于相冲突的正文；只作历史定位
- 说明：用户提供资料；不把其旧状态当成 10-13 进度

## SRC-19 — 交付前增量

最后核对到 3e2fe59；4 个新增提交，两个关键 critic blob 未变。正文 pin 不变，执行状态由 `docs/08_DELIVERY_UPDATE_CN.md` 补充。

- [固定比较](https://github.com/W-Yinghao/VCS_QMI/compare/54e1715e29dd2caaa6be7ed8e670829310951d39...3e2fe59c4861abae46537574c4c59f9bedfe91c6)
