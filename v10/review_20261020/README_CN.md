# V10 审查动作：交付物（审查包 2026-10-20，第 6 节）

这里的结果都来自已有记录，没有新增训练，没有重开 CAL，也没有读 TEST134/112。包原样提交在 `package/`。

| 第 6 节条目 | 文件 | 生成方式 | 状态 |
|---|---|---|---|
| 1 当前状态 | `V10_STATUS_CURRENT.json` | `status_current.py OUT.json` | 已生成；TEST v5、seed 扩展和评估噪声仍在运行或排队 |
| 2 基线资格 | `V10_BASELINE_QUALIFICATION.csv`；`rulsif_v2/cand5_B_export.json`、`cand5_per_core.csv` | `baseline_qualification.py`、`rulsif_v2_cand5_B.py` | 完成；候选 5 与 FIT 门禁记录逐位一致 |
| 3 重评审计 | 最终版 `repeat_final/`（47 组同环境配对，6 组环境不同的单列）；中间版 `repeat_with_v5/REPEAT_AUDIT.json`、`REPEAT_PER_UNIT.csv`、`cross_gpu_pairs.json`、`HASH_PROVENANCE.json`；逐配准长表在 `handoff/v10_review_actions_20261020/repeat/` | `repeat_rows.py OUT --v5`，再跑包内 `repeat_eval_audit.py` | 完成：v5 全部 42 个单元已跑完；47 组同环境配对，6 组环境不同（驱动）单列 |
| 4 EPFL | 官方分析 `handoff/v10_epfl_production_20261019/analysis/EPFL_RESULTS{.json,_CN.md}`（含逐单元表） | `ssl_pilot_v10_epfl_prod/scripts/v10_epfl/analyze.py`（含 `--check`） | 十族与 RuLSIF v2 T1024/T2048 完成；评估噪声作业待重交 |
| 5 M1 | `m1/M1_GAP_STATS.csv`、`M1_STOP_REASONS.json` | `m1_stats.py` | 完成（30 个单元） |
| 6 文案修订 | `WORDING_REVISIONS_CN.md` | 手写 | 完成；EPFL 分析部分已改入代码 |
| 7 可选重评诊断 | 见下文 | — | 建议不启用 |
| 8 T03 | `t03/T03_BIAS_COMPONENTS.csv`、`T03_M1_PRECISION_DISAGREEMENTS.json`、`T03_RULSIF_ADAPTER_CONTRACT.json` | `t03_tables.py` | 完成；接入合同的微型数值检查 UNRUN |
| §3.3 CoMIR | `COMIR_RECORDS_COMPARISON_CN.md` | 手写，数字来自记录 | 原因未查明，下一步核查已列出 |

## 第 7 项：建议不启用可选重评诊断

包里设想的诊断对象，是 MF_VCS_M1 与 InfoNCE_host[J_perm1] 的 seed 1001，在 L40S 上各重评 2 次。现有数据已经覆盖了这个用途：

- TEST_REPEAT v1/v2 的门禁，加上 v5 的同作业 DEV 重评，已经给出这两个单元在同 GPU 型号、同 checkpoint 下的重评：MF_VCS_M1 1001 有 2 次，InfoNCE 1001 有 1 次；
- 另有 33 组其他单元的同环境重评。

再开 4 次新重评，不会改变“同 checkpoint 重评会让单个配准跳变，学习型 critic 的 <9 双向翻转可达 35/320”这个观察，也不足以估计方差。按包的规定，所需 GPU 不超过 2 小时，但没有必要。

## 尚待作业完成的部分

- TEST_REPEAT v5 其余单元：跑完后重跑第 3 项的两条命令。TEST134 的结果由主会话另行分析，这里不读。
- EPFL 评估噪声的 27 个作业：已撤下，等待重交。重交跑完后重跑 `analyze.py`，并把 EPFL 的同 checkpoint 配对加进第 3 项。
- seed 扩展 1006–1008 中待重交的 10 个单元：状态文件会随之更新。
