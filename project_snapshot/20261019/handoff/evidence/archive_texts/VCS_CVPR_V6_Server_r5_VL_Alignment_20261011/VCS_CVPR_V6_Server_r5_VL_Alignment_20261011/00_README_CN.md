# VCS-QMI：CVPR V6 / Server r5
## 真实候选中的参考分布、正例目标与定位决策

**日期：2026-10-11。论文仍为 V6；r5 是服务器交接包编号，不是新方法名。**

本包承接已完成的 r4。基准截点为 `2bd405af4279ddbe7a28e6062cf740be281e037a`（2026-10-11 12:04:02 UTC）。在这个截点，R4-VL1 已完成全部 18 次完整微调；R4-VL2 有 12 次运行的提交记录，但没有完整结果报告。接入时先核对最新 HEAD 和真实作业状态，不重复提交。

本轮围绕两个具体问题：
1. 改变候选参考分布的收益，是否超过仅在测试时使用检测器分数？
2. 对所有合法框拟合均匀概率，与只要求命中任一合法框，是否解释了部分方法差距？

## 阅读顺序

- `01_SERVER_AGENT_PROMPT_CN.md`：直接发给 agent 的接入消息。
- `docs/02_SERVER_PLAN_CN.md`：主执行方案；同内容阅读版为 `VCS_Server_Plan_r5_CN.pdf`。
- `docs/03_TARGETS_AND_LOSSES_CN.md`：统计对象、损失、正例分布／集合区别、代码接入规范。
- `docs/04_DIAGNOSTICS_AND_FUSION_CN.md`：不新增训练的诊断、融合对照和共同读数。
- `docs/05_EXECUTION_STATS_AND_DECISIONS_CN.md`：六次新微调、统计、决策分支和最终评价准备。
- `docs/06_EVIDENCE_AND_SOURCES_CN.md`：来源、实测结果、推导与待验假设的区分。
- `08_R4_TO_R5_DELTA_CN.md`：与旧包衔接。

## 最小执行范围

| 类型 | 本轮工作 |
|---|---|
| 复用 | 收齐 R4-VL2；不重做 R4-VL1、E1、T2、P151 修复 |
| 分析 | 既有完整微调 checkpoint 的评分层梯度、合法框数分层、同查询嵌套候选、CAL 融合 |
| 新训练 | 分布式候选 softmax，2 数据集 × 3 seeds = **6 次完整双塔微调**；完成预算与协议核对后执行 |
| 条件扩展 | 冻结语义教师定义的参考分布，只形成备选，不自动启动 |
| 最终评价 | 准备冻结清单；没有独立批准不打开官方评价 |

本包不提供真实服务器结果，不修改 GitHub，不提交作业。参考代码验证数学与张量语义，**不是完整 VL runner**。新自然图像学习证据仍使用完整双塔微调；重读已有 checkpoint 不等于新开冻结特征探针。

## 文件与测试

`reference/r5_reference.py` 包含可移植的损失与诊断参考；`tests/` 包含独立 CPU 测试。运行：

```bash
python -m unittest discover -s tests -v
python reference/check_package.py
```

运行环境版本和测试数量以 `validation/` 实际日志为准。源码与模板适合服务器 agent 接入现有仓库，先比对再移植。全部文件校验见 `MANIFEST.sha256`。
