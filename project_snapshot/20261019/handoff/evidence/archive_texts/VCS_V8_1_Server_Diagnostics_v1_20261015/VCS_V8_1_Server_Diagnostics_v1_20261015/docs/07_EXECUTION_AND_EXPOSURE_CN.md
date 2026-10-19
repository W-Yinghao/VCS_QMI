# 07 · 数据、执行与并行

## 1. 不触碰既有主结果

新报告建议`reports/V8_1_DIAGNOSTICS/`；新输出`/projects/EEG-foundation-model/yinghao/outputs_v81/`，实际可写路径由接入时确认。读取V8目录，但不修改best.pt、log、pred、confirm产物。

新分支建议`v8.1/utility-diagnostics`。若服务器已开始对应审查，复用其分支和编号。main/V8/V9的来源、协议和结果仍由原路径维护；更正报告采用明确附录/erratum，不覆盖原数值文件。

## 2. 标签用途表

| 数据 | V8.1 默认可做 | 默认不做 |
|---|---|---|
| EvalMuse FIT | 固定审计子集、原生损失/输入诊断 | 从FIT性能声称泛化 |
| CAL_SELECT | 原epoch/lr选择复核；新的选模假说只能另列 | 用新规则改原V8主模型 |
| CAL_PROB | 既有温度的次级核对 | 根据DEV重新拟合温度 |
| DEV | 本次开发诊断、原方法比较 | 伪称新的独立确认 |
| GenAI-Rank | 引用已发布聚合结果；需要时做明确的无标签预测一致性 | 默认不新增逐例标签读取或据此调方法 |
| GenAI-Bench 850非Rank | 等原V8-05固定报告，用于原V8结论 | V8.1的模型/阈值/结构选择；参考CLI拒绝此角色 |
| TIFA160 | 沿用V8-05已授权次级流程 | 额外新标签读取或改变用途 |
| V9/REG/SSL数据 | 不用 | 变成V8.1新训练集 |

即使确认报告在执行中到达，V8.1改动理由也应来自原开发诊断。记录设计卡时间；若团队已受确认具体错误启发，应如实标记暴露，不能自称完全独立。

## 3. 第一轮资源上限

新增完整训练0。CPU优先；现成预测优先；缺失推理上限为24正式模型的best与最终resume在固定FIT-AUDIT及CAL/DEV必要角色上的推理，每模型路径先去重。先在相同GPU类型测标准batch吞吐/峰值显存，填预算，不套用旧L40S/RTX时间作保证。

D06梯度检查最多20个固定FIT batch，先运行CS/RANK/LIST的实现检查；它不是训练网格。对任一临时模型的改动只发生在隔离输出，绝不写回原权重。

配额与nice遵循项目现有决定；本包不新设抢占优先权。确认评价和已批准V9/REV10作业不取消、不重排。没有权重的epoch不自动重训。

## 4. 时间与软件状态

来源截至2026-10-15的指定提交。记录服务器实际UTC、设备、torch/numpy/scipy版本、job snapshot和数据hash，不升级环境来迎合本包参考依赖。参考代码只需CPU numpy/scipy；pytest用于测试。

已有数据按负责人决定使用，保留来源事实，不新建许可阻塞项。本包不含第三方数据、模型权重或字体文件，也不授权再分发它们。[R8]

## 5. 交付结构

```text
reports/V8_1_DIAGNOSTICS/
  V81_00_INTAKE.md
  V81_01_EPOCH_AND_CODE_AUDIT.md
  V81_02_DECISION_AGREEMENT.md
  V81_03_SCALE_VS_SELECTION.md
  V81_04_DISTRIBUTION_AND_TRAINING.md
  V81_05_INPUT_GRADIENT_CHECKS.md
  V81_06_EXTERNAL_CONTEXT.md
  V81_08_PHASE_REPORT.md
  results/*.json, *.csv
```

机器结果字段保留NA原因、每列样本数、元数据、生成命令和引用文件hash。任何“所有方法一样”“VCS更好”“过拟合”“视觉不足”都要指明具体对象和统计范围。


**交付状态更新见 docs/09：V8-05确认已完成。上述协议保护不变；不再把等待该作业作为前置条件。**
