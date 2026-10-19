# VCS-QMI estimator 优化：研究与执行文档

## 两份主要文档

1. `VCS_QMI_Estimator_Research_Plan_v1_CN.md/.pdf`：研究问题、最新已知证据、数学主线、三条优化路线和应用关系。
2. `VCS_QMI_Estimator_Server_Agent_Spec_v1.md/.pdf`：实际实现、数据角色、损失归一化、对照、数值测试、阶段任务和结果格式。

两份文档基于上一轮已核对的 `7b7402c05161c33d77a4301f6efc27bb55420e6a` 快照整理；本次未再次检查 GitHub HEAD。服务器执行时应先核对当前代码。

## 使用方式

研究讨论阅读第一份；发给服务器时发送第二份及 support 目录。收到明确执行指令后，首批完成阶段 0–2 的小试和固定 checkpoint 诊断，再提交报告。本文档包没有启动任何服务器作业或修改 GitHub。

## 支持文件

`support/bounded_regression_core.py` 是小型 NumPy 参考，支持原始 J、平衡风险、两 critic 步长和至多八候选的 simplex 二次求解。它不是完整 neural estimator 实现。

本地运行：

```bash
python support/validate_spec.py
```

结果见 `support/validation_report.json`。验证覆盖代数、固定输出求解及有限差分；没有进行神经训练、CIFAR 实验、数据划分集成或 GPU 调度验证。

`support/previous_round/` 原样保留上一轮分析和合成检查。本轮重新运行了原检查，结果 JSON 与原记录完全一致。

## 实施默认与来源

合作者的 M、S、J、tanh critic 和回归 gap 保持不变。字典形式、残差组合和观测尺度是拟议扩展；服务器文件中的样本量、网络层宽、数据角色、噪声网格与先后顺序属于本轮实施默认。来源链接和推导条件分别写在研究稿及执行稿中。
