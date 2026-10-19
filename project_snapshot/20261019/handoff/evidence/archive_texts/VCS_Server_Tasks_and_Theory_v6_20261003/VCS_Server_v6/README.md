# VCS 服务器材料 v6

先阅读 **VCS_Server_Tasks_v6_CN.md**；理论细节见 **VCS_Theory_to_Experiments_v6_CN.md**。

- `experiments_v6_logical.yaml` 是逻辑任务清单，不是可直接交给仓库 trainer 的 YAML。不得不查重就自动提交。
- `evidence_index.json` 固定来源、证据日期和状态；它不声称能看到服务器实时进度。
- `support/geometry_evidence_core.py` 提供精确离散计算和固定曲率 scorer 的参考接口，不是完整训练系统。
- `support/test_geometry_evidence_core.py`、`support/local_checks.json` 是本地检查及实际结果。

复核命令：

```bash
python support/run_checks.py
```

依赖 NumPy；安装 PyTorch 后运行所有 scorer/API 测试（缺少 PyTorch 时相关测试会显式 skip）。材料制作时 27 项全部通过，无 skip。没有使用真实图像、没有启动服务器训练、没有写入 GitHub。

## 本轮主线

原始 J 不变。先完成 CIFAR-100 控制，分清表示、相似度和后验评分；然后用保持 f(0)、f(1) 不变的固定曲率候选检验评分形状。a、kappa 允许按数据集选择，不能根据测量 J 最大直接挑最佳 SSL 配方。

## 阅读优先级

主任务书第 0、1、6、11 节明确接续和首批范围。第 2–5 节是理论驱动的只读实验，第 7 节是新候选，第 8 节是独立调优预算，第 9–10 节为估计表达能力及 CIFAR-100 任务粒度。附录用于讨论定义和证明，不是前置新增大项目。

## PDF 与验证

服务器任务书 PDF 为 12 页；理论与实验对应附录 PDF 为 5 页。两份 PDF 均已逐页渲染检查，未发现越界文本。

`support/local_checks.json` 记录 27 项本地检查通过。检查覆盖精确恒等式、评分/梯度和矩阵路径，不包含 CIFAR 预训练。
