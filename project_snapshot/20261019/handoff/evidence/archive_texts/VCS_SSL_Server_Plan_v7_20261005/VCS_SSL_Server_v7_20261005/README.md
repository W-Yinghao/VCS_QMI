# VCS SSL 服务器方案 v7

主文档：`VCS_SSL_Server_Plan_v7_CN.md`，PDF 为同内容的阅读版。

证据截点：`W-Yinghao/VCS_QMI@fba18ffec5bca4b9396c70f427f7fda66494416f`，2026-10-05 19:50:17 UTC。执行前重新查 HEAD 与运行登记；已有单元查重后复用。

## 使用顺序

先读主文档第 0–3 节，收齐已有结果与来源。首批新性能实验是两个 CIFAR 数据集、三个方法的 projector 输出维度对照，共六个完整训练。PAIR 只读检查和 ImageNet 管线并行；PAIR、STRESS、ImageNet 的完整训练分批确认范围和预算。

`task_manifest_v7.yaml` 是逻辑清单，不是 trainer 配置，也不包含 shell 提交命令。19 个跨阶段 seed-0 训练是全部阶段的总提案，不表示现在同时提交；确认种子与可选核路线另计。

## 参考代码

```bash
cd support
python -m unittest -v test_reference
```

`pair_sampling_reference.py` 是可读、可检验的 PyTorch 数学核心；需服务器接入实际 trainer、CUDA 分块和分布式设施。它没有性能优化承诺。17 项本地 CPU/float64 检查已运行通过，实际日志及环境见 `verification_log.txt` / `verification.json`。

## 来源与结果

`sources/latest_observations.json` 只包含从已读仓库结果中摘录的字段及明确标记的旧基准，不是完整原始评价文件。`source_index.json` 指向来源。所有新方案均与已有结果分开；图像数据上没有伪造 oracle。

本包未启动服务器训练、未修改 GitHub、未包含数据集、模型权重或字体文件。
