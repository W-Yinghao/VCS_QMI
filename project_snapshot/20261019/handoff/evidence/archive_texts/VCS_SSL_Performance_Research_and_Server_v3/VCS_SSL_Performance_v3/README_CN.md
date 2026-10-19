# VCS 完整 SSL 性能优化 v3

目标：保持原始 VCS 二次目标与 tanh 有界输出，通过 critic 几何刻度、完整配对和噪声损失平均，争取超过同协议调优 SimCLR。

## 交付

- `VCS_SSL_Optimization_Analysis_v3_CN.md/.pdf`：9 页研究分析。包括已有证据、两条结构推导、三条主优化线、完整实验与后续方向。
- `VCS_SSL_Server_Execution_v3.md/.pdf`：9 页服务器规范。八个完整训练候选、配置继承、精确 mask、噪声评价、梯度控制、复用规则与结果要求。
- `configs/experiments_v3.yaml`：逻辑实验清单，需要映射到当前仓库 schema；不是现成训练 CLI 的可直接加载配置。
- `support/vcs_perf_core.py`：PyTorch 参考计算核心。
- `support/validate_core.py`、`validation_report.json`：25 项已运行的 CPU 检查与记录。
- `SOURCE_MANIFEST.json`：资料与证据截点。

## 给服务器

先读服务器规范。收到执行授权后核对实际 HEAD、运行已有测试，接入参考计算，然后进行真实数据 smoke 与八组完整 800-epoch 训练。P95/P98/P99 等已有工作继续复用，不覆盖配置，不用早期 J 或 rank 结束有效候选。

本轮没有修改 GitHub、访问服务器队列或启动训练。参考代码并未接入服务器主训练器。目标性能尚未验证。

## 本地检查

```bash
python support/validate_core.py
```

检查环境为 Python + PyTorch 2.10.0 CPU、float64。代码使用普通 PyTorch 操作，服务器需在其实际环境重新验证。

数学检查涵盖：二次风险、公共分量的 affine 等价、全部视图 pair 数与梯度、单侧负表示梯度的 1/2 关系、噪声平均方式与 RNG 重放。噪声点积方差的 100,000 次 Monte Carlo 相对误差约 0.30%。未运行 CIFAR、CUDA 速度测试或新的 estimator 性能比较。

PDF 使用 LaTeX 排版并逐页检查；源文档保留 Markdown 数学公式。PDF 重建脚本依赖 Pandoc/XeLaTeX 与系统字体；包内不含字体文件。
