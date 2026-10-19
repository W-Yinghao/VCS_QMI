# 本地验证范围

日期：2026-10-13。

- 参考单元测试：**41 项通过，0 failures / 0 errors**；见 unit_test_log.txt 和 unit_test_summary.json。
- 覆盖：ties/常数、共同置换、模式恢复与 eval 下 autograd、mask 四格、有限预算质量、风险恒等式与保守误差界、条件归一化、有限决策 regret、CSV role guard。
- 所有 Python 源码 AST 解析成功；JSON/YAML/CSV 结构校验通过；3 个 CLI help 和无 allow-import 的拒绝路径正常。
- PDF：7 页，已渲染全部页面并检查总览及重点页；未发现越界、重叠或缺字。

## 没有完成的验证

没有在本地导入用户私有仓库的生产模块，没有读取真实 checkpoint、FIT/CAL cache、宿主权重或自然数据；没有运行完整 VLM、评估 GPU 任务、读取 live Slurm 队列、修改 GitHub 或提交作业。

repo_runtime_probe.py 的实际生产模块导入与 Qwen 可变 N 路径，需服务器核对后运行；本包作者只验证了其语法与显式允许保护。它不是一个已经通过真实数据验证的 patch。

数学随机测试只证明被测有限例子的公式与实现一致，不代替普遍定理证明或自然性能实验。

环境：Python 3.13.5 / NumPy 2.3.5 / SciPy 1.17.0 / PyTorch 2.10.0+cpu。不要求服务器升级依赖。
