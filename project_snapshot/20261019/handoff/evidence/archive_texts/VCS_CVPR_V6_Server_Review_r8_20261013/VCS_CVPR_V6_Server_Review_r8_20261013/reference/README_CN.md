# 审查辅助代码

这些脚本不是训练实现，不替代服务器原有 host、dataset、metrics 和统计模块。

| 文件 | 能做什么 | 不能做什么 |
|---|---|---|
| audit_core.py | 平均 ties 的 rank、四格分解、风险恒等式、有限 mask 决策、模式 context | 自然数据性能或因果解释 |
| repo_source_audit.py | 本地源码 AST、blob、HEAD、函数范围，只读检查 | 证明实际 GPU 作业走了相同路径 |
| repo_runtime_probe.py | 显式允许后，用微型合成输入导入生产 LLaVA selector，检查 mode/梯度/constant rank | 真实 checkpoint、Qwen 可变 N 运行、完整 VLM |
| audit_rows.py | 对预先导出的 FIT/CAL 逐 mask CSV 计算描述性 rank/有限 regret | 模型拟合、独立性检验、读取 official test |

依赖：Python 3.10+、NumPy、PyTorch；本地测试用 SciPy 交叉检查。不要求服务器改变既有训练环境。脚本在本包目录运行，实际 repo 路径由参数提供。输出只写用户指定文件，不提交 Git、不修改源码、不调用 sbatch。

`--allow-newer` 只表示明确允许检查另一版本，输出记录实际 blob；不表示旧审查结论已针对新代码验证。

`legacy_spearman` 故意复现被审查实现的两次 argsort，仅用于测量差异，不应导入生产训练。

本包作者只运行了参考单元测试；没有本地挂载该私有仓库源码，所以未执行 `repo_runtime_probe.py` 的真实生产模块导入路径。交付前做了语法和 CLI 拒绝路径检查。服务器应先审代码再执行。
