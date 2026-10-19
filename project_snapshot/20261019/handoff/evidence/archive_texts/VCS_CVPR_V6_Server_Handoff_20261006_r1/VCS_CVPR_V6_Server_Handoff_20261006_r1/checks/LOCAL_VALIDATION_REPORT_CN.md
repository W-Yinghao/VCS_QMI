# 本次材料包的本地核对

本记录只涵盖材料包制作和小型数学/工具测试，不属于服务器的新科学结果。

| 检查 | 实际结果 | 覆盖范围 |
|---|---|---|
| 原 V6 文件忠实性 | 22/22 文件与收到的原 ZIP 逐字节一致 | 正文、PDF、source、原证据文件未改写 |
| 新 contract checks | 14/14 通过 | 有向pair计数、同image排除、分母、block累计、scorer、logit因子、有限分布gap/压缩 |
| 原 V6 数学脚本 replay | 10/10 通过；最大绝对残差 4.44e-16 | 原有限分布恒等式和已报舍入表格统计 |
| 结构与工具检查 | 43/43 通过 | 原文件保真、Python语法、JSON/schema、toy Git只读性、PDF可渲染、无字体文件 |
| 原PDF | main 7页、full 12页、中文说明5页，全部页面可渲染 | 文件可读性；没有编辑或新排版 |

## 检查范围没有覆盖什么

未运行真实服务器上的神经训练、checkpoint复评或conditional tests；没有测试实际数据的null水平、DDP或PyTorch梯度；没有查当前HEAD和Slurm作业，没有提交新GPU实验或更改GitHub。

`intake_repo.py` 的接口测试使用隔离的toy Git仓库，不是用户服务器。它区分baseline、HEAD和dirty文件；在该测试中没有改变HEAD/index/工作文件；拒绝仓库内输出和覆盖既有输出；baseline本地缺失会明确记录。

原数学replay在临时目录执行，结果复制为 `v6_replayed_verification_results.json`；`reference_v6/evidence/` 不被重新写入。原脚本依赖NumPy/SciPy，本地版本见 `local_environment.json`，不是服务器依赖锁。

本包文件完整性可通过 `python tools/verify_package.py` 重新核验；校验通过不等于论文实验已完成。新的实验结果仍应使用独立的protocol、raw artifacts和状态记录。
