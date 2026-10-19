# V10 v1 验证报告

日期：2026-10-16。本报告描述本轮实际执行范围；未执行项保持明确标记。

## 本轮实际执行

| 部分 | 结果 | 证据文件 | 覆盖 |
|---|---:|---|---|
| 独立NumPy/SciPy数学检查 | 47/47通过 | `core/MATH_NUMPY_AUDIT.json` | V/U/独立池、dense/矩一二阶、有限差分、HVP、逐场景加权、permutation均值、盒QP与ridge/M1例子 |
| 离散真值/数据生成器 | 138/138通过 | `synthetic/validation/CHECK_RESULTS.json` | 两家族、12条件、精确概率/gap、角色分隔、抽样事件及稳定RNG |
| 包语法与数据预检契约 | 15/15通过 | `results/PACKAGE_CHECK_RESULTS.json` | 9个Python文件AST、有效manifest、跨group/path角色冲突、未验证CONFIRM、空manifest、只解析宿主而不导入 |
| 计划单元生成 | 114条设计单元 | `configs/EXPERIMENT_UNITS.csv` | 生成成功；全部`designed_not_launched`；没有发作业 |

数学参考环境：Python 3.12.14、NumPy 2.3.5、SciPy 1.17.0。实际Torch模块不可用。

数学核对主要数值：V/U/独立池矩与显式表的float64误差约5.55e-17，HVP最大误差约1.11e-16，特征/logit/W有限差分最大误差约5.62e-11。NumPy投影算法与SciPy盒QP核对的最终gap小于1e-9。这是对独立数值公式和算法的检查，不是执行`mf_vcs.py`后得到的Torch成绩。

离散实例包括12个条件，每条件FIT/CAL/EVAL分别257/131/509条，共10,764条基本联合观测。S与独立闭式最大差1.11e-16，原gap最大差2.22e-16。完整pair没有被当成新的独立基本样本。

## 历史证据另存

原v0.1包曾记录19项CPU检查。本轮读取并保留于`provenance/MATH_CHECK_RESULTS_v0_1.json`，没有把这19项重新计入本轮结果，也没有把旧脚本通过当作新PyTorch后端通过。

## 明确未执行

- PyTorch真实后端、CPU/CUDA autograd和checkpoint分块：NOT_RUN。
- 真实Torch头部solver与运行时API兼容：NOT_RUN。
- 当前G1/r6_reg_pilot源码重新获取与实际adapter：NOT_RUN。
- 图像读取/encoder/checkpoint/BatchNorm/dropout状态接入：NOT_RUN。
- 连续warp与几何参数梯度：NOT_RUN。
- GPU时间、allocated/reserved峰值、encoder-inclusive成本：NOT_RUN。
- 完整自然数据训练、配准成功率、S估计优劣和论文新颖性确认：NOT_RUN。
- 当前服务器V9完成状态、manifest及确认集独立性：待实际预检，没有猜填。

任意空间forbidden edges、Gaussian/cubic生成器和真实完整应用入口没有在本版实现；主文已给出后续接入条件。现有小型数学检查不证明这些功能可运行。

## 服务器验收顺序

1. 填写真实宿主/数据预检与接口映射。
2. 运行`core/verify_torch.py`，检查V/U/独立池、所有梯度和求解器。
3. 在真实warp链补完整几何梯度核对，确认测试固定W。
4. 运行T02小表并测encoder-inclusive成本，填实际预算。
5. 按主文继续T03–T05，回传完整结果/失败和统计单位。

阶段结果分别登记，不使用一个笼统的“PASS”覆盖所有执行层次。
