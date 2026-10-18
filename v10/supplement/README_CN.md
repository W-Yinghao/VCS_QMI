# V10 剩余基线实现：首版代码，测试延后

本目录对应用户 2026-10-19 CEST 的决定：“V10 的实现可以先开始，测试之后来”。基于 BF1 `b8f59ea` 建立独立分支 `v10/remaining-implementation-20261018`。本次写代码和做静态检查，不运行 Torch、数据实验、GPU门禁或全量训练，也不改变现有调度。原 BF1 冻结 clone 和科学产物独立保留。

## 已写入的代码

| 方法 | 本次实现 | 当前状态 |
|---|---|---|
| MLP_VCS_perm8 | 复用既有 perm8 训练；配准固定8个普通排列，首个与宿主一致，其余由独立随机流产生；逐排列计算负一阶/二阶矩 | WRITTEN_UNTESTED |
| InfoNCE_host[J_perm1] | 宿主同场景原始 logits 交叉熵训练；与VCS同初始化、batch和优化器；配准用固定 perm1 的 J(tanh f)，训练日志不将InfoNCE loss冒充J | WRITTEN_UNTESTED |
| MI_native、NMI_native | 原宿主64-bin Gaussian Parzen强度直方图、0.5/99.5百分位范围；共同支撑/搜索；不生成虚构训练seed重复 | WRITTEN_UNTESTED |
| CoMIR_native | 锁定已有官方DenseUNet源码；FIT32完整23×32×32训练、独立SGD状态/RNG续跑；一次编码后配准，计入latent生成成本 | WRITTEN_UNTESTED |
| RuLSIF_exact | 可分Gaussian核、完整scene V矩、alpha=1/2无约束ridge解；FIT内核中心和带宽；冻结g后用J(g−1)配准，不refit；底层另支持U矩 | WRITTEN_UNTESTED |
| MIND_2D_adapted | 显式四轴邻域、Gaussian patch SSD、本地方差和最大值归一化；二维descriptor一次生成后配准 | WRITTEN_UNTESTED |

实现入口：`src/v10/supplement/`。InfoNCE仅在本分支的`src/v10/adapter.py`增加训练分支，原MF和MLP路径保留。统一执行入口为`scripts/v10_supplement/run.py`；Slurm脚本为`slurm/v10_supplement/run.sbatch`，本次未提交。

## 理论和方法边界

RuLSIF使用成对核中心，`g(x,y)=Σ_j θ_j a_j(x)b_j(y)`。令`Z=a*b`，则完整V负参考的`h_Q=mean(a)*mean(b)`、`H_Q=(aᵀa)*(bᵀb)/n²`；U版本从分子分别扣除`Σ_i Z_i`及`ZᵀZ`，分母换为`n(n−1)`。逐scene先归一化，再按相等scene权重累积。用`H=0.5H_P+0.5H_Q`解`(H+λI)θ=h_P`，与[RuLSIF原文公式(4)–(6)](https://www.ms.k.u-tokyo.ac.jp/sugi/2013/RuLSIF.pdf)的无约束解相符。核宽度、有限核中心数和同场景产品参考属于本项目适配，不声称重现论文的所有实验。

alpha=1/2、T=g−1时，配准读出写成`J=2E_Pg−(E_Pg²+E_Qg²)/2−1`；这是展开式。代码不裁剪系数、g或T，负估计保留。线性方程用fp64求解，特征/配准沿用共同fp32设置；这项精度差异显式记录。128个中心在32个FIT core间均分，每core取2048个正配对位置；带宽中位数只由预选FIT中心确定。0.5/1/2倍带宽与1e−4/1e−3/1e−2 ridge共9个候选均在读取新增方法结果之前定义。

MIND依据[原论文的局部自相似描述子](https://pubmed.ncbi.nlm.nih.gov/22722056/)做二维适配，采用四轴邻居、3×3 Gaussian patch（sigma0.8）、局部方差下限1e−6；descriptor通道按图像坐标轴定义，配准时不额外重排方向。该选择及旋转响应必须实测；不能标为原三维方法或MIND-SSC的直接复现。

MI/NMI的“native”指当前已审计宿主的Shannon MI和Studholme NMI能量，不是ITK Mattes或另一个外部系统的完整复现。CoMIR采用宿主已登记的官方DenseUNet及原生SGD配方，不把736次原生更新等同于3000次宿主critic更新；保存每epoch checkpoint，但本首版附录只在完整第23 epoch做CAL，规则在方法冻结前需经实现审查和计时核对。

## 执行和数据接口

- 仍只激活BF FIT32/CAL8/DEV32。训练和核中心/带宽仅来自FIT，候选及checkpoint选择仅用CAL；最终完整DEV32另报固定DEV24敏感性。
- 主体数据缓存、角色manifest及代码身份均检查哈希。TEST_REPEAT与EPFL仍未激活，`evaluator_only`不进入训练或选择。
- checkpoint保存全部模块/缓冲区、优化器、局部采样和Torch/CUDA RNG；复用时校验身份、完整配置、字节哈希、步数和恢复后的模型内容。
- COMPLETE须有预定完整训练长度、全部CAL checkpoint、必要产物和哈希。所有FAIL/BLOCKED保持异常退出；原宿主会将NaN网格改写为−inf，新入口在该操作之前拒绝非有限能量。
- 成本记录涵盖job启动、训练、checkpoint、完整CAL/DEV，以及参考图编码、移动图编码或descriptor构造。中断的未完成配准只体现在实际allocation账本，不混入“已完成配准耗时”。
- CAL每族最多9候选、8 L40S allocated GPUh，保留30秒记账余量；中断尝试也计费。上一CAL job没有实际退出回执时，拒绝继续消耗预算。FINAL必须验证CAL选择和对应exit0，不能只看SELECTED字样。
- 运行入口需要另行生成且提交时固定SHA的`FROZEN_VALIDATED` release，其中T01、T02、runner验收必须对应本方法和当前代码；V9完整收尾PASS、原V10暂停标记移走也是条件。本次没有生成这样的release。

## 尚未执行的验证与后续工作

当前只有静态代码检查；所有Torch/CUDA数值、checkpoint续跑、真实配准、计时与完整实验均为NOT_RUN。`METHOD_ATTACHMENTS_DRAFT.json`是实现草案，不能作为通过门禁或正式科学冻结的证据。

后续按以下顺序验证，均通过Slurm：

1. MLP perm8固定排列、逐排列参考值/梯度、perm1一致性；InfoNCE宿主值/梯度与采样RNG重放，确认训练loss和J读出分开。
2. RuLSIF对完整小矩阵V/U逐项枚举的矩、解及输入梯度核对，核中心仅FIT、退化带宽失败、负系数不隐式裁剪；MIND常量/平移/强度变换响应、边界及几何梯度。
3. MI/NMI宿主一致性与退化范围处理；CoMIR官方模型/配方重放、含BatchNorm/dropout的续跑，以及encode-then-register路径。
4. 所有方法的CPU/CUDA必要门禁、完整checkpoint恢复、数据角色隔离、坏COMPLETE和非零退出处理；保留原4族的受影响回归范围。
5. 全部方法的完整训练/完整CAL registration计时和显存、实际成本边界。根据验证而不是DEV结果修复实现，再冻结新版本及方法附录。
6. 接入共享normal8卡的调度：先各族CAL，成功核验后交最终seed，适用时保持8个pending；不得直接复用旧dispatcher的四族枚举。

该首版草案有19个CAL候选、15个最终单元（MI/NMI/MIND各一个无训练单元，其余各3 seed），共34个完整单元，不含原BF1，也不含验证任务。`scripts/v10_supplement/plan.py`只输出清单，不提交。完整训练受8h上限约束，预算截断必须保留并从fixed-steps选择中排除，不能保证草案所有候选都能在上限内完成。

**仍未在本次完成：**实际数值/执行验收和科学结果；新方法的正式冻结与自动feeder；RuLSIF的T03完整接入；EPFL图像QC与适配；TEST_REPEAT未来激活；完整方法集的最终统计分析。这些继续归入既有V10剩余工作单。七个适配器首版代码写完不等于整个V10完成。
