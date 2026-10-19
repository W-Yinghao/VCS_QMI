# V10 第一阶段审核决定与服务器后续协议

更新日期：2026-10-17。项目：VCS_QMI。决定版本：V10-phase1-decisions-20261017。

**接受第一阶段 T00–T02 报告。五项主建议采用报告选项 1A、2A、3A、4B，以及数据角色的 A 立即生效、B 冻结后生效。确认第 8 节的梯度门控修改。T03 以 U 为主，自然配准继续使用同场景 V。**

本文件供服务器端实施与接续。它明确修改 2026-10-16 交接包中相应工程条款；原包保留为历史基线。决定已经作出，但这些改动尚未在服务器实现、验证或运行；本轮没有推送 GitHub 提交或提交远程训练作业。配套 JSON 是决定契约，不是可以直接运行的配置文件。

## 1 审核版本与阶段结论

审核分支为 `v10/mf-vcs`，HEAD 为 `15b1c548a7dbeb8ee043fecfee209b118657202b`；最近的测量结果提交为 `11746ddb914e1fb370215f13026fadb2dd4c571c`。HEAD 后续报告变动没有改变当前测量代码。逐项读取了报告、对应源码及已提交的 T00–T02 记录；没有在本地重新运行服务器 Torch 或 L40S 实验。[报告与提交依据][R1]

| 核对项 | 审核结果 |
|---|---|
| 测量代码 hash | `bf8942645f0167b0`，按 adapter 的定义独立重算一致 |
| 宿主 hash | `6c6f094684fbc50a` |
| 数值后端 SHA256 | `cdf5df2e43bacb74b963f7b6b519a962f91fc8e41f28b4f6a4e2696816024973` |
| 原包清单 | 134 个文件的 SHA256 全部匹配 |
| T01 | 2078 项；2031 个必需项全部 PASS；47 个诊断项中 27 PASS、20 FAIL |
| 宿主接入 | 已实现并通过当前代码身份下的 CPU/CUDA、梯度、几何及冻结状态验收 |
| T02 | L40S 已测完整训练步及配准成本；计时使用的模型没有任务表现结论 |
| T03、T04、T05 | 尚未运行；没有新的 S 估计质量或配准准确率结论 |

以上接受的是阶段证据和后续决策依据。M1 默认规则改动会产生新代码身份，旧 T01 的 PASS 不自动覆盖新实现。T00 的元数据清单也不等于数据独立性或 EPFL 图像级接入已经通过。[T01 原始记录][R2] [宿主实现][R3]

## 2 决定一 M1 使用相对零头 gap

### 2.1 主规则

对当前批次、当前固定特征和同一负参考定义头目标 \(F_b(W)\)，令

\[
\Delta_b=\nabla F_b(0),\qquad
g_{\mathrm{ref},b}=G_b(0)=\|\Delta_b\|_1,\qquad
\tau_b=0.01\,g_{\mathrm{ref},b}.
\]

当 \(G_b(W)\le\tau_b\) 时按 gap 停止；否则运行至迭代上限。CAL 候选上限为 **10、30、100**。主训练保持 `require_converged=False`，达到上限必须标记为近似解并保留实际 gap。

这里的基准是**当前批次 W=0 的 gap**，不是当前 warm start 的 gap。第一次 W=0；以后沿用上一批次 W，并单独记录实际起点的 gap。不要用随机非零初始化、M0 预训练或绝对容差下限改变这项决定。数值审计保留原有严格绝对容差模式。

### 2.2 证据与解释范围

真实宿主初始化的零头 gap 为约 \(3.176\times10^{-5}\)，小于旧的 \(10^{-4}\)，所以旧规则在该批次上返回零次迭代、W=0、J=0、encoder 目标梯度为零。强制 30 次迭代后 W 的最大绝对值约 0.0143，encoder 梯度范数约 \(8.72\times10^{-6}\)，确认可以发生目标驱动的学习。[报告 §6.1][R1]

需要修正“永远不训练”的表述：T02 旧规则的前 10 步为零次迭代，第 11 步随批次 gap 超过绝对阈值才开始更新。问题是启动依赖批次波动。另一方面，T01 强制 30 步的 gap 仅从约 \(3.176\times10^{-5}\) 降至 \(3.147\times10^{-5}\)，下降约 0.92%；这不证明新目标能在 100 步内达到。报告中的计时 gap 中位数不能当作逐批相对零头 gap 的中位数。

gap 仅证实固定特征经验头问题的次优程度；它不自动界定 encoder 梯度误差，也不证明总体 S 已估准。[头部目标与 solver][R4]

### 2.3 实现与验收要求

- 从同一个已构建的 `QuadraticHead.delta` 取得参考值并求解，避免为了参考值额外重建矩。若实现增加矩遍历或 HVP，计数和成本必须真实更新。
- 若精确计算的 \(\Delta_b=0\)，明确返回 W=0 和 `zero_reference_gap`。此时零头是该负半定二次经验目标的最优点；不要除以零，也不要给当前要求正容差的 solver 传入零容差。这不等同于总体独立。
- 对接近 fp32 分辨能力的参考值，保留一个同输入 fp64 数值诊断；没有实际失败证据时不把主训练整体改成 fp64。
- 逐批记录 `gap_ref_zero`、`gap_start`、`gap_final`、相对参考 gap、实际阈值、迭代数、停止原因、HVP 次数和矩遍历次数。保留 W 更新量与 encoder 梯度抽查。
- 新增有针对性的验收：零头正常启动、warm start 与零头基准不同、零参考值、达到上限仍正确返回、计数正确。新代码下重跑全部 T01 part，重新合并 provenance；T02 只重测受影响的 M1 配置和由此改变的预算。
- encoder、特征映射与 W 作为同一个 checkpoint 保存；头 gap 对应本步外层更新前的特征。主协议不增加最终 FIT refit，配准阶段固定模型参数并保留几何梯度。

## 3 决定二 所有方法共用已记录的 Torch 精度环境

采用报告 2A，保留当前宿主的有效设置。协议应固定实际版本和开关，而不是只写容易随环境变化的“torch 默认”。当前证据如下。[报告 §6.2][R1]

| 项目 | 当前测量环境 |
|---|---|
| PyTorch / CUDA / cuDNN | 2.9.0+cu128 / 12.8 / 91002 |
| GPU | NVIDIA L40S，compute capability 8.9 |
| float32 matmul precision | `highest` |
| cuDNN convolution fp32 precision | `tf32` |
| 原始 cuda_matmul_fp32_precision 字段 | 字符串 `none`；按原值保存，不改写为 ieee，也不把该字符串当设置值 |
| deterministic algorithms | false |
| cuDNN benchmark / deterministic | false / false |
| CUBLAS_WORKSPACE_CONFIG | `:4096:8` |
| 两项 TF32 环境覆盖变量 | 旧记录未覆盖；下一次必须记录 `TORCH_ALLOW_TF32_CUBLAS_OVERRIDE` 和 `NVIDIA_TF32_OVERRIDE`，不得假定为空 |

Torch 文档将矩阵乘法与 cuDNN 卷积的精度控制分开；`highest` 不表示所有卷积也使用 IEEE。按 2.9 的同一套 API 管理并在全部导入完成后记录有效值，不混用新旧 TF32 开关。[Torch CUDA][P1] [matmul 精度][P2] [环境覆盖变量][P3]

主表中的 Torch 方法共用该设置。原生系统若有额外精度约束，应明确命名和计费。IEEE 卷积或 fp64 可作为命名数值诊断；本批不以此另开完整科学网格。已有几何诊断说明 IEEE 可降低该比较中的数值误差，尚未证明 TF32 是 V9 配准不可重复的原因或影响任务准确率。

## 4 决定三 InfoNCE 保留宿主共同 J 能量

主表命名为 **`InfoNCE_host[J_perm1]`**：

- 训练使用 V9 的 InfoNCE 目标。
- 配准使用 `J(tanh f)`，普通 perm1，保持同场景参考；一次配准内固定负排列、搜索支撑与比较规则。
- 使用同一完整冻结 checkpoint；不在测试时重新训练 critic 或 W。
- 表注必须写明：这是“InfoNCE 训练的 critic + 共同 J 读出”，与原包中“原生目标 / 候选归一化”的表述不同。

实际 V9 的训练与配准代码支持这一命名。它可以回答共同读出下训练方式的影响；不能据此代表所有原生 InfoNCE 配准系统。[InfoNCE 宿主训练][R5] [共同 J 配准读出][R6]

原生 InfoNCE 配准能量如以后实施，作为另一命名变体，先定义候选归一化、核对数值再测完整成本。它不是当前首批的阻塞项。不能将已测 perm1 成本移作原生能量成本。

## 5 决定四 按同一冻结协议分批补齐基线

第一批为 **MF_VCS_perm1、MF_VCS_M0、MF_VCS_M1、MLP_VCS_perm1**。完成新 M1 验收和本文件所列协议落地后，可以开展 BF 的 CAL 与 DEV 工作；其余基线同步实施。[基线现状与提议][R1]

| 后续方法 | 补齐要求 |
|---|---|
| MLP_VCS_perm8 | 训练已接入；实现 8 个固定排列的配准能量，核对排列均值与梯度 |
| InfoNCE_host[J_perm1] | 接入 V10 数据门、seed、checkpoint 和上述宿主能量 |
| MI、NMI | 接入 V10 的数据、搜索、支撑与计费规则 |
| CoMIR_native | 接入原生完整训练及 encode-then-register；训练和 latent 生成成本计入 |
| RuLSIF / 核精确矩 | 实现所声明参考下的精确矩；V9 的 RFF ridge perm1 不充当该基线 |
| MIND_2D_adapted | 实施真实的二维适配并披露，不能标成已经直接复现原三维方法 |

每个新适配器先做对应 T01 核对与 T02 完整计时，再按冻结的数据、预处理、seed、几何、CAL 和精度规则补跑。可以分别记录进度、失败和实际成本；必需基线补齐之前不发布方法优劣、强基线优势或新颖性比较结论。RuLSIF 精确矩也应作为后续 T03 的二次风险对照；matched_BCE 等校准扩展另登记，不阻塞第一批实现。

全局协议冻结后，新增方法可以补充自己的实现说明和预先定义的候选；不能因第一批 DEV 表现改变公共规则。若公共实现有实质 bug，应升协议版本并对受影响方法作匹配重跑。完整方法集合和方法内选择全部固定以后才能启用 TEST 重复块。

## 6 决定五 数据角色与 EPFL 激活

### 6.1 BF 开发表与重复基准

当前维持 **FIT32、CAL8、DEV32**。DEV32 是开发主表。134 个官方 TEST core 已由 G1 读过，当前仍 HOLD；V10 完整协议冻结后，以新 manifest 版本启用单独的 TEST_REPEAT134 表。DEV 与 TEST_REPEAT 不合并成 166 个独立确认样本。当前没有独立 CONFIRM；这不阻止开发和重复基准工作。[数据 manifest][R7] [暴露记录][R8]

同时预登记一个保守敏感性分析：按 record_id 的下划线后部分作为 TMA 位置，排除与当前 FIT/CAL 同位的评估 core。跨 1B/2B 同位是否同一 specimen 尚未证实，此分组是敏感性假设。

| 评估块 | 完整 core / 位置组 | 排除 FIT/CAL 同位后 core / 位置组 |
|---|---:|---:|
| DEV | 32 / 32 | 24 / 24 |
| TEST_REPEAT，未来激活 | 134 / 97 | 112 / 75 |

DEV24 排除的 8 个 core：`1B_F4, 1B_H4, 1B_H9, 2B_D5, 2B_G3, 2B_G5, 2B_H2, 2B_I5`。TEST 的 22 个排除 ID 和可复现规则见配套 JSON。以上计数由当前 manifest 独立重算；不改原始角色。

统计以 core 为单位并在方法间配对；每个 core 的多个扰动不当作独立组织样本。另报告按 TMA 位置组重采样的保守敏感性。DEV 与旧 TEST 同位本身不等于 FIT/CAL 泄漏，不能把所有 24 个“与其他角色同位”的 DEV 一概删除。未来若确认同位为同一组织，下一版需正式按组制定划分。

### 6.2 EPFL 需要图像级核对

元数据级检查是必要条件；激活前还要完成：

1. 在 V10 路径上读取并核对 477 对图像的 hash、解码、形状、模态顺序和范围。
2. 按已登记规则复现 33 个对齐排除、24 个尺寸排除及 420 个保留窗口，不根据 V10 方法表现重新挑选。
3. 按冻结 ID 顺序每类确定 3 个代表，共 9 类 × 3 个 overlay，并检查所有已标记异常；完成 G2/P3 对齐及预处理重放。
4. 对跨类或不同尺寸的近重复候选作统一缩放缩略图检查，结合原图或元数据判断关系；候选相似不直接等同重复。若影响划分，先冻结新的组记录。
5. 通过 V10 数据门、warp、几何梯度、像素单位、有效支撑和冻结 checkpoint 核对，随后实测 EPFL 完整成本；旧代理成本不能改标“实测”。

报告中的“约 2 native px 残差”需要收窄：P3 的检查是半分辨率梯度相关的整数最佳平移落在每轴 ±1 像素附近，并非已知真对齐误差的 2 像素上界；对角分量对应的原生距离也可达 \(2\sqrt2\) 像素。2.25 native px 严格阈值可用于已知施加扰动的恢复误差，不能据此声称真实 RGB–NIR 对齐误差小于约 2 像素。[EPFL P3 检查][R9]

## 7 确认第八节 fp32 梯度门控

确认 `a8f4c5d` 的有限范围修改，容差保持不变：

| 项目 | CPU 最大绝对梯度误差 | CUDA 最大绝对梯度误差 | 门控角色 |
|---|---:|---:|---|
| fp32 全图 vs 同输入 fp64 | 2.91217e-5 | 4.13737e-8 | 诊断；CPU FAIL 保留 |
| fp32 分块 checkpoint vs 同输入 fp64 | 2.94138e-7 | 1.64116e-8 | 必需 |
| fp32 分块 vs fp32 全图 | 2.93627e-5 | 2.98023e-8 | 诊断；CPU FAIL 保留 |
| fp64 分块 vs fp64 全图 | 通过既有门限 | 通过既有门限 | 必需 |

被测实现确实在 fp32 运行；参考由相同 fp32 输入和权重转成 double 后重新计算梯度。比较包含 du、dv 两项输入描述符梯度及 6 项 critic 参数梯度，共 8 个张量、71873 个元素。报告第 8 节的“fp64 值”应改为**同输入和同权重的 fp64 梯度**。[实际门控实现][R10]

fp32 仍为 atol=1e-5、rtol=1e-4；fp64 仍为 atol=1e-10、rtol=1e-8。目标函数值的分块与全图比较保持必需，其他数学、梯度和几何门控不降级。浮点求和顺序不同可以造成全图与分块不逐位相同；本次接受的依据是独立更高精度参考支持分块实现，而不是简单消去失败行。[Torch 数值精度说明][P4]

CPU fp32 全图 dense-MLP 仍是已知超界路径；保留旧失败与新诊断。当前 T04/T05 没有使用该路径，不要求现在修复闲置路径。如果以后进入正式科学运行，它必须成为必需验收项并通过，或按原包规定使用显式更高精度积累、计入成本并重新验收。

## 8 自然配准调参网格与预算

### 8.1 第一批固定网格

本批 MF 主表固定 **K=16、temperature=1、ridge=0、W weight decay=0**。自然宿主 encoder 与 MLP critic 保持已有结构。所有由优化器学习的模块共享同一个候选学习率；M1 的 W 由 solver 求解，不给它额外学习率。其余优化器、采样和裁剪规则继承审核版本：AdamW，非 W 参数 weight decay=1e-4，clip_norm=1.0，16 场景 × 128 位置，crop=160；M0/perm1 的 W 按既定规则参与 clip 后投影。

| 方法 | 学习率 | M1 上限 | 候选数 |
|---|---|---|---:|
| MF_VCS_perm1 | 1e-4、3e-4、1e-3 | 不适用 | 3 |
| MF_VCS_M0 | 同上 | 不适用 | 3 |
| MF_VCS_M1 | 同上 | 10、30、100，完整笛卡尔积 | 9 |
| MLP_VCS_perm1 | 同上 | 不适用 | 3 |
| 合计 | | | **18** |

这是对原包 K×lr 网格的明确修改：先用共同 K16 回答机制问题。K8/K32 如用于自然图像，另记容量扩展，不暗中并入主表，也不增加没有入账的 27 格。

调参 seed=1017；正式算法 seeds=1001、1002、1003。每次 3000 步，每 250 步一个 checkpoint，共 12 个。CAL 为 8 core×10 扰动；DEV 为 32 core×10 扰动。候选与 checkpoint 仅按 CAL 的下列字典序选择：

1. success <42 native px 较高；
2. success <9 native px 较高；
3. median corner error 较低；
4. 更早的 checkpoint；
5. 冻结的候选顺序。

候选顺序为学习率升序；M1 在每个学习率内按 cap 10、30、100。允许调度并行，但完成、超预算和失败不得根据 DEV 表现选择性处理。公共搜索器、负排列、几何支撑、随机流和 checkpoint 规则保持一致。[宿主 CAL 与配准][R6]

### 8.2 预算决定

每族调参上限为**最多 9 个候选、最多 8 个 L40S 已分配 GPU 小时**，计入训练、checkpoint、CAL 配准及作业内开销；两者先到为准。排队时间另报。无须为少候选方法人为耗满预算；报告每族实际候选数和实际总成本。预算边界一旦达到，标记未完成部分，不因结果好坏临时延长。

8 小时是本次设定的工程上限：当前 capbound 计时对 M1 九格的投影约为 6.435 小时，留有余量；它不是新相对规则的实测承诺。新增原生基线先计时，再将其忠实配置与受此上限约束的候选写入冻结附件。若原生配置无法在此上限完成，披露为资源受限，而不删去成本或私自放宽。

由现有 L40S 原始计时、按新候选数重算：

| 调参部分 | 旧计时投影 GPU-h |
|---|---:|
| MF perm1，3 格 | 0.8934 |
| M0，3 格 | 0.9798 |
| M1，3 lr × 3 cap | 6.4345 |
| MLP perm1，3 格 | 1.3575 |
| 调参合计 | **9.6651** |

加上四族各 3 个正式 seed，若最终 M1 cap 为 10 / 30 / 100，总计分别约 **15.3618 / 15.8500 / 17.5608 GPU-h**。这些是基于旧计时的条件投影，未计新增验收、I/O、失败、缺失基线、T03、TEST 或 EPFL。新相对 M1 实测后必须重写 BUDGET；报告旧的 21.089 或 29.78 小时不能继续作为此网格的总预算。[T02 原始计时与预算][R11]

相同步数主表称为 same-data / fixed-steps。same-total-cost 视图使用实际累计成本和预算内已产生的 checkpoint，并只按 CAL 选取；不能用超预算训练完成后的最佳 checkpoint 回填较早预算。预算上限相同也不等于实际花费相同。

## 9 T03 目标与完整网格

### 9.1 估计对象与排列规则

T03 主模式为 **U**，真实目标仍为包生成器的 \(Q=P_XP_Y\)、\(S\) 与 \(\eta\)。对 n 个 iid 基本观测，经验 U 参考为

\[
\widehat Q_U=\frac1{n(n-1)}\sum_{i\ne j}\delta_{(x_i,y_j)}.
\]

MF 精确矩按 U 计算。所有 perm1/perm8 对照也使用**均匀无固定点排列**：从均匀全排列拒绝有固定点者，或经验证等价的均匀算法。不能只写 `randperm`，不能用“修补固定点”而不核对分布；固定循环平移也不代替本次选定的均匀无固定点方案。小 n 穷举验证所有无固定点排列的平均值和梯度等于完整 U。

V 仅作**同一个冻结 critic 上的 U/V 评估诊断**，不另开第二套 V 训练网格。自然配准保持同场景 V，普通排列允许固定点。固定、独立于当前评估样本的 T 下，V 的 Q 项含 1/n 的联合分布成分；这个有限样本公式不证明经训练及选模后的最大 J 是 S 的无偏估计。[原包与 T03 提议][R1] [V/U 数值定义][R4]

### 9.2 主训练方案

| 项目 | 冻结决定 |
|---|---|
| 数据条件 | two_block、diagonal_mixture × m=8、32 × rho=0、0.25、0.75，共 12 条件 |
| 每条件 FIT / CAL / EVAL | 4096 / 2048 / 8192 个独立基本观测 |
| 输入与 MF | 整数 one-hot；复用 SimplexBilinearCritic(m,m,K,K)，每侧 Linear+softmax，主 K16，temperature=1，W=0 |
| MLP | 拼接两侧 one-hot；Linear(2m,64)→GELU→Linear(64,64)→GELU→Linear(64,1)，T=tanh f；perm1/8 同结构同初始化 |
| 主方法 | MF perm1、M0、M1、MLP perm1、MLP perm8，共 5 族 |
| 训练与优化 | FIT 全批，2000 步，每 250 步 checkpoint；AdamW；非 W 参数 decay=1e-4、W decay=0、ridge=0、clip=1 |
| 学习率 | 1e-4、3e-4、1e-3，学习模块共享候选；M1 同时调 cap=10、30、100 |
| CAL 候选数 | 每条件 3+3+9+3+3=21；12 条件共 **252 次 CAL 训练** |
| 超参数调参套 | 独立数据 root seed=1017，算法 seed=1017 |
| CAL 选模 | 只最大化独立 CAL 的完整经验 U-J；并列取更早 checkpoint，再取上述冻结候选顺序 |
| 真实量访问 | truth 仅交 evaluator；训练和 CAL 选择只读样本 |

使用原包生成器及其样本/真值分离方式，记录每套数据的生成器与样本 hash。data RNG、模型初始化及排列 RNG 分流；相同 data seed 的全部方法共享同一套观测。每个正式重复的超参数固定自调参套，checkpoint 仍只按该重复的 CAL 样本选择。MF 的 Linear 初始化继承当前 PyTorch 默认并冻结 RNG 顺序，三个 MF 方法共用对应初始状态；不为某方法另找更有利的初始化。

### 9.3 数据重复与算法重复

**改变数据 seed 才能估计数据抽样变异。** 固定同一套观测只换三个训练 seed，不能称为有限样本估计方差。

- 主实验使用 10 套独立完整数据，root seeds 为 **2026101700 至 2026101709**。每套均有独立 FIT/CAL/EVAL，算法 seed 固定 1001。
- 第一套数据 2026101700 另加算法 seeds 1002、1003，用于该固定数据条件下的算法波动；不与 10 个 data 重复合并为 12 个同分布重复。
- 以 data replicate 为主配对单位计算 bias、variance、RMSE，并报告有限重复数的不确定性；条件算法波动独立成表。

### 9.4 预定容量诊断与训练数量

容量诊断限定在两个 family 的 **m32、rho=0.25/0.75**，共 4 个条件。三个 MF 族增加 K8 和 K32，继承各自 K16 在独立调参套选定的 lr/cap；同为 2000 步、每 250 步 checkpoint，按该重复的 CAL 样本 U-J 选 checkpoint，不访问 oracle。使用相同 10 个 data seed 和算法 seed1001，不另加算法重复或调参网格。

其含义是**固定训练规则下的容量响应**，包含优化因素，不称为每个 K 独立充分优化后的最优容量曲线。

| 部分 | 计算 | 训练次数 |
|---|---|---:|
| 主实验 | 12 条件 × 5 族 × 10 数据重复 | 600 |
| 固定首套数据的额外算法重复 | 12 × 5 × 2 | 120 |
| 容量诊断 | 4 × 3 MF族 × 2 额外 K × 10 | 240 |
| 正式与诊断合计 | | **960** |
| 独立 CAL 候选 | 12 × 21 | **252** |
| 合计，不含 smoke 或后补基线 | | **1212** |

先完成一个条件的五族 smoke 和实际全训练成本测量，再按此冻结网格分批实施。原报告以数值头代替完整 T03 训练的时间仅为粗估，不能作为实测预算。若实际资源超过原数量级，先记录新预算及可完成范围，不通过删 seed、挑条件或事后改网格掩盖差异。

### 9.5 误差分解与低成本解析诊断

分别报告：固定特征头部优化 gap、学习后 critic 的总体逼近误差、独立 EVAL 的经验 J 抽样误差，以及在同一基础数据上的排列条件方差。主 EVAL U-J 的负数保留，不截成零。

有限状态下 evaluator 可以枚举最多 m²=1024 个单元，核对

\[
\mathbb E_M(T-\eta)^2=S-J_{\mathrm{pop}}(T).
\]

另外保留一个仅用 FIT 计数的饱和表诊断：

\[
\hat p_{xy}=N_{xy}/n,\quad
\hat q^U_{xy}=\frac{N_xN_y-N_{xy}}{n(n-1)},\quad
\hat q^V_{xy}=N_xN_y/n^2,\quad
T^*_{xy}=\frac{\hat p_{xy}-\hat q_{xy}}{\hat p_{xy}+\hat q_{xy}},
\]

分母为零时取零。它作为 FIT 全表函数类的经验最优参考，不使用真实 p/q 训练，也不冒充已训练的 MF 神经特征。离散场景中 MLP 同样可以按单元频数聚合；不要以这个小支持集实验推断 MF 对任何 MLP 都有普遍计算优势。

RuLSIF 精确矩实现并核对后补入二次风险比较。非 S 输出的 InfoNCE、DV、NWJ 等不按 S 真值计算 RMSE。

## 10 服务器下一批执行顺序与完成标准

1. 将本决定转入版本化协议与配置，修正报告中梯度参考及 EPFL 阈值的措辞；保留原包和旧失败记录。更新新候选数、T03 单元表及数据敏感性规则。
2. 实现 M1 相对零头规则与日志，补充定向核对，在新代码 hash 下重跑并合并全部 T01；受影响的 M1 T02 与 BUDGET 随后更新。
3. 冻结首批公共协议、manifest、候选顺序和成本上限；开展 BF 的四族 CAL/DEV。并行编写 T03，先生成检查与 smoke，再按已定网格实施。
4. 补齐基线适配、T01/T02 和冻结附件，在同一公共协议下补跑；记录成功、失败、超预算及完整累计成本。
5. EPFL 满足图像级和宿主接入条件后单独激活；完整 V10 方法选择冻结后再启用 BF TEST_REPEAT。所有当前数据结果标为开发或重复基准。

并行开展的 T03 与基线开发使用独立 worktree 或固定提交；BF 作业始终运行已经验收的冻结代码快照。当前 code hash 覆盖 src/v10 与 scripts/v10 的全部已跟踪代码，新代码合入后按既有 provenance 规则更新验证，不沿用旧代码身份的 PASS；T02 仅重测实际受影响的路径。

继续遵守已有服务器资源限制、作业代码 pin 和 V9 保护规则；提交前读取实际队列，不把报告旧快照当实时状态。完成阶段一验收不等于已完成上述后续运行。

## 11 依据

[R1]: https://github.com/W-Yinghao/VCS_QMI/blob/15b1c548a7dbeb8ee043fecfee209b118657202b/v10/V10_PHASE1_STAGE_REPORT_20261017.md
[R2]: https://github.com/W-Yinghao/VCS_QMI/blob/15b1c548a7dbeb8ee043fecfee209b118657202b/v10/T01/TORCH_VALIDATION.json
[R3]: https://github.com/W-Yinghao/VCS_QMI/blob/15b1c548a7dbeb8ee043fecfee209b118657202b/src/v10/adapter.py
[R4]: https://github.com/W-Yinghao/VCS_QMI/blob/15b1c548a7dbeb8ee043fecfee209b118657202b/src/v10/mf_vcs.py
[R5]: https://github.com/W-Yinghao/VCS_QMI/blob/15b1c548a7dbeb8ee043fecfee209b118657202b/src/v9/reg/g1.py
[R6]: https://github.com/W-Yinghao/VCS_QMI/blob/15b1c548a7dbeb8ee043fecfee209b118657202b/src/v9/reg/r6_reg_p2.py
[R7]: https://github.com/W-Yinghao/VCS_QMI/blob/15b1c548a7dbeb8ee043fecfee209b118657202b/v10/T00/V10_DATA_MANIFEST.csv
[R8]: https://github.com/W-Yinghao/VCS_QMI/blob/15b1c548a7dbeb8ee043fecfee209b118657202b/v10/T00/EXPOSURE_NOTES.md
[R9]: https://github.com/W-Yinghao/VCS_QMI/blob/15b1c548a7dbeb8ee043fecfee209b118657202b/src/v9/reg/r6_reg_p3.py
[R10]: https://github.com/W-Yinghao/VCS_QMI/blob/15b1c548a7dbeb8ee043fecfee209b118657202b/scripts/v10/t01_validate.py
[R11]: https://github.com/W-Yinghao/VCS_QMI/blob/15b1c548a7dbeb8ee043fecfee209b118657202b/v10/T02/BUDGET.json
[P1]: https://docs.pytorch.org/docs/2.9/notes/cuda.html
[P2]: https://docs.pytorch.org/docs/2.9/generated/torch.set_float32_matmul_precision.html
[P3]: https://docs.pytorch.org/docs/2.9/cuda_environment_variables.html
[P4]: https://docs.pytorch.org/docs/2.9/notes/numerical_accuracy.html

以上 GitHub 依据均固定到同一审核 HEAD；协议中的新网格、预算上限、T03 重复与容量诊断是本次决定，尚不是测量结果。
