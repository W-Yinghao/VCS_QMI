<!-- Owner message of 2026-10-17 (summary version of V10_PHASE1_DECISIONS_20261017_CN.md), stored verbatim.  The three deliverables it links
     (V10_PHASE1_DECISIONS_20261017_CN.md, V10_PROTOCOL_DECISIONS_20261017.json, VCS_QMI_PROJECT_CONTEXT_CN.md) live in the owner's sandbox and were
     not transferred at first; the owner then uploaded two of them (2026-10-17): V10_PHASE1_DECISIONS_20261017_CN.md is byte-identical to the copy
     stored from the message (sha256 2e60f963ad0cf874...), and V10_PROTOCOL_DECISIONS_20261017.json is stored unchanged next to it as the canonical
     decision contract; v10/protocol/ is checked against that JSON. -->

我已核对 **`v10/mf-vcs@15b1c54`** 的报告、对应源码和 T00–T02 原始记录。**我接受第一阶段报告；五项主建议均采纳，第 8 节的梯度门控修改也确认接受。T03 选择 U 为主，并按下面的网格推进。**

当前测量代码 hash 为 `bf8942645f0167b0`，T01 的 **2031 个必需项全部通过**；真实宿主接入已经完成这一阶段的验收。下面的新决定需要落实到新代码与协议，旧代码的 PASS 不自动覆盖这些改动。[阶段报告](https://github.com/W-Yinghao/VCS_QMI/blob/15b1c548a7dbeb8ee043fecfee209b118657202b/v10/V10_PHASE1_STAGE_REPORT_20261017.md)

## 1. M1：采用相对容差，明确以当前批次的零头 gap 为基准

决定采用报告的 **1A**：

\[
g_{\mathrm{ref},b}=G_b(0)=\|\Delta_b\|_1,\qquad
G_b(W)\le 0.01\,g_{\mathrm{ref},b}.
\]

迭代上限 **10、30、100** 作为 CAL 维度。第一次仍从 \(W=0\) 开始，后续沿用上一批次的 W；`require_converged=False`，达到迭代上限时明确记录为近似解。

这里必须区分两个量：

- **参考 gap**：当前批次、当前特征下的 \(G_b(0)\)。
- **实际起点 gap**：warm start 的 \(G_b(W_{\mathrm{start}})\)。

**1% 乘前者。** 如果 warm start 已经很好，再要求相对它缩小 100 倍，会把停止标准变成另一套更严格、也更依赖训练历史的规则。

你指出的初始化问题成立，但“根本不训练”需要稍微收窄：T01 的初始化批次确实立即停止，W、J 和 encoder 目标梯度均为零；T02 旧规则的前 10 步也如此，第 11 步才因为批次 gap 超过 \(10^{-4}\) 开始更新。问题是**启动时刻被批次波动决定**。另外，强制 30 次迭代虽然恢复了学习，gap 只下降约 0.92%，所以新规则解决启动问题，并不证明 100 次迭代足以达到目标。[报告 §6.1](https://github.com/W-Yinghao/VCS_QMI/blob/15b1c548a7dbeb8ee043fecfee209b118657202b/v10/V10_PHASE1_STAGE_REPORT_20261017.md)

实施时还要固定三点：

1. 记录零头参考 gap、warm-start gap、最终 gap、相对 gap、迭代数、停止原因及实际成本。
2. 参考值从同一个 `QuadraticHead.delta` 取得，避免为了计算容差额外重复构建矩。
3. 精确零参考值单独处理；严格绝对容差模式继续用于数值审计。

修改后按现有 provenance 规则**重跑全部 T01 part**；T02 只重测受影响的 M1 路径和预算。

## 2. TF32：保持当前共同设置，并冻结实际环境

采用 **2A**。当前主协议继续使用已测环境：

| 项目 | 决定 |
|---|---|
| PyTorch | `2.9.0+cu128` |
| float32 matmul precision | `highest` |
| cuDNN convolution fp32 precision | `tf32` |
| 方法间要求 | 共用同一有效配置，运行时记录 |

协议不能只写“torch 默认”，因为默认行为与环境可能变化。`highest` 控制矩阵乘法精度，不表示 cuDNN 卷积也使用 IEEE；Torch 对这两者分别提供控制。[PyTorch 2.9 documentation](https://docs.pytorch.org/docs/2.9/notes/cuda.html?utm_source=chatgpt.com)

下一次补记 `TORCH_ALLOW_TF32_CUBLAS_OVERRIDE` 和 `NVIDIA_TF32_OVERRIDE`；旧记录没有覆盖这两项，不能假定为空。IEEE 卷积和 fp64 保留为命名数值诊断，当前不因此另开完整实验网格。

现有证据支持“IEEE 在这组梯度比较中误差更小”，尚不支持“TF32 是 V9 不可重复的原因”或“关闭 TF32 会改善配准表现”。[报告 §6.2](https://github.com/W-Yinghao/VCS_QMI/blob/15b1c548a7dbeb8ee043fecfee209b118657202b/v10/V10_PHASE1_STAGE_REPORT_20261017.md)

## 3. InfoNCE：主表使用共同 J 能量，名称写成 `InfoNCE_host[J_perm1]`

采用 **3A**：

- **训练**：沿用 V9 的 InfoNCE 目标。
- **配准**：使用 \(J(\tanh f)\) 的普通 perm1 能量。
- **评估**：同一冻结 checkpoint，同场景负参考，一次配准内固定排列和搜索规则。

表注明确写：

> InfoNCE 训练的 critic 使用共同 \(J(\tanh f)\) 读出进行配准；这与原包中原生目标／候选归一化的描述不同。

这符合实际 V9 路径，可以用于研究共同读出下训练目标的影响。它不能代表所有原生 InfoNCE 配准系统。[训练源码](https://github.com/W-Yinghao/VCS_QMI/blob/15b1c548a7dbeb8ee043fecfee209b118657202b/src/v9/reg/g1.py) · [配准源码](https://github.com/W-Yinghao/VCS_QMI/blob/15b1c548a7dbeb8ee043fecfee209b118657202b/src/v9/reg/r6_reg_p2.py)

原生 InfoNCE 配准能量以后作为另一命名变体，定义、核对并测量完整成本后再加入；不阻塞当前首批。

## 4. 缺失基线：分批推进，补齐之前不下比较结论

采用 **4B**。首批为：

**MF_VCS_perm1、MF_VCS_M0、MF_VCS_M1、MLP_VCS_perm1。**

M1 新规则验收、公共协议冻结以后，这四族可以先跑 BF 的 CAL/DEV。与此同时补齐：

| 方法 | 必须完成的工作 |
|---|---|
| MLP perm8 | 配准能量与对应核对 |
| InfoNCE | V10 训练和上述共同 J 配准适配 |
| MI、NMI | V10 数据、搜索及成本记录 |
| CoMIR | 原生完整流程，计入训练和 latent 生成成本 |
| RuLSIF | 所声明参考下的精确矩实现 |
| MIND-2D | 明确、可核对的二维适配 |

V9 的 RFF ridge perm1 不充当 RuLSIF 精确矩基线。每个新增适配器先做对应 T01、T02，再在同一公共协议下补跑。[报告 §6.4](https://github.com/W-Yinghao/VCS_QMI/blob/15b1c548a7dbeb8ee043fecfee209b118657202b/v10/V10_PHASE1_STAGE_REPORT_20261017.md)

**可以报告进度、失败和实际成本；必需基线补齐前，不发布方法优劣、强基线优势或新颖性结论。** 后续不得根据第一批 DEV 结果修改公共规则。

### 自然图像的调参网格现在也固定下来

首批 MF **固定 K16**，temperature=1、ridge=0。学习率在优化器学习的各模块之间绑定；M1 的 W 由 solver 求解。

| 方法 | 学习率候选 | M1 cap | 候选数 |
|---|---|---|---:|
| MF perm1 | \(10^{-4},3\times10^{-4},10^{-3}\) | — | 3 |
| M0 | 同上 | — | 3 |
| M1 | 同上 | 10、30、100 | 9 |
| MLP perm1 | 同上 | — | 3 |
| **合计** | | | **18** |

这明确修改了原来的 K×lr 网格：首批用共同 K16回答机制问题，K8/K32另记容量扩展，避免把 lr×K×cap 的 27 格藏在旧预算里。

其余固定为：3000 步、每 250 步 checkpoint；调参 seed1017，正式 seeds1001/1002/1003。CAL 按 success<42 px、success<9 px、median corner error、更早 checkpoint、固定配置顺序依次选择。

每族调参上限定为 **最多 9 个候选、8 个 L40S 已分配 GPU 小时**，训练和 CAL 都计入，报告实际花费。按旧计时重算，新网格调参约 **9.665 GPU-h**；加上四族各三个正式 seed，M1 最终 cap 为 10/30/100 时，总投影分别约 **15.36/15.85/17.56 GPU-h**。这些尚需新 M1 实测替换，也不包含缺失基线、T03、TEST、EPFL 或新增验收成本。[预算依据](https://github.com/W-Yinghao/VCS_QMI/blob/15b1c548a7dbeb8ee043fecfee209b118657202b/v10/T02/BUDGET.json)

## 5. 数据角色：DEV 先行，TEST 只作冻结后的重复基准

采用 **5A 立即生效，5B 延后生效**：

- 当前使用 **FIT32、CAL8、DEV32**，DEV32 是开发主表。
- **TEST134 当前继续 HOLD**；完整 V10 协议和方法选择冻结后，启用单独的重复基准表。
- DEV 与 TEST 不合并为独立确认结果。
- 当前 **CONFIRM=0**，如实写入结论边界。

这与已有暴露记录一致。[数据 manifest](https://github.com/W-Yinghao/VCS_QMI/blob/15b1c548a7dbeb8ee043fecfee209b118657202b/v10/T00/V10_DATA_MANIFEST.csv) · [暴露记录](https://github.com/W-Yinghao/VCS_QMI/blob/15b1c548a7dbeb8ee043fecfee209b118657202b/v10/T00/EXPOSURE_NOTES.md)

我还增加一项**预登记的同位敏感性分析**：排除与 FIT/CAL 同 TMA 位置的评估 core，得到 **DEV24**，以及未来的 **TEST112**。这些数量已经从 manifest 独立重算。跨切片同位是否同一 specimen 尚未证实，所以这是保守敏感性假设；不擅自改变原始划分，也不把每个 core 的多个扰动当独立组织样本。

**EPFL 要完成图像级复核后再激活。** 除适配器和 T01 外，还要重放既定排除与窗口构造、核对代表图及异常、检查近重复关系，并实测完整成本。

报告中的“约 2 native px 残差”也要收窄：P3 的整数相关检查不是真实对齐误差小于 2 像素的证明。2.25 px 严格阈值应解释为**已知施加扰动的恢复误差**。[P3 实现](https://github.com/W-Yinghao/VCS_QMI/blob/15b1c548a7dbeb8ee043fecfee209b118657202b/src/v9/reg/r6_reg_p3.py)

## 6. 第 8 节：确认修改，但门控对象应写成“梯度”

**确认接受 `a8f4c5d` 的修改。**

我核对了实现：被测分块/checkpoint 路径仍然实际运行 fp32；参考由**相同 fp32 舍入输入和权重转为 double 后重新计算梯度**。不是把被测路径改成 fp64 后再声称 fp32 通过。

CPU 上的关键结果为：

| 比较 | 最大绝对梯度误差 | 决定 |
|---|---:|---|
| fp32 全图 vs fp64 | \(2.91\times10^{-5}\) | 保留失败诊断 |
| fp32 分块/checkpoint vs fp64 | \(2.94\times10^{-7}\) | 作为必需门控 |

fp32 容差仍是 atol=\(10^{-5}\)、rtol=\(10^{-4}\)；fp64 容差不变。fp64 分块对全图梯度、目标函数值比较，以及其他数学和几何门控继续保留。[门控源码](https://github.com/W-Yinghao/VCS_QMI/blob/15b1c548a7dbeb8ee043fecfee209b118657202b/scripts/v10/t01_validate.py)

报告的“fp64 值”应改成 **“同输入和同权重的 fp64 梯度参考”**。

接受的边界也明确：CPU fp32 全图 dense-MLP 仍是已知超界路径，旧失败不能删除。当前正式单元不用它，可以继续；以后若用于正式科学运行，就必须通过必需验收，或提高积累精度并计入成本。

## 7. T03：U 主模式，独立数据重复，采用以下完整网格

### 目标模式

**U 为主。** MF 精确矩用 U；perm1/perm8 也必须使用**均匀无固定点排列**。普通 `randperm` 的期望对应 V，不能给它贴 U 标签。

V 只做**同一个冻结 critic 上的 U/V 评价诊断**，当前不另开完整 V 训练网格。自然配准继续使用同场景 V。

### 主网格

| 项目 | 决定 |
|---|---|
| 条件 | two_block、diagonal_mixture × m8/32 × rho0/.25/.75，共12条件 |
| FIT/CAL/EVAL | 每条件4096/2048/8192个独立基本观测 |
| 主方法 | MF perm1、M0、M1、MLP perm1、MLP perm8 |
| MF 宽度 | 主表K16 |
| 输入 | 整数one-hot；MF使用包内双侧Linear+softmax |
| MLP | 两层64维GELU隐藏层，\(T=\tanh f\) |
| 训练 | 2000个全批步，每250步checkpoint |
| 学习率 | \(10^{-4},3\times10^{-4},10^{-3}\) |
| M1 cap | 10、30、100 |
| CAL | 只按独立CAL样本的完整经验U-J选择，truth仅交评估器 |

调参使用独立数据 root seed1017、算法 seed1017。每条件 **21 个候选**，合计 **252 次 CAL 训练**。

### 重复方式

我不采用“同一套数据只换三个训练 seed”来估计抽样方差。决定是：

- **10套独立完整数据**：root seeds为2026101700至2026101709，算法seed固定1001。
- **第一套数据**另加算法seeds1002、1003，单独估计固定数据下的算法波动。
- 两种重复分别报告，不合成12个同分布重复。

容量诊断仅放在两种family的 **m32、rho=.25/.75** 四个条件上：三个MF族追加K8/K32，继承对应K16由调参套选定的lr/cap，沿用相同训练步数及CAL规则。它回答固定训练规则下的容量响应，包含优化因素。

| 部分 | 训练次数 |
|---|---:|
| 主实验：12条件×5族×10套数据 | 600 |
| 首套数据额外两个算法seed | 120 |
| 预定容量诊断 | 240 |
| **正式与诊断合计** | **960** |
| 独立CAL候选 | **252** |

先做一个条件的五族 smoke，测完整训练成本，再按冻结网格分批实施。评估保留负的经验 J，区分头部优化 gap、总体 critic 误差和独立样本估计波动；不能把三者归为同一种“估计误差”。

## 下一步执行顺序与交付文件

服务器端可以按这个顺序继续：**落实协议与 M1 修改 → 新代码下 T01 → 受影响的 T02 与预算 → 冻结后跑 BF 四族，同时开展 T03 smoke → 基线核对后补跑。**

并行开发 T03 和基线时使用独立工作树或固定提交。当前代码 hash 覆盖 `src/v10`、`scripts/v10` 全部已跟踪代码；在同一树继续加代码，会使旧验收身份和已排队作业 pin 过期。

完整决定、参数契约和项目接续记录已完成：

- :chatgpt-content-reference{index="2"}[V10_PHASE1_DECISIONS_20261017_CN.md](sandbox:/workspace/scratch/28c8bcecf9af/deliverables/V10_PHASE1_DECISIONS_20261017_CN.md)
- :chatgpt-content-reference{index="3"}[V10_PROTOCOL_DECISIONS_20261017.json](sandbox:/workspace/scratch/28c8bcecf9af/deliverables/V10_PROTOCOL_DECISIONS_20261017.json)
- :chatgpt-content-reference{index="4"}[VCS_QMI_PROJECT_CONTEXT_CN.md](sandbox:/workspace/scratch/28c8bcecf9af/deliverables/VCS_QMI_PROJECT_CONTEXT_CN.md)

本轮完成的是**审核与协议决定**；JSON 明确标为待服务器实施的决定契约。后续代码修改和训练作业尚未执行。
