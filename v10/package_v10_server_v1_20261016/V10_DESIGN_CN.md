# V10 server 端设计与执行交接包 v1

日期：2026-10-16。项目：VCS_QMI。方法工作名：Moment-Factorized VCS（MF-VCS）。

## 0. 本次恢复到哪里，交付覆盖什么

本包接续 ChatGPT 对话「系统学习变分互信息」。已恢复的最后明确决定是：2026-10-16 的 V10 v0.1 方法设计被接受，随后要求继续补齐数据集、实现、对比工作与 benchmark，准备完整 server 端包。原包只有设计稿、数值参考、19项历史检查记录、来源和校验文件，尚无真实宿主接入。

本版给出可交给 server agent 的完整实施顺序、数值后端、验证/计时入口、受控数据生成器、实验配置、数据与结果表结构。服务器需要在现有仓库中完成宿主 adapter，读取真实 manifest，并执行 PyTorch/GPU 和自然图像实验。**本版的交付状态是设计与数值工具交接完成，真实宿主适配和自然实验尚未执行。**

| 项目 | 本轮状态 | 使用边界 |
|---|---|---|
| V10 v0.1 设计与参考 | 已恢复原始文件 | 保留于 provenance/，不把旧记录写成本轮重跑 |
| MF-VCS 核心 PyTorch 源码 | 已提供 | 当前环境没有 PyTorch；实际执行状态见 VALIDATION_REPORT_CN.md |
| 独立 NumPy/SciPy 数学核对 | 已执行，结果附包 | 不替代 PyTorch、CUDA、连续几何梯度验证 |
| T00–T05 任务、配置、比较单元 | 本版已具体设计 | 单元表是 planned ledger，不是作业完成表 |
| BF–SHG / EPFL 官方来源及旧协议 | 已核对 | 官方样本数不等于服务器实际使用数 |
| 当前 GitHub/服务器提交与 V9 状态 | 本轮未重新取得 | 由 T00 只读预检填入，不沿用旧提交当 current HEAD |
| 图像训练、配准结果、GPU速度 | 未执行 | 不能据本包声称性能、加速倍数或新颖性成立 |

本轮网页与 Git 入口未取得 VCS_QMI 实际源码；恢复的 V9/R6 包也没有 g1.py、r6_reg_pilot.py 或已填 manifest。因此本包明确给出 adapter 契约与核验要求，没有猜造宿主函数签名。宿主历史路径为 `src/v9/reg/g1.py` 和 `src/v9/reg/r6_reg_pilot.py`，历史提交 `27117c9ae78d64d39a6548c40b00ff58f70cc4d4`。[P0–P3]

## 1. 保留的理论与候选贡献

保持合作者原始对象：

\[
M=(P+Q)/2,\quad \eta=(p-q)/(p+q),\quad S=E_M\eta^2,
\]
\[
J(T)=E_PT-E_QT-\tfrac12E_PT^2-\tfrac12E_QT^2,
\qquad S-J(T)=E_M(T-\eta)^2.
\]

`S` 不是 Shannon MI。原来的混合参考 CS 变换仍为 `2 log((1+S)/(1-S))`。训练使用原始 J；不通过额外损失、重新定义 S 或换任务标签制造新的泛函。

新实现的结构为

\[
a(x)\in\Delta^{r-1},\quad b(y)\in\Delta^{s-1},\quad
T(x,y)=a(x)^T Wb(y),\quad W\in[-1,1]^{r\times s}.
\]

T 是 W 元素的凸组合，因此全域有界。a/b 是可学习的单纯形特征，不解释为语义类别或真实后验；W 不要求对称、正定。有限 r/s 有表示误差，去掉输出 tanh 也不继承旧 tanh 的抗污染结论。[P0]

**本轮要验证的主问题：在同样的独立数据与计算预算下，这个结构能否通过精确负对矩计算和可诊断的头部求解，得到更好的误差—成本与配准—成本曲线。**代数等价只支持实现正确性；任务收益需要实验。

### 1.1 与先行工作的逐式关系

原 J 已与平衡 RPC 目标重合，不能把它登记成两种独立损失。进一步，对 RuLSIF 的半混合参考，令 `g=1+T`，去掉不影响优化的常数后的相对最小二乘风险满足

\[
\widetilde R(g)=\tfrac12E_Mg^2-E_Pg,
\qquad \widetilde R(1+T)=-\tfrac12-\tfrac12J(T).
\]

这是直接展开得到的等价关系。因此 RuLSIF 不仅是泛泛的近邻；相同参考下的基础二次风险也有仿射等价。不同方法的函数类、约束、核、正则、负分布和数值求解仍可能不同。正式 novelty ledger 必须写清这一点。[L1,L2]

谱对比学习已经有双线性关系和平方负项，variable projection 与 Frank–Wolfe gap 也有成熟先行工作。候选新增内容应落在具体的有界函数类、对应的精确负对计算、可行头部求解，以及实测结果上。不能使用“首次二次头”“首次传矩不传 pair”“首次神经配准”等表述。[P0,L3]

## 2. 负分布与矩计算的实现规格

设某一场景中的配对特征为 A、B，各有 n 行，`t_i = a_i^T W b_i`；`u=sum_i a_i`、`v=sum_i b_i`、`G_A=A^T A`、`G_B=B^T B`。

| mode | q1 | q2 | 对应抽样 |
|---|---|---|---|
| V | `u^T W v / n²` | `tr(W^T G_A W G_B) / n²` | 普通随机 permutation 的条件平均，包含固定点 |
| U | `(u^T W v - sum t_i)/(n(n-1))` | `(tr(...) - sum t_i²)/(n(n-1))` | 排除对角的完整负对；n必须大于1 |
| 独立边缘池 | 两池分别求矩，分母 `n_x n_y` | 同前，使用独立池 | 没有理由仅因同号索引就删掉一项 |

所有模式中 `J=mean(t-.5t²)-q1-.5q2`。产品矩与显式 pair 必须使用完全相同的正对、场景、权重与归一化。

### 2.1 G1 的主接入固定为 V

恢复的 V10 设计记录：历史 G1 使用 `permutation(n_pos)`，允许固定点；污染正对的空间排除不等于负对的对角排除。**本版配准默认 `mode=v`。**服务器先在当前源码确认这一点；若当前实现已经变化，先登记 target 差异，再使全部方法使用相同的新目标。[P0]

原参考 `off=True` 默认 U 只适合其数学示例，不能直接当 G1 默认。本版后端明确模式。U 保留为受控 iid 估计实验和命名清楚的目标敏感性分析，不能悄悄替换 V 来提高结果。

### 2.2 按场景计算，保留条件参考

对于自然配准，Q 是“先选择场景，再在同一场景独立取两侧位置”的混合。逐场景计算 J，再用锁定的场景权重相加；不把所有图像的特征拼起来做全局乘积。

若场景 ID 不作为 critic 输入，优化的是混合后的可观测 P/Q。它一般既不是 global `P_X P_Y` 的 QMI，也不等于允许每个场景任意独立最优 critic 的条件 S 平均。文中应直接称同场景参考下的 J / CS discrepancy，避免偷换对象。

本版快速后端支持 V、U、独立边缘池及固定非负场景权重。**任意空间禁止边没有在生产后端实现。**遇到这类宿主条件时，先使用精确 dense 参考或专门补齐禁止集合的加权扣除与相同归一化，再做梯度核对。不能以忽略禁止边的方式“保持快速”。

### 2.3 数值与复杂度

K=r=s 时，一次固定头部求值/梯度约 `O(nK²+K³)`；显式 pair 约 `O(nK²+n²K)`。这没有计入图像 encoder、warp 或 M1 多次头部迭代。K增大、GPU kernel 吞吐和小矩阵开销都可能缩小收益。

fp32 中 U 的大项相减需要实测；小型正确性用 fp64。不要把负 q2 直接 clamp 后宣称等价；超出误差阈值应保留原值并定位。W每步维持可行；a/b 必须有限、非负且行和为1。训练保持 raw J，不将负 J 截为0。若展示 CS 变换时需要裁剪，原读数与裁剪比例一并保存。

## 3. M0 / M1 与真实图像接口

### 3.1 M0 为首个完整训练版本

从真实宿主提取两侧局部 encoder 输出，在两侧增加投影与 softmax，生成 K维单纯形特征。原 encoder 的输入、感受野、采样、预处理与训练角色保持一致。encoder、feature maps 和 W 均训练；W的 AdamW weight decay 设为0，更新后投影到盒内。其他可训练模块的正则与宿主一致并记录。

接口 smoke 用 K=16、温度1、V 形式；这些是本版建议的起始配置，不声称它们是历史宿主配置。正式 K/学习率由 CAL 按第7节的对称预算选择。

### 3.2 M1 是同一经验目标上的 profiled head

固定本次 minibatch 实际生成的特征 tensors，用矩 HVP/梯度求解盒约束凹二次问题。默认 ridge=0。然后固定求得的 W，在**同一 tensors、同一场景权重、同一目标**上更新 feature maps 与 encoder。内层不重新抽增强、dropout、位置或负池，也不更新 BatchNorm 状态。

若使用 `F=J-lambda||W||²/2`，内外层都采用相同 F；同时报告未正则 J、lambda 和 gap。不能内层拟合 FIT、外层换独立 CAL 后继续使用未经推导的 stop-gradient。不能对刚更新的 encoder 使用上一轮过期矩。

对可行 W，令 g 为当前 F 的 W梯度，盒的一阶 gap 为

\[
\mathcal G(W)=\sum_{kl}|g_{kl}|-\langle g,W\rangle.
\]

它上界固定特征、当前经验问题的优化不足。迭代到上限仍未达到阈值，记录实际 gap 和 `converged=false`。头部 gap 小不代表特征类好、总体 S估得准或整个网络全局最优。

有限容差下的M1外层更新仍是近似profiled更新；仅有目标gap不自动得到外层梯度误差界。无ridge时最优W也可能不唯一，需按Danskin的条件理解方向导数，不能直接宣称光滑、唯一或精确包络梯度。

**最终W的规则锁定如下：**主M1保存每个评估checkpoint当时的encoder/feature maps与W，作为不可拆分的一对由CAL选择。W来自最后一次已完成训练batch的内层求解，随后与该次outer更新后的encoder一起保存；它不是全FIT的最优关系头。内层gap记录的scope为`inner_pre_outer_batch`，不能当作更新后checkpoint或全FIT的证书。主协议不额外做最终FIT refit。若研究冻结encoder后的全FIT共享头重拟合，另命名`FIT_refit`变体、固定FIT reference pool和场景权重、同样给M0对照，并计入求解与选模成本；不得按测试表现选择哪个W。

### 3.3 共同能量表的几何评估契约

以下规则适用于MF/MLP/InfoNCE等沿用宿主共同搜索器的能量。CoMIR等原生系统遵循第5节的原生流程，并在配置中单独覆盖几何顺序。

1. 测试时 encoder、feature maps 和 W 全部固定；每个候选变换都不重新拟合关系头。
2. MF/MLP/InfoNCE共同表保留 `warp-then-encode`。将MF改为`encode-then-warp`需要另命名、核验近似与成本；CoMIR原生的先生成共同表示再配准，按其原生系统定义执行。
3. P与Q的 moving 侧均经过同一个变换，梯度必须穿过正项和负项的一二阶统计。
4. 先复用并记录真实宿主的支持、可见域、固定位置和越界处理规则，各方法保持一致。若新增“全部候选变换有效支持的交集”，必须登记为同步应用于所有方法的协议变化，核对交集非空、P/Q和有效样本数，并区分旧结果的可比性；padding或重叠处理不能成为方法特有线索。
5. 当前旧能量在一次搜索中使用固定 permutation；V10去除的是初始 permutation 选择依赖，不是修复不存在的每步重抽bug。
6. 写清像素中心、方向、旋转中心、角度单位、compose顺序、插值、align_corners与native坐标换算。

详细接入任务见 `docs/HOST_ADAPTER_CN.md`。只有在当前源码上通过值、A/B/W梯度和连续几何梯度核对后，才能写 `host_adapter_verified=true`。

## 4. 两个自然数据协议

### 4.1 BF–SHG：主数据

官方 full-size TMA-core 版本记录206对2048×2048图像，角色为训练40、训练调参验证25、配准调参验证7、测试134。官方页只支持core/image-pair级数量，不足以称206名患者或证明patient-disjoint。[D1]

**实际协议沿用服务器已有BF–SHG版本、原图ID、物理样本映射与R6/V9划分。**若用的是裁剪版本，必须记录与full-core的父ID关系，不能把两版本当作独立数据。所有同源patch、窗口和重复视野继承同一个group与角色。若患者关系不可恢复，统计单位写可核验的core/scene，明确其限制。

沿用已冻结的刚体扰动和角点误差定义，BF–SHG主阈值42 native px、严格阈值9 native px；服务器核对缩放后再执行，不能将网络分辨率像素混用。[P1]

本版主表先做全量FIT、干净对应、完整encoder训练。训练数据25%/100%是估计—数据预算扩展；不把V9的20%错配轴自动移来当V10主故事。

### 4.2 EPFL RGB–NIR Scene Dataset：第二场景

官方页列477个RGB/NIR场景条目、9类，下载图像为1024×768 TIFF。原始配准使用SIFT、RANSAC similarity transform及最小二乘重估；它不提供原始图像严格逐像素对齐真值。EPFL RGB–NIR与CoMIR使用的Zurich Summer航拍数据是不同数据源。[D2,L4]

沿用项目已有场景/窗口映射、33个历史排除及约2px分辨限制，先恢复排除ID与原因；不依据V10表现重新挑图。原图的多个窗口和近重复不得跨角色。继承EPFL主/严格阈值10.5/2.25 native px，单独分表。[P1]

本任务恢复的是对已近似配准图像施加的已知人工刚体扰动；人工变换恢复正确不等于消除了真实传感器残余错位。严格阈值接近既有对齐限制，应连同残余误差解释。

### 4.3 确认来源的决定

V9已读或参与方法设计的集合，在V10仍是开发/重复benchmark。新训练seed不能恢复数据的未见性。EPFL也已有项目历史，不自动获得“全新第二数据集”的资格。

server先填写 `DATA_MANIFEST_TEMPLATE.csv` 的group、原始role、exposure和source_version，并维护暴露说明。只有有证据的未读独立物理/scene组才能标CONFIRM；否则本轮仍可完成开发比较，结论标为开发结果。若现存数据没有新的独立组，报告该事实，不在本包内凭空指定确认来源。

## 5. 对比方法与归因矩阵

| 方法/比较 | 保持的因素 | 改变的因素 | 回答的问题 |
|---|---|---|---|
| 原MLP+原J，perm1 | 历史encoder/critic与目标 | 无 | 当前强度与实际成本 |
| 原MLP+原J，perm8/完整分块 | 函数类和目标 | 负对计算预算 | 原方法增加预算后能达到什么水平 |
| bounded bilinear，perm1 | P/Q、encoder输入与预算规则 | critic函数类 | 新结构本身的代价与表达力 |
| bounded bilinear，moments，M0 | 同一bilinear类和优化方式 | 精确配对利用 | 去掉配对抽样后的变化 |
| bounded bilinear，moments，M1 | 同一类、数据、经验目标 | 头部求解程度与总成本 | 更充分求解是否值得 |
| InfoNCE宿主实现 | 相同局部观测与对应数据 | 原生目标/候选归一化 | 成熟非CS学习目标的任务表现 |
| RuLSIF/核相对比 | 相同参考和样本机会 | 核函数类与求解 | 二次风险的强传统路线 |
| CoMIR | 相同允许训练数据 | 原生表示/训练/配准流程 | 外部完整系统 |
| MI、NMI、MIND 2D适配 | 相同图像与几何真值 | 统计/描述子能量 | 非学习或经典强基线 |

MLP→bilinear不能只归因为“更好的负样本”；perm→moments才隔离配对计算；M0→M1才隔离求解。dense与moments在固定输入下等价，不能把它们重复训练后当两个独立算法，再期待神秘精度差。

匹配BCE、学习型核和共同局部观测上的CS-KDE保留为完整估计/归因对照。BCE不是整个工作必须先跨过的唯一门槛。学习型核不能被默认为固定弱特征。传统原生CS散度与混合参考S不同，原生真值与共同S评价分表。[P0,L2,L3]

核/谱对照也应使用合理的精确矩优化。例如可分核基函数`g(x,y)=sum_l c_l k_x(x,x_l)k_y(y,y_l)`，令两侧基函数矩阵为U、V，则完整乘积负项为`c^T[(U^T U) ⊙ (V^T V)]c/(n_x n_y)`。同池排除对角时再扣同号项并改分母。不能让MF使用矩计算，却强制核基线枚举全部pair。固定核和学习型核的表示、训练成本分别登记；共同S读数是明确适配，不能冒称原生MMD方法也估计S。[L2,L3]

具体地，核的U形式为`q1_U=(n² q1_V-sum_i g_ii)/(n(n-1))`，`q2_U=(n² q2_V-sum_i g_ii²)/(n(n-1))`；独立池保持`n_x n_y`。转换`T=g-1`时，一阶矩减1，二阶矩为`q2(g)-2q1(g)+1`。如果再逐点clip g到[0,2]，通常会破坏可分矩快捷式，clipped变体需要显式求值或新的严格推导。

主应用输出两张表：

- **共享搜索器表：**相同输入支持、初值、变换族和优化预算，比较可兼容的能量；核/MLP不能偷用额外对齐信息。
- **原生完整系统表：**CoMIR、MI/NMI、适用的MIND使用各自合理流程，报告完整成本。预训练来源与权重是否见过评价图像单列。

CoMIR本来训练每模态网络生成稠密共同表示并接配准器，不能仅把其名字贴到一个普通InfoNCE critic上。[L4] MIND若采用2D offsets/patch适配，明确标注本项目适配，不抄原3D论文数字。单纯边缘MMD对位置排列可不敏感，不能不加定义就当配准能量。

## 6. T00–T05 的执行顺序与产物

| 任务 | 具体输入/动作 | 必交产物 | 进入下一步的依据 |
|---|---|---|---|
| T00 恢复实际宿主 | 当前HEAD、两份reg源码、V9当前状态、manifest、现有资源 | `PREFLIGHT.json`、`HOST_ADAPTER_AUDIT.md`、来源/暴露说明 | 目标、接口和角色可核验；不要求方法先赢 |
| T01 数学和接入核对 | V/U/独立池/多场景；值、W/A/B梯度；小型dense-QP；连续warp梯度 | `TORCH_VALIDATION.json`、`GEOMETRY_GRADIENT_CHECK.json` | 相同对象下误差达阈值，边界条件明确 |
| T02 计算实验 | 同A/B/W，perm1/8、dense、重计算分块dense、moments | 长表耗时/显存/误差/抽样方差，encoder-inclusive补表 | 得到真实预算，记录OOM，不按输赢删格 |
| T03 受控真值 | 新独立离散样本，后续Gaussian/cubic与多峰 | MSE、S偏差/方差/RMSE、gap、时间曲线 | 区分表示误差、数据误差、求解误差 |
| T04 完整训练 | BF–SHG主表，EPFL审计后；机制矩阵与外部学习对照 | checkpoints、训练曲线、CAL选型与成本 | 完成预定单元，不只保存最好seed |
| T05 完整配准 | 同一扰动/支持/搜索；原生系统另表 | success/median/p90、失败、耗时、每图记录与能量剖面 | 给出实际结论及统计范围 |

### 6.1 T01 的可执行验收

先跑小尺寸fp64 CPU/PyTorch，再跑同尺寸CUDA fp64/32。比较原始J、A/B/W梯度、矩HVP与dense HVP。小型默认 `atol=1e-10, rtol=1e-8`（fp64），`atol=1e-5, rtol=1e-4`（fp32）；接近零梯度采用绝对误差，不用纯相对误差。

对真实warp，固定小图/特征网络、位置、W与有效支持，用双精度中心有限差分核对平移/旋转梯度；采样点避开插值折点，至少两个差分步长核验稳定区间。dense和moments都要经过完整warp链。再覆盖零特征差异、n=1的V可用/U拒绝、不等场景n、独立池、不可行W、非simplex/NaN输入、正则与无正则头。

这些阈值用于定位实现错误，不能放宽到恰好让新方法通过。若大规模fp32消减错误超限，使用显式更高精度积累并把成本计入。

### 6.2 T02 的实际计时

初始计算格：N=128/512/2048/8192；K=8/16/32。N是每场景位置数，K是simplex宽度，permutation重复数另记R=1/8，避免符号混淆。脚本默认只跑smoke；完整表显式打开。

每格同一A/B/W、同一个V目标、相同dtype/device；先warmup，同步CUDA，再统计前向+反向的中位数及分位数，记录峰值allocated/reserved、实际重复数和失败。perm与moments值/梯度不同属于抽样误差，不要求逐次相同；另固定模型重复permutation报告条件方差。dense/分块dense与moments应在阈值内一致。

真实流式或checkpoint分块dense是必要成本对照；简单分块但保留全部反向score图不自动消除N²显存。要注明用到的重计算及反向成本。OOM记录为该配置受资源限制，不静默减N或改dtype。

三层计时分别为：数值头；包含两个encoder/warp的完整训练步；完整注册。M1另报求解迭代、gap、重算矩次数。训练、调参、注册均计成本，不只报fast inner kernel。

### 6.3 T03 的真值与选择

包内首先实现balanced two-block与diagonal-mixture的有限可观察表：m=8/32，rho=0/.25/.75；p/q/eta/S可以精确计算。FIT/CAL/EVAL分别使用独立随机流，truth单独保存在评估目录，训练器不读取。

主指标为 `E_M(T-eta)²`、独立评价J对S的bias/variance/RMSE、同一基础数据内permutation方差、求解gap与总时间。增大完整pair数不增大独立样本数。CAL以独立经验J/明确的任务指标选型，oracle不参与调参。不同方法的随机流、重复数和基本样本预算保持对称。

后续Gaussian、逐坐标可逆cubic与多峰是明确定义的扩展，不在本轮工具中冒充已实现。若使用蒙特卡洛S，必须报告独立truth样本量和标准误；不能把精确潜变量S当作不可恢复观测的真值。MINE/DV/NWJ/SMILE/InfoNCE以各自原生量和共同下游任务比较，不用S真值给非S输出算RMSE。

## 7. 预算、调参和实验单元

本包不填写虚构GPU小时。先用T02和每种方法的短训练片段实测，记录本服务器已允许的资源，形成 `BUDGET.json`：设备、可用显存、并发、每单元步数/秒数、预计总时长、现有V9资源占用。只读记录现有V9状态，不取消或重算其任务。

主训练seed固定为1001/1002/1003；调参seed为1017。seed更换用于分开新训练轨迹，不代表数据是新确认。各方法共享可共享的encoder初始化、数据顺序与扰动ID；架构不同部分单独记录。

初始调参上限每个方法族9个候选，且附相同总调参时间预算。MF的K∈{8,16,32}与feature/head lr∈{1e-4,3e-4,1e-3}构成9格；encoder lr先继承真实宿主并记录。核方法得到同等机会调带宽/正则；InfoNCE调原生温度/学习率；CoMIR保留官方合理配方并给出对等机会。若两种公平标准无法同时满足，分别报告same-data和same-total-cost表。

M1主ridge=0，额外lambda∈{1e-6,1e-4}仅作显式敏感性。max_iter∈{10,30,100}的求解成本也算调参；复用M0已选结构只能减少搜索成本，不能把它当作免费额外信息。gap目标1e-4是起始工程容差，达到iteration cap时报告不足，不硬称精确包络梯度。

为落实这一有限预算模式，M1训练调用显式设置`require_converged=False`并逐批保存gap/收敛状态；这是预算内近似profiled head。严格数学验收使用默认`require_converged=True`。服务器不得遇到未收敛后静默改变最大迭代数、丢弃失败批次或把近似模式写成精确求解。

`make_units.py`生成所有计划单元；它不发作业。EPFL和完整训练激活取决于数据/资源就绪，不取决于新方法在BF上是否获胜。未完成单元写incomplete；不能把缺失作业当配准失败，也不能从表中悄悄删除。

## 8. 评价、统计与可展示结果

主结果是几何成功率与误差—成本曲线，不是训练J。分别报告主阈值与严格阈值成功率、角点误差median/p90、旋转/平移误差、注册时间与能量调用数。数值失败、超时、不收敛均保留在完成试验的成功率分母；median/p90若仅对成功样本计算，必须明确命名，另报全样本失败尾部。

独立统计单位为可核验的物理specimen/core或scene group。同组多个patch、扰动和同一模型的重复注册是条件重复。先在组内对预定trial与seed求平均，再对组汇总。训练seed的离散程度另报；按group bootstrap的CI只解释为对这批训练seed条件下的数据组不确定性，不能冒充3个seed提供了大量模型重复。

推荐2000次成组bootstrap、固定seed10191；比较差值使用共同group/扰动配对。主要比较预定为 `MF_M0 vs MF_perm1`（配对利用）与 `MF_M1 vs MF_M0`（求解），外部任务表完整列MI/CoMIR/RuLSIF等。如果给多项显著性主张，应预先锁家族和校正；探索项直接给估计与CI，不靠挑p值定故事。

每图保存：输入ID、group、已知扰动、估计变换、native误差、失败原因、checkpoint/hash、manifest/hash、目标mode、总时间。可视化固定挑选规则（典型、中位、尾部和全部失败索引），展示对应位置/变换叠图与能量剖面，不能只挑最好图。

能量与搜索错误分开：若真变换附近能量不高，属于能量/可识别性问题；能量正确但搜索走错属于优化问题。更低固定变换配对方差不自动保证能量在真变换处最大。

## 9. 根据结果作出的决定

- dense与moments不等价：先修目标/梯度/归一化，不能进入性能比较。
- 矩计算更快、总体MSE无改善：保留计算结论，检查特征类逼近误差，不把速度写成精度贡献。
- gap更小但M1同预算更差：M0作为主方法，M1作为负结果或诊断。
- MF函数类明显弱于MLP/强核：报告结构与表达力的折衷；不能只削弱外部基线。
- 仅J改善、配准不改善：结论限定于能量/估计层，不称任务提升。
- 组合被先行方法完整覆盖且无新增有效结果：降为实现或分析贡献，不重新命名认领算法。

## 10. 交付后server需要回传的最小结果

先回传当前 `PREFLIGHT.json`、宿主接口映射、数据暴露/划分说明、T01执行结果和T02实测小表。随后按已记录资源继续T03–T05，回传单元ledger、方法差异登记、完整训练/注册长表与每项失败。不要仅回一句“全部PASS”，不要把任务表中的planned改成executed却没有日志、checkpoint或结果文件。

完整公式与早期19项检查记录见 `provenance/V10_DESIGN_v0_1_CN.md` 和原始结果；本轮实际验证见 `VALIDATION_REPORT_CN.md`。官方来源核验范围见 `docs/PROTOCOL_SOURCES_CN.md`。

## 来源

- [P0] 恢复的2026-10-16 V10_DESIGN_CN.md、v10_reference.py及原始ZIP；本包provenance目录保留。
- [P1] V9 server v1，docs/06_REG_EXTENSION_CN.md、07_DATA_AND_BENCHMARKS_CN.md；本轮读取的历史协议，不是当前运行状态。
- [P2] V81/V9 SERVER_UPDATE_CN.md；它自述为旧版补充件，不是V10。
- [P3] R6 New Applications的04_REGISTRATION_CN.md与reference/README；旧设计建议不当作实际宿主函数签名。
- [D1] BF–SHG full-size TMA cores 官方数据页：https://zenodo.org/records/4550300 。核验数量、分辨率和4个角色，未核验patient-disjoint。
- [D2] EPFL RGB–NIR 官方数据页：https://ivrlwww.epfl.ch/supplementary_material/cvpr11/index.html 。核验场景条目、9类、分辨率与原始配准方法。
- [L1] RPC：https://arxiv.org/abs/2103.11275 。原J重合关系见恢复稿；本轮补查官方论文入口。
- [L2] RuLSIF：https://arxiv.org/html/1106.4729v1 。相对密度比与二次风险来自原文；本版的g=1+T关系为显式代数推导。
- [L3] Spectral contrastive：https://arxiv.org/abs/2106.04156 。二次/双线性近邻，非声称相同完整J。
- [L4] CoMIR论文：https://arxiv.org/abs/2006.06325 ，官方代码：https://github.com/MIDA-group/CoMIR 。原生表示与配准流程不同于普通局部InfoNCE critic。
