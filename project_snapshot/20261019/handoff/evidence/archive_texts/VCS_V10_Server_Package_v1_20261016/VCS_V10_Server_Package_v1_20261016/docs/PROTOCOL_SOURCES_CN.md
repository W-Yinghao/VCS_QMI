# V10 MF-VCS：真实应用协议、基线与先行工作核验

核验日期：2026-10-16。本文为 server 包的协议附件；状态是**设计与公开来源核验完成，数据下载、宿主实物核验、模型训练、GPU 计时和确认实验均未在本文完成**。

本文接续已接受的 V10 方法定义，不调整合作者的混合参考 M、原始 J、S 及 CS 变换。项目执行事实以服务器冻结 manifest、版本化配置和日志为准。官方数据集介绍只能说明公开来源，不能证明服务器当前使用了同一个版本、尺寸或划分。

## 1. 可立即锁定的决策

1. BF–SHG 是第一自然应用；沿用 R6/G1 宿主的物理来源分组、图像、固定刚体扰动、CAL 规则、搜索器和 native-pixel 指标。曾参与 V9/R6 设计的图像保持 development/repeated-benchmark 属性。
2. “EPFL”必须写为 **EPFL RGB–NIR Scene Dataset（Brown–Süsstrunk, CVPR 2011）**。它不是 CoMIR 使用的 Zurich Summer 航拍集。项目历史已提到 EPFL 的 33 个旧排除和约 2 px 对齐精度限制，因此不能把它自动宣布为新独立确认集。
3. V10 第一训练机制表比较原 MLP、结构化 critic、全负对矩计算和内层求解。所有神经方法都训练完整 encoder；固定特征实验仅用于计算和估计误差隔离。
4. 共同搜索器表保留 warp-then-encode；测试过程中固定 W、encoder 和归一化状态，不在每个候选变换上重新拟合 critic。
5. 同时做外部原生系统表：CoMIR 的表示生成后配准流程、合理调参的 MI/NMI 多分辨流程，以及注明实现来源的 MIND 系统。共同搜索器表与原生系统表不能互相替代。
6. RPC 的平衡设置与 J 完全相同；RuLSIF 在 alpha=1/2、g=1+T 时的未正则最小二乘风险也是 J 的仿射改写。V10 的贡献候选应落在有界可分函数类、完整乘积计算和求解—成本结果，而不能认领基础二次目标。

本节 1–5 为根据本项目历史形成的执行约束与建议；第 6 项包含文献目标核验和本文代数推导。具体依据见后文。

## 2. 数据身份：公开事实与本项目状态分开登记

### 2.1 BF–SHG 的两个官方入口

| 公开来源 | 已核实事实 | 不能据此推定 |
|---|---|---|
| CoMIR NeurIPS 2020 正文 [D1] | 206 对 BF/SHG TMA core 图像；原图 2048×2048；论文使用 834×834 中心区域 | 服务器当前图像尺寸、实际保留数、患者人数 |
| Full-size TMA Cores，Zenodo 4550300 [D2] | 40 对 train；25 对 validation1 用于训练超参数；7 对 validation2 用于配准超参数；134 对 test；原图均为 2048×2048 | 这四份划分是 patient-disjoint；本项目角色恰好与官方角色相同 |
| Cropped benchmark，Zenodo 3874362 [D3] | 由 CoMIR 官方仓库和 full-core 记录明确链接 | 本次读取工具不能打开该页面；不能把它写成已下载、失效或不存在 |

Full-core 官方记录列出四个 zip，总大小约 1.8 GB，且给出 md5。文件名示例显示测试集使用 BF/SHG 后缀，训练/验证示例使用 WB/MI 后缀。这是数据适配器应显式映射的模态命名差异，不能仅按字符串包含 “SHG” 搜索训练图像。[D2]

**独立单位判断：**公开 CoMIR 材料支持“core/image pair”这个观察单位，没有在已读材料中给出 206 个条目到患者、同一组织块或同一切片的完整映射。不能写“206 名患者”，也不能将官方 40/25/7/134 自动称为患者独立划分。原数据论文还使用了更大的乳腺与胰腺样本集合，不能把它的总人数、组织数或百万 patch 数转移到这 206 对基准上。[D4]

建议适配器采用分层 ID：

    patient_id          如可验证；未知时为 null，不能用文件名伪造
    physical_sample_id  宿主可核验的物理样本/切片/组织块 ID
    core_id             例如保留原始文件 stem 的完整 core 身份
    original_pair_id     BF 与 SHG 的配对身份
    window_id           从原图裁出的窗口；继承上级 role
    transform_id        同一窗口上的固定人工扰动；继承上级 role

若有证据表明同一患者/切片多个 core，则用最高可识别关联层级分组。若只能保证 core 不重叠，报告 core-grouped 结果，并披露更高层关联未知。对旧 benchmark 不能为了改善独立性而静默改划分；应另建版本化协议，保存旧结果。

### 2.2 官方 CoMIR 划分的角色不可互换

| 官方角色 | 公开数量 | 原用途 | V10 处理 |
|---|---:|---|---|
| train | 40 对 | 表示训练 | 仅当宿主 manifest 指向同一实物时才能映射 FIT |
| validation1 | 25 对 | 训练超参数 | 可对应模型 CAL，但以已有角色记录为准 |
| validation2 | 7 对 | 配准器调参 | 搜索超参数与训练超参数分开锁定 |
| test | 134 对 | 原论文几何评价 | 若已被 R6/V9 读取结果，不再是 V10 首次确认 |

官方数量来自 [D1,D2]。V10 角色列是建议协议，不是已执行映射。下载新的 full-core zip 不会使同一 core 的旧 test 重新获得未暴露属性；从同一 core 改裁窗口或换扰动 seed 也不会。

### 2.3 EPFL RGB–NIR Scene Dataset 是什么

官方页面与作者论文共同核实以下信息：[D5,D6]

- 477 对已配准 RGB 与近红外图像，9 个场景类别：country、field、forest、indoor、mountain、oldbuilding、street、urban、water。
- 下载版为 TIFF、1024×768，官网标注约 1 GB；原始/更高分辨率数据需要联系作者，不能当作现成公开包。
- RGB 与 NIR 是先后拍摄的两个曝光；原始对齐使用 SIFT、RANSAC similarity transform，再重采样到共同坐标。
- 原论文的任务是场景识别。其实验每轮随机取 99 幅测试图像，即每类 11 幅，重复 10 个随机划分；这不是一个官方固定的刚体配准 FIT/CAL/TEST 划分。
- 原论文记载部分图像有拍摄间场景运动。由已对齐图像添加人工变换，真值是相对于发行版对齐状态的人工扰动，不能声称发行版每一像素都具有无误差物理对应。

主入口：[EPFL RGB–NIR Scene Dataset](https://ivrlwww.epfl.ch/supplementary_material/cvpr11/index.html)。本次能够读到介绍与下载入口，未取得数据实物或核验 archive；没有核实明确的再分发许可。因此 server 包可以记录公开来源与已有使用依据，不应附带重分发数据或替作者编写许可证。

本项目历史文件 [H1,H2] 已记录 **33 个旧排除**与**约 2 px 对齐精度限制**。这些是项目历史约束，不是官网公开数据清洗规则。V10 必须恢复逐 ID 排除原因、原始文件/窗口映射、像素坐标及 exposure ledger；本文没有看到那 33 个 ID，不能写出有效保留数或假定都属于某一种缺陷。尤其不能将 “477−33” 的算术结果当作已核验可运行数据量。

### 2.4 EPFL 与 Zurich 不可混同

| 项目 | EPFL RGB–NIR Scene | CoMIR 的 Zurich Summer |
|---|---|---|
| 图像内容 | 地面场景的 RGB/NIR 摄影 | Zurich 城市航拍 |
| 已核实规模 | 477 对、9 场景类 | 20 幅四通道图像，约 930×940 |
| 两模态获取 | 分开曝光，发行版经过配准 | 同一传感器，RGB 与 NIR 同分辨率 |
| V10 地位 | 已有项目历史暴露的条件扩展 | 尚未在本协议中授权替换 EPFL 或追加训练 |

Zurich 事实来自 CoMIR 原文 [D1]。两个数据集同为 RGB–NIR 并不意味着划分、精度、场景 ID 可以互用。

## 3. 暴露账本与确认资格

### 3.1 主协议沿用宿主 manifest

开始 V10 自然数据训练前，对每条数据生成一份只读快照；不得从官方数量反推当前状态。至少保存：

    dataset_name, edition, source_url, source_sha256
    original_pair_id, scene_or_core_id, physical_sample_id, patient_id
    modality_a_file, modality_b_file, native_shape, crop_bounds
    native_to_model_3x3, channel_conversion, interpolation, padding
    host_role, v10_role, parent_manifest_sha256
    exclusion_status, exclusion_reason, exclusion_evidence
    seen_by_versions, first_result_read_at, exposure_evidence
    confirmation_eligibility, eligibility_reason

变换、裁剪、重采样的派生条目必须回溯到原始 ID。关联图像、近重复或同一场景的多个窗口不允许跨 FIT/CAL/CONFIRM。无法判断先前是否使用时，登记 INDEPENDENCE_UNVERIFIED，而不是默认为未读。

### 3.2 BF–SHG 没有新的独立确认时怎么做

仍然可以完成 V10 端到端训练、相同数据机制比较、误差—计算成本曲线和配准失败诊断；结果标为 DEVELOPMENT 或 REPEATED_BENCHMARK。不得为了“需要确认表”而重新命名 V9 test、挑未显示过的扰动，或换训练 seed 后称独立复现。

这一状态只限制可作出的确认性结论，不应阻断不依赖新确认集的 T00–T04 设计/计算工作。最终报告直接说明确认来源仍待落实。

### 3.3 EPFL 能否提供确认

只有暴露账本证明存在足够的未参与 V8/V9/V10 设计、训练、调参或结果选择的 scene groups，才能将这部分作为**候选确认来源**。在查看其方法结果之前锁：

1. 全部原图、窗口与近重复的 scene-group 映射；
2. 保留原有 33 个排除及原因，所有方法使用同一资格表；
3. 未读 groups 的 ID 清单、角色、分组算法、hash 和数据使用依据；
4. 方法族、训练/校准政策、主指标、阈值、搜索预算及失败计数；
5. 唯一的模型选择规则及确认执行时点。

不能预先承诺 EPFL 会满足这些条件。如果 EPFL 全部已暴露、群组太少、历史映射不清或使用依据缺失，则保持 development/independence-unverified；另寻来源须单独提出，不在此包悄悄扩展任务。新训练 seed 只创造新的训练随机性，不能创造新场景。

## 4. 完整配准任务定义

以下是对已接受 V10 与 R6/G1 接口的执行规范。真实张量布局、采样坐标和函数名需从冻结宿主代码核对，不能假定本文已经实现。

### 4.1 输入、目标与学习样本

每个物理样本或场景 s 提供参考图像 A_s、移动模态图像 B_s、原始对齐关系和预生成的固定刚体扰动。目标是估计二维旋转与平移，恢复被施加的几何变换。

训练 P_s 由同一图像中对齐位置的局部内容组成。Q_s 由同一个 s 内的边缘位置重新配对形成。每场景先计算自己的目标，再按宿主已锁的场景权重平均；不能为了一个大矩乘把它改成跨场景 Q，从而让模型识别场景身份。

保留宿主普通 permutation 允许固定点的定义时，矩实现使用 V 形式；排对角 U 形式另列协议并为所有对照同步改变。训练正对应污染中的空间排除，与负 permutation 的对角或邻域排除分别记载，不可混用。

### 4.2 几何变换与特征计算顺序

在共同搜索表中，候选变换 phi 的特征来自：

\[
x_i=f_A(A_s)(u_i),\qquad
y_i^\phi=f_B(\mathrm{warp}(B_s,\phi))(u_i).
\]

即先对移动原图执行几何变换，再编码，再读取预定位置。实际 pullback/forward 方向、旋转中心和像素中心约定必须与宿主一致，并在转换到 native 坐标时显式保存 3×3 映射。不能把 encode-then-warp 当成无条件等价加速。

测试固定所有可训练参数；M1 的 W 只在训练内层求解，不能在每个测试候选变换重新训练。测试不更新 BatchNorm 统计，dropout 关闭或保持宿主已锁的确定规则。

### 4.3 可见区域、空间支持与边界

所有方法使用相同可见区域、采样位置、变换初始化、可接受候选域和失败定义。记录每次候选有效像素数与 out-of-bounds 比例。保持宿主 support 规则；若需要新建固定支持，选择整个候选域内都可见的共同区域并单独版本化。

对需要比较不同有效支持的扩展任务，必须明确归一化和支持策略，并让所有方法使用同一规则。不能让某方法通过 padding、可见面积缩小或样本数改变来获得更高能量。分辨率、图像 crop 或 mask 改动会改变任务，需列为新条件。

### 4.4 指标与统计单位

主量保持 R6 的 native-pixel 角点误差：

\[
e_{s,j}=\frac14\sum_{k=1}^4
\left\|\widehat H_{s,j}c_k-H^\star_{s,j}c_k\right\|_2.
\]

这里 H 的方向按宿主定义；若使用逆变换比较，则两者同时求逆，不混用。c_k、H 与误差必须在同一 native 坐标系；从模型分辨率回到 native 的关系不能只猜一个整数缩放倍数。

| 数据 | 项目主成功阈值 | 项目严格阈值 | 解释 |
|---|---:|---:|---|
| BF–SHG | 42 native px | 9 native px | 来自 R6/V9 历史协议 [H1] |
| EPFL RGB–NIR | 10.5 px | 2.25 px | 尺度不同，分表；严格阈值接近历史约 2 px 对齐精度限制 [H1,H2] |

CoMIR 原文在它的历史 834×834 任务中使用误差超过 100 px 判失败。[D1] 这个外部历史数值不能覆盖上述项目阈值。结果页面和 CSV 必须显式区分“原论文复现协议”与“本项目宿主协议”。

同时报告 median/p90 角点误差、旋转/平移误差、失败率、每图注册时间、能量评估次数、encoder 前向次数和峰值内存。先在同一物理组/scene 内汇总重复扰动，再做组间汇总与 paired bootstrap；训练 seeds 分开报告或使用分层重抽，不能把 seed×patch×transform 个条目计作独立样本数。

人工扰动恢复的精度不等于跨模态真实组织逐像素对齐精度。EPFL 的发行版残余错位和 BF–SHG 人工对齐误差都应保留在解释中。

### 4.5 能量诊断

对固定模型、固定样本和固定位置，输出平移/旋转的能量剖面，以及真值、初始点、搜索结果的能量与几何误差。将“真实位置的能量不是最好”与“搜索器没有找到能量最好点”分开。

矩版与 dense 版同一目标应产生相同值、梯度和几何预测；其比较目标是计算成本。sampled→moments 才测试负对条件抽样噪声。新的函数类/优化器改变带来的任务差异必须通过机制矩阵归因。

## 5. 两张比较表与可运行基线定义

### 5.1 表 A：共同图像、共同搜索器

| 名称 | 训练与配准能量 | 该行回答什么 |
|---|---|---|
| MLP-VCS sampled | 原 concat-MLP+tanh，宿主负对，完整 encoder | 复用原关系模型 |
| MLP-VCS more-neg / block-dense | 原函数类，更高负对预算或完整分块 | 增加负对能否解释收益，计实际成本 |
| bounded-bilinear sampled | V10 两侧 simplex + W，宿主随机负对 | 单独改变函数类的影响 |
| MF-VCS M0 | 同一个 bounded critic，完整矩 J，普通外层优化与 W 投影 | 单独隔离完整负对利用 |
| MF-VCS M1 | 同一 critic/矩目标，固定特征内层求 W | 内层求解是否改善误差—成本 |
| InfoNCE-local | 宿主原局部关系模型，所有 eligible 候选或锁定的相同候选预算 | 不把不同 objective 的数值称为 S |
| CoMIR + shared-search | 完整训练 CoMIR；在共同支持上按声明的 MSE/cosine 能量搜索 | 控制搜索器后比较表示 |
| MI / NMI | 原图统一标量通道规则、同支持的 histogram/Parzen 读数 | 非学习传统能量 |
| MIND-2D + shared-search | 明确二维 descriptor，描述子距离 | 非学习局部结构基线 |
| kernel-relative-ratio | 调好的 kernel/RuLSIF 关系模型，独立读数与相同搜索 | 避免只有固定弱核对照 |

InfoNCE-local 的原始实现须由宿主源码与 checkpoint 锁定：正对索引、温度、score scale、分母是否包括正对、是否对称、候选是否仅限同 scene。标准单向写法为

\[
\ell_i=-s(x_i,y_i)/\tau+
\log\sum_{j\in\mathcal E_i}\exp(s(x_i,y_j)/\tau).
\]

不能为了给 VCS 提供完整负对而减少 InfoNCE 已有全部候选；反过来也不能在 InfoNCE 中悄悄引入跨图像负对。分块 logsumexp 是合理的内存优化，但不具有与二次目标相同的有限低阶矩恒等式。

V10 先完成 clean、完整训练量的机制主表。V9 的 25%/100% × 0/20% 训练错配是另一个已经存在的研究轴，不自动复制成 V10 主贡献或新增默认大网格。

### 5.2 表 B：各方法的合理原生流程

**CoMIR native。** 原方法为每模态各训练一个无参数共享的 dense U-Net，用带 C4 旋转约束的 InfoNCE 学习 image-like 表示，再用 SIFT/RANSAC 或 alpha-AMD 完成单模态配准。它不是在每个位置最大化 VCS-J 的系统。官方仓库给出 train-biodata notebook、SIFT/Fiji 路径和 MI/Matlab 对照；MultiRegEval 提供后续统一评估入口。[D1,B1,B2]

执行时保留原生表示生成和配准顺序，报告其全部时间、推理缓存和调参。若将它放到表 A 使用 warp-then-encode，要明确这是 common-search 适配；不能将表 A 的适配结果冒充原生 CoMIR。

**MI native。** CoMIR 使用 Mattes MI 与 (1+1) evolutionary 策略。[D1] 可采用官方 SimpleITK/Elastix 实现建立合理多分辨、多起点路线，锁定 histogram bins、sampling seed、pyramid、优化器、插值和物理坐标。SimpleITK 明确提供 SetMetricAsMattesMutualInformation；Elastix 的 MI/NMI 路线需分别登记。[B3,B4]

**NMI native。** 这里选择 Studholme 的

\[
\operatorname{NMI}(A,B)=
\frac{H(A)+H(B)}{H(A,B)}
\]

作为明确指标，不混用聚类里的 normalized_mutual_info_score。Elastix 有 Parzen-window NMI metric，且文档公开 histogram 和 B-spline 选项。scikit-image 的 NMI 可用于固定输入数值核对，但其不等尺寸输入默认零填充，因此共同配准应先构造同一有效支持，不能直接比较大小不同的数组。[B4,B5]

**MIND。** 原论文以局部 self-similarity 构造多维 descriptor，并以描述子间 SSD 用于线性/形变配准。[B6] 本次未成功获取作者软件页或完整原始 2D 实现，不能写“已验证官方 MIND2D 可直接运行”。

如果 server 采用自行适配，名字写 MIND-2D，固定公式、局部 patch、offset 集、方差下限、平滑核与归一化。例如以二维四邻域估计局部方差，在预定二维 offsets 上计算局部平方差，指数化并归一化，再以均方 descriptor 差作能量。这个具体二维配方是适配协议，不是原论文已发布的统一默认设置；应先与获得的作者实现或公式参考进行数值核对。MIND-SSC 也是不同 descriptor，不能不标记地替换。

### 5.3 共同预算

给各可训练方法相同的数据访问、初始化 policies、模型选择角色与可比较调参机会。分别保留“相同独立训练数据量”和“相同总计算预算”两张曲线；epochs 相同不等于计算预算相同。非学习方法只计实际参数选择和配准成本，不为了与 3 个训练 seeds 对齐而虚构 3 次训练。

不填任何预测 GPU 小时或加速倍数；先由 T02 取得 encoder/critic/solver/search 实测，再冻结 T04/T05 支出。

## 6. RPC、RuLSIF、谱对比和 learned kernel 的逐式关系

### 6.1 RPC：精确相同的基础目标

RPC 原文的三个参数版本是：[L1]

\[
J_{\mathrm{RPC}}(T)
=E_PT-\alpha E_QT-\frac{\beta}{2}E_PT^2
-\frac{\gamma}{2}E_QT^2,
\quad
T^*=\frac{p-\alpha q}{\beta p+\gamma q}.
\]

因此 alpha=beta=gamma=1 给出本项目 J 和 eta，完全相同。正文/算法表应写“same objective, different critic/computation/solver”。

平衡 RPC 与 MLP-VCS 若共享函数类、采样、超参数和实现，只保留一行；重复标签不是额外基线。若另测论文推荐的非平衡参数或 dot-product critic，应注明它改变了目标/参数化，按原生目标和共同几何任务评价。

### 6.2 RuLSIF：不仅是笼统近邻

RuLSIF 定义 r_alpha=p/[alpha p+(1-alpha)q]，用核线性模型最小化相对密度比的平方误差并加 ridge；固定核下求解线性方程，核宽度与正则可用平方误差 cross-validation 选择。[L2,L3]

下面是本文代数核对，不是新文献结论：

令 alpha=1/2、M=(P+Q)/2，则 r=p/m=1+eta。忽略与 g 无关的常数，RuLSIF 风险为

\[
\widetilde R(g)=\frac12E_Mg^2-E_Pg.
\]

设 g=1+T，

\[
\begin{aligned}
\widetilde R(1+T)
&=\frac12(1+2E_MT+E_MT^2)-1-E_PT\\
&=-\frac12-\frac12(E_PT-E_QT)
+\frac14(E_PT^2+E_QT^2)\\
&=-\frac12-\frac12J(T).
\end{aligned}
\]

并且

\[
\mathrm{PE}_{1/2}(P\|Q)
=\frac12E_M(r-1)^2=\frac12S.
\]

结论是**未正则风险在这种参数与平移下仿射等价**，并非仅“都是二次”。但有限函数类、常数项是否可表示、pointwise bounds、核中心、ridge、采样和有限样本读数仍可能不同。原生 RuLSIF 的系数正则不一定等于 V10 对 W 的 Frobenius 正则。

建议报告两种明确的读数：

- 原生 RuLSIF：alpha=1/2，按作者配方选择 kernel/ridge，报告其相对 PE 估计；若另选 alpha，则原生目标已变。
- 共同 S 读数：固定训练好的 g，定义 T=g−1，在独立样本上计算 J(T) 与 eta-MSE。若额外裁剪 g 到 [0,2]，命名为 clipped variant，并同时记录；不能把后处理隐藏在“原生 RuLSIF”里。

只要平方可积，J-gap 恒等式本身不要求拟合出的 T 已在 [-1,1]；但 V10 的结构性有界类与未约束核拟合是不同实现条件，要如实说明。

### 6.3 可分 kernel 也应获得完整负对加速

为了避免给传统核基线人为制造 O(n²) 劣势，可以采用乘积核基

\[
g(x,y)=\sum_{\ell=1}^m c_\ell
k_x(x,\bar x_\ell)k_y(y,\bar y_\ell).
\]

若 U_{i\ell}=k_x(x_i,\bar x_\ell)，V_{j\ell}=k_y(y_j,\bar y_\ell)，则在包含对角的经验乘积中：

\[
E_Qg=
\frac{(\sum_iU_i\odot\sum_jV_j)^Tc}{n_xn_y},
\]

\[
E_Qg^2=
\frac{c^T[(U^TU)\odot(V^TV)]c}{n_xn_y}.
\]

这是本文由乘积结构直接推出的精确公式。非对角版减去对应 self-pair 项即可。中心和带宽从 FIT/CAL 选择，中心数量与 Nyström/ARD 预算公平记录。learned product-kernel features 也可能保留类似结构，因此“通过矩避免枚举所有 pair”不能脱离这些先行结构单独夸大新颖性。

### 6.4 Spectral contrastive：二次负项相近，目标不同

原文的谱对比损失为：[L4]

\[
\mathcal L_{\mathrm{SC}}(f)
=-2E_P[f(x)^Tf(y)]
+E_Q[(f(x)^Tf(y))^2].
\]

它基于 augmentation graph 的谱/低秩分解，与 J 的负对平方项有关，但缺少 J 的正对平方惩罚及负对线性项。单一表示的 dot product 还是一个对称 PSD Gram 结构；MF-VCS 的两侧映射和一般 W 不要求对称或 PSD。

推荐比较方式是原生谱目标的完整表示训练，加上明确的跨模态适配说明。若使用不同的两侧 encoder，说明从原单域设置如何改为跨模态图；不能把原文线性 probe 保证直接移到配准。

谱损失的完整负对平方本来也可写成二阶矩运算。应允许官方谱基线采用合理 moment/Gram 实现，并计完整训练/表示成本，而不能只有 V10 得到计算优化。

### 6.5 Learned kernels：有效强对照，但不要冒称在估计 S

Learning Deep Kernels for Non-Parametric Two-Sample Tests 学习神经特征上的核，优化 MMD² 相对其标准差的比值以提高检验功效，并将核训练样本与最终检验样本分开。论文和官方代码都明确了这一点。[L5,L6]

其核例子为

\[
k_\omega(z,z')=
[(1-\epsilon)\kappa(h_\omega(z),h_\omega(z'))+\epsilon]\,q(z,z').
\]

这与固定弱 Gaussian kernel 明显不同；V10 的“传统核”表应保留学习型核或学习特征上的相对密度比拟合。

但要区分两个实验：

1. 采用作者原生 MMD/检验功效训练和读数：它是原生 deep-kernel 基线，评价检验功效或明确适配后的配准任务；不对其原始 MMD 数值直接计算 S-RMSE。
2. 学习核特征并用 RuLSIF/J 风险训练 ratio head：这是 learned-kernel relative-ratio 适配，可给共同 S 读数；不能冒充对 ICML 2020 原算法的完全复现。

配准时，单纯比较两个模态各自的边缘样本集合 MMD 会对位置排列不敏感，不能当作配准能量。若用依赖检验，观察对象须是联合 pair 分布 P_{XY} 与重配对乘积分布 Q，或明确的 HSIC 类目标；空间依赖、训练/测试分离及分组置换需要单独处理，不能套用独立同分布两样本检验的现成显著性。

## 7. 数据与实现可行性台账

| 项目 | 本次已核验 | 仍需 server 完成 |
|---|---|---|
| BF–SHG 官方 full-core | 公开记录、四个 archive、公开 md5、角色说明 | 实物下载/已有缓存核验、hash、与宿主裁剪/模态映射 |
| BF–SHG cropped | 官方来源链确实指向 3874362 | 实际文件可达性、版本、尺寸、transform 文件与 manifest |
| EPFL | 官方介绍与下载入口、作者论文 | archive 校验、原ID映射、33旧排除、使用依据、exposure ledger |
| CoMIR | 作者仓库训练 notebook、native 注册说明；MultiRegEval 接口 | pin commit、依赖适配、完整 encoder 训练、原生注册可运行性 |
| MI/NMI | 官方 SimpleITK/Elastix API 与指标定义 | 版本固定、宿主通道规则、坐标对齐、CAL 调参及运行 |
| MIND | 原论文公开摘要及 descriptor 原理 | 原公式/作者代码取回与2D核验；当前不能写 native-implementation-ready |
| RPC | 原文目标与最优 critic 已核对；作者仓库检索入口已找到 | 具体版本与参数复现；相同设置不重复计行 |
| RuLSIF | 作者论文公式、作者软件入口 | 软件实物、license、数值对照、产品负对适配 |
| Spectral | 原文谱目标与作者 PyTorch 仓库 | 跨模态适配、moment 优化与完整训练 |
| Deep kernel | 原论文及官方 DK-for-TST 仓库 | native objective 或 ratio adaptation 选型、独立读数与预算 |

本文没有使用服务器数据、运行完整配准、预测设备吞吐或测量 GPU 时间。能否执行自然数据确认由实物核验和历史暴露证据决定。

## 8. 来源索引

所有网页均来自作者、论文出版社/会议、官方数据入口或软件官方文档。ref ID 与核验深度另存 sources.json，便于主进程重新打开后引用。

### 数据与外部任务

- [D1] [CoMIR: Contrastive Multimodal Image Representation for Registration, NeurIPS 2020](https://proceedings.neurips.cc/paper_files/paper/2020/file/d6428eecbe0f7dff83fc607c5044b2b9-Paper.pdf)。已读数据、训练、native registration 与评价段落；未复现结果。
- [D2] [Multimodal Biomedical Dataset for Evaluating Registration Methods — Full-size TMA Cores](https://zenodo.org/records/4550300)。已读数据记录及 archive metadata。
- [D3] [Cropped CoMIR dataset, DOI 10.5281/zenodo.3874362](https://doi.org/10.5281/zenodo.3874362)。来源链由 [D2,B1] 确认；本次直接页面读取失败。
- [D4] [Non-disruptive collagen characterization in clinical histopathology using cross-modality image synthesis](https://www.nature.com/articles/s42003-020-01151-5)。已读样本/成像部分；不将其更大样本数移作 CoMIR 数量。
- [D5] [EPFL RGB–NIR Scene Dataset](https://ivrlwww.epfl.ch/supplementary_material/cvpr11/index.html)。已读官网说明、尺寸与下载入口；未核验数据 archive。
- [D6] [Multispectral SIFT for Scene Category Recognition, CVPR 2011](https://ivrlwww.epfl.ch/supplementary_material/cvpr11/BS11.pdf)。已读配对数、对齐/运动限制与原 scene-recognition 划分说明。

### 应用软件

- [B1] [MIDA-group/CoMIR](https://github.com/MIDA-group/CoMIR)。已读官方 README；未安装或运行。
- [B2] [MIDA-group/MultiRegEval](https://github.com/MIDA-group/MultiRegEval)。已读统一评价入口与依赖说明；未运行。
- [B3] [SimpleITK ImageRegistrationMethod](https://simpleitk.org/doxygen/latest/html/classitk_1_1simple_1_1ImageRegistrationMethod.html)。核对 Mattes MI API。
- [B4] [Elastix NormalizedMutualInformationMetric](https://elastix.dev/doxygen/classelastix_1_1NormalizedMutualInformationMetric.html)。核对 Parzen-window NMI 与配置项。
- [B5] [scikit-image metrics API](https://scikit-image.org/docs/stable/api/skimage.metrics.html)。核对 NMI 公式、bins 和自动 padding 行为。
- [B6] [MIND: Modality independent neighbourhood descriptor for multi-modal deformable registration](https://www.sciencedirect.com/science/article/pii/S1361841512000643)。核对出版社摘要；未取得并逐行审核原实现。

### 目标与方法先行工作

- [L1] [Self-supervised Representation Learning with Relative Predictive Coding, ICLR 2021](https://arxiv.org/pdf/2103.11275)。正文与附录均支持参数化目标及最优 critic。
- [L2] [Relative Density-Ratio Estimation for Robust Distribution Comparison](https://arxiv.org/pdf/1106.4729)。核对相对比、PE、风险、ridge 线性求解与 CV。
- [L3] [作者 RuLSIF 软件页](https://riken-yamada.github.io/RuLSIF)。核对软件入口与公式；未下载代码。
- [L4] [Provable Guarantees for Self-Supervised Deep Learning with Spectral Contrastive Loss, NeurIPS 2021](https://papers.nips.cc/paper/2021/file/27debb435021eb68b3965290b5e24c49-Paper.pdf)。核对谱分解、正负对定义与 Eq.6；[作者代码](https://github.com/jhaochenz96/spectral_contrastive_learning)。
- [L5] [Learning Deep Kernels for Non-Parametric Two-Sample Tests, ICML 2020](https://proceedings.mlr.press/v119/liu20m/liu20m.pdf)。核对 kernel Eq.1、标准化功效目标 Eq.3–5 与 train/test 分离。
- [L6] [fengliu90/DK-for-TST](https://github.com/fengliu90/DK-for-TST)。核对作者公开仓库身份；未安装运行。

### 本项目提供材料

- [H1] VCS_V9_Reliable_Correspondence_Server_v1_20261016/docs/06_REG_EXTENSION_CN.md。核对原 R6 指标、分组、DEV 属性、EPFL 旧限制、两张方法表。
- [H2] 同包 docs/07_DATA_AND_BENCHMARKS_CN.md。核对 EPFL 来源身份、33旧排除、约2px限制、数据审计要求。
- [H3] V10_DESIGN_CN.md。已接受候选设计：bounded bilinear、普通 permutation 对 V 形式、M0/M1、full encoder、warp-then-encode 与测试固定 W。

这些本地材料提供的是恢复出的版本化历史，不等于已检查当前服务器作业或新运行状态。
