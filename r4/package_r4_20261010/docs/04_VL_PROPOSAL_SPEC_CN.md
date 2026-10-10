# VL 实施规范：从给定标注候选到实际 proposals

本文件实现最后讨论中的有限重设计。继承 VL3 已完成全微调和 VL1-22 的 clean detector，不回到 VL1 冻结特征大筛选。新训练须满足完整实验约束。[R1,R2]

## 1. 最新结果所支持的起点

给定框的平均 VCS-softmax 差约 +0.01 点；实际检测候选变为 −2.05 点，12 条件全部负向。目标 proposal 覆盖率约 97%–99%。这支持检查候选分布，不证明原因已经是“margin 太小”或“有界梯度压不住困难负例”。

主要缺口有三种：候选数变多；裁剪框变松或截断；候选出现原来未训练的对象/部件/背景。先用只读干预分别测，不自动扩大 backbone。

## 2. 固定对象与评价池

首批：RefCOCOg、RefCOCO+，SigLIP 2 B/16，VCS/JS/softmax，各三个既有 VL3 seed。已有 18 个 checkpoint，不产生新预训练。

继承 `VL1_22_TABLE_C_FROZEN_20261010.md` 的裁剪、文本 tokenization、候选 score≥0.05/top-100/NMS 等约定。实际 API 或文件不同，以本地版本核对并记录，不凭文件名推断参数。

模型评分按原生 cosine/单调 logit；不因 tanh 浮点饱和造成 ties。优先复用全部 pair 分数。如果只保存 top-1 结果，做一次冻结前向补齐分数，并与原表在同一分母下核对。

## 3. 只读诊断 R4-VL0

### 3.1 必须先恢复主表

全部 proposals、全部 eligible DEV queries、image-macro Acc@IoU≥0.5 为基准。没有 proposal 覆盖真实目标的 query 计错；不要为了漂亮结果移除。另给 covered-query 条件结果、query-weighted 结果及候选覆盖率。

每条记录保存 image_id、query_id、target set、proposal_id/box、IoU、detector score、各模型得分、正确候选数、top-1正确性。标注 ID 仅用于评价与抽样，不进入模型输入。

### 3.2 数量：区分“增加候选”与“改变候选类型”

第一条可部署曲线：使用 proposal 固定 score/hash 顺序的前 K 个，K=4/8/16/32/全部（不足 K 取实际数量）。候选选择不依赖当前文本或方法得分。每个 K 同时报覆盖率，避免将 coverage 变化混成排序变化。

第二条 oracle-assisted 曲线：在 covered-query 池中，强制保留相同的一个合法候选，其余从相同类型的非目标 proposals 用固定随机排列逐步增加。可用标注筛选负框，但必须标明这是诊断，不进入部署结果。这样更直接测多干扰项的最大值效应。

对所有方法共享 K、query 子集和随机排列；不能只给 VCS 挑困难/简单候选。

### 3.3 框质量与干扰类型：小型 2×2 诊断

在目标可覆盖的同一批 query 上，目标框取 A=标注框，B=与其 IoU 最高的检测框；干扰取 C=原标注非目标框，D=同数量的检测干扰框。使用确定的去重和 target-set 规则，报告四格：A+C、B+C、A+D、B+D。

该实验明确使用真值选择目标，属于 oracle-assisted。选框只按 IoU，不按任何模型分数。多合法目标使用预先固定目标集合或统一最大 IoU 规则，不能对不同方法换答案。

干扰进一步按与已标注物体的空间关系分为：完整其他对象、同类其他对象、part/重复框、弱对象重叠框。后两类先称几何分层，不假装自动拥有准确语义标签。“没有对应标注”不能直接等于真正背景。

### 3.4 配对分数如何读

报告 best-valid 与 best-invalid 的 logit/cosine margin、错误候选类型、正确/错误切换数。没有 valid proposal 时 margin 为空，不用人为极值填充。

主要比较采用同图片/同 query 配对差；图片级 bootstrap，保留种子维度。不同种子的同一图片结果不是额外独立图片；先按 seed 给差，再报告 seed 平均的图片 bootstrap 及 seed spread。已多次用过 DEV，因此这些区间是探索性描述。

不计算“各因素的百分比因果贡献”：这些干预改变了分布，2×2 结果是可重复的受控诊断，不能无条件作总体因果分解。

## 4. Proposal-aware 训练的 P/Q 规范 [D]

### 4.1 主对象不是原条件 QMI 的静默替换

令 G 为图片，W 为一个符合闭集训练条件的表达，R 为该图片的冻结 detector 候选。A_G(r,w)=1 表示 candidate 对目标框的 IoU 达到统一训练阈值。首轮建议 0.5；处理近阈值/多目标时规则一次冻结，另报 0.75 评价，不每方法挑阈值。

先按图等权、图内 query 等权取 π_G(w)。对至少存在一个合法 proposal 的训练 query，定义

$$p_G(r,w)=\pi_G(w)\frac{A_G(r,w)}{\sum_s A_G(s,w)},$$
$$q_G^{\nu}(r,w)=\pi_G(w)\nu_G(r).$$

ν_G 为 query-independent、全支持的 proposal 参考先验；首轮在该图全部合格 proposals 上均匀。于是

$$\frac{p_G(r,w)}{q_G^{\nu}(r,w)}=\frac{p_G(r\mid w)}{\nu_G(r)}.$$

Qν 可以抽到合法目标，这不矛盾：二元标签指采样机制，不是纯语义真假。Qν 也能包含从未被任何表达指向的 proposal；正关系的区域边缘通常不是 ν，所以该目标命名为 **proposal-reference CS discrepancy**，不是原条件 joint-vs-product QMI。

同一 S/J 原理仍适用。若希望将 Q 改为确定的 semantic nonmatch，必须另命名、另定先验与模糊标注规则，本轮不同时引入两套 Q。

### 4.2 对部署排序的联系

最优 critic 满足 $e^{2f^*}=p_G(r\mid w)/\nu_G(r)$；排名可以用

$$\log\nu_G(r)+2f(r,w,G).$$

均匀 ν 时与 f/cosine 排名一致。ν 的值只能由候选生成规则和部署可用信息得到，不能读取测试 query 的真实目标。训练与测试 class-conditional 分布变化不是先验修正能单独解决的问题。

$f=0$ 不是无目标拒绝阈值；当前训练仍条件于存在合法 proposal。查询没有候选覆盖只计闭集系统失败，不因此变成已训练的 null 类。

### 4.3 零正例、多目标和漏标

FIT 中没有合法 proposal 的 query，三个方法都不做这条闭集匹配梯度，记录排除数。不能只给 VCS 注入真框保底。CAL 的选择指标、DEV 的最终任务指标仍使用全部 eligible queries，包含无覆盖失败。

多目标 query 使用统一 A；多处合法框不强迫只有一个正确答案。部分候选与未标注语义目标的关系不确定时，先说明标签是官方 IoU 操作定义，不称其为完整语义真值。

## 5. 三个损失与共同训练条件

**VCS：** 每图 p/q 各归一化，使用原 J。

**平衡 JS：** 同输入/同 p/q、同 f，$L_{JS}=E_P softplus(-2f)+E_Q softplus(2f)$。共同评价 T=tanh f 后再算 J；JS native 另列。

**候选 softmax：** 在完全相同的 proposals 中，使用多正例候选似然 $-\log\sum_{r\in A_w} softmax(u)_r$。u 的温度允许按相同 CAL 预算选择；不把其平移自由度当成已标定配对后验。多正例集合事件 CE 与均匀软标签 CE 并不相同，本轮默认集合事件形式，历史实现若不同须披露。

仍用当前双塔 cosine＋affine 评分结构，完整更新两塔；不追加一个只有 ours 能用的更大 MLP、上下文模块或空间规则。若只调 a,b 而冻结 encoder，不能改变 ranking；这种实验不能替代本任务。

## 6. 训练范围与公平比较

主计划：2 数据集 × 3 目标 × 3 seed =18 个新选定 full fine-tune 单元。统一原始 SigLIP 2 B/16 权重，不能从各自已经微调的 encoder 继续训练后把结果当同起点。

初始使用历史 10 epochs、同 effective image batch、同数据版本和 CAL 选择规则。proposal 数导致 crop 曝光增加，记录图片、crops、pairs、更新步、总 FLOPs/计时；图像数相同不等于计算量相同。

优先复用完全兼容的 VL3 标注候选控制。若改变 optimizer、预处理、数据拆分或主终点，需新增对应控制，先返回成本；不把旧轨迹与新 recipe 作单因素归因。

### 有限超参数检查

先核对已有搜索，缺失时使用两个学习率和两个 gain 起点的对称 CAL 探索，不扩大成大网格。三种目标都给同一选择机会，gain 的实际 logit 标度要写清（JS 使用 2f，softmax 使用自己的 u）。选择种子和确认种子预先区分；不得只有第一 seed 赢的条件才补 seed。

### 全候选算不动怎么办？

先记录真实显存与耗时，再提交一种预先固定的候选抽样方案。VCS/JS 若用重要性权重近似全 Q，需要完整支持与概率；softmax 分母的 subsampling 不能默认无偏。最简单的公平替代是将“共享候选子集生成器”纳入实验定义，三目标同用，再对完整 proposals 评价；此时目标和计算条件已经变了，另建协议，不能叫全候选精确等价。

## 7. 决策与替代路线

| 观察 | 解释范围 | 下一步 |
|---|---|---|
| 三方法均因 proposals 训练改善，差距仍接近 | 候选支持适配有效，共同方法收益 | 保留应用，停止同质扩展 |
| VCS 的公共风险改善，但任务无改善 | 更好读取规定 p/q，不足以改善排序 | 检查 margin/多目标定义，不再以 J 单独宣称任务优势 |
| VCS 对困难候选仍差 | 该配方下目标/评分器不适合该任务 | 保留结果，不强行换数据寻找优势 |
| 固定候选适配能用，部署覆盖仍不足 | 提议器或框质量限制 | 改更强统一 proposer 属后续系统工作，不算 estimator 独有改进 |
| 差距仅在某小子集正向 | 探索线索 | 新样本/新协议确认前不升级主张 |
| 需要表达“无目标” | 新的语义事件 | 暂缓；独立 null 数据/校准流程先定义 |

RefCOCOg 与 RefCOCO+ 都来源于 COCO，不是独立域。后续要做真正跨域可采用已列数据资源，但本轮不新增下载，不自动开启 Cops-Ref/gRefCOCO 或全局 CLIP+λJ。

## 8. 必交表

表 1：实际 proposals 的旧/新训练结果，三方法、三 seed、全部分母、候选覆盖。

表 2：只读诊断的数量/框质量/干扰类型，明确 oracle-assisted。

表 3：同 p/q 的公共平方风险、初始化 spread、成本；自然图像没有 oracle S，真值字段为空。

表 4：数据、预训练来源、CAL/DEV 使用历史、detector exposure 与外部 baseline 条件。

只有部署表使用的流程可以写成方法。不能把真值辅助选框、DEV 调参或公共 J 后处理藏进实际测试推理。
