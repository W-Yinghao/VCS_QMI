# 完整训练与强对照矩阵

## 1. 相同表示与新模型起点
复用V7的数据身份与公开BLIP-2 COCO初始化，不能从见过本地DEV的发布FGA权重开始。保留FGA原生冻结ViT选择，训练Q-former/query tokens及新的评分head。主输入364完整图像、完整提示上限128；不翻转、不裁剪改变方向/数量事实。模型加载、参数量和每次sample exposure均记录。[G02]

主head：32个图像查询token的mean→5个logits，softmax为自然评分直方图。所有8臂使用相同5logit head和相同初始logits偏置（FIT总体直方图的log值，平滑仅为数值初始化并记录，不改标签）；每seed相同权重。MSE/L1/Rank/List从同一head计算μ，因此容量相同。

## 2. 八个完整配方
| ID | 目标 | 主输出 | 要回答的问题 |
|---|---|---|---|
| U-CS | 四阈值原J，FIT先验平衡，a=1/4 | μ=1+Σu | 原方法在效用对齐后的表现 |
| U-BCE | 同四阈值的平衡BCE，使用ell=logit(u)−logitπ | 同μ | 严格近邻损失控制 |
| U-RPS | 自然采样下平均平方CDF误差 | 同μ | 强非CS正规分布评分对照 |
| U-CE | −Σh empirical_hist_h log qh | 同μ | 标准评分分布最大似然 |
| U-MSE | ((μ−Hbar)/4)² | 同μ | 直接优化均值效用的强对照 |
| U-L1 | |μ−Hbar|/4 | 同μ | 任务常见L1与FGA目标风格 |
| U-RANK | U-MSE + 同提示RankNet-style BCE，λ=1 | 同μ | 外部排序机制，不假设必须CS |
| U-LIST | U-MSE + 同提示ListNet top-one CE，λ=1 | 同μ | 候选竞争的强任务对照 |

三seed={0,1,2}全部正式训练：24单元。每配方用独立tuning seed=17，lr={5e−6,1e−5,2e−5}，完整5epochs：24调参单元。所有运行都计成本。不能单独为U-CS加入辅助头、额外标签或更长训练。

## 3. group batch与监督公平性
冻结一种所有臂共用的grouped image batch：按seed/epoch洗牌提示，每提示图像洗牌切成≤7图的小组，将组打包到≤14图batch；全FIT每图每epoch恰出现一次（末批也处理），同seed所有臂读取相同batch-index manifest。单图提示仍训练均值/分布目标，不因排名目标没有pair而删除样本。

点式项使用自然目标的prompt→image权重（每提示质量一致），按有效样本权重精确累计。Rank/List只在同一子组≥2图时计算；组内平均后按prompt的子组数量归一，另记录pair/list实际曝光量。它们总有共同MSE项，单图组不会没有监督。任务基线同时改变损失形状是合法的，但不能称为“只换一项proper loss”。

RankNet任务适配：对所有无序图像对，目标为Hbar_i>Hbar_j的1/0，严格平局目标0.5；模型logit=(μ_i−μ_j)/τ，τ=1。基于观察到均值的pair并不是真实潜在偏好标签，保留标注噪声限制。ListNet：target softmax(Hbar/τ)、prediction softmax(μ/τ)，τ=1。温度敏感性另行申请，对两臂对称，不能看外部结果后搜索。

### 共同优化器与数值模式（本轮设计）
AdamW betas=(0.9,0.999)、weight_decay=0.05；100个优化步从1e−8 warmup至选定学习率，随后按实际总优化步cosine降至0。全部配方相同；AMP用于多模态forward，softplus/logsumexp/阈值风险与归一分母至少fp32。浮点类型、GradScaler和有效更新计数入配置。不为某个损失单独缩放梯度或改clip后仍称匹配。

ViT参数按共享约定冻结且保持eval；训练Q-former和head时启用同一dropout配置。所有预测/校准model.eval()。没有像旧D1那样单独冻结一个有dropout的critic再优化gate的步骤。

## 4. checkpoint与读数选择
统一主选择终点为CAL_SELECT的prompt-macro均值MSE（同量表、同样本），各epoch保存模型；平局优先早epoch，再低lr。原因是主要部署效用为期望评分，不以某一损失的native值选择其他方法。记录CAL同提示选择质量及native最优epoch仅作次级诊断，不根据DEV或GenAI重选。

主输出raw μ，不用混合α或V7的E后验平均。独立CAL_PROB仅做预登记的次级概率温度校准，主结果不以校准版取代。概率校准对分布head保持单纯形/顺序，不分别对四尾概率做可能交叉的任意Platt后还宣称一致性。

## 5. 数据规模与独立训练
5epochs按全部合格FIT图，不是小样本探针；训练量因无效overall标签的排除而改变须在第一步公布。使用相同有效FIT集合，不因bad element decomposition排除总体标签本身有效的图。两级质量标签有不同缺失规则，具体见数据章。

有限数据(10%/25%)、第二backbone、额外ranking辅助不是首轮自动网格。若完成主表后有明确问题，再冻结扩展；所有扩展必须有传统/非CS对应对照，不能只追加VCS。

## 6. 可选而非默认：有序结构对照与辅助项
U-EMD2：按NIMA公开r=2公式的平方根版本，3seed；CORAL/CORN：不同参数化与条件风险集，不能只替换名称。它们有新代码/结构预算，若要宣称“优于专门有序方法”则必须完成，否则只在Related Work定位。

CS+ranking只在概率好但同提示选择不足时作为下一次计划；同时给BCE/RPS相同辅助项、同λ选择机会。不要第一轮就把所有有利模块打包，导致无法判断哪一项有效。
