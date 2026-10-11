# D3 服务器实验方案｜条件生成分布比较

**性质：候选方案，尚无本方案自然数据结果。** 先读本目录综述和共同理论/评价规范。数据条目：DS13, DS12, DS14, DS15, DS16, DS17, DS18, DS19, DS20；全部定义见 `../../datasets/DATASET_ATLAS_CN.md`。

## A. 统计对象

固定条件权重π(c)，P(c,z)=π(c)P_ref(z|c)，Q_m(c,z)=π(c)P_model_m(z|c)。主读数S；CS变换按全局S单独提供，不能对每条件CS求均值后冒充同一值。共同观测编码器固定，不为VCS偷偷提供更强文本条件特征。

首次原型先在低维latent条件和已知观测映射上做精确truth，再渲染为CLEVR型场景。若渲染/特征变换不单射，先计算pushforward分布或放弃oracle标签；不要把scene label的S直接当CLIP特征真值。自然数据只报held-out common J与不确定性，不报未知S的MSE。

## B. 可控实验设计

条件c取对象/属性/关系组合，图像中的未指定位置、材质、背景视为合理变化。固定生成P_ref和四类Q：条件误配（保持总图像边缘不变）、类内模式缺失、视觉噪声/质量下降、混合污染。各类型强度3档，加clean reference，共13个规定分布。采用相同条件频率，生成/采样seed在角色间独立。

每条件FIT样本数{8,32,128}；条件组数先固定16，再以更多组测试可迁移性；3次独立抽样。主要6读数：VCS、matchedLogistic→J、same-target核relative fit、grouped RBF MMD、CFID/cFreD适用实现、无条件CMMD负控制。只有VCS、Logistic→J与同目标核回归进入posterior/S误差表；MMD、CFID/cFreD、CMMD只比较各自的分辨、稳定性与外部用途，绝不拿它们的原始值减去S。正确信息保持的null和恒定估计器同时列出，不能靠恒零的“低方差”获胜。

在多个Q_m比较时，每个critic都用相同FIT/SELECT/EVAL预算；不能用同一evaluation pool选网络/带宽。可增加共享条件critic，但“跨条件泛化”必须在未用于训练的condition/template上另测。

## C. 自然数据评价

第一部分选择已存在多真实实例的语义条件组，明确reference究竟是这批受限组，不是所有自然图像。COCO对象/属性group需要剔除歧义、共现偏差并记录；分组条件本身不赋予真实密度。不同生成器用固定版本/seed/prompt，先用已有可再分发样本，生成新图前估算预算与许可。

第二部分用GenAI-Bench/Rank固定release的人类判断，按prompt-group和generator-family做留出。评价模型级和prompt组级相关性；同一prompt中样本不足以拟合条件分布时聚合到预定组，不为得到显著性事后重分组。第三部分用GenEval/CompBench版本和ConceptMix检验语义缺陷排序，不把这些分数当S数值真值。

## D. 主表与指标

| 表 | 必报 |
|---|---|
| 受控同目标估计 | posterior MSE、S bias/RMSE、评价方差、null错误与样本量；fit/eval成本 |
| 缺陷分辨 | 各类型强度的排序/分辨概率；marginal不变但条件破坏的专门对照 |
| 多样性与忠实性 | GenEval/CompBench分项、Density/Coverage、固定人评相关性；不能只报整体好坏 |
| 自然分布评价 | 样本数曲线、跨generator/template一致性、独立bootstrap；不写oracle MSE |

输入尺度/特征/样本数必须匹配；核方法的带宽/维数在相同SELECT预算内选。无偏MMD负值照实保留，confidence不是自制分数。采用prompt群组重采样；生成次数是独立图像而不是一个样本的多裁剪。

## E. 完整应用与候选新增点

完整pipeline应能接收固定reference和新generator样本，输出标量、每条件诊断、有效样本量、fit/eval分离日志与不确定性。预注册一个外部用途：在受控质量/覆盖变化的生成器集合中正确识别退化，或根据独立评价选择生成配置；选择后再用新prompt评价，不能用同一人评挑模型再报相关性。

可选后续是条件生成适配，但minimizing a lower bound J会受critic不足影响；必须交替重拟合和独立外部评价，且不声称小J认证分布匹配。

## F. 失败后的处理

若CFID/groupedMMD已经稳定解决，研究应收束到VCS在哪些非高斯或高维条件有可复现增量，而不是撤强对照。若critic靠来源风格胜出，增加同风格/同生成器条件置乱对照，并固定与文本相关的观测，而非继续扩大网络。若自然reference不足，限定为有限参考评价或暂停自然主张；不能依靠“总会有真分布”继续跑。

首轮不是生成十万张图：先做13×3样本档的受控估计和一个固定人评release的资源核验。自然生成量由第一个完整timing决定，所有新生成/付费人评需按实际授权。
