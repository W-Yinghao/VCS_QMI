# D4 服务器实验方案｜多对多、不完整标注与概率图文关系

**性质：候选方案，尚无本方案自然数据结果。** 先读本目录综述和共同理论/评价规范。数据条目：DS12, DS21, DS22, DS23, DS24, DS25；全部定义见 `../../datasets/DATASET_ATLAS_CN.md`。

## A. 首先选关系，不先选损失

主候选是“多个合法语义对应下的检索与集合预测”。不再只在同一图的2–4个框中选择。以原图像为split单位；基础双塔先固定CLIP B/16或当前已存SigLIP2 B/16的一个，不能每格挑最有利checkpoint。

首轮分两张表：受控CUB类相关多正例；自然COCO/Flickr训练→ECCV Caption/CxC评价。CUB的真关系是class relevance；自然评价的真关系仅限官方judged universe。若自然训练没有已判断负例，必须列出这一缺口，不能使用eval标注来补。

## B. 模型和损失设计

使用共同多原型或Gaussian embedding宿主时，固定它的容量、采样次数和所有辅助正则。VCS定义为已标注/按协议构造的P/Q上的原J；Logistic在同P/Q同函数类拟合。InfoNCE或multi-positive softmax按同一合法正例集合训练。若选择机制Q=乘积，其T读取机制后验；需要语义支持概率时另做带人工标签的CAL映射。

未知pair不要赋硬负标签。第一轮可比较原始训练标注、受控删除一部分已知关系的标注变稀，及保留的已judge子集；所有方法看到相同删除mask。对COCO未知关系仅做敏感性分析，不声称有真实删标oracle。语义伪正例教师及阈值对所有方法共享；无教师和有教师分开。

## C. 最小完整训练

| 单元 | 内容 |
|---|---|
| D4-00 | 重现原双塔、PCME++的检索评估输入和ECCV evaluator fixture；清理COCO/Flickr/CUB角色 |
| D4-10 | 同宿主VCS、Logistic、multi-positive softmax、PCME++共4方法×3seeds，完整双塔适配 |
| D4-20 | 固定删标比例{0,.3,.6}在已知关系数据上检验；先共享训练budget，不额外每档多调参 |
| D4-30 | 未用于选择的ECCV Caption/CxC/跨Flickr与SUGARCREPE++语义读数；ProLIP作为完整系统参考 |

D4-10是12个训练单元提案；D4-20为条件扩展，不自动把12变36。训练都从相同原始权重开始；不能一边给VCS已有task-finetuned权重一边让基线从pretrain开始。

## D. 评价

主表两方向mAP@R与R-Precision，COCO5k/Flickr R@1/5/10同时保留。COCO1k五折另列，不与5k比较。CUB按PCME的class-heldout与类相关定义，不能和caption semantic truth混排。CxC相关性在公布judgements上算；支持概率/校准只在真正人工二值judge子集算。

新增集合预测用CAL选择阈值或coverage预算，比较相同平均返回条目数下的precision/recall。所有方法都可temperature/Platt校准；T有界不等于天然概率更好。按原图群组bootstrap，重复caption不计独立图像。

## E. 我们希望什么、什么不算成功

希望缺失关系下的多正例召回、概率风险或集合覆盖更好，且标准检索没有不合理损失。仅在原始单正例R@1上平局，不是新增能力；仅uncertainty与caption长短相关也不是正确的多义表达。

若所有收益来自prototype数、伪标签或更长context，则报告共有结构而非平方损失优势。若不完整标注导致无法评价语义真值，保留标准检索与annotation-sensitivity，暂停“准确估计真实匹配概率”的主张。需要新标注时明确人工成本/许可，不偷偷使用评价集训练。

下一步可转到D2的有真实支持标签核验，或D5的数据覆盖用途；不继续加内部余弦head直到某一列赢。
