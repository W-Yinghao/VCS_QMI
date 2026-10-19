# 06｜忠实于来源的对齐与新增限定

## 原计划 [O1]

原题为“Variational Cauchy–Schwarz Quadratic Mutual Information — A Stable Alternative to MINE”，原目标投稿范围是SPL/Signal Processing；本项目后来按用户要求拓展到CVPR。r3沿用其理论中心，但不将原稿当成CVPR已获论证的新颖性或篇幅规定。

| 来源位置 | 来源实际提出的内容 | r3如何使用 |
|---|---|---|
| §1–4 | 不是Shannon MI；选择equal-mixture得到sample-only J与精确gap | 定义不变，same-target与native-target分别评价 |
| §5、7 | tanh(PMI/2)、平衡二元后验与平方风险 | 测量critic的目标；不把固定SSL scoring当成天然准确posterior |
| §8 | triangular discrimination联系、可逆不变性/DPI、必须核对先行工作 | 有条件的测量压力检查；不宣称散度本身全新 |
| §9 | joint/product samples、K移位/U-statistics、原J训练 | 优先复用已有生成器与采样实现 |
| §10 | bounded optimal critic/empirical summands、无指数矩、透明误差对象 | 分估值波动、尾部与梯度；不将log转换值说成均匀低方差 |
| §11 | consistency与finite-sample package；处理product样本依赖 | 区分独立池与共享U-statistic，未知拟合误差不伪装成置信区间 |
| §12、17 | 误差、gradient-norm variance、batch/lr、直方图、失败与成本；关键图 | R3-E0/E1是最直接对齐，不由属性功效表替代 |
| §12.4、17 | 一个依赖测量本身居中的应用 | 固定SSL两视图关系的测量与辅助模型解释 |

## 补充意见 [O2]

- §2.4：Gaussian加噪只是相关系数重参数化；本包仅作实现检查。
- §2.5：原总RMS噪声太小；本包明确标准化坐标和每坐标方差，仍不保证降到目标S。
- §2.6：突出平方增量与平方根误差界；保留该关系，但不自动把它变成文章唯一中心。
- §3.3：可以由其他正规损失拟合posterior，再用平方结构测增量；r3允许logistic→common-J，不将其改名成直接VCS训练。
- §3.7：不能撤掉MINE/NWJ/InfoNCE/SMILE；r3保留修正后的强对照，不用单负例失败概括。

## 本次推导/实施限定，不冒充来源原话

1. 置换功效在严格单调分数变换下不变，因此不是数值精度的充分证据。它不否定检测应用的价值。
2. 条件期望的平方范数差与所给误差传播属于二次几何；其他评分规则有各自增量表达。补充强调的优势在本包中按这个具体形式验证，不扩成“其他方法不存在任何误差界”。
3. 望远镜相消可由差值定义自动成立；小残差或全部为零并非精度证书。将其保留为实现一致性而非独立oracle验证。
4. 有界T保证a±有界，但不自动约束网络Jacobian或所有参数梯度；需要实际gradient记录。
5. 原tanh参数化的边界最优需闭包/极限理解；其具体SSL固定a,kappa没有任意分布最优保证。
6. 新视觉Brownian通道、运输MLP权重、144分析格、独立评价三池是本次建议。不会作为合作者原文已定计划或既有正结果引用。

原计划与补充原样保留。以上限定只是把来源与新的统计/实施判断分开，不在reference文件中静默修正原文。
