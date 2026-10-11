# 共同评价、数据与资源规范

## 1. 三层对照必须同时有，不能互相替代

**同条件组件层。** 同输入、同可训练函数类、相同数据/PQ、相同初始化分配与调优机会，比较VCS平方拟合、平衡Logistic以及合理的核/密度比估计。同一个T最终共用J时，分别记录fit objective与readout objective。平衡RPC恒等配置作实现check，不重跑成不同方法。

**任务方法层。** 每条路线保留最接近的已发表方法；若复现需要不同的预训练特征或外部教师，不把它当纯损失比较。官方原生代码优先，改变网络/输入后标为适配版本。

**完整系统层。** 最终答案、生成文本、匹配列表、选中数据训练出的学生或视频区间必须被独立评价。内部J、critic功效和teacher一致性不能单独构成应用成功。

## 2. 基础样本而非pair是统计单位

COCO派生集按原image ID，文档按document/page family，视频按原video ID，图像编辑与改述按parent family；生成按prompt/concept family。先划分、再产生mask/caption/负例/渲染/生成样本。

固定FIT/CAL/DEV/FINAL manifest。CAL可以选超参数、checkpoint或校准器；DEV只用于开发期结果解释且必须记录曝光；FINAL只在预注册协议下读。新分割不能消除此前在同一基础图像上的使用。若eval-only数据是POPE、HallusionBench、AMBER、ECCV Caption或GenEval，不能反过来挖负例训练模型。

跨数据源建立`original_asset_id`、`family_id`、`source_domain`、hash/pHash可疑重复列表。没有可获得的预训练清单时标“pretraining exposure unknown”，不得声称完全无污染。

## 3. 选择预算、随机性和置信区间

选定路线的主对照全部运行预定seeds0/1/2，探索种子与确认种子记录分开，不见seed0落后就停止它而只给我们补种子。训练失败也列入表，说明是否因共同硬件/数据错误允许重试。

报告图片/视频/文档/提示级paired bootstrap与训练seed离散度两种不确定性，说明它们各自覆盖什么。样本内很多token/声明/frames不能当独立样本。临界显著结果在三个种子下不升级为普遍定理。

“CI包含0”只能写没检出差异；等效/非劣需先设实际可接受margin，不能事后调margin到通过。多指标与多子集保留完整表，不只突出显著一项。连续预算曲线使用固定横坐标和整曲线摘要，不事后挑一个最优token比率。

## 4. 成本

每个完整系统计入：teacher/评分器计算、vision encoder、selector、prefill、decode、后处理/反事实次数、数据筛选/caption生成及下游训练。训练时teacher成本与部署成本分开，摊销计算的假设明确。

同GPU型号、dtype、分辨率、输入长度、batch和生成长度下实测。报告batch1端到端中位/P95延迟、峰值显存、TTFT、decode速度；FLOPs仅辅助。不用“减少token50%”代替“加速50%”。D1若仍计算完整视觉主干，只能声称后续语言模型节省；D6使用缓存特征不能声称原视频端到端加速。

先测一个完整训练步和一个完整推理循环，再估算预算。小时数未测前留空，不沿用r4小双塔微调的1.3小时估计7B VLM。

## 5. 官方评估与自定义评估分开

锁定 evaluator repo、commit、配置、输入字段、样本清单和数值容差；先跑发布的golden prediction fixture。任何API裁判固定版本、prompt、temperature、缓存与失败重试。只生成平均分而没有原始预测不算完成。

若官方服务停运，使用其已发布本地脚本/groundtruth并记录；公开测试标签不允许用于训练。例如QVHighlights官方README现称Codalab关闭且test真值公开，不能继续要求向已关闭服务上传。[V02]

原始发布分数只用于配置一致的复现锚，不直接写进同一因果对照表。未能复现应查权重、输入/候选、数据子集、预处理与evaluator，不删除不利基线。

## 6. 资源和权限

代码LICENSE、模型LICENSE、数据annotation LICENSE、原始图像/视频权利及API条款分开记录。可下载不等于允许重新分发；本包不附第三方图像、模型、完整论文或字体。下载需账号时不规避；新外部服务或付费裁判按用户已有授权执行。

本包不自动取消R5/R6已批准任务，不自动提交六条线全部网格，不改论文，不打开未用最终测试。服务器依据本包科学优先级和真实既有授权执行范围明确的工作。
