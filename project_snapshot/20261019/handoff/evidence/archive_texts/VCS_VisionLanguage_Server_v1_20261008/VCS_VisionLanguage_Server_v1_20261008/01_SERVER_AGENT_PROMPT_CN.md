# 发给服务器 Agent 的接入消息

请按附件 `VCS-VL-SERVER-v1-20261008` 准备视觉—语言实验。它是当前VCS estimator论文的一个候选应用，不改变V6主理论，不取代正在进行的图像SSL/ImageNet实验，也不要求回退仓库。

## 本轮目标

先在Flickr30k Entities建立**有限标注、同图区域—短语匹配**：冻结视觉和文本双塔，比较VCS、平衡Logistic、同目标RFF以及条件softmax任务对照；再视阶段结果做一个中等数据规模的identity adapter实验。首先验证estimator和任务是否一致，不从零训练大CLIP。

读取顺序：`00_README_CN.md` → `docs/02_SERVER_PLAN_CN.md` → `docs/03_DATASET_ATLAS_CN.md` → `docs/04_FALLBACKS_AND_DECISIONS_CN.md`。来源与既有工作见`docs/05_SOURCES_AND_RELATED_WORK_CN.md`。`contracts/task_manifest.yaml`是逻辑计划，不是现有trainer配置；本包没有完整的数据下载或训练runner。

## 接入后先做什么

1. 核对当前HEAD及已有VL目录/数据/授权。包基于`6f30f9b5cf85b8af1b4780a24c2103824d96c3cb`，不要为此checkout旧版本。复用P72/P79/P80可复用的模型加载/ID清单，但其topic/exact对照不是新任务，不能拼接旧数值。
2. 按图片身份建立split，解析实体/短语/框/多目标集合，审查许可和实际资源。主数据取不到，先在未查看新EVAL前选择RefCOCOg UMD；不是临时从未知镜像凑图。
3. 运行包内参考测试：`python -m unittest discover -s tests -v`。移植条件配对到仓库并补dataset/缓存/精确矩阵与chunk梯度检查。
4. 仅用TRAIN/DEV做原CLIP基础能力检查，先报告数据质量、候选数量、天花板和输入问题。随报告提交资源实测，再按项目预算流程冻结VL1-10/20。

## 四个必须明确的实现点

- 原始`J`不改；每图先构造P/Q，P的区域边缘均匀；Q是同图边缘乘积，**不删除碰巧正确的配对**。一图一个总权重，不能把所有pair直接均值后让候选多的图占更大权重。
- `C=+1/-1`是来自P/Q的机制标签，不是每个Q配对都语义错误。所有合法phrase aliases和多目标集合由标注记录；ID只用于管理，不送入评分器。
- 固定特征上正单调`a*cos+b`只会改标定，不会改变排序；要获得新的排名能力用同类MLP或adapter。VCS和JS函数类、输入、数据、搜索预算对齐。
- 估計面板用CAL选共同J，任务面板用DEV选Top-1，分开保存；已知有限模型才有oracle，图像端不编造S真值。candidate priors、独立图片数和多图cluster不确定性都要记录。

## 结果不好时怎么办

不要无方向地扩大网格。按`docs/04`先归因为数据、表示、拟合、目标/评价对齐或没有新增价值，再选择一条备用：RefCOCOg/RefCOCO+上下文、DCI细描述、COCO/Flickr多caption、CUB细粒度描述、CLEVR-Ref+诊断。每次保留原负结果，新的目标/数据/预算先冻结。

旧VL实验没有验证VCS天然更耐噪；平衡JS也有相同后验意义，不撤掉这个对照。FLAIR/FG-CLIP等可以作强特征参照，但额外预训练数据单列。不要为了VL扩成生成模型、自动数据清洗系统或开放VQA。

## 第一轮交付

`VL1_INTAKE.md`、`VL1_DATA_AUDIT.json`、`VL1_FROZEN_PROTOCOL.yaml`、原CLIP的DEV表、门槛检查与实际运行预算。使用`VL1-*`逻辑ID与负责人协调P编号；不自行占用已有编号。不删除别的任务checkpoint，不自动下载所有18项数据，不因本包停止获批SSL作业。
