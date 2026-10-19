# 数据与文献入口：服务问题，不按数据集铺开

核对日期：2026-10-11。下面是本轮核验的公开原论文/官方数据入口，以及仓库来源；不是穷尽性新颖性证明。数据字段“待本机核验”意味着尚未下载或检查字节、许可与split。

## 1. 原计划与仓库来源

| ID | 来源 | 本包引用的内容 |
|---|---|---|
| O1 | 用户原文件 `Variational_CS_QMI_Research_Plan (1).pdf`，本包sources内原样副本 | 一般P/Q、混合参考、S/J、gap、风险解释、稳定性限制及应用选择 |
| O2 | 用户 `VCS_QMI_Estimator_Package_v1_Supplement.md`，历史副本 | 二次增量几何、对JS和原生标定的提醒；运行状态已过时，独有性需结合后续修正 |
| R0 | https://api.github.com/repos/W-Yinghao/VCS_QMI/branches/main | 工作基线ba3bc3e；交付核对e6eee63，18:18:11 UTC；QN/X1协议，不是完成结果 |
| R1 | https://github.com/W-Yinghao/VCS_QMI/blob/ba3bc3ee6ae8e5906ba309795c296931e6f72c00/reports/R4/R4_RPC_DELTA_20261011.md | 平衡RPC同一泛函、RuLSIF、标准统计工具与待新增内容 |
| R2 | https://github.com/W-Yinghao/VCS_QMI/blob/ba3bc3ee6ae8e5906ba309795c296931e6f72c00/reports/R5/R5_PHASE_REPORT_CN_20261011.md | 当前VL已完成的有限对照，参考融合/多正例解释的边界 |
| R3 | https://github.com/W-Yinghao/VCS_QMI/blob/ba3bc3ee6ae8e5906ba309795c296931e6f72c00/reports/R5/R5_QNORM_PROTOCOL_FROZEN_20261011.yaml | QN已批准的12个运行范围，按实际状态接续，不额外重复 |
| R4 | https://github.com/W-Yinghao/VCS_QMI/blob/ba3bc3ee6ae8e5906ba309795c296931e6f72c00/reports/R4/R4_T2_REPORT_20261011.md | 实际配对统计的标准实例验证；不能用固定critic区间覆盖全部学习误差 |
| R5 | https://github.com/W-Yinghao/VCS_QMI/commit/e6eee6386c441ae70f8b48071d68df91026d95c8 | RefCOCO复制R4-VL1，9个完整运行的冻结协议/提交清单；本次未读到结果 |

上述GitHub状态为截点记录，不是实时队列；自然REG/GAIN结果在这些来源中不存在。

## 2. REG 数据入口

| ID/数据 | 已从官方来源核验的事实 | 首轮用途/需要进一步核验 |
|---|---|---|
| D1 Zurich Summer | 官方记录提供20个4波段NIR/R/G/B原始tile和标注；建议报告训练/验证/测试tile ID；数据归档约138MB | 首轮轻量成像/已知扰动；20tile不能当成patch级大样本；相邻tile/同次采集关联披露。许可按归档文件，不将仓库代码MIT当数据许可 |
| D2 BF–SHG | CoMIR官方代码链接旧记录3874362；搜索可核实全尺寸版本4550300含206对2048²图像，40训练、25训练调参、7配准调参、134测试 | 明确版本，不混合旧crop与full-size文件；按组织来源核对。网页打开有失败，本机字节访问未确认；测试不用于方法选择 |
| D3 EPFL RGB-NIR Scene | 官方页面为477对、9类别，来自不同曝光的RGB/NIR图像，说明processed/aligned | 可选非医学替代；残余位移/运动、场景邻近性检查；不能假设每像素原始对齐精确 |

D1: https://zenodo.org/records/5914759  
D2旧入口: https://zenodo.org/record/3874362  
D2全尺寸: https://zenodo.org/records/4550300  
D3: https://www.epfl.ch/labs/ivrl/research/downloads/rgb-nir-scene-dataset/  
CoMIR代码及数据指针: https://github.com/MIDA-group/CoMIR

不直接引用D1页面中未经核验的物理分辨率单位；pilot先用像素/归一化坐标。真实物理TRE需读取可靠元数据后再报。

## 3. GAIN 数据入口

| ID/数据 | 已核验事实 | 使用限制与角色 |
|---|---|---|
| D4 AV-MNIST / MultiBench | 官方文档提供图像+音频loader，官方任务10类识别；由MNIST/FSDD及音频增强组成 | 适合完整小型流程；音频可能重复配对，核对原始ID；已融合特征不用于声称原始模态时延 |
| D5 CREMA-D | 原始项目为7442段、91演员的音视频，含情绪标签/评分；Git LFS媒体，普通ZIP可能只有指针 | 正式自然多模态备用；演员级拆分；核对视频解码。TensorFlow及部分镜像只有audio，不可替代音视频数据 |
| D6 MM-IMDb | MultiBench列出海报/文字及genre任务入口 | 备用的任务相关信息获取，重回V+L但不是候选框排序；按电影身份切分，核对raw images/text |
| D7 Kinetics/视频音频 | MultiBench/AdaMML有接口或训练数据说明 | 仅在本机已有或访问明确时扩展；失效视频/下载量记录，不首批大规模下载 |

D4: https://multibench.readthedocs.io/en/latest/start/datadownload.html  
MultiBench官方项目: https://github.com/pliang279/MultiBench  
D5: https://cheyneycomputerscience.github.io/CREMA-D/  
D5原仓库: https://github.com/CheyneyComputerScience/CREMA-D  
D6/D7任务说明: https://multicomp.cs.cmu.edu/multibench/datasets/

数据门户描述不等于本机已可用。本轮不申请临床受限数据、不绕过许可证，也不将用户其他EEG项目的许可沿用到这里。

## 4. 两条优先路线的直接相关文献

| ID | 工作与核验入口 | 需要继承/比较的内容 |
|---|---|---|
| L1 | CoMIR, NeurIPS 2020. https://proceedings.neurips.cc/paper/2020/hash/d6428eecbe0f7dff83fc607c5044b2b9-Abstract.html | 学习跨模态稠密表示并用配准评价；使用aligned pairs。原生pipeline与共享优化器对照分开 |
| L2 | MIDA-group/CoMIR官方实现. https://github.com/MIDA-group/CoMIR | 训练/注册脚本，部分Fiji/Matlab依赖，公开权重可能见过官方训练图；移植不自动叫逐字复现 |
| L3 | Hasanbelliu et al., Information Theoretic Shape Matching, TPAMI 2014. DOI 10.1109/TPAMI.2014.2324585. https://pubmed.ncbi.nlm.nih.gov/26353150/ | CS已有点集分布对齐用途；这不等于局部图像joint-product，但禁止称首次CS配准 |
| L4 | EDDI, ICML 2019. https://proceedings.mlr.press/v97/ma19c.html | 预期信息增益指导特征获取，Partial VAE；不是我们首次提出按需信息 |
| L5 | AdaMML, ICCV 2021. https://openaccess.thecvf.com/content/ICCV2021/html/Panda_AdaMML_Adaptive_Multi-Modal_Learning_for_Efficient_Video_Recognition_ICCV_2021_paper.html | 逐视频片段的模态选择，完整计算/任务权衡 |
| L6 | Stochastic Encodings for Active Feature Acquisition, ICML 2025. https://proceedings.mlr.press/v267/norcliffe25a.html | 主动获取已有超越贪心CMI的潜变量策略；需要匹配任务与访问权限后比较 |
| L7 | Learning-To-Measure: In-Context Active Feature Acquisition, ICML 2026. https://proceedings.mlr.press/v306/kobayashi26a.html | 跨任务、缺失特征及无需每任务重训的AFA；非首批必复现，但创新审查应涵盖 |
| L8 | InfoNet, ICML 2024. https://proceedings.mlr.press/v235/hu24h.html | 无测试时优化的神经MI估计，摊销路线的直接先行工作 |
| L9 | Snaauw et al., Mutual information neural estimation for unsupervised multi-modal registration of brain images. https://arxiv.org/abs/2201.10305 | 已有神经MI指导配准；本次仅核对预印本，不猜测其正式venue |
| L10 | Active feature acquisition via explainability-driven ranking, ICML 2025. https://proceedings.mlr.press/v267/guney25a.html | 已有用实例级解释排名训练获取策略的路线 |

RPC: https://arxiv.org/abs/2103.11275  
RuLSIF: https://arxiv.org/abs/1106.4729  
Proper composite losses: https://www.jmlr.org/papers/v11/reid10a.html

这些工作的任务、算力、监督和原生模型并不都一致。公平目标对照先共用架构/数据；原生系统保留其关键结构并另列，不把方法削弱后当论文原算法。

## 5. 近期配准方向入口，尚不作为已复现基线

- CARL，CVPR 2025：https://openaccess.thecvf.com/content/CVPR2025/html/Greer_CARL_A_Framework_for_Equivariant_Image_Registration_CVPR_2025_paper.html 。关注等变配准，多步3D场景；本轮2D pilot不硬比其原表。
- MIFNet，预印本：https://arxiv.org/abs/2501.11299 。单模态训练、跨模态关键点描述，可能是正式扩展的强参照；本次不猜测正式场馆。
- Search-MIND，2026预印本：https://arxiv.org/abs/2604.09743 。无训练多模态配准及宽收敛范围；只作为方法审查入口，不把摘要数字当复现实验。

## 6. 备选路线的已有工作入口

- DANN：https://www.jmlr.org/papers/v17/15-239.html 。已有域对抗表示学习。
- ReBias：https://arxiv.org/abs/1910.02806 。新增去偏方案需先核对其任务与依赖约束。
- VID（CVPR 2019）：https://openaccess.thecvf.com/content_CVPR_2019/html/Ahn_Variational_Information_Distillation_for_Knowledge_Transfer_CVPR_2019_paper.html 。已有变分信息蒸馏。
- CRD：https://arxiv.org/abs/1910.10699 。已有对比表示蒸馏。
- Xu et al., Multi-Task Learning with Knowledge Distillation for Dense Prediction（ICCV 2023）：https://openaccess.thecvf.com/content/ICCV2023/html/Xu_Multi-Task_Learning_with_Knowledge_Distillation_for_Dense_Prediction_ICCV_2023_paper.html 。已使用CS蒸馏处理多任务稠密预测；只是换CS损失不是新贡献。

## 7. 补检索要求

自然pilot冻结前，服务器对选定路线再查近两年直接竞争方法，保存论文主张/统计对象/监督/开源实现/评测协议/本项目新增部分六列。发现重合时先修改贡献归属，不能只添加一条citation后仍保留“首次”表述。
