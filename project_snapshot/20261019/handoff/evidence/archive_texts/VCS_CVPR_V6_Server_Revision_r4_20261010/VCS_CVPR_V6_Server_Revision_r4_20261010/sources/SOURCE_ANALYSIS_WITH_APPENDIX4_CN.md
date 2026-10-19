# VCS-QMI 项目：进展、问题、文献组织模式与下一步（2026-10-10）

依据：GitHub `W-Yinghao/VCS_QMI` 截至 2026-10-10 18:39 UTC（最新 commit `55155f2`；10-08 以来 141 个 commit）、本地克隆（10-08 14:34 快照）、以及本次对约 120 篇文献的核实（场馆均经官方页面或 DOI 核对；未核实的在清单里标明）。全文不写算力预测。

---

## 0. 三个问题的直接回答

**项目进展到哪了。** 五条线里四条已经关闭：估计器基准（P85/P86/P108/P151/P152）、SSL A-P3（P104–P154，owner 10-09 叫停）、真实图像测量（P116–P150）、CLIP-adapter R 线（P49–P94，负结果）。只有 VL 线在跑：VL1/VL2（冻结特征）、VL3（全微调 12 格 × 3 seed，108 次运行）已完成并出报告，Table C（VL1-22，检测器候选框）和 add. 7（Large 20 epoch）在跑、未读。会议（10-08）把论文重定位成"估计器论文"，但服务器端随后执行的 VL 设计并不是会后方案里的 CLIP 损失 + λJ，而是 RefCOCO 系列"给定框内排序"的 J-only 方案。

**最严重的问题是什么。** 全部证据链里没有一处 VCS 对 matched JS 的可宣称优势：合成基准上 JS 的后验 MSE 等于或低于 VCS（高斯阶梯 5/5 层，N 轴 3/4，差 2–10 个 pooled sd）；SSL 上两者在共享评分器下持平、各自调优后 JS 略优；测量线上两者功效相同；VL3 上 +0.07 [−0.01, +0.15]。于是"估计器优越性"这句话目前只能对核方法（RFF/Nyström/KDE/rLS）和单负样本 KL 估计器说，而文献认为这两者是"预期结果"而不是贡献。其余问题（见 §2）大多是报告与流程层面的，可修。

**VL 应用是否按设想"保留 CS 优势、突出估计器"。** 没有。VL3 的设计使它只能回答"全微调后，一个有界的图内成对损失能否追平图内 softmax"，答案是平局（+0.01 [−0.35, +0.36]）。图内 P_G/Q_G 是精确有限分布，什么都没有被估计；每张图的目标 S_G=(m−1)/(m+1) 是候选数的常数（sympy 已核：m=2 时 1/3，D_CS 上限 2 log m）；评价是 argmax cosine，对 a、b 和任何单调变换不变；唯一的"估计器视角"数字 J_recal ×1.33 对 JS 也是 1.33。CS 的七条性质（有界 critic、精确回归间隙、线性类闭式解、后验标定、正负解耦、Hoeffding 界、对称有界 f-散度与 DPI）在 VL 线里一条也没被读出。详见 §4。

---

## 1. 仓库进展（自 9/28 至 10/10）

### 1.1 估计器基准线（合成数据，已关闭）

| 单元 | 内容 | 结果 |
|---|---|---|
| P85/P86 | 85 格 × 3 seed，同目标 S 的后验 MSE | 对核方法 5/5 pad、4/4 N、15/15 阶梯全部"better"（pad d=100：VCS 0.354 vs RFF 0.540、rLS-tanh 0.541、KDE 0.537；N 256→16384：VCS 0.364→0.029 vs rLS-tanh 0.491→0.104） |
| P86 对 JS | 同网络同数据 | JS 在绝大多数格更低（N 16384：0.0255 vs 0.0292；I 10：0.018 vs 0.023）；唯一例外 d_total=100（0.354 vs 0.509，seed 双峰） |
| P108 | 变分读数 Ĵ vs 平方 plug-in | plug-in 更准 70/72（偏差）、69/72（RMSE）；对两种拟合损失都成立 |
| P151 | 三面板重汇总 | 高斯阶梯 VCS .065→.027，JS .061→.018；"bounded S 并不唯一可分辨"；**脚本 bug 未修**（见 §2） |
| P152 | 稳定性字段 | 梯度范数 CV：InfoNCE 最低，JS 第二，VCS 第三；非有限步 0；VCS 饱和最少、无溢出（但阈值未按尺度匹配） |
| CS-K-native | 经典核 CS，median heuristic | d=20 时只恢复自身真值的 ≤13% |

r3 的理论到证据台账（10-08）结论："no further r3 measurement is needed"；可写的句子被压缩为"可训练的同目标 critic（VCS 或 JS）在合成数据上远优于固定特征核估计器；VCS 与 JS 在精度与稳定性上相当"。

### 1.2 SSL A-P3 线（CIFAR 尺度，已关闭）

| 比较 | 数字 |
|---|---|
| CIFAR-10 线性，5 seed | A-P3 89.02±0.04 vs SimCLR 88.31±0.21，+0.72 [+0.48, +0.95]；kNN −0.45 [−0.92, +0.02]（SimCLR 4/5 领先） |
| 强增广 | A-P3 88.25 vs SimCLR 89.49（反转；差异来自 SimCLR 的小裁剪增益 +1.58） |
| matched JS，CIFAR-10 | +0.21 [−1.03, +1.46]（close） |
| matched JS，CIFAR-100 | 共享评分器 +0.80 [+0.06, +1.54]；各自调优 −0.49 [−0.77, −0.20]（JS 优）；JS 抗腐蚀更好（mCA −1.36） |
| 固定尺度 vs 学习尺度 | +6.11（C10）、+3.26 [+2.01, +4.51]（C100）：这是最大、最稳的效应 |
| P130 ResNet-50 CIFAR-100，3 seed（10-09） | 线性 64.39±0.82 vs JS 63.23±0.75 vs SimCLR 59.21±0.62，+5.17 [+2.68, +7.67]；kNN −1.59 [−3.25, +0.08] |
| P154 ImageNet-100 pilot（10-09，单 seed） | VCS 73.58 / JS 74.72 / SimCLR 76.66 线性；kNN 64.92 / 65.44 / 70.24；预注册触发器触发但未跟进 |
| 其他 | P128 SimCLR 调参撤回；P138 K=16 仅 VCS-C10 非劣；P145 污染 ε 0.10/0.20 无差异；P153 弱裁剪 −2.20 |

关键结构性发现（P122）：固定 (a,κ) 的评分器是"训练装置而不是测量 critic"，对同一 s 的标量重读可再回收 +0.07–0.09 的 J；让 critic 学习恰恰摧毁 SSL 结果。所以论文"理论→估计器→SSL"的链条在 SSL 这一环断开。

### 1.3 真实图像测量线（已关闭）

- Table 3（P116/P142）：固定 SimCLR 特征上，VCS ridge-tanh 与 matched logistic 检测功效 0.75/0.99/0.97 vs 0.82/1.00/0.94；同目标 RFF critic 0.59/0.96/0.80；条件 HSIC 0.15–0.58；零假设 0.03–0.07。
- Table 4（P143）：colour 0.1 在两个平衡后验编码器里远比 SimCLR/VICReg 可检测（H 0.31–0.95 vs 0.08–0.19），**属于目标族共性而非 VCS**；且与配对准确率效应无关（SimCLR 的准确率掉得一样多）。
- P144/P150：同样的样本上 J_h ≤ 0（−0.0003 到 −0.0075）而检测率 0.60，所以 Tables 3–4 只能读成"检测表"，S 的量级（1e-3）不可估。
- P149 + add. 1：两视图 J 在可逆坐标下稳定到 ~2 SE；三条路线全都能沿 Brownian 通道排序；VCS-MLP ≈ JS-MLP（t=0 相差 ≤0.005）；RFF 受函数类限制（0.11–0.34 vs 0.81–0.87）；S_plugin 在真实图像上不可用（0.80 vs J 0.35）；z 位的 J 高于 h 位 0.10–0.17 尽管 z=g(h) → W11（跨位比较不是 DPI 陈述）。
- 唯一保留的 VCS 专属优点"成本 30–35×"在匹配收敛（logistic@50）下消失：0.04–0.09 s vs 0.05–0.09 s。

### 1.4 CLIP-adapter R 线（已关闭，对 VCS 负面）

原生标定只在"critic 在中等依赖关系上训练"时成立（P50 setting 2：ECE 0.023 vs Platt 0.022），在精确配对上失败（0.046/0.089 vs 0.004/0.036）；绝对阈值迁移和 cosine 阈值完全相同（tanh(a·cos+b) 单调）；J-only adapter 损失 CLIP 的实例对齐（R@1 0.267 vs raw 0.368）；注入错配时 VCS 损失最多 R@1（13.1 点 vs InfoNCE 1.9，P80）；错配识别 JS 在每格最好（P72 R3）；P94 表明 P72 中 VCS 漂移小是选中训练预算的产物。

### 1.5 VL 线（进行中）

**会后计划与实际执行的替换。** 你的 10-08 方案是：COCO 双向检索，L = L_CLIP − λĴ，对照 A–F（含 D 核 J、E J-only、F CS-Aligner），再到 ROCOv2 与文本引导分割。同日服务器移植了外部包 `VCS_VisionLanguage_Server_v1_20261008`，写了 CVPR gap analysis，§5 建议保留 RefCOCOg 给定框设计（"isolates the critic"），你批准"就按这个设计来"。仓库中没有任何文件引用会后方案；实际跑的 VCS 行是方案里的对照 E（J-only），"任务损失"是图内多正例 softmax 而不是对照 B 的 CLIP/InfoNCE，C 与 F 不存在，D 只在冻结特征上有。

**VL1/VL2（冻结特征，10 个特征集 × 3 数据集）。** CLIP B/16 crops：raw 65.99 → VCS 68.80±0.48 / JS 68.34±0.08 / SigLIP 式 67.41 / RFF 67.40 / softmax（DEV 选）70.75。强特征上 critic 几乎不加分（FG-CLIP 2：72.29→72.51；SigLIP 2：75.48→76.00，JS 76.43；So400m −0.07）。softmax 在 30/30 格领先（像对像 0.8–2.9）。VL1-14：critic 搜索无必要；VL1-15：CAL J 与学习后 Top-1 的 Spearman 0.90–0.96，报告自称"一致性性质，不是外部验证"。

**VL3（全微调，12 格 × 3 seed，gate 108/108）。**

| 读数 | 结果 |
|---|---|
| 汇总 VCS − softmax | +0.01 [−0.35, +0.36]（预设措辞"VCS matches the task loss"） |
| 逐格 | RefCOCOg×CLIP-B +0.89 [+0.03, +1.75]，RefCOCOg×SigLIP2-L +0.47 [+0.18, +0.76]；RefCOCO+×SigLIP2-B −0.54 [−0.88, −0.21]，RefCOCO+×SigLIP2-L −0.48 [−0.82, −0.13] |
| VCS − JS | +0.07 [−0.01, +0.15] |
| J_recal 比 | VCS/softmax 1.326 [1.288, 1.363]；JS/softmax 1.328 [1.286, 1.371] |
| add. 4 调度（20 vs 10 ep，B/16） | +0.37 [−0.11, +0.85]；softmax 是唯一在 20 ep 下降的目标（−0.20 [−0.41, −0.00]） |
| add. 5 分层 | 长短表达式 −0.02 [−0.36, +0.33]；方位词交互 +0.35 [+0.14, +0.56]（未校正、8 个非独立格） |
| add. 6 跨数据集迁移 | −0.10 [−0.39, +0.18]；RefCOCO↔RefCOCO+ 分区错误只剩 389/390 of ≈1590 DEV 图 |
| 微调增益 | 相对冻结 critic +9.0 到 +23.2，三种目标对零样本基线的增益 +14.8/+14.7/+13.9（可互换） |

**Table C（VL1-20/21/22）。** 干净检测器 AP 36.4（gate D ≥33 通过），DEV 候选框 recall@100 97–99%；12 个 Table C 作业已喂入（优先于 add. 7，add. 7 暂停并取消 11 个待跑）。Grounding DINO-T DEV 行为外部锚点（但其 RefCOCOg val 60.5 vs 论文 67.46 未复现）。

**仍未做的事。** 官方 val/test 一次性读取没有日期；STATUS 页面停在 10-08；RefCOCO 系列数据许可"确认中"。

---

## 2. 问题清单（按严重度）

### 高

1. **全链条无 VCS-over-JS 证据。** P86/P151：JS 在高斯阶梯 5/5 层、N 轴 3/4、两条 batch 轴 3/3 更低（2–10 pooled sd，不是报告写的"1–3 sd"）；P108 plug-in 更准与拟合损失无关；P152 VCS 不是最稳；SSL 各自调优 JS 优；测量线功效相同（JS 在 own-statistic 下 2/9 更高，common-statistic 下 3–4/9）；VL3 +0.07。台账自己写着"no 'superior' wording unless a claim survives against JS (none has so far)"。
2. **VL 设计不能产生估计器结果**（§4）。目标是候选数常数，η 两值，评价对 critic 不变，无 InfoNCE 对照，绝对分数从未被消费。
3. **VL3 的"任务损失"是弱化基线。** 图内 2–4 路 CE，logit 尺度 10（CLIP 原生 100），b 在损失中抵消，CAL 选 2–4 epoch，是唯一 20 epoch 下降的目标；在批内跨图 InfoNCE（CLIP/MDETR/GLIP/Grounding DINO 真正用的损失）缺席的情况下，"VCS matches the task loss"读作"VCS 追平一个未调参的基线"。P80 已给出 InfoNCE 36.3 vs VCS 26.7 的先验。
4. **SSL 调参不对称、增广区制依赖、DEV-only。** SimCLR 在 2 视图/200 epoch 搜到的 recipe 从未在 4 视图/800 epoch 下重调（P128 撤回），VCS 侧 ~13 个筛选格 + ~70 个探索格全部在 DEV 上选择；A-P3 领先只在标准增广；所有 headline 数字都在 5000 图的选择集上，官方测试集从未读过；+0.72 的量级与选择乐观度同量级。
5. **ImageNet-100 负结果且触发器未跟进。** CIFAR 尺度的结论不外推；SimCLR 也不在其发表 recipe 上。
6. **估计器基准的"better"裁决依赖预注册排除过的负样本预算不对称。** P85 写明"product = the same-target comparison partner of the kernel rows"，P86 却用 K=8 cyclic（8N 负样本）对核方法的 N；在 product 负样本下 VCS 在 N=256（0.563 vs 0.491）、N=1024（0.223 vs 0.179）、gaussian I=2 输给 rLS-tanh，裁决应是"on par"。闭式核方法本可零成本用全部 N² 交叉对。
7. **VL3 seed 规则在评价数据上选择。** 哪些格加 seed 1–2 由 seed-0 DEV Top-1 决定，留在单 seed 的五格恰是 VCS 落后的格（7 格池 +0.29 vs 12 格 +0.01）。add. 3 已修复，但 B/16 报告和 add. 1 interim 是在有偏规则下发布的。
8. **核方法对照偏弱。** RFF 选在网格边界（m=4096, multiple 0.5）且自认优化受限；Nyström 带宽未调；KDE 在 d=2 被 Scott 规则以 3× 击败；pad 轴上没有 ARD 核——该轴正是各向同性核必然失败的轴，所以"VCS 胜核方法"部分是函数类结果（JS 同样胜）。

### 中

9. **P151 汇总脚本 bug 未修。** `scripts/p151_e0_aggregate.py` 第 129 行把 JS 行的 native 值填进公共 J 字段（Gaussian I=6 in-batch JS 应为 −0.0409，文件写 −0.0653）；E-Stability 面板同样受影响；GitHub main 10-10 仍是同两行。
10. **J_recal ×1.33 近乎同义反复。** 它是 CAL 上重拟合的同类仿射 critic 的公共 J，即 VCS 直接最大化的量；JS 同为 1.33；softmax 的 CAL 选中 epoch 系统性更早（[2,2,3] vs VCS [6,7,7]），没有 epoch 匹配对照；迁移无后果。add. 6 的"the estimator objectives' advantage"超出证据。
11. **add. 7 在一个随后被抹掉的 interim 池上冻结。** 冻结引用 add. 4 五格池 +0.37 [+0.08, +0.66]，add. 4 完整六格报告（12 分钟后）改为 +0.24 [−0.16, +0.64]。
12. **外部基线未复现仍在表里。** FG-CLIP 2 Base COCO box 64.8 vs 论文 74.9（Large 55.06 被排除却 Base 保留）；MDETR RefCOCO+ 76.00 vs 79.52；GD-T RefCOCOg 60.5 vs 67.46；ReCLIP 启发式在 val 上调过。
13. **任务狭窄且不可对标。** 图内 2–4 候选，随机 ≈40%，同类别占 90%（RefCOCOg）/99.3%（RefCOCO/+）；VL3 的 80–91% 与任何已发表数字不可比；Table C 是第一个标准读数，但它评价的是微调后编码器而不是 Table A/B 的冻结 critic，两套表从不评价同一模型。
14. **Table 3 省略了存在差异的 fixture。** P116 的 9 格里 6 格在 recipe-VCS 特征上，JS 显著更强的格全在那 6 格；Table 3 只报 SimCLR 特征的 3 格，P142 也只扩展了 SimCLR fixture。
15. **测量线的主读数不一致。** P118 的规则只在 nested_exact + 采样读数下通过，P144/P150 却用 nested + 精确 Q 作主读数；JS 的"3 of 4 criteria"是对全有或全无规则的事后软化。
16. **R 线设计缺陷。** P72 测的是"离 raw CLIP 多远"而不是鲁棒性（服务器自认"The design error is mine"）；P79 在看到 P72 后设计；P71 阈值"按 owner 放松"；P94 单 seed。
17. **添加 seed 无多重性控制、合并筛选 seed。** 3-seed"inconclusive"后延到 5 seed 再标"clear"（P120 add.1 +0.80 [+0.06, +1.54]；P138 add.2）；A-P3 的 5 seed 含筛选 seed 0。
18. **线性与 kNN 读数不一致且规则跨单元不一致。** P107 add.2 要求两读数一致才给排序；P130 add.1 在 kNN −1.59 下仍写"+5.17 clear"。
19. **VL 冻结表选择不对称。** softmax 行在 DEV（报告集）上选，VCS/JS/RFF 在 CAL 上选；正文用 query 加权、表用 image-macro。
20. **数据与许可。** RefCOCO/+/g 标注（来自 MDETR json）与 COCO-2014 图（symlink 到 2017）无许可声明；CLIP（WIT-400M）/SigLIP 2（WebLI）预训练污染未披露；add. 6 分区错误。
21. **P116 成本声明不成立。** "0.04 s vs 1.2–1.4 s"比的是 A1 对过度预算的 JS；JS 在 50 次闭包已收敛，P142 同批次计时 logistic@50 0.08 s vs VCS 0.11 s。

### 低

22. P86 §5 与表不符（InfoNCE EVAL 界应为 log 1024=6.93 而非 log 256；xor 8→10 排序概率 NWJ 0.93/DV 0.99/VCS 0.96 而非 1.00；product DV 4→6 反序为 0.00）；"S-Kernel RFF (selected)"按字母序取子组（N=256 应为 pooled 0.521±0.021 而非 0.508）；事后引入 1−MSE/S 指标；SMILE 作废行（1.9e7）在主汇总中无标记；P108 正文仍写"guaranteed lower readout"而只附加更正；P118 第 27 行勘误（JS seed 2 colour h 应为 0.44）未改写；P142 承诺的 Wilson/McNemar 区间缺失；P149 addendum 无"Not claimed"节；所有测量在编码器的预训练图像上做，无留出图像复现；两种 bootstrap 约定给出两套分辨率表（P86 vs P151）；基础设施（GPU 混用、requeue）使墙钟比较只在同机型内有效；VL3 interim 文件未加 supersession 头；J_recal 范围改过两次；add. 5 六个次要对比无多重性校正却突出其一。

---

## 3. 文献：与我们类似的工作如何组织

### 3.1 CVPR/ICCV/ECCV 目标函数类论文的固定骨架（21 篇已核实，2022–2026）

1. **先量化标准目标的一个具体缺陷**：批内耦合（SigLIP）、假负例（FFF）、共享温度（Logit Standardization）、梯度冲突（DCR）、过度分散（COVER）、最小充分表示（Rethinking MSR）、噪声视图（RINCE）。
2. **最小改动的即插即用损失**，开销写明；理论可选，通常 1–3 条刻画损失的命题（FroSSL、PFML、DCR），21 篇里 10 篇无定理，COVER 的界放附录。
3. **插入网格**：≥3 个宿主模型/骨干 × 3–6 数据集（SSL/度量学习），或 ≈11 个零样本分类集 + Flickr/COCO 检索（+分割）（VL）；主基线由作者在同数据同 recipe 下复现（SigLIP、SILC、FLAIR、nCLIP）。
4. **一条基线没有的次级证据轴**：收敛速度（FroSSL）、小批量（SigLIP）、密集预测迁移（SILC/FLAIR/COSMOS）、标签噪声（PFML）、跨域（ADL）、自建性质基准（EqBen、CountBench、FLAIR 细粒度检索）。
5. **只有新损失才支持的分析节**。
6. **声明层级**："consistently improves within the matched setting"是常态；"competitive"只在配有区分轴时被接受（FroSSL；HCL 明说"not another SOTA method"）。

**平局何时被接受（VL 线核实到的三种形式）：** (a) 大规模持平 + 受限区制严格胜出 + 效率（SigLIP：≤16k batch 胜，更大"similar"，ICCV oral）；(b) 标准检索持平 + 针对性诊断大幅提升（NegCLIP：ARO +10–20，ICLR oral）；(c) 平均略优 + 新能力（ProLIP：不确定性，ICLR 2025）。纯平局从不单独撑起论文。

**全预算下的实际增益很小**：DCL +6.4 仅在 bs256/200ep；RCL +2.0 vs MoCo v3；RINCE +0.4–0.7；SigLIP 峰值 +0.2；FroSSL 持平；SSL-HSIC +0.5 vs BYOL；CorInfoMax 持平；f-MICL +0.2–2.3；MINC +1 vs 父目标。Augmentations-vs-Algorithms 独立估计损失变化在 ImageNet 线性探针上移动不到 1 点。这些论文赢在区制、同代码等预算交换、第二任务族、损失专属分析。

**评审指南原文（CVPR 2026/ICCV 2025）**："the fact that a proposed method does not exceed the state-of-the-art accuracy on an existing benchmark dataset is not grounds for rejection by itself"；"Be specific about novelty… MUST be backed up with specific references"；rebuttal 不得要求大量新实验——所以**缺失的基线在投稿后无法补**。

### 3.2 MI/依赖估计器论文：全部在 ML 场馆，有固定协议

- 所有以估计器为主贡献的已核实论文都在 ICML（MINE、Poole、CLUB、InfoNet、D3RE）、NeurIPS（f-GAN、FLO、BMI、Lee–Rhee、Butakov、f-DIME、VCE）、ICLR（SMILE、DIM、MINDE、CS-IB）、AISTATS（McAllester–Stratos）、AAAI（Mroueh）或 TMLR。**没有一篇 CVF 主会论文以估计器为 headline**；最近的 CVF 条目（RankMI CVPR 2020、MIRO ECCV 2022、CS 多视图聚类 CVPR 2020/21/23）都把估计器当嵌入工具、按任务指标评价。
- 2019–2025 稳定模板：统一理论节（把新目标放进 DV/NWJ/InfoNCE/JS 家族，至少一个定理）→ d=20 高斯阶梯 + cubic（偏差/方差/MSE 随真值、batch、critic 结构）→ 2023 起 BMI 40 任务套件 → 2024 起带构造真值的真实数据（Lee–Rhee 同类采样 + 二元对称信道）→ SMILE 自洽性测试（独立基线、数据处理、可加性）→ 1–2 个下游应用（从不作主证据）→ 成本列。**2026 年缺 BMI 或自洽性即是拒稿信号。**
- "优越性"的标准已迁移：2018–20 追阶梯；2020–22 偏差方差权衡、批量无关、稳定性；2023–25 跨异质分布的真值误差、d>20 小 N 的样本效率、尾部鲁棒性、标定的置信区间与失败标志、推断速度、跨数据集 MI 排序。精度平局只在配有新操作点或性质时被接受（CLUB 上界、InfoNet 速度、VCE 样本效率、MINDE 自洽性）。
- f-DIME（NeurIPS 2024）已证明 JS/logistic critic 是阶梯上最好的 f-散度 critic；DEMI（logistic critic 直接读密度比）至今未发表。**一个在阶梯上与 JS 打平的新 critic 目标，按现行标准不是贡献，除非附带分离轴。**
- 文献中对有界二次形式最自然的分离轴候选：固定 d 下更低的每样本方差（VCE 式 N 扫）、高散度区制不饱和（D3RE 区制）、闭式值域带来的标定区间或置换零分布（HSIC/C2ST 式"依赖度量 + 检验功效"框架，此时 JS 平局无关紧要）、RuLSIF 式的有界相对比→更快速率与与复杂度无关的渐近方差（可证明而不仅观察）。

### 3.3 CS 散度论文：文献真正证明的 CS 优势，以及排序任务里哪些会丢失

文献证实的 CS 优势：(1) 混合分布闭式（Kampa IJCNN 2011、Nielsen ICPR 2012、CSRAE JMLR 2022）；(2) 无密度比/无变分界/无 critic 的 Gram 矩阵批量泛函（Jenssen 2006 → DDC、CVPR 多视图聚类三篇、CS-IB ICLR 2024、CS-adv UAI 2024、CS-MVIB、CS-MIM TIP 2026）；(3) 高斯假设下 D_CS ≤ min(KL, 反 KL)（CS-IB Thm 1；CS-adv Props 1–4 需非重叠条件）；(4) 条件 CS 的检验功效与忠实性（TPAMI 2025、UAI 2024）；(5) 数值稳定、少超参（ACM MM 2025、ICCV 2023 蒸馏）；(6) 对噪声/离群的鲁棒（TPAMI 2014 配准、CS-MVIB）；(7) 允许无配对数据的分布级对齐（CS-Aligner ICLR 2026）。

场馆模式：**没有 ECCV 论文用 CS；CVPR/ICCV 的 CS 论文（EAMC 2020、SiMVC/CoMVC 2021、DeepMVC 2023、Xu ICCV 2023）都把 CS 当聚类头或蒸馏损失里的组件，从不论证散度优越性；所有以 CS 为 headline 的论文都去了 ICLR、UAI、ACM MM、TPAMI、TIP、Neurocomputing、Information Fusion、JMLR。**

若估计器只作为排序任务的损失，(1)(3)(4) 这些"作为度量的性质"对 Recall@K 不可见；(5)"无超参"对核估计器不成立（带宽仍在）；批内排序正是 InfoNCE 近乎最优的区制（与 P41 和 oracle 分析一致）。能存活的只有 (6) 对噪声配对的鲁棒、(2) 无 critic 的跨 batch 稳定、(7) 无配对数据——而 (7) 已被 CS-Aligner 用标准估计器占据。

### 3.4 REC/候选框排序：绝对分数唯一能改变报告数字的地方

标准 REC 协议（MAttNet、UNITER、ViLBERT、TransVG、MDETR、ReCLIP、CLIP-VG、CVPR 2024 零样本）全部是 argmax：Acc@IoU 0.5，图内 hinge/softmax/BCE/InfoNCE 或直接回归；**没有一篇报告配对分数的 ECE 或可靠性图**。绝对、有界、标定的分数只在"这对 (box, expression) 是不是指称"而不是"哪个框最好"的协议里有意义：

| 槽位 | 基准与指标 | 当前做法的缺口 |
|---|---|---|
| 无目标拒绝 | gRefCOCO N-acc/T-acc（GRES CVPR 2023）、Ref-ZOM 一对零（ICCV 2023）、R-RefCOCO mRR/rIoU（TIP 2024） | ReLA 加独立分类头仍漏 ~40% 无目标；GREC 显示 MDETR 的 1−P(no-object) 需手调阈值（Pr 37.0→51.2 随阈值 0.5→0.9） |
| 跨图候选池 | Cops-Ref Full（CVPR 2020，13 图池） | 图内 softmax 从 ~78 掉到 ~26 |
| 阈值跨域迁移 | RefCOCO→gRefCOCO/ReferIt/RefGTA/RefEgo；VIRO Balanced Accuracy（arXiv，自称 CVPR 2026） | VIRO 需在 ImageNet 上按标签标定 CLIP 阈值 |
| 选择性预测 | MA-PaPSP AURC（ICLR 2026）；ECE under shift（Murugesan ECCV 2024） | 明说 CLIP cosine 未标定，需局部重归一化 |

MLLM 线（RC-GRPO，2026 预印本）用训练出的拒绝替代分数，并明确否认概率标定，暴露 P-acc/N-acc 由训练负例比例决定。**"REC 里没有标定分数"是文献记录在案的空缺。**

### 3.5 已核实的文献清单（节选，按用途）

- **CVF 目标函数骨架**：SigLIP（ICCV 2023 oral）、FroSSL（ECCV 2024）、VNE（CVPR 2023）、HIER（CVPR 2023）、SILC（ECCV 2024）、FLAIR（CVPR 2025）、Logit Standardization（CVPR 2024 highlight）、FFF（CVPR 2024）、nCLIP（CVPR 2023）、Jiang et al. 模态结构（CVPR 2023）、β-CLIP（CVPR 2026）、COSMOS（CVPR 2025）、Context-Aware（CVPR 2025 highlight）、DCR（CVPR 2026）、COVER（ICCV 2025）、PFML（CVPR 2025）、EqSim（ICCV 2023 oral）、Count-CLIP（ICCV 2023）、DCL（ECCV 2022）、RINCE（CVPR 2022）、Rethinking MSR（CVPR 2022 oral）、RCL（ECCV 2022）、HCL（CVPR 2023）。
- **VL 目标与绝对分数槽位**：CS-Aligner（ICLR 2026）、CLIP-Lite（AISTATS 2023）、CyCLIP（NeurIPS 2022 oral）、NegCLIP/ARO（ICLR 2023 oral）、FG-CLIP（ICML 2025）、PCME++（ICLR 2024）、ProLIP（ICLR 2025）、NCR（NeurIPS 2021）、RCL（TPAMI 2023）、L2RM（CVPR 2024）、Murugesan（ECCV 2024）、MA-PaPSP（ICLR 2026）、MDETR（ICCV 2021）、GLIP（CVPR 2022）、Grounding DINO（ECCV 2024）、ReCLIP（ACL 2022）、RefCLIP（CVPR 2023）、GRES（CVPR 2023 highlight）、Ref-ZOM（ICCV 2023）、Cops-Ref（CVPR 2020）、R-RIS（TIP 2024）、CVPR 2024 结构相似零样本 REC；未核实场馆：GREC（arXiv 技术报告）、VIRO（arXiv，声称 CVPR 2026）、RC-GRPO（arXiv）。
- **MI 估计器**：MINE（ICML 2018）、f-GAN（NeurIPS 2016）、Poole（ICML 2019）、SMILE（ICLR 2020）、McAllester–Stratos（AISTATS 2020）、DIM（ICLR 2019）、CLUB（ICML 2020）、Mroueh（AAAI 2021）、FLO（NeurIPS 2022）、BMI（NeurIPS 2023）、Lee–Rhee（NeurIPS 2024 D&B）、InfoNet（ICML 2024）、MINDE（ICLR 2024）、Butakov（NeurIPS 2024）、f-DIME（NeurIPS 2024）、VCE（NeurIPS 2025）、D3RE（ICML 2025）、HSIC 检验（NeurIPS 2007）、RuLSIF（NeurIPS 2011）、C2ST（ICLR 2017）；预印本/工作坊：MIST（ICML 2026 workshop）、DEMI、Abdelaleem。
- **CS 散度**：Jenssen 2006（J. Franklin Inst.）、Kampa（IJCNN 2011）、Nielsen（ICPR 2012）、Hasanbelliu（TPAMI 2014）、Hoang（TIT 2015）、Rami（SPIC 2016）、DDC（Neural Networks 2019）、EAMC（CVPR 2020）、SiMVC/CoMVC（CVPR 2021）、CSRAE（JMLR 2022）、DeepMVC（CVPR 2023）、Xu（ICCV 2023）、CS-IB（ICLR 2024）、CS-adv（UAI 2024）、CS-MVIB（Inf. Fusion 2025）、条件 CS（TPAMI 2025）、GCSD（Neurocomputing 2025）、CSD 检索（ACM MM 2025）、CS-Aligner（ICLR 2026）、CS-MIM（TIP 2026）。
- **SSL 目标（ML 场馆）**：Tschannen（ICLR 2020）、Spectral（NeurIPS 2021 oral）、SSL-HSIC（NeurIPS 2021）、CorInfoMax（NeurIPS 2022）、f-MICL（TMLR 2023）、Poly-View（ICLR 2024）、Matrix-SSL（ICML 2024）、X-CLR（ICLR 2025）、SimDINO（ICML 2025）、MINC（AISTATS 2026）；Explicit-MI（ICASSP 2025，不是未发表）；预印本：Augmentations-vs-Algorithms、LeJEPA。

---

## 4. VL 应用是否"保留 CS 优势、突出估计器"：逐条判断

三个独立视角（估计器优势、CS 理论、CVPR 领域主席）得出同一结论：**没有**。

### 4.1 它实际检验了什么

"同一编码器特征、同一图内候选集上，三种损失形状（1 − mean_G J_G 配 T=tanh(a·cos+b)；matched 平衡 logistic；多正例候选 softmax）作用于同一个 cosine 的两标量函数，微调后 argmax-cosine 排序是否不同。"答案：不同不了。

### 4.2 结构性原因（每条都可验证）

1. **什么都没被估计。** P_G、Q_G 是由标注矩阵 A 决定的精确有限分布，J_G 是有限和；P86 的后验 MSE 优势来自 d=20–100、N=256–16384 的样本拟合，在 2–4 个格子上没有对应物。
2. **目标是已知常数。** A 块对角 ⇒ η_G 在真对上为 (m_G−1)/(m_G+1)，假对上为 −1 ⇒ S_G=(m_G−1)/(m_G+1)（m=2:1/3，m=3:1/2，m=4:3/5），D_CS,max=2 log m_G。**sympy 已核**（`oracle_analysis/vl_identities_check.py`）。DEV 上 J_recal 0.14–0.25 是仿射 critic 对该常数的恢复比例，是"cosine 标定指数"，没有依赖含量；跨数据集 J 比较被 m 分布混淆。
3. **图内构造把对 InfoNCE 的唯一目标级优势交了回去。** 每图 CS 散度上限 2 log m_G，正是 oracle 分析里 InfoNCE 的 log-候选数天花板；m≤4 把"强依赖、多候选"这个有界性可能起作用的区制排除在设计之外，而且没有跑批内 InfoNCE 来证明。
4. **η 两值 ⇒ 二次损失退化为平方分类损失**，与 logistic 共享总体最优，VCS−JS +0.07 和 1.33 vs 1.33 是预期而非发现。这正是 9/27 brief 里预检 A 的"二值化不是标定"的退化区制（P1 违反）。
5. **评价对 critic 不变。** argmax cosine 与 a、b 无关；T*=2P(joint|·)−1 的绝对值没有被 Top-1、Acc@IoU、Table C 消费（P2 违反）；标定被断言（VL1-10）而未测量。
6. **置信界不适用。** 负样本是正样本的确定性函数而不是 Q 的独立抽样，1/n+1/m 方差界与 Hoeffding LCB 前提不成立；J_own 还在选择集上读。
7. **D_CS 在 VL 线里从未计算。** "CS 散度"只以名字出现。
8. **VL1-11 不是 O1 的检验。** 先验从评价图自身标注读出，对 softmax 的帮助（+0.56）与 VCS（+0.66）、JS（+0.63）相同。
9. **VL1 的"估计器作探针"**（ρ=0.90）按 addendum 3 自述是"一致性性质，不是外部验证"，JS 每个符号相同。
10. **可能显示真实图像估计器性质的弱信号测量已知失败**（P150 J 不可估；P72/P80 无鲁棒优势、JS 错配识别最好、R@1 26.7 vs 36.3）。

### 4.3 哪些 VL 表述能存活、哪些不能

存活：微调对冻结 critic 的增益 +9.0 到 +23.2（三目标相同）；弱特征上 critic 的排序增益（CLIP +2.81 [+1.62, +4.00]，ReCLIP +3.38 [+3.01, +3.75]）但强特征上 ≤+0.52——这是关于 F2r critic 类的陈述，不是 CS 目标；一个描述性数据集模式（RefCOCOg +0.60，RefCOCO −0.11，RefCOCO+ −0.47）。

不能存活："the VCS critic is a calibrated estimator that also ranks"、"improvement is in calibration"、"fine-tuning reverses the frozen-feature picture"、"gain largest for the estimator objectives"、"theory prediction O1 holds on real data"、"the estimator objectives' advantage is in what the encoder exposes"、以及任何"VL 中保留了 CS 优势"的句子。

### 4.4 CS-Aligner 的占位

"CS 散度用于视觉语言对齐"已被 ICLR 2026 占据（Parzen 估计器 + 保留 InfoNCE，FID 11.27 vs 23.67，检索 88.7/87.4 vs 87.2/85.4）。会后方案的 D（核 J）与 F（CS-Aligner）两个对照从未运行，所以即便 VL 出现增益，也无法归因于混合参考估计器而不是 CS 散度或有界损失。任何 VL 报告都必须对 CS-Aligner 定位，目前为零。

### 4.5 什么样的 VL 设计才"保留 CS 结构"

唯一保留 CS 结构的 VL 形式是**测量形式**：全局 P 对独立池乘积 Q（冻结或微调后的 VL 特征）、已知或植入的 S、critic 在一个块上拟合另一块上读、1/n+1/m 界与 Hoeffding LCB、核同目标与批内 InfoNCE 对照、非排序主读数——即把 P86 协议移植到 VL 特征上。会后方案"CLIP 损失 + λJ"保留了损失结构但通过 R@K 读取（P2 违反）且不可归因，只应在归因门之后运行。

---

## 5. 下一步：正面主张 → 预测 → 推翻条件

排序依据：先做不需要新训练的、能改变论文句子的；再做决定投稿场馆的；条件性大实验最后。每项只写依赖关系，不写算力。

### A. 用现有 108 个 VL3 检查点做绝对分数读数（仅评价）

- **主张**：标定后验 T*=2P(joint|x,y)−1 是任务损失给不了的东西：一个固定、未调的阈值能分开匹配/不匹配对并在跨图池化下存活；softmax 的 b 抵消、分数无绝对含义。
- **做法**：(a) 每条查询的候选跨 13 张 DEV 图池化（Cops-Ref 协议）+ VL1 定义过而 VL3 删掉的 all-objects 集（raw CLIP 52.95 vs 65.99）；(b) 以同类别错图构造无目标子集，VCS/JS 用固定阈值 T≥0（后验 ≥1/2）判定，softmax 用 CAL 调优阈值与未调阈值，raw CLIP 作底；(c) 延伸：gRefCOCO N-acc 固定阈值、无分类头。措辞与阈值在第一个数字前冻结。
- **预测**：跨图池化下 VCS/JS 的 Top-1 下降不超过 softmax 下降的一半；无目标子集上 VCS 固定阈值与 CAL 调优 softmax 相差 ≤2 点、高于未调 softmax ≥10 点，JS 与 VCS 相差 ≤1；gRefCOCO N-acc ≥56.4（ReLA）。
- **推翻**：VCS 固定阈值落后 CAL 调优 softmax >5 点，或三种目标的池化下降相同 ⇒ 标定性质在 VL 无下游后果，VL 线仅保留平局句，J_recal 整体删除。

### B. VL3 平局句的基线有效性检查（两格）

- **主张**："全微调后 VCS 追平任务损失"在正确调参的 softmax 与跨图对比基线下仍成立。
- **做法**：RefCOCOg×CLIP B/16 与 RefCOCO+×SigLIP 2 B/16，softmax 在 CAL 上重调（logit 尺度 {10,50,100}，编码器 lr {1e-5,3e-5}），加批内跨图 InfoNCE 臂（batch 内 32 图全部区域），3 seed，其余冻结，10 epoch 表为主。
- **预测**：调优 softmax 移动 DEV Top-1 <0.3；InfoNCE 与 VCS 相差 ±0.5 内。
- **推翻**：调优 softmax 或 InfoNCE 高出现 softmax ≥0.5 且领先 VCS ≥0.5（区间在 0 以上）⇒ 108 次运行表是对未调基线的比较，平局句撤回，VL3 只以"微调增益 +9.0–23.2"进附录。

### C. 对 JS 的分离轴：污染恒等式（合成真值 + VL 特征）

- **主张**：二次（Brier 型）目标的梯度 2(T−η)(1−T²) 在饱和对上消失，critic 已经确信的污染对对 VCS 的拉扯小于对 log-loss critic；ε=0 时两者相等，污染下 VCS 的干净后验 MSE 退化更少；有界目标值域 [0,1] 给出 logistic 没有的标定区间。配套的闭式去污染映射 η_ρ=(1−ρ)η/(1−ρη)、η=η_ρ/((1−ρ)+ρη_ρ)（**sympy 已核**）。
- **做法**：d∈{20,64,128} 高斯与 cubic，Monte-Carlo S/η 真值；用乘积对替换 ε∈{0,0.1,0.3,0.5} 的联合对；报告干净对后验 MSE、S 误差、20 次抽样方差；方法 VCS、matched JS、批内 InfoNCE（N=256）、SMILE、KDE-CS、uLSIF/RuLSIF；同一次运行补齐阶梯/cubic、BMI 任务密度、图像上的 SMILE 自洽性，使估计器节满足 2026 协议。VL 特征版本：RefCOCOg FIT 全局 crop–表达式对，植入错配率 ρ，用步骤 D 的标签信道真值。
- **预测**：ε=0 时 VCS/JS MSE 相差 <10%；ε=0.3、d=64 时 VCS MSE ≤0.7× JS，20 次区间排除 1.0；KDE-CS 误差 ≥3× VCS；InfoNCE 在真 KL 超过 log 256+2 后饱和而 VCS 继续跟踪；VCS 通过三项自洽性测试；去污染后 MSE/干净 MSE 在 1.0–1.2。
- **推翻**：ε=0.3 时 VCS/JS 比 ≥0.9 或 JS 更低 ⇒ 污染也不是分离轴；论文把 VCS 写成"平衡 logistic 家族里带精确间隙恒等式与有界目标的 CS 目标成员"，删除所有"优于 JS"。

### D. VL 特征上的真值测量块（P86 协议移植）

- **主张**：P86 的优势迁移到 VL 特征：d=512 时神经 VCS critic 的后验 MSE 比每个核同目标估计器低 ≥3×（核方法停在平凡估计附近），与 JS 相差 ≤1.5×，以概率 1.00 排序相邻 ε 层，LCB 以 ≥95% 覆盖真 S；批内 InfoNCE（B=32）读数 ≤log 32≈3.47 nat，无法排序信道 MI 超过它的层（K=80 类允许到 log 80≈4.38）而 S 仍可估。
- **做法**：冻结 CLIP B/16 与 SigLIP 2 B/16 的 crop/表达式特征；通过 K 类标签信道（翻转率 ε）配对，使密度比等于信道比、S(ε) 闭式；P=信道对，Q=独立池乘积；FIT/TUNE/EVAL/TRUTH 分块同 P85；方法 VCS-N（cyclic 8 与 in-batch）、matched JS、InfoNCE（B=32 与 256）、S-KDE、RFF、Nyström、rLS-tanh；读数后验 MSE、|J_eval−S|、五层排序概率、固定 critic 的 LCB 覆盖。
- **预测**：P86 形状的表，VCS-N 与 JS 在 0.02–0.08，四个核行距 S 0.05 以内；InfoNCE-32 顶层排序概率 <0.8；LCB 覆盖 0.95–0.99；核方法 log MSE–log N 斜率平于 −0.3，神经 critic 约 −0.6。
- **推翻**：任一核估计器在 d≥512 时进入 VCS 的 1.5× 内，或 LCB 覆盖 <0.90，或 VCS-N 排序概率 <0.95 ⇒ VL 线没有估计器内容，从证据链整体移除。

### E. 用 108 个检查点做两项无训练重读

- **主张**：(a) J_recal 比是 cosine 标定指数：按候选数分层 m∈{2,3,≥4} 后 VCS/softmax 比在每层落在同一 1.24–1.43 带内并逐层等于 JS/softmax；天花板归一化后跨数据集 J 差至少缩一半。(b) 图内训练不改变全局区域–短语依赖：以独立池 Q 的跨图 DEV 对、CAL 上拟合 a·cos+b、DEV 上固定 critic 读 J 并带 1/n+1/m 界与 LCB，三目标的全局 Ŝ 相差不到一个 LCB 宽度。
- **推翻**：(a) 比在 m 层间移动 >0.15 ⇒ J 带候选数结构，VL1/2/3 所有跨数据集 J 陈述按层重述。(b) VCS 或 JS 的全局 Ŝ 高于 softmax 超过两个 LCB 宽度且在换数据集后仍存在 ⇒ VL 线第一个非同义反复的估计器效应，升格为论文的 VL 测量结果；否则 J_recal 降为一行标定注记，删除 add. 6 的"advantage"。

### F. 会后方案"CLIP 损失 + λJ"的归因门（仅在 D 通过后）

- **做法**：四臂同 λ 同 batch：CLIP；CLIP+λJ_VCS（全局批内乘积负样本、等混合、tanh critic）；CLIP+λJ_JS；CLIP+λD_CS-KDE（CS-Aligner 估计器）；另加 VCS-only 与 InfoNCE-only；λ 减半、加倍各一次。主读数 COCO 5k R@1，次读数 E(b) 的全局 Ŝ。
- **主张与预测**：三个正则臂互差 ≤0.3 R@1、高出 CLIP 0–1.5；VCS-only 落后 InfoNCE-only 5–10 R@1（P80 模式）。
- **推翻**：J_VCS 臂在同 λ 下高出 J_JS 与 D_CS-KDE ≥1.0 R@1 且在 λ 减半加倍时仍在，全局 Ŝ 差超两个 LCB 宽度 ⇒ 升格为可归因的学习应用；否则降为与 CS-Aligner 的一行相关工作比较。

### G. 只在 CVPR 路线仍开放时才做的两项大实验

- **G1 等预算 VL 预训练区制扫（SigLIP/RCL 模板）**：同编码器（ViT-B/32 或 B/16）、同 CC3M、同 recipe；臂为批内 InfoNCE、SigLIP sigmoid、matched JS、VCS 单独、核 J（至少一格）、CS-Aligner 基本版；batch {256,1k,4k,8k} × 错配率 {0,20,50%}（公开 NCR 噪声索引）；COCO 5K、Flickr30K 双向 R@1 + 零样本 ImageNet；3 seed；协议先冻结。**预测**：batch 256 时 VCS−InfoNCE ≥+1.0（区间在 0 上），8k 时 ±0.5；50% 错配时 VCS−InfoNCE ≥+2.0 且与 RCL 相差 ≤1.0；VCS−JS 每格 ±0.5。**推翻**：没有任一格 VCS−InfoNCE ≥+1.0 ⇒ CVPR 路线关闭，VL 在 ML 场馆论文里缩为一张平局表。仓库自身的 P80 指向这个结果。
- **G2 ResNet-50 ImageNet-1K 100 epoch 同代码损失交换**：SimCLR InfoNCE / matched JS / VCS（A-P3 recipe），batch {256,4096}，线性探针为主、kNN 为辅，256 处 3 seed。**预测**：batch 256 时 VCS−SimCLR ≥+1.0，4096 时 ±0.5；VCS−JS ±0.5。**推翻**：256 处 <+0.5 或 4096 处落后 >0.5（CIFAR 强增广与 P41 指向此）⇒ SSL 节变成一段平局应用。

### H. 分析层面的修正（不需要 GPU，投稿前必做）

1. 修 `scripts/p151_e0_aggregate.py` 第 129 行及稳定性面板的同一错误，重生成 `reports/P151_E0/e_precision.json`、`e_stability.json` 与所有依赖图。
2. 在预注册的 product 负样本伙伴下重读 P85 规则 1（或给核方法匹配的 N² 交叉对负样本预算），明确写出 product 下 rLS-tanh 在 N=256/1024、gaussian I=2 胜 VCS。
3. 加强核对照：ARD/可学带宽 RFF、Nyström 带宽调参、KDE 低 d 更细带宽网格、m>4096 与 multiple<0.5。
4. 统一 RFF"selected"行的 seed 合并（P86 与 P151 一致）；修 P86 §5/§1 文字（InfoNCE 界 log 1024；xor 排序概率；product DV 反序；"worse than trivial"改"at trivial"）；标记作废 SMILE 行；P108 正文删除"guaranteed lower readout"；明确两种 bootstrap 约定中哪一种是注册的。
5. 测量线：删除/改写 P116 成本句；补 P142 的 Wilson/McNemar 与 P143 的二项区间；选定 Table 3 fixture（并补 VCS 特征格或说明省略）；P118 规则在精确 Q 下重判或在论文中说明；W9/W11 全面应用（B5 机制图 (c) 面板改为同位通道阶梯）；P118 第 27 行勘误落实。
6. VL：给 VL3_INTERIM、VL3_B16_REPORT、VL1-12 add. 2 §3 加 supersession 头；Table A 改为对称选择（softmax 也 CAL 选，或全部 DEV 选并标注）；RefCOCO/+/g 许可与 CLIP/SigLIP 2 预训练污染声明；FG-CLIP 2 与 MDETR RefCOCO+ 复现差距在任何数字印出前解决或该行移除。
7. 口径：论文中不得出现"stable alternative to MINE"（无 in-batch 限定）、"VCS is uniquely calibrated"、"converges to S"（无度量）、"lower variance than MI estimators"、"J is a per-draw lower bound"、"more accurate than JS"；VL 文本不写"CS divergence"直到某个 VL 单元真的报告了 S 及其 CS 变换。

### 停止 / 降级清单

- **停**：VL3 add. 7（add. 4 已给出零调度效应；即便 12 格为正也是非标准任务上对未调基线的亚点增益）；所有图内给定框的新 addendum（分层、先验 N 曲线、迁移变体、"估计器视角"、Large 冻结扩展）；DEV 上的序贯数据依赖预注册（DEV 已承载 ≥6 个主读数、~15 个次读数）。
- **让其跑完但只按冻结措辞读**：VL1-22 Table C（第一个标准 Acc@IoU 0.5 读数，Grounding DINO-T DEV 行作外部锚），不开任何逐格跟进。
- **降级**：J_recal（附录或删除；若保留必须 epoch 匹配并写明 JS 1.328 vs VCS 1.326）；VL1-11 先验校正（若引用须用 FIT 估计 p_G）；VL1/VL2 Table A（附录、对称选择）；估计器探针 ρ=0.90（一行一致性注记）；add. 5 方位词对比（不报告）；P143 Table 4 的"更可检测"（不得写成优点）；VL 线整体从"展示估计器优势的应用"降为"第二应用：微调后与候选 CE 平局，加上若成立的绝对分数读数"。
- **不开官方 val/test**，直到最终一次通过，且只对经上述步骤存活的行。

---

## 6. 论文形态与场馆

**形态 A（证据现在支持的）：ML 场馆估计器论文**（TMLR/AISTATS/ICLR 类；先例 CS-IB ICLR 2024、f-MICL TMLR 2023、MINC AISTATS 2026——"理论 + 核比较 + JS 平局 + 平局应用"唯一被接受过的场馆族）。

1. 引言：混合参考 CS 二次依赖 S=E_M[η²] 的可训练估计；只做三个声明——精确间隙恒等式、标定后验读数、高维下对核同目标估计器的优势；应用在摘要里就写成平局。
2. 背景与定位：CS-QMI 与 Parzen/Gram 估计器（Jenssen 2006；CS-IB）、分类器两样本/DIM/f-DIME 的 JS critic、CS-Aligner（P_V 对 P_L 的分布匹配 + KDE + 保留 InfoNCE——与成对依赖是不同问题）、RuLSIF（有界相对比）。
3. 理论（合作者冻结）：S、J、S−J=E_M[(T−η)²]、T*=tanh(PMI/2)；一条新命题走 RuLSIF 路线（有界相对比的速率/方差），条件对照 CS-IB/CS-adv 先例写明；两条小恒等式（每图天花板、污染映射）。
4. 估计器基准（主证据，约四成篇幅）：P86 + 阶梯/cubic（d=5/20/更大，joint 与 separable critic）+ BMI 任务密度 Monte-Carlo 真值 + 图像 SMILE 自洽性 + uLSIF/RuLSIF 与 HSIC 基线 + 成本列；JS 报告为相等（或略优）；若 C 成立，污染分离轴作唯一 headline 图。
5. 学习应用（各一表，平局写成平局）：SSL A-P3（89.03±0.06 vs 88.32±0.30；强增广 88.25 vs 89.49；JS 88.81，+0.21 [−1.03, +1.46]）；REC 微调 12 格表（+0.01 [−0.35, +0.36]；VCS−JS +0.07）仅在 B 通过后，Table C 并列，微调增益作背景。无 J_recal、无先验校正、无探针。
6. 真实图像测量：植入属性检测（声明 J 对弱信号不可估）+ 若 A 为正的绝对分数读数——VL 数据能服务估计器声明的唯一位置。
7. 局限：JS 平局；CS 作为度量的优势未被排序任务调用；DEV-only 与单次官方测试；recipe 由估计器线设定。附录：VL1/VL2 对称选择表、critic 筛选、带宽与 batch 敏感性、完整流程台账。

**形态 B（CVPR 2027 要求的）：只在 G1 出现交叉时提交。** 一句诊断、一句声明（"批解耦的有界成对目标，小批量与错配对下保持检索精度，并带标定的配对分数"）；§2 量化 softmax 不足；§3 损失 + 1–3 条命题；§4 三块证据按顺序：受限区制、匹配算力下标准检索/零样本平局、绝对分数任务（ECE 迁移、选择性 ITM AURC、gRefCOCO N-acc 固定阈值）；每块都要对 CS-Aligner 与 JS 臂归因。没有交叉就不投 CVF 场馆——CVF 对 CS 的先例是"CS 使某种视觉能力成为可能"，平局行提供不了这种能力。

**预计的审稿决定句**（若按现状投 CVPR）："VCS is indistinguishable from the JS critic in every reading, so the CS estimator is not what is being evaluated." 形态 A 回避这句话的方式是把 JS 平局写成设计事实并把分离轴放在对核方法、对 InfoNCE 天花板、以及（若 C 成立）对污染上。

---

## 附录 1：本次核实的恒等式

`oracle_analysis/vl_identities_check.py`（sympy 1.14，输出见 `vl_identities_check_output_20261010.txt`）：

- 图内 m 候选、块对角标注、区域均匀 P_G、乘积 Q_G：η_true=(m−1)/(m+1)，η_false=−1，S_G=(m−1)/(m+1)，D_CS=2 log m。m=2/3/4：S_G=1/3, 1/2, 3/5。
- 污染 P_ρ=(1−ρ)P+ρQ：η_ρ=(1−ρ)η/(1−ρη)；逆 η=η_ρ/((1−ρ)+ρη_ρ)。

## 附录 2：本次输入的来源

GitHub（API 与 raw）：VL3_COMPLETE_12CELL_REPORT、VL3 add. 1–7 冻结与报告、VL1_22_TABLE_C_FROZEN、VL1-20/21 报告、VL1_FROZEN_PROTOCOL、VL1_10_11/12/13/14/15 报告、VL2 冻结与报告、VL_LESSONS_20261009、PROGRAMME_GAP_ANALYSIS_CVPR_20261008、EXPERIMENT_ORDER_20261008、P130 add.1、P138 add.2、P145 add.3、P153、P154、P149 add.1、fig_mechanism 说明、`scripts/p151_e0_aggregate.py`。本地克隆：P85/P86/P108/P151/P152、r3 台账、P100–P148、P116–P118、P142–P144、P149/P150、MV6 handoff、P49–P94、SECOND_APP_FINAL_SUMMARY、V2 reconciliation、CVPR 证据台账、配置文件。会后文档：`output/VCS_QMI_VLA_会后实验方案_2026-10-08_CN.md`、会议纪要与发言稿。

---

## 附录 3（10-10 补充）：VCS–matched JS 比较的角色，以及被遗漏的 P134

**数学事实。** 平衡混合 M=(P+Q)/2 下，η=(p−q)/(p+q)=2P(joint|x,y)−1。VCS 的 J 是对 η 的平方损失（Brier）：J(T)=S−E_M[(T−η)²]；matched JS 是对同一后验的对数损失（logistic），σ(f*)=p/(p+q)，故 tanh(f*/2)=η。两者是同一后验的两个严格 proper scoring rule，总体最优相同。这不是 SSL 场景的巧合，也不限于平衡情形：对 α-混合参考，带类先验权重 α 的 logistic 回归给出同一后验 p/(αp+(1−α)q)。

**比较的正确角色。** 在每个应用里 matched JS 都保持 critic 形式不变、只换损失，所以它隔离的恰是论文的贡献（二次变分形式），而不是 critic。它应作为对照而不是对手：经典理论预期对数损失（MLE）在后验估计上至少同样有效，Brier 不会更准，所以 P86/P151 的"JS ≥ VCS 后验 MSE"是预期结果，告诉论文不要在精度轴上宣称。台账的规则"no 'superior' wording unless a claim survives against JS"应改为"只在平方损失有别于对数损失的轴上宣称"。

**平方损失专属的内容。** (i) 精确间隙 = 后验 MSE，优化即报告，任意 T 的 J(T) 是 S 的认证下界；(ii) 嵌套增量恒等式 S_B−S_A=E_M[(T_B*−T_A*)²]（JS 的间隙是 Bregman 量，无三角不等式；P118 中 VCS 全规则通过、JS 3/4）；(iii) 线性类闭式解（= RuLSIF α=½ 岭回归；成本优势在匹配收敛下消失，P116）；(iv) tanh 输出上的 redescending 影响：对 Q 标签对，梯度 2(T+1)(1−T²) 在 f=2/4 时为 0.28/0.005，logistic 为 1.96/2.00（sympy 核）。

**P134（2026-10-05，之前综合中只列名未读）支持 (iv)。** P85 高斯阶梯格（d=20，I=4，N=4096，3 seed，product 负样本），污染 ε∈{0.01,0.05,0.1,0.2}：每 0.1 ε 的最大归一化位移（自身分辨率单位）VCS 0.88/0.18/0.62 vs JS 1.40/0.13/1.64（独立对/离群/重尾），预设读法"supported"（JS 不满足"同样有界且在 20% 内"）。更有信息的是误差分解：ε=0.2 时对污染后真值的误差变化 VCS −0.032/−0.006/−0.049 vs JS −0.066/−0.026/−0.127（原生单位），InfoNCE/NWJ 在离群下 ≈ −1.0 nat。限制：单一设置、单一 critic 类、无差值区间、JS 的归一化超额主要来自小 ε 的加权。这是目前唯一在理论预言方向上 VCS 优于 JS 的估计层面单元，应扩展（d、N、critic 类、VL 特征）为分离轴 headline（§5 步骤 C）。

**框架暴露的更深问题。** S = 2·PE_{1/2}（Yamada 等 RuLSIF 的相对 Pearson 散度，α=½）= 三角判别/2；D_CS^(M)=2 log((1+S)/(1−S)) 是单调变换；且 S/2 ≤ D_JS ≤ (log 2)·S（g(η)=((1+η)/2)log(1+η)+((1−η)/2)log(1−η) 的 Taylor 系数全正，g(1)=log 2），即 S 与 JS 散度双 Lipschitz 等价。因此"估计器是创新点"必须精确为"对相对 Pearson/CS 依赖的神经变分估计 + 有界 critic + 精确间隙工具箱"，并对 RuLSIF/LSMI 与分类器路线（DEMI、C2ST、f-DIME）定位。

**SSL。** 同意用户的判断：A-P3 的评分器固定，没有 critic 拟合，J 是编码器的训练目标；作为编码器目标 S 与 D_JS 是 η 的双 Lipschitz 等价泛函，平局是预言。SSL 中的 VCS–JS 平局既不是失败也不是估计器证据；SSL 的真实发现是固定尺度效应（+6.11/+3.26）。

---

## 附录 4（10-10 晚）：对外部审查意见的回应与勘误

一份独立审查（用户转来）核对了本文档与附录 3。以下是经我复核后的勘误；凡标"接受"者，本文档正文以此处为准。

### 4.1 接受的数学勘误

1. **"什么都没被估计"措辞过强（§4.1–4.2）。** 准确说法：图内 P_G、Q_G 的和可精确计算，但 T_φ 对配对后验的逼近仍需学习，对未见图片的泛化仍是统计问题。VL 设计检验的是"已知对应结构下的监督学习"，不是"对未知自然图文总体依赖的高维数值估计"。结论不变：Top-1 不测量后验质量，P86 的估计优势在此没有对应物。
2. **"不同损失由构造上不可能改变排序"错误（§4.1"不同不了"）。** 固定编码器时 argmax 对 a、b 与单调变换不变，这是对的；但 VL3 训练编码器，两种损失的 f-梯度不同（(T−c)(1−T²) 对 T−c），可以得到不同编码器与不同排序。正确陈述：设计里没有任何估计器性质起作用的通道；损失之间的差异只能通过训练中对配对的梯度加权产生，这是损失形状效应，实测 ≤0.07。
3. **附录 3"对数损失在后验估计上至少同样有效，Brier 不会更准"：删除。** 不存在这样的一般结论。在受限/错设函数类下，Brier 风险恰是后验 MSE 加常数，所以 Brier 的最小化者按定义有更低的后验 MSE（审查给出的两点反例成立：交叉熵选 B，平方风险选 A，A 的后验 MSE 0.00405 < B 的 0.0100）。由此，P86 中"JS 后验 MSE 略低"不是理论预期，而是有限预算下的优化效应（VCS 梯度在饱和处消失，拟合更慢；P152 的"饱和最少"与此一致）。这是一个可查否的预测：预算或 N 增大时差距应缩小或反转。
4. **附录 3 把(i)精确间隙/认证下界与(ii)嵌套增量恒等式列为"平方损失专属"：分类错误。** 二者是目标与读数的性质：任何 T（含 logistic 拟合后 T=2D−1）都满足 J(T)=S−E_M[(T−η)²]，嵌套增量恒等式是关于总体最优 critic 的恒等式。平方损失专属的只有"优化即报告"（训练目标等于所报告的间隙）、线性类闭式解、以及 tanh 输出上的 redescending 影响函数。
5. **α-混合先验公式漏系数。** 带先验权重 α 的 logistic 回归给出的后验是 αp/(αp+(1−α)q)；p/(αp+(1−α)q) 是相对密度比（= 后验/α），不是概率。实质不变：仍可由加权 logistic 恢复。
6. **S/2 ≤ D_JS ≤ (log 2)·S 不蕴含相同排序或相同编码器。** 审查的反例成立：A（η=±1/√2 各 1/2）S=0.5，D_JS≈0.2767；B（η=±1 各 0.225，0 占 0.55）S=0.45，D_JS≈0.3119，排序相反。g(η)/η² 从 1/2 增到 log 2，JS 相对更奖励完全饱和的配对。因此"SSL 平局是预言"改为"平局是可预期的量级结果，不是定理"。
7. **InfoNCE 的 log B 上界限制数值恢复，不禁止上界之外的单调排序**（P86 自己已记录饱和值仍单调移动）。§5 D 中"InfoNCE-32 顶层排序概率 <0.8"撤回。
8. **Shannon MI 的可加性自洽测试不能原样用于 S**（S≤1）。D_CS^(M) 也不可加（M 不在独立副本上分解）。自洽性套件须对 S 重新推导（独立基线、DPI 方向可用；可加性换成对乘积构造的精确 S 的数值核对）。

### 4.2 对三个新方向的修正

- **A（绝对分数/无目标拒绝）。** 在图内均匀先验下 (1+T*)/(1−T*) = p(r|w,G)/p(r|G) = m·p(r|w,G)，故 T*≥0 ⟺ p(r|w,G) ≥ 1/m，是 PMI 符号检验，不是"语义匹配概率 ≥ 1/2"，且随候选数变化；图内训练没有定义无目标事件的概率。修正：要么保留联合对乘积证据并用独立校准集映射到拒绝任务，要么把 P、Q 定义为语义匹配/不匹配分布（无目标表达式进入 Q），此时阈值应是由先验算出的 log 先验比（α-混合/logit 调整），"可迁移"的检验对象是算出的阈值而不是固定零点。原预测数字（N-acc ≥56.4、领先 ≥10 点）无推导依据，撤回；改为先做小规模定义与校准实验。
- **C（污染）。** 恒等式成立，但：两种损失在污染问题上仍共享最优 η_ρ，恒等式本身推不出"VCS 清洁后验误差更小"；参数梯度含 ∇_φ f，logit 梯度衰减不等于整个估计器的抗污染能力；逆映射的导数 dη/dη_ρ=(1−ρ)/((1−ρ)+ρη_ρ)² 最大为 (1+ρ)²/(1−ρ)，ρ=0.5 时 4.5，去污染放大误差；若评价样本来自 M_ρ，还需 dM/dM_ρ=1/(1−ρη) 重加权。原预测"≤0.7×""1.0–1.2"撤回，须先做误差传播推导。
  **P134 的同目标重读（本次从 51 个单元文件的 posterior 字段计算，对污染后真值 η_ε 的后验 MSE，3 seed 配对）：** 绝对水平上 JS 在几乎所有单元仍更低（清洁 +0.006–0.008 的差距基本保留）；但退化量（MSE_ε − MSE_0）在"独立对"污染下 VCS 明显更小，ε=0.1/0.2 时比值 0.53–0.70，三种负样本构造一致，ε=0.2 时 VCS 在三种构造上反超（−0.0086/−0.0048/−0.0018）；在 product 构造下重尾与离群也略有利（0.25–0.87）；但 cyclic8/in-batch 的重尾比值 ≈1.0，outlier_indep ≈1。所以同目标证据是"独立对污染下的斜率优势"，不是绝对优势，也不覆盖所有污染类型。这与 redescending 机制一致（独立对污染正是 P_ρ=(1−ρ)P+ρQ 的形式），可作为 C 的出发点，但不能写成鲁棒性定理。
- **D（VL 特征真值）。** 标签信道只有在类别能从特征无损恢复时才给出观测特征的密度比；一般 p(u,v)=Σ p(c1,c2)p(u|c1)p(v|c2)，且由 DPI，S(U;V) ≤ S(C1;C2)，等号当且仅当类别可恢复。修正：把标签信道量作为上界/参照；或用有限支持构造、或可恢复标签的受控表示。Lee–Rhee 的无误差解码假设须显式检查。
- **置信界。** "负样本非独立 ⇒ 不能构造置信界"错误。以图片为独立单位、critic 在评价外固定，J_G(T)∈[−3,1]，N 张图平均满足 P(|Ĵ−E_G J_G|≥ε) ≤ 2exp(−Nε²/8)；此界对任意固定 T 成立，不要求平方损失训练。它界定的是固定 critic 在未见图片上回收的天花板份额。

### 4.3 文献与定位的勘误

- **RPC（Tsai 等，ICLR 2021）是最直接的先行工作，本文档正文遗漏。** 其式(2) J_RPC=E_P f − αE_Q f − (β/2)E_P f² − (γ/2)E_Q f²，引理 1 f*=(r−α)/(βr+γ) 且 −α/γ ≤ f* ≤ 1/β；命题 1 有界性与 O(√((d+log(1/δ))/n')) 偏差；命题 2 方差 O(c₁/n + c₂/m)；并给出 J_RPC=((β+γ)/2)E_{P'}[r²]（P' 为混合）。取 α=β=γ=1 即 J，f*=η，J*=E_M[η²]=S，且"1/n+1/m 方差界"就是 RPC 命题 2。实验：CIFAR-10/100、STL-10、ImageNet（SimCLRv2 框架，100 epoch，RPC 74.43 对 CPC 73.48；R152-2×SK 78.40 对 77.80）、LibriSpeech；并用 RPC 估计 MI。合作者的稿件 v3/v4 已写明"与三角判别、相对密度比拟合和平衡 RPC 的关系明确陈述……不把新散度或无限制的新变分原理归于已相关的二次泛函"，理论核心限定为"构造与采样分析"。因此真正的新颖性问题是：在 RPC/RuLSIF 已有的二次目标、相对密度比、有界性与方差界之上，本文新增的是什么。目前可辩护的候选：CS 变换与解释、精确间隙作为后验 MSE 的测量用途（认证下界、嵌套增量、标定读数）、tanh 参数化与 PMI/2 的联系、对核 CS 估计器的系统基准。这应先于场馆讨论，并交由合作者处理。
- **f-DIME 的表述改为**："在所测的三种 f-散度与指标上综合推荐 GAN(JS)-DIME，低 MI 时 KL-DIME 略准"，不是"JS 最优"的定理。
- **场馆判断改为风险陈述。** "缺 BMI 即拒稿信号""无 +1 点即关闭 CVPR"等是策略判断；官方指南要求综合权衡新颖性与影响。保留的事实：未找到 CVF 主会以估计器为 headline 的论文；CS 为 headline 的论文均在 ML/期刊场馆。
- **公平性措辞。** 核方法"零成本用全部 N² 交叉对"改为"可按实际算法匹配负样本预算并计时"；应分别报告独立数据量、配对计算量、容量与调参预算，同时保留 product 与多负例两种比较。图内候选 softmax 是任务适配目标，不因 2–4 候选自动成为弱基线，需对称调参；批内跨图 InfoNCE 是补充臂而非唯一合法对照。外部基线复现不符应先排查权重/划分/候选/预处理，分表说明，不直接删行。
- **推翻条件的范围须与主张匹配。** 例如"某核方法进入 VCS 的 1.5× 内"只推翻"该条件下 VCS 大幅优于核方法"，不推出"VL 线无估计器内容"。§5 各步的推翻条款按此收窄。

### 4.4 修订后的下一步优先级

H（分析与报告修复）立即；B（两格对称调参 + InfoNCE 补充臂）有限确认；E（检查点重读）作为诊断；A 先定义语义事件与校准规则再做；C 先做同目标小确认（已给出 P134 重读）并补误差传播推导；D 先补数学条件；F 暂不作为默认；G 不是自动门槛。首要事项是与合作者明确相对 RPC 的新增内容。
