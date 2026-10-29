Publication copy (sample names aliased; Learn2Reg case ids withheld; numbers and conclusions unchanged).

# 配准应用：与 SOTA 的对照分析（2026-10-26）

目标：找到一个配准应用，使 VCS 明确胜过当前 SOTA。本文先核对我们自己对所有基线的结果，再整理文献里的 SOTA，最后判断“和 NCE 比”在论证里应该放在什么位置，以及下一步做什么。
内部数字均来自已有分析文件（来源见各节）；没有新读 TEST。TEST134 的数字来自 10-20 用户批准的一次性 TEST_REPEAT134。

## 0. 结论

1. **在 BF-SHG 上，我们的系统已经是系统级 SOTA，而且差距明显。** 用 VCS 训练的稠密描述子 + 全局网格搜索 + Adam 精修，在 CoMIR 基准的同一批 134 个 TEST core、同样的 1%/5%（<9 / <42 全分辨率 px）阈值上，严格/宽松成功率约 0.75 / 0.98。同一评估框架下重跑官方 CoMIR 只有 0.64 / 0.83，MI 0.57 / 0.78，MIND 0.33 / 0.77。文献中最好的数字是 CoMIR+SIFT 的 0.56 / 0.73（CoMIR 原协议）和 DD_RoTIR 的 0.27 / 0.89（另一测试集）。
2. **但这个优势来自整个系统，不来自 VCS 目标本身。** 同一系统换成 InfoNCE 训练，TEST134 为 0.76 / 0.98，与 VCS 各族持平（MF_VCS_M1−InfoNCE 宽松 +0.004，core 95% 区间含 0）。
3. **所以“和 NCE 比”本身不是错的，错在把它当成唯一的成功标准。** NCE 同框架对照是归因用的消融，论文里必须有。但“应用是否成功”的标准应该是公开 SOTA。按这个标准，BF-SHG 上我们已经大幅胜出；按“VCS 胜过同框架 NCE”的标准，R22 全部系列都没有成立，R23 还在跑。
4. **EPFL RGB-NIR 不适合作展示。** 经典 MIND 严格成功率最高（0.879），中位误差最小（原生 0.41 px）；学习型方法（含 VCS）严格 0.85–0.87、中位原生 0.65–0.86 px，都没有超过它。
5. **真正公开、仍被引用的 SHG-BF SOTA 基准是 Learn2Reg 2024 COMULISglobe SHG/BF。** 它用真实配对和人工地标 TRE。前五名 TRE 在 2.64–3.21 px，评分者间误差 2.00 px，精度已接近上限，真正的区分点是失败数（TRE>15 px 的病例 1–7/40）。冠军 VROC 是经典变分方法，第二名用的正是 CoMIR（InfoNCE）特征。
6. **建议的应用叙事**：VCS 训练的稠密描述子配准在 BF-SHG 上达到并明显超过已发表 SOTA；同时如实报告同框架 InfoNCE 消融持平。若 R23 或真实数据上的鲁棒性实验能显示 VCS 相对 NCE/BCE 的失败率优势，再把“VCS 特有”的论点加进来。要让系统级 SOTA 的结论站得住，还缺两件事：现代学习型匹配器基线，以及与 CoMIR 原协议对齐的评估（§4）。

## 1. 我们自己的结果：所有基线，不只 NCE

### 1.1 BF-SHG（V10，TEST134 与 DEV32；严格 <9 / 宽松 <42 全分辨率 px）

| 方法族（seed 数） | TEST134 宽松 | TEST134 严格 | TEST134 中位 | DEV32 宽松 | DEV32 严格 |
|---|---:|---:|---:|---:|---:|
| MF_VCS_M1（3） | 0.984 | 0.747 | 5.12 | 0.957 | 0.643 |
| MLP_VCS_perm8（3） | 0.982 | 0.767 | 4.83 | 0.946 | 0.688 |
| MLP_VCS_perm1（3） | 0.984 | 0.752 | 4.72 | 0.953 | 0.651 |
| InfoNCE_host（3） | 0.980 | 0.762 | 4.98 | 0.957 | 0.685 |
| CoMIR_native（3，官方代码） | 0.827 | 0.640 | 5.78 | 0.798 | 0.528 |
| NMI_native（1） | 0.791 | 0.574 | 6.82 | 0.759 | 0.506 |
| MI_native（1） | 0.777 | 0.574 | 6.72 | 0.741 | 0.494 |
| RuLSIF_exact_v2（3） | 0.769 | 0.612 | 6.36 | 0.787 | 0.581 |
| MIND_2D_adapted（1） | 0.766 | 0.334 | 15.37 | 0.709 | 0.350 |

core 配对差（宽松，TEST134）：MF_VCS_M1−CoMIR +0.158 [+0.112, +0.205]，−MI +0.207，−NMI +0.193，−MIND +0.218；−InfoNCE_host +0.004 [−0.000, +0.011]。
来源：`ssl_pilot_v10_test_repeat/v10/test_repeat134/V10_TEST_REPEAT134_REPORT_CN.md` §3–§5。

R22 共享余弦族（同一编码器，T=cos，只换训练目标，30k 步，DEV）：BF 严格 VCS exact 68.3、BCE 67.5、NCE 69.3；宽松 98.2 / 97.2 / 95.2。这是同框架消融，不是 SOTA 对比。

### 1.2 EPFL RGB-NIR（V10，DEV，72 个 scene 单元宏平均；严格 <2.25 / 宽松 <10.5 原生 px）

| 方法族 | 宽松 | 严格 | 中位（原生 px） |
|---|---:|---:|---:|
| MLP_VCS_perm8 | 0.996 | 0.870 | 0.72 |
| MF_VCS_perm1 | 0.984 | 0.873 | 0.68 |
| InfoNCE_host | 0.996 | 0.850 | 0.86 |
| CoMIR_native | 0.968 | 0.858 | 0.60 |
| MI / NMI | 0.940 | 0.831 / 0.829 | 0.47 |
| MIND_2D_adapted | 0.972 | **0.879** | **0.41** |

宽松已饱和；严格和中位误差上经典 MIND 最好。来源：`V10_EPFL_EXPERIMENT_CN.md`、`handoff/v10_epfl_production_20261019/analysis/EPFL_RESULTS_CN.md`。

## 2. 文献中的 SOTA

### 2.1 CoMIR 的 BF-SHG 基准（Pielawski et al., NeurIPS 2020）

- **数据与协议**：206 对 BF/SHG TMA core（2048×2048），训练 40 对（834×834 中心裁剪），测试 134 对。旋转 ±30°，平移 ±100 px（834 patch）；误差为四角点平均位移；成功阈值 <1%（<9 px）与 <5%（<42 px）。
- **结果（134 对）**：

| 方法 | <1% | <5% |
|---|---:|---:|
| SIFT on CoMIR（MSE） | 56.0% | 73.1% |
| MS α-AMD on CoMIR（MSE） | 53.7% | 67.2% |
| CurveAlign（领域专用） | 16% | 25% |
| MI（原图） | 5% | 22% |
| CycleGAN / Pix2Pix / DRIT 翻译后 | ≤5% | ≤23% |

- **后续工作**：
  - DD_RoTIR（2024）在 0–30° 上达到 <1% 27.3%、<5% 89.0%（作者说明测试集与 CoMIR 不同）。
  - Lu et al.（PLOS ONE 2022）的比较研究在 Histological 集（±20°、±80 px、<2% 阈值）上：CoMIR+α-AMD 81.3%，MI 56.5%，MIND 29.1%；GAN 翻译方法普遍失败。
- **与我们协议的差别**：我们用整个 core 下采样到 512（×4），平移最大 51 px@512（约 204 全分辨率 px，大于 CoMIR 的 ±100 px），每个 core 10 个分层扰动；阈值相同。所以文献数字只能作参照。公平对照是我们框架内重跑的官方 CoMIR（§1.1）。

### 2.2 Learn2Reg 2024 COMULISglobe SHG/BF（MELBA 2025 综述）

- **数据**：乳腺与胰腺癌组织的 SHG 与 H&E 明场真实配对，训练 156（无标注）/ 验证 10 / 测试 40，每例约 25 个人工地标；CC BY 4.0，源自 Eliceiri 组的 Multimodal Biomedical Dataset（Zenodo 4550300），与我们的 BF 数据同源。**使用前必须核对样本重叠，避免泄漏。**
- **指标**：TRE（全部地标）、TRE30（初始误差最大的 30% 地标）、失败数（TRE>15 px）。
- **测试结果**：

| 名次 | 队伍 / 方法 | TRE | TRE30 | 失败数 /40 |
|---|---|---:|---:|---:|
| 1 | VROC（经典变分 + U-Net 翻译，NCC + Demons） | 2.64 | 2.86 | 3 |
| 2 | next-gen-nn（CoMIR 特征 + XFeat 预对齐 + ConvexAdam） | 2.93 | 3.18 | 1 |
| 2 | greedy（仿射基线） | 3.03 | 3.29 | 4 |
| 3 | IWM（SuperPoint+SuperGlue 穷举旋转 + 形变实例优化） | 3.10 | 3.42 | 4 |
| 4 | globalign（频域 CMIF 全局 MI 基线） | 3.21 | 3.64 | 7 |
| — | 未配准 | 69.6 | 102 | — |

  评分者间 TRE 为 2.00 px。综述的结论是：各范式调好后精度都在窄区间内，鲁棒性才是仍待解决的问题。
- **其他参赛工作**：
  - 验证阶段第一的 B-NCE + XFeat + ConvexAdam（LNCC+CMIF）TRE 1.943（10 例验证集）。
  - IWM 报告 SuperPoint+SuperGlue 初始对齐成功率 87.95%（全部 166 对，目视判定），OmniGlue 48.8%，基于 MI 的多级仿射 38.0%，SIFT+RANSAC 34.3%。
- **测试阶段已关闭**，是否仍接受赛后提交未确认；可自评的只有 10 个验证对。

### 2.3 还没比过的现代方法

- **学习型稀疏/稠密匹配器**：SuperPoint+SuperGlue/LightGlue（需穷举旋转）、XFeat、OmniGlue、RoMa、MINIMA（CVPR 2025，用生成数据训练的模态不变匹配器，未在显微数据上评测）。
- **基础模型特征**：DINOv2 特征的免训练配准（DINO-Reg，MICCAI 2024 / TMI 2025）。
- **医学多模态通用模型**：uniGradICON / multiGradICON（主要是 3D 医学影像，参考价值有限）。
- **RGB-NIR 方向**：EPFL 场景数据集本身是用 SIFT+RANSAC 相似变换对齐的分类数据集，不是专门的配准基准；2024 年的 RGB-NIR-IRegis 基准（含视角变化）和 SGFormer 更有代表性。

### 2.4 与理论相关的旧工作

已有用 Cauchy–Schwarz 散度作配准相似度的工作（MR/PET 刚性配准，2005 前后；CS 散度点集配准，TPAMI 2015），但都是固定的直方图/核密度估计，不是变分、可学习 critic 的 VCS-QMI。

## 3. 判断：VCS 在哪里可能“明确胜过 SOTA”

1. **系统级 SOTA（BF-SHG）：已经有，而且幅度大**：相对同框架重跑的官方实现，宽松 +16–22 pp、严格 +11–41 pp；相对文献最好数字，宽松 +9 pp（DD_RoTIR，测试集不同）、严格约 +19 pp（CoMIR+SIFT，协议不同）。要写成论文结论，还需要：
   - 补现代匹配器基线（§2.3）。SuperPoint+SuperGlue 在真实 SHG-BF 上初始对齐成功率约 88%，可能是最强的对手；
   - 在 CoMIR 原协议下再评一次，或两种协议并列；
   - 如实报告同框架 InfoNCE 消融持平。论文可以说“VCS 目标达到最佳对比目标的水平”，不能说“优势来自 VCS”。
2. **VCS 特有优势：目前没有证据。** 有理论依据、值得检验的方向只有：
   - **局部结构不一致时的鲁棒性（R23）**：容量小实验显示 VCS 会丢弃稀疏细节的身份、保留广泛结构，但 BCE 和高温 NCE（τ=0.30）表现相同。即使 R23 出现收益，也很可能与它们共享。
   - **对应到真实数据**：Learn2Reg SHG/BF 的开放问题恰好是鲁棒性（失败数），这是 R23 假设最自然的真实场景。
   - **免温度调参**：VCS 没有 τ，NCE 需要调 τ。可以作附带结论（“不调温度即达到调好的 NCE”），力度较弱。
   - **成本**：描述子层 N2048 时约 10 倍加速，完整训练步只快 1.06 倍，不足以作主张。
3. **风险**：BCE 与 VCS 有相同的最优 critic。凡是 VCS 胜出的地方，都要同时检验 BCE，否则审稿人会指出这不是 VCS 特有的。

## 4. 建议的下一步

| 项 | 内容 | 资源 | 需用户确认 |
|---|---|---|---|
| A | 在我们的 BF 协议 DEV32 上补现代匹配器基线：SuperPoint+SuperGlue/LightGlue（穷举旋转，按 IWM 做法）、XFeat、RoMa、MINIMA、DINOv2 稠密特征 + 我们的搜索器、globalign（CMIF MI） | 只做推理，约 5–15 GPUh；需下载公开预训练权重 | 下载模型权重 |
| B | 按 CoMIR 原协议（834 patch、±100 px、每对一个变换）重评我们的系统与基线；先用 DEV | 约 3–6 GPUh | 若要再读 TEST134，需另行批准（属重复使用） |
| C | 接入 Learn2Reg 2024 SHG/BF：下载数据（CC BY 4.0），先核对与我们 BF 数据的样本重叠，再用我们的描述子做仿射配准，加一个简单形变精修，在 10 个验证对上自评 TRE 与失败数 | 实现约 1–2 天；GPU 少 | 下载数据；赛后提交渠道需再确认 |
| D | R23（已在跑）：看 VCS 相对最强 NCE / BCE / RINCE 在局部不一致下的失败率；有信号再迁移到 C 的真实数据 | 已在途 | 否 |

推荐顺序：先做 A 和 B，它们决定“系统级 SOTA”能否写进论文，代价也小；D 照常进行；C 在 A/B 确认我们的系统确实领先之后再做，或与 D 的结果合并成“真实数据鲁棒性”实验。

## 来源

- CoMIR：[arXiv 2006.06325](https://arxiv.org/abs/2006.06325)；[NeurIPS 2020](https://papers.neurips.cc/paper_files/paper/2020/hash/d6428eecbe0f7dff83fc607c5044b2b9-Abstract.html)
- Lu et al., Is image-to-image translation the panacea for multimodal image registration?：[PLOS ONE / PMC9704666](https://pmc.ncbi.nlm.nih.gov/articles/PMC9704666/)
- DD_RoTIR：[arXiv 2407.11223](https://arxiv.org/abs/2407.11223)
- Learn2Reg 2024 综述：[arXiv 2509.01217](https://arxiv.org/abs/2509.01217)；任务页：[learn2reg.grand-challenge.org](https://learn2reg.grand-challenge.org/learn2reg-2024/)
- Learn2Reg 2024 SHG-BF 参赛：[arXiv 2410.20812](https://arxiv.org/abs/2410.20812)（B-NCE+XFeat+ConvexAdam）；[arXiv 2409.15931](https://arxiv.org/abs/2409.15931)（SuperPoint+SuperGlue+形变）
- MINIMA：[CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Ren_MINIMA_Modality_Invariant_Image_Matching_CVPR_2025_paper.html)
- DINO-Reg：[arXiv 2402.15687](https://arxiv.org/abs/2402.15687)
- multiGradICON：[arXiv 2408.00221](https://arxiv.org/abs/2408.00221)
- RGB-NIR 配准：[arXiv 2405.19914](https://arxiv.org/abs/2405.19914)；EPFL RGB-NIR 数据集：[IVRL](https://www.epfl.ch/labs/ivrl/research/downloads/rgb-nir-scene-dataset)
- DeeperHistReg（ANHIR/ACROBAT）：[arXiv 2404.14434](https://arxiv.org/abs/2404.14434)

## 5. 更新：B（CoMIR / MultiRegEval 协议，DEV32，2026-10-26）

同一批 640 题（32 DEV core × 10 变换 × 2 协议），学习型方法为 3 seed 均值：

| 方法 | CoMIR <9 | CoMIR <42 | MultiRegEval <2% |
|---|---:|---:|---:|
| InfoNCE_host | .635 | .936 | .883 |
| MLP_VCS_perm8 | .614 | .928 | .881 |
| MF_VCS_M1 | .562 | .935 | .849 |
| soft MI（同一全网格搜索器） | .534 | .787 | .719 |
| CoMIR（同一搜索器） | .425 | .688 | .645 |
| MIND（同一搜索器） | .300 | .700 | .559 |
| CoMIR + SIFT（我们重训版） | .230 | .403 | .372 |
| SimpleITK MI（CAL 调参） | .197 | .294 | .484 |

- 相对已发表风格的流程（CoMIR+SIFT、SimpleITK MI），学习型能量严格高 33–44 pp，区间都不含 0。
- 但相对**同一搜索器**下的 soft MI，严格只高 3–10 pp，区间含 0；学习型能量的稳定优势在尾部（<42、p95）。也就是说，我们系统领先的主要来源是全网格搜索 + 局部精修，而不是学习到的能量本身。
- VCS 没有胜过 InfoNCE（MF_VCS_M1 − InfoNCE 严格 −7.3 pp [−14.3, −0.3]）。
- 限制：CoMIR+SIFT 不是论文原版（重训于 FIT32、1/4 尺度、OpenCV SIFT），会低估原方法；SimpleITK MI 的登记配置失败后改为 CAL 调参，两种结果都报告；α-AMD 未跑；DEV 已在 V10 开发中使用。
- 对论文叙事的影响：“系统级 SOTA”成立，但要把搜索器作为贡献的一部分，并如实给出“同搜索器 soft MI”这一消融；它也说明 VCS 替换 MI（散度实验）正好落在同一搜索器上比较，是更公平的测试。

## 6. 更新：VCS 替换 MI 作配准相似度（同直方图、同搜索器，DEV，2026-10-26）

| 散度 | BF 严格/宽松 | EPFL 严格/宽松 |
|---|---|---|
| MI（KL） | .525 / .753 | .829 / .938 |
| VCS（½Σ(p−q)²/(p+q)） | .553 / .791 | .826 / .934 |
| JS（BCE 最优） | .541 / .803 | .824 / .931 |
| Hellinger | .531 / .813 | .817 / .924 |
| CS-QMI（经典） | .572 / .816 | .686 / .785 |
| χ²（无界） | .453 / .575 | .794 / .915 |
| MIND（历史） | .350 / .709 | .879 / .972 |

- VCS−MI：BF +2.81 / +3.75 pp（97% core 不变差），略低于 +3 pp 门槛；EPFL −0.35 / −0.42。预登记信号在两个数据集都不成立。
- 有界或二次的散度在 BF 上普遍比 MI 宽松高 4–6 pp，无界 χ² 崩溃——“有界”这一性质有作用，但 VCS 不是其中最好的，JS、Hellinger、CS-QMI 相当或更好。
- 能量面光滑度和盆地捕获率 MI 与 VCS 几乎相同，“更宽盆地”的机制没有得到支持。
- 结论：在直方图相似度这个最直接的理论落点上，VCS 与其它有界散度同属一类，没有特有优势；EPFL 上仍是 MIND 最好。

## 7. 更新：VCS / BCE 替换 CoMIR 的 InfoNCE（DEV，2026-10-26）

| 臂 | BF 严格/宽松 | EPFL 严格/宽松 |
|---|---|---|
| NCE-MSE τ2.0 | 58.8 / 87.0 | 86.1 / 96.6 |
| NCE-cos τ0.1（CAL 选中的 NCE） | 56.5 / 82.8 | 84.4 / 94.8 |
| VCS-exp τ0.5 | 22.7 / 41.8 | 78.1 / 89.4 |
| BCE-cos | 29.2 / 47.7 | 75.3 / 87.1 |

最佳 VCS − 最佳 NCE：BF 严格 −33.8 pp（0/3 seed），EPFL −6.3 pp（0/3）。在对训练目标敏感的 CoMIR 框架里，有界目标（VCS、BCE）都学不出可用于单模态配准的表示，VCS 输得更多而不是更少。这个替换方向不成立。

## 9. 更正（2026-10-27，依据外部复核）

1. **“我们的系统已经是系统级 SOTA”证据不足，收回。** 对照的 CoMIR 是 FIT32、1/4 尺度重训版本并接在我们的搜索器上；现代匹配器（LightGlue、RoMa、MINIMA 等）DEV 尚未完成，CAL 上 RoMa/MINIMA-RoMa 已接近 100% 严格；真实地标任务（Learn2Reg COMULIS）还没有接入。目前只能说：在我们的合成刚性协议和同框架重训的对照下领先。
2. **“CoMIR 替换失败说明 VCS 学不出可用于配准的表示”过宽。** 该替换同时改变了损失形状和负样本结构（NCE 用 2B 个点、含同模态负例；VCS/BCE 只用 B×B 跨模态矩阵），并且训练 critic 与部署读出（原始 latent MSE）不完全一致。能确认的是：在当前参数化、负例构成、736 步配方和 MSE 读出下，VCS 不能直接替代 CoMIR 的 NCE。
3. **“稀疏细节”假设对 BF-SHG 不宜作默认。** SHG 的共同内容本身是稀疏的胶原纤维，稀疏结构可能正是定位线索；R23 的前提需要按“有定位价值 / 模态特有 / 无信息背景”重新划分。

## 10. 更新：CoMIR 收尾实验（2026-10-27，`handoff/vcs_comir_close_20261027`）——修正 §7 的结论

- **只用跨模态负例的 NCE**：相对完整 NCE，BF 严格 NCE-MSE −5.2 pp（seed 间 +4.4/−10.3/−9.7）、NCE-cos −1.2 pp；EPFL −1.1/−0.4 pp。去掉同模态负例只解释一小部分差距；VCS/BCE 仍比它低 24–41 pp（BF）。
- **读出交叉评测（同 checkpoint、同搜索器，只换配准能量）**，BF/EPFL 严格：

| 臂 | 原 MSE 读出 | NCC 读出 | 余弦读出 |
|---|---|---|---|
| NCE-MSE τ2.0 | 58.8 / 86.1 | 63.5 / 86.9 | 62.4 / 86.1 |
| NCE-cos τ0.1 | 56.5 / 84.4 | 63.0 / 87.3 | 62.0 / 87.3 |
| VCS-cos | 27.5 / 71.2 | 61.4 / 85.0 | 55.5 / 82.6 |
| BCE-cos | 29.2 / 75.3 | 61.9 / 85.7 | 58.5 / 83.3 |
| VCS-exp τ0.5 | 22.7 / 78.1 | 43.3 / 82.9 | 39.1 / 79.2 |

- **结论修正**：§7 的巨大差距主要来自训练 critic 与部署 MSE 读出不匹配，而不是 VCS 学不出表示。换成 NCC 读出后 VCS-cos 在 BF 上 +33.9 pp，与 NCE、BCE 基本持平；但 NCE 同样换成 NCC 后仍略好（VCS−NCE BF −2.2、EPFL −2.4 pp）。NCC 读出与“最佳”臂是看过 DEV 后选的，属描述性结果。
- **诊断**：VCS-exp 正对梯度没有饱和（d/τ 中位约 0.3、梯度因子约 1.5），L1/L2 正则梯度可忽略；“梯度消失”“正则过强”两种解释都不成立。
- **副产品**：CoMIR 类基线应同时报告 NCC 读出——NCE 自己在 BF 上也因此提高 5–7 pp。

## 8. 现代匹配器基线（A，2026-10-27，`handoff/r23_sota_matchers_20261026/RESULTS_CN.txt`）

V10 BF DEV32 同一批 320 题，CAL 8 选配置后锁定；现成公开权重，不在 BF 上训练。严格 <9 / 宽松 <42 / p95：

| 方法 | 严格 | 宽松 | p95 |
|---|---|---|---|
| SuperPoint+LightGlue | 72.8 | 98.8 | 20.0 |
| MINIMA SP+LG | 72.5 | 100 | 17.6 |
| MINIMA-RoMa | 69.4 | 100 | 17.6 |
| V10 InfoNCE_host（3 seed） | 69.0 | 95.6 | 23.4 |
| V10 MLP_VCS_perm8（3 seed） | 68.0 | 94.6 | 117.5 |
| globalign | 67.5 | 100 | 17.4 |
| RoMa / XFeat / LoFTR 等 | 64–66 | 92–99 | 17–117 |
| V10 MF_VCS_M1（3 seed） | 64.1 | 95.7 | 27.0 |
| V10 CoMIR_native | 52.8 | 79.8 | 423 |
| DINOv2 特征 + 宿主搜索 | 5.3 | 57.5 | 229 |

配对 core 级差（32 core bootstrap）：SP+LG − MF_VCS_M1 严格 +8.8 pp [+1.4, +17.9]，− MLP_VCS_perm8 +4.8 [−0.7, +11.4]，− InfoNCE_host +3.9 [−2.5, +11.0]；宽松全部 +3–4 pp。

**结论：系统级 SOTA 说法不成立。** 现成的 SuperPoint+LightGlue（不在 BF 上训练）在严格成功率上与我们最好的系统持平或更好，在宽松成功率和尾部误差上明显更好（98.8–100% 对 94.6–95.7%）。我们的学习型系统只对已发表的 CoMIR/MI/MIND 类流程有明确优势。这与 §9 的收回一致；后续配准工作应以这些匹配器（及 R24 的 Neon 强宿主）为基线，并把 VCS 放到局部对应评价或候选选择的位置，而不是替换整个能量。

## 11. 配准收尾：按图像对校准的 VCS 结构匹配器（2026-10-28，`handoff/reg_close_pairvcs_20261027/ANALYSIS_CN.txt`）

冻结 MINIMA-RoMa 特征、FIT 固定低维基底，每对图像在固定支持集上做一次有界二次 VCS 校准后冻结，生成新对应并刚性求解；只做推理期诊断，不训练基底。

- **oracle 支持上限（CAL）**：校准后的新对应在 2.25/3/6 px 内正确率仅 0.10%/0.18%/0.74%，只用新对应严格 0%。真实支持与之几乎相同。
- **完整任务（DEV 320 题）**：原生 69.06 严格 / 100 宽松；等预算额外搜索 69.06；仅支持点 67.81；全局 VCS、逐对 VCS、逐对 matched BCE 全部 68.75。C5−C1 −0.31 pp [−0.94, 0]，C5−C6 与 C5−C4 为 0。宿主额外搜索能产生 40% 正确的新对应，VCS 校准模块只有 0.08%。
- **结论**：属于设计中“连理想支持都不能改善新匹配”的情形，推理期即插即用的逐对校准切口无价值，VCS 与 BCE 无差别。**配准线就此关闭**。局限：基底低秩（有效秩约 3.4）、二次问题病态，可训练基底/双层训练版本未运行，不能推出该版本必然无效，但不再投入。
