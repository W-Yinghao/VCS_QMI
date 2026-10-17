# VCS-QMI 项目地图与解读（2026-10-14）

**性质。** 本文由协调 agent 于 2026-10-14 写成，供 owner 评估项目与规划论文；数字抽样核对脚本见提交信息。这是对整个项目（2026-09-24 至 2026-10-14；主仓库 962 次提交，147 个 P 编号实验线，加 R4–R7、REV8/9、V7/V7X 各轮）的综合解读：理论贡献到底是什么、为什么 matched JS / logistic 是固定对照、各条应用线做到了哪里、暴露了什么根本问题、下一步怎么走。每个数字都注明来源文件；解读与数字分开标记。**实验证据是仓库里的；论文由 owner 写。**本文不替代各线的冻结协议和报告，只是它们的索引和一页判断。

目录：§1 理论与贡献定位 · §2 为什么对手总是 matched JS / logistic · §3 第一应用（CIFAR SSL） · §4 估计器层（受控数据） · §5 第二应用预检 · §6 R4–R5 视觉定位（RefCOCO） · §7 R7 视觉 token 剪枝（D1） · §8 V7/V7X 文图对齐 · §9 横向总结：暴露的根本问题 · §10 VL 相对 SSL 的成熟度 · §11 下一步 · 附录：证据索引

---

## 1. 理论与贡献定位

### 1.1 对象（原计划 §3，不变）
对任意两分布 P、Q，M=(P+Q)/2，η=(p−q)/(p+q)∈[−1,1]：

- S = E_M η²  （"混合参考 CS"，∈[0,1]；CS 散度 = 2 log((1+S)/(1−S))）
- J(T) = E_P T − E_Q T − ½E_P T² − ½E_Q T²，T=tanh f ∈[−1,1]
- **恒等式 S − J(T) = E_M (T − η)²**：J 是 S 的变分下界，gap 是 T 到 η 的 L²(M) 距离。

SSL 里 P = 联合（两视图配对），Q = 边缘乘积（循环移位负对）；语义任务里 P/Q = 满足/不满足。实现只需 T，不需密度。（来源：`VCS_QMI_SSL_Server_Agent_Spec_v1.md` §3.1；`r4/package_r4_20261010/docs/03` §1。）

### 1.2 定位中已经确认的事实（这些不是新贡献）
| 事实 | 出处 | 含义 |
|---|---|---|
| J = balanced RPC F₁,₁,₁（α=β=γ=1），T* = η，F(T*)−F(T) = ½∫(p+q)(T−T*)² | r4 docs/03 §2；R4-T0；R4_02 ledger "VCS = balanced RPC" | 目标泛函不是新的；是 Tsai et al. 2021 RPC 的平衡实例 |
| RuLSIF α=½ 的相对比值 = 1+η | 同上 | 核/密度比文献早有同一对象 |
| JS（logistic）与 VCS 的总体最优解相同（D*=p/(p+q)，T=2D−1），只差评分规则：∂ℓ_V/∂f=(T−c)(1−T²)，∂ℓ_L/∂f=T−c | r4 docs/03 §3 | 两者是**同一后验、不同 proper scoring rule**（平方 vs 对数） |
| 嵌套三项分解 S_W − J(T) = E(η−η_Z)² + E(η_Z−η_s)² + E(η_s−T)²、数据处理不等式、梯度公式，在有限模型上到 1e-16 成立 | `reports/P121_V6_THEORY_REPORT_20261003.md` §2 [identity] | 几何恒等式全部成立；但 P121 §4 也写明"是否对真实 SSL 重要未测" |
| 原生校准 (1+T)/2 = σ(2f) 对 JS 同样成立 | CVPR ledger §D.9 | 校准不是 VCS 独有 |

### 1.3 那么贡献还剩什么？（r4 docs/03 §9 的决策表，照抄其尺度）
| 结果类型 | 可写 | 不足以单独声称 |
|---|---|---|
| RPC 恒等、完成平方 | 清楚定位原目标 | 新泛函 |
| 正确 U-statistic / 保守界 | 纠正相关 pair 的误差计算 | 新统计原理 |
| 明确条件下的新有效区间 / 配对设计及成本收益 | 经证明与验证的具体估计贡献 | 任意依赖、任意网络均有效 |
| 更高公共 J | 更好的规定函数类读数 | 自然总体信息一定更多 |
| proposal-aware 微调提升 | 候选任务的适配收益 | 独有 CS 理论优势（除非对照支持） |

**我的判断（解读，不是数据）：** 到今天为止，项目能站住的"贡献"是**估计器层面的一个干净对象 + 一整套诚实的对照方法学**：混合参考 S、精确的二次变分形式、有界最优 critic、闭式/神经二重性（CVPR ledger §D.11 原话），以及"任何优越性都要先过 matched JS 这一关"的实验纪律。**在任何一条自然数据线上，VCS 相对 matched JS 的精度优势都没有出现过**（§2–§8 逐条给数）。所以论文的主张只能放在：(a) 这个对象的性质与证明；(b) 它与 JS 共享的"学习型后验路线"相对传统估计器/外部系统的收益（V7X 已有正面证据）；(c) 二次评分规则独有的、可证明的性质——鲁棒性（P134 支持）、嵌套增量的 L² 可加性（P102/P118 部分支持）、有界 T 下的保守下界（R4-T2：只在强依赖、n=512 时能认证）。

---

## 2. 为什么对手总是 matched JS / logistic

1. **它是唯一能把"损失"从"其他一切"里分离出来的对照。** 同一函数类、同一 P/Q、同一参考测度、同一读数 T=tanh f、同一先验/权重，只换评分规则（平方 ↔ 对数）。任何其它对手（SimCLR、SparseVLM、FGA、VQAScore、核方法）同时换了主干、数据、结构或目标，赢了也归不到"VCS 的二次形式"头上（v2 包 docs/13 §2 的"系统表 vs 匹配表"）。
2. **因为它们估计的是同一个后验。** 总体最优 T 相同；有限样本差异只来自评分规则导致的梯度权重 (1−T²) 与优化动态。这正是 R4 审查把它的台账角色改成"同一后验、不同评分规则"的原因，也是为什么"VCS 打败 logistic"不是一个有理论支撑的预期——没有定理说平方规则在有限样本下精度更高。
3. **它是最严格的对手，所以也是最容易输的。** 这不是设计失误，而是纪律：如果连这个对照都赢不了，"二次形式带来精度"就不能写。

**各线 VCS − matched JS 的结果汇总（全部为匹配条件）：**
| 线 | 读数 | VCS − JS | 出处 |
|---|---|---|---|
| 受控数据，P85 阶梯 45 cell | 后验 MSE | JS ≥ VCS 于 42/45 | `reports/R4/R4_02_EVIDENCE_LEDGER_20261011.md` l.33 |
| 受控数据，R4-E1 直接 gap | 后验 MSE | +0.0082（N 4096）→ +0.0027（N 16384），不反转 | `reports/R4/R4_E1_REPORT_20261011.md` |
| 受控污染 P134（ε 轴） | 归一化偏移（越小越稳） | VCS 0.88 / 0.18 / 0.62 vs JS 1.40 / 0.13 / 1.64（P134 主读数 supported）；**R4-E0 在共同污染后验上重读：JS 绝对误差更低 19/48 cell（VCS 0），VCS 相对自身干净种子退化更小 9/48 → 台账判定 mixed：局部退化优势，不是鲁棒性** | `reports/P134_R1_CONTAMINATION_REPORT_20261005.md` §1；`reports/R4/R4_02_EVIDENCE_LEDGER_20261011.md` l.25 |
| RefCOCOg / RefCOCO+ 全微调（R4-VL1） | Acc@0.5 | +0.38 [−0.60,+1.36] / +0.17 [−0.43,+0.77] | `reports/R4/R4_VL1_REPORT_20261011.md` |
| 同上，检测分参考 ν（R4-VL2） | Acc@0.5 | +0.13 [−0.30,+0.57] / −0.08 [−0.21,+0.05] | `reports/R4/R4_VL2_REPORT_20261011.md` |
| SSL，A-P3 配方 vs 匹配 JS-AP3（CIFAR-10，3 seed） | linear 探针 | +0.21 [−1.03, +1.46]（kNN −0.29，JS 在 3/3 seed 更高） | `reports/P114_V5_JS_AP3_REPORT_20261004.md` |
| SSL，CIFAR-100（5 seed） | linear | P120 +0.80 [+0.06, +1.54] → P129 各自选定 scorer 后 **−0.49 [−0.77, −0.20]（JS 更好）**，"lr-dependent" | `reports/P120*`、`reports/P129*` 附录 3/4 |
| R7 D1 token 剪枝（LLaVA 10k，官方测试） | S1−S2 准确率 @L25 | GQA −3.24，ChartQA −0.87（S2 领先） | `reports/R7/R7_50_PHASE_L_REPORT_20261013.md` |
| R7 D1 FULL（官方测试） | 同上 | GQA −0.78 / ChartQA +0.75，各由单 seed 驱动 | `reports/R7/R7_50_PHASE_F_REPORT_20261014.md` |
| V7 完整模型 M2−M1（EvalMuse DEV） | 自然 Brier（负为好） | +0.0005 / +0.0005 / +0.0003，区间均含 0 | `ssl_pilot_v7/reports/V7/V7_TABLE_A_results/v7_table_a.json` |
| V7X 共同特征 N-VCS-F − N-BCE-F | 自然 Brier | +0.0009 / +0.0002 / −0.0001，区间均含 0 | `ssl_pilot_v7/reports/V7/V7X_X21_results/x21_dev_scores.json` |

**结论：** 精度上 VCS ≈ JS（五条自然数据线、SSL 的 P114）或 JS 略优（受控后验 MSE 42/45、D1 10k、CIFAR-100 P129）。VCS 一侧的正面读数只有两个，且都带限定：污染下的退化斜率（P134，经 R4-E0 重读为"局部退化优势"）和嵌套精确拟合（P118：VCS nested_exact 通过全部 4 条判据，JS 失败 1 条）。这就是"把优越性论证从精度移到鲁棒性、嵌套增量和认证下界"的全部数据依据——§4.4 会说明这三条各自的证据强度。

---

## 3. 第一应用：CIFAR-10 SSL（2026-09-24 → 10-09；v1/v2 配方 155 个 run，之后 v3–v7 配方线；已关闭）

**问题。** 用 J 作为 SSL 目标（两视图 P=联合、Q=循环移位乘积），ResNet-18，能否达到 SimCLR / VICReg 的线性探针精度。

**结果（`reports/CVPR_EVIDENCE_LEDGER_20260928.md` §A1–A2，`reports/SYNTHESIS_20260925.md` §1/§7）：**
| 预算（1× = 2 视图 200 ep） | VCS v2 配方（论文数字）Sel / Test | SimCLR 等预算调参 Test | VICReg 等预算调参 Test | 差距（Test, SimCLR / VICReg） |
|---|---|---|---|---|
| 1× | 83.19 / 82.98 | 86.94 ± 0.12 | 87.00 ± 0.25 | +3.96 / +4.02 |
| 2× | 84.54 / 84.52 | 87.99 ± 0.07 | 87.11 ± 0.31 | +3.47 / +2.59 |
| 4× | 85.95 / 85.87 | 88.21 ± 0.10 | 86.91 ± 0.15 | +2.34 / +1.04 |
| 8× | 87.01 / 86.65 | 88.13 ± 0.07 | 86.87 ± 0.06 | +1.48 / +0.22 |

原始配方 74.34 ± 0.46 → v2 配方 87.01 ± 0.53（4 视图、800 ep、3 seed；官方测试一次 86.65 ± 0.26）；100+ 个配方搜索把差距从 ~12 点缩到 1.5 点（SimCLR）/ 0.2 点（VICReg），但**等预算调参后两个对照在每个预算点都高于 VCS**（预先承诺的措辞，ledger §E；R4_02 §4.1 要求 SimCLR 标为"SimCLR recipe at VCS's view / epoch budget (P41)"而非 "tuned SimCLR"）。

**v3–v7 后续配方线（2026-09-29 → 10-09，仅开发划分）：** G2（固定 (2,−1) scorer）88.69 ± 0.15，vs SimCLR +0.38 [−0.05, +0.81]（不成立）；**A-P3（固定 tanh(2s−1)、全视图 token、全梯度）89.02 ± 0.04 vs SimCLR 88.31 ± 0.21，Δ +0.72 [+0.48, +0.95]（5 seed；kNN −0.45 [−0.92, +0.02]；强增强下 SimCLR 反超 1.95）**（`reports/P107_AP3_LAYER2_REPORT_20261003.md`）。限定：R4_02 §6.1 "No v3–v7 recipe was ever confirmed on held-out data"——CIFAR-10 测试集已被 v2 用掉。外部协议 solo-learn：SimCLR 91.24 vs VCS 87.33（REFUTED，`reports/P76_S2_SOLO_LEARN_REPORT_20260929.md`）；ImageNet-100 试点 P154：SimCLR +3.08；ResNet-50 CIFAR-100：VCS − SimCLR +5.17 linear / kNN −1.59（探针敏感，解释已撤回）。

**确立的机制（SYNTHESIS §3）：** critic 的函数形式决定 encoder 几何（拼接 MLP critic 把 h 秩压到 13–21，余弦 critic 到 58），与容量无关；负分支 detach 在余弦 critic 下 +2.2、在 MLP 下塌缩；J 不是表示质量（跨 critic 族不可比）；优化配方不是瓶颈。h-均匀性排序所有 VCS run。

**匹配 JS 在 SSL 侧（§2 表已列）：** P114 A-P3 vs JS-AP3 +0.21 [−1.03, +1.46]（"no statement that the VCS quadratic objective has its own contribution"）；CIFAR-100 的 +0.80 在各自选定 scorer 后翻成 −0.49（JS 更好）。CIFAR-10-C 腐蚀 mCA：A-P3 70.58 ≈ SimCLR 70.63，JS-AP3 最高 71.94（P131）。

**关闭。** Owner 2026-09-26 23:00："后续不用再提交更多的 SSL 实验了"；2026-10-09 再次确认 VL 优先。SSL 在论文里的位置：主应用、带等预算对照、不写"superior"（ledger §D.11）。

**解读。** SSL 线回答了"J 能不能当 SSL 目标"（能：v2 配方差 SimCLR 1.5 点；A-P3 在开发划分上反超 0.7 点但无 held-out 确认）和"差在哪"（critic 几何，不是优化）。它从未设计成能分离"二次形式"的实验——SimCLR 换了目标、负对构造和温度；而真正的匹配对照 JS-AP3 与 A-P3 持平。所以 SSL 的可写结论是"一个 CS 型目标在精心设计 critic 后可与 SimCLR 同级"，不是"二次形式更好"。

---

## 4. 估计器层：受控数据（P82–P152，R4；2026-09-28 → 10-11）

这是项目里**唯一同时有真值、有匹配对照、有重复**的层。核心产物：`reports/P86_ESTIMATOR_BENCHMARK_REPORT_20260929.md`（P85，85 个 cell JSON）、`reports/P108_V4_E_ESTIMATOR_REPORT_20261002.md`（+ 勘误 1ea973d）、`reports/R4/P151_r4/`（P151 的修正聚合）、`reports/P152_R3_E1_REPORT_20261008.md`、`reports/R4/R4_02_EVIDENCE_LEDGER_20261011.md`（总台账）、`reports/R4/R4_E1_REPORT_20261011.md`、`reports/R4/R4_T2_REPORT_20261011.md`。

### 4.1 精度（后验 MSE 对已知 η；越小越好）
- 参考 cell（gaussian d 20, I 4, N 4096；R4_02 l.92–99）：VCS product 0.095 / cyclic-8 0.059 / in-batch 0.060；JS 0.089 / 0.052 / 0.052；RFF 0.185；RuLSIF-tanh 0.163；Nyström 0.533；S-KDE 0.602。
- N 轴（P86）：VCS-N 0.364/0.146/0.059/0.029 vs JS 0.323/0.122/0.052/0.026（N 256→16384）；log-log 斜率 VCS −0.61 = JS −0.61，核方法 −0.37 / −0.30 / −0.06 / ≈0。
- 台账句（R4_02 l.117）："Matched JS is equal or 5–25 % better than VCS on S in most cells"；JS ≥ VCS 于 42/45 阶梯 cell。例外：无关维度 d_total 100 时 VCS 0.354 vs JS 0.509（恢复后验方差份额 0.35 vs 0.06）。
- R4-E1 把 VCS−JS 的 gap 分解：+0.0082（N 4096）主要是"优化路径效应"，拟合准则本身贡献 +0.0004（不显著）；N 16384 时 gap 缩到 +0.0027，**不反转**。
- 对比传统估计器：学习型（VCS/JS）在 N 轴上全面优于 RFF (3/4)、S-KDE (4/4)、Nyström (4/4)，与 RuLSIF-tanh 持平（N 大时赢、N 小时输）。

### 4.2 读数与下界（P108、R4-T2）
- Ŝ_plug（T 代入 E_M T²）比 Ĵ 偏差更小 70/72、RMSE 更小 69/72（P108 勘误）；Ĵ 的评价 SE ≈ 3.2× Ŝ_plug；Ĵ 偏差 72/72 为负。**"Ĵ 是逐次运行的下界读数"被撤回**：J(T) ≤ S 只在总体成立；逐次下界要用 held-out 集中区间（docs/03 §5：max(0, Ĵ − √(8 log(2/δ)/N))）。
- R4-T2：学习型 critic 的方差在 EVAL 32 768 时 85 %（VCS）/ 77 %（JS）来自训练；有界 LCB 半径 0.43（n=512），**弱依赖（S≈0.05）在 n ≤ 512 永远无法认证**，只在强依赖可认证 S>0。pair-iid SE "两个方向都无效"。

### 4.3 稳定性、分辨率、污染
- P152：梯度范数 CV VCS 0.22/0.20/0.56 vs JS 0.13/0.16/0.25（I 2/6/10）；饱和 |T|>0.95 VCS 低于 JS；0 个非有限步。台账判定 "VCS-lower-variance: not_supported"。VCS 的独特性质是"共享目标 S、不饱和输出、无分数溢出"，不是更低梯度方差。
- 分辨率（P86 阶梯相邻档）：InfoNCE 118/114/81/52 > VCS 88/57/41/16 ≈ JS 63/45/34/20；InfoNCE "分辨 1.3–6× 更细"（台账 MI-resolution: supported）。
- P134 污染（§2 表）：主读数支持 VCS 斜率更缓；R4-E0 重读 → "mixed"。docs/05 l.37："不能直接叫清洁鲁棒性"。
- 嵌套增量：P121 恒等式到 1e-16；P118（Gaussian 链，有真值）VCS nested_exact 通过 4 条判据，JS 通过 3/4；但真实正增量仍低估 20–30 %；P102（CIFAR 特征，n=3000）规则 1 在 0/8 组通过，正交增量"不可见"。

### 4.4 这一层对"三条替代论证"的裁决（我的解读，数据见上）
| 论证 | 证据强度 | 能写什么 |
|---|---|---|
| 鲁棒性 | 弱–中：P134 主读数支持，R4-E0 重读为局部退化优势；SOURCE_ANALYSIS App.4 l.345："不能写成鲁棒性定理" | 独立对污染下的退化斜率优势（ε 0.1–0.2，退化比 0.53–0.70），作为经验观察 |
| 嵌套增量 | 中：恒等式精确；P118 有真值链上 VCS 完整通过、JS 差一条；P102 自然特征上不可见 | L² 可加性是平方规则的性质（JS 的是 Bregman/KL 增量）；nested_exact 作为一种拟合纪律 |
| 认证下界 | 弱：J ≤ S 对任意有界 T 成立（App.4 l.335 纠正为"非平方独有"）；有限样本只在强依赖可认证 | 保守一侧界的正确算法（jackknife LCB 覆盖 89–100 %），不是效率主张 |

台账里真正被标为"平方规则独有"的只有：优化即报告（训练目标 = 报数的 gap）、线性闭式解、再下降型影响函数（App.4 l.335）。

---

## 5. 第二应用预检 A–D（P45–P68，2026-09-27/28；完成，未选出任务）

`reports/CVPR_EVIDENCE_LEDGER_20260928.md` §B、`reports/SECOND_APP_FINAL_SUMMARY_20260928.md`：
- A 校准/阈值迁移：A-S1 不成立；原生 (1+T)/2 的校准性是平衡 proper scoring rule 的共性（JS 同样 σ(2f)），非 VCS 独有。
- B 闭式 critic：B1 成立（闭式线性 critic + tanh ≈ 训练 critic，gap ≤ 0.012）；B2 配准（RGB–NIR / IXI / fMRI）不成立（J* 成功率 0.50/0.32 vs MI 0.70/0.52）。
- C batch 解耦：不成立（冻结塔对 batch 不敏感；从头训练无动量编码器时 VCS 0.35、SimCLR 0.12 都塌）。
- D 依赖/泄漏检验：有条件成立——置换校准的 VCS 检验与 HSIC 持平；Hoeffding 界比置换多耗 20–50× 样本。

**解读。** 预检的价值是方法学教训（ledger §B 末："固定扰动零假设、循环移位零假设、字面 vs 意图、敏感 ≠ 依赖、持平 ≠ 优势、等预算改变标题"），这些教训直接塑造了后面每一轮的冻结协议。

---

## 6. R4–R5：指代表达定位（RefCOCO 系列，SigLIP 2 B/16；2026-10-10 → 10-12；完成，含官方测试）

- **VL3（给定框）**：12 cell × 3 seed；VCS − softmax +0.01 [−0.35, +0.36]，VCS − JS +0.07 [−0.01, +0.15]（`reports/VL3/VL3_COMPLETE_12CELL_REPORT_20261010.md`）。给定框时 VCS 与任务匹配的 softmax 持平。
- **R4-VL1（真实 proposal，全微调，18 run）**：候选支持适配对所有目标都有效（+13.8 ~ +20.6）；但 **VCS − softmax −1.85 [−2.41, −1.29] / −3.22 [−5.08, −1.37]**，VCS − JS +0.38 / +0.17（含 0）。差距随干扰框数增长 −0.3 → −1.0 → −2.6 → −3.5（RefCOCOg）：根因是成对 critic 缺少同一 query 内候选间的竞争，不是 ν、不是正样本。
- **R4-VL2（检测分参考 ν）**：VCS(ν) − VCS(uniform) +2.08 / +2.47（>0），但等信息对照 C3b 显示"收益是测试时分数融合，不是估计律"；VCS(ν) − JS(ν) +0.13 / −0.08。
- **R5（query 内归一化 VQ = KLIEP 式）**：DEV 上 VQ − VCS +1.13 / +2.22 / +7.56（RefCOCOg / RefCOCO+ / RefCOCO；`reports/R5/R5_QNORM_REPORT_20261012.md`、`R5_QX1_REFCOCO_REPORT_20261012.md`），关闭 VCS−E 差距的 61 % / 69 % / 86 %；**官方测试（owner 批准 2026-10-12）**：RefCOCO testA +5.57 [+4.22, +6.92]（Holm p 0.0127）、testB +8.73 [+7.90, +9.55]（p 0.0024）→ "VQ improves VCS"；RefCOCOg / RefCOCO+ 不可检出（Holm p 0.1143）。VQ − E / VQ − D 仍 −0.4 ~ −1.9；VQ ≈ JQ（区间全含 0）。RefCOCOg test：VCS 72.77，VQ 73.60，D 74.82，E 74.66。（`reports/R5/R5_FINAL_EVAL_REPORT_20261012.md`、`reports/R5/final_eval/results.json`）
- 披露：63 个预定行未评价（无 checkpoint / 无 CAL λ）；测试标注未曾封存在 closed/。

**解读。** 这条线给出了项目最清楚的一个"结构性"发现：**成对二分类 critic（VCS 和 JS 一样）在"从 N 个候选里选一个"的任务上天然输给 set-level softmax，差距随候选数增长；补救（query 内归一化）确实有效且过了官方测试，但它是 KLIEP 归一化的实例，不是 VCS 独有**，而且补救后仍落后 softmax 0.4–1.9。这个发现在 §7 和 §8 以另一种形式重现。

---

## 6b. R6：跨模态配准与按需采集（2026-10-11/12；DEV 结论，未读官方测试）

`reports/R6/R6_SYNTHESIS_20261012.md` 及各 REG/GAIN 报告：
- **REG（学习型局部关系 critic 的配准能量）**：BF/SHG 平移 VCS 成功率 100.0 vs MI 81.9 vs CoMIR+我们的优化器 90.9（VCS − MI +18.1）；刚体 ±30° VCS − MI +20.1 [+15.1, +25.2]；EPFL RGB–NIR VCS − MI +5.5 [+4.9, +6.1]，VCS − CoMIR +3.1 [+0.7, +5.5]。**但 VCS ≈ logistic ≈ InfoNCE 处处相同**——owner 措辞"learned-critic class effect"，不归 VCS 独有。这反转了 09-28 预检 B2 的否定（那时用的是原始像素闭式 critic）。
- **GAIN（按需采集第二模态；`reports/R6/R6_GAIN_P1_CREMAD_REPORT_20261012.md`、`R6_GAIN_P1_AVMNIST_REPORT_20261012.md`）**：CREMA-D gain_h − entropy −0.0204 [−0.0279, −0.0136]（Brier，负为好），工作点 Brier 0.564 @ 9.7 ms vs 常开 0.486 @ 33.3 ms；AV-MNIST 效应极小 −0.0028，且其音频只有 1 601 个独特数组各复用 ~37 次（数据缺陷）。
- Owner 决定（2026-10-12，`reports/OWNER_DECISIONS_20261012.md` P2/P3）：REG 作为正面应用、归因改写；GAIN 入附录作边界分析。NUISANCE / COMPRESS / AMORTIZE 未跑。EPFL 数据使用依据待补；BF/SHG 测试集未下载。

---

## 7. R7 D1：问题条件下的视觉 token 剪枝（LLaVA-1.5-7B / Qwen2.5-VL-7B；2026-10-11 → 10-14）

**问题。** 用 VCS 风险 r(K)=(T_B−T_A(K))² 训练一个选择器，决定保留哪些视觉 token；同架构对照 S2（logistic 后验→同平方读数）、S3（直接回归教师实现风险 h）；外部对照 uniform grid、SparseVLM（LLaVA）、MMTok / HiPrune（Qwen，E1）。主终点：匹配时延（L25/L50/L75）下的官方任务质量。预注册规则 DF-8：只有 10k-FIT 与 full-FIT 两个尺度同号且都过 Holm 才写 claim。

### 7.1 结果（准确率点差，95 % 区间；n.s. = Holm p ≥ 0.05）
**D1-FULL（LLaVA，DEV/val，`reports/R7/R7_D1_FULL_REPORT_20261013.md` §1；9 claims / 11 scale-dependent / 7 no-claim）**
| 对比 | GQA DEV | TextVQA val | ChartQA val |
|---|---|---|---|
| S1−S2 @L25（full / 10k） | −0.06 n.s. / −2.86 | **+2.01 / −4.37（反转，两尺度都显著）** | −0.10 n.s. / −1.13 |
| S1−S3 @L25 | +0.38 n.s. / −1.19 | +1.18 / +8.76（claim：S1 高） | −0.19 n.s. / +1.32 |
| S1−SparseVLM @L25 | **−3.59 / −4.12（claim：S1 低）** | **−19.59 / −16.97（claim）** | **−3.59 / −2.60（claim）** |
| S1−SparseVLM @L50 | −1.19 / −1.09（claim） | −11.11 / −7.52（claim） | −2.62 / −2.00（claim） |

**官方测试（R7-50，owner 批准 2026-10-12，各读一次）：**
- Phase L（10k 选择器）：GQA testdev S1−S2 @L25 −3.24 [−3.76, −2.66]，S1−SparseVLM −4.03；ChartQA test S1−S2 −0.87，S1−SparseVLM −1.65。与 DEV 同号 18/18。
- Phase Q（Qwen，10k，固定比例）：S1−grid @25 % GQA −4.81，ChartQA **−16.25**；S1−S2 @25 % −1.40 / **−13.28**。Qwen 上网格处处优于所有学习型选择器。
- Phase F（FULL 选择器）：S1−SparseVLM @L25 −3.06 / −1.99，@L50 −1.47 / −1.49（全部 seed 为负）；S1−S2 @L25 GQA −0.78、ChartQA +0.75（各由单 seed 驱动）。
- 固定预算 25 %（Phase F，GQA testdev，full 61.95）：grid 58.70 > SparseVLM 58.79 > S2 56.57 > S1 55.73 > S3 53.41。（前两项顺序有误，见文末 Corrections 2026-10-17。）
- E1（Qwen，`reports/R7/R7_E1_REPORT_20261013.md`）：GQA f=0.125 grid 82.08、S1 76.78、MMTok 83.23、HiPrune 82.36；ChartQA f=0.25 grid 56.77、S1 39.51、HiPrune 69.95。"学习型选择器在 36/36 cell 低于网格列"。

### 7.2 D1 确立了什么
- **确立：** S1（VCS）在 L25/L50 低于预算匹配的 SparseVLM（三个基准、两个尺度、官方测试同号）；S1 在 TextVQA 上高于 S3（唯一对 S1 有利的 claim，无官方测试对应）。
- **未确立：** VCS 选择器优于 logistic 兄弟（10k 时 S2 领先 2.9–4.4 点，full 时消失或在 TextVQA 反转 → 尺度依赖，不写）；任何对 SparseVLM / 网格 / MMTok / HiPrune 的优势。
- **根因未识别：** REV8（critic-only 诊断，36 单元）给出三个候选：(i) VCS / logistic critic 在同一问题、同一预算内对三种 mask 机制的选择**处于随机水平**（成对准确率 0.498–0.504，top-1 0.33 ≈ 1/3，regret = 随机选择；S3 略高 0.525–0.544）；(ii) 软–硬 gap 是松弛而非 Gumbel 噪声，只在 phase-A lr 3e-4 下可观（−0.057）；(iii) phase B 的 dropout 扰动门控输入（train-vs-eval top-k Jaccard 0.62–0.67）。
- **REV9（已冻结、运行中）：** 对 9 个冻结单元从 phase-A 末状态配对重跑 phase B（I1 eval-mode；I2 + straight-through hard top-k），同一评价器在固定预算上比较——这是回答"候选因素是否致因"的实验。LLaVA 10k 与 FULL 两套都在队列里。

### 7.3 解读
D1 的失败形态与 R5 一样：**成对 critic 学到的 r(K) 能排序"整体上哪个预算更安全"（pooled Spearman 0.09–0.21），却分不清同一问题内三个候选 mask 的优劣（随机水平）。**选择任务要的是候选间的竞争，而二分类后验（VCS 和 logistic 一样）不提供。S3 直接回归实现风险略好，说明问题在"后验→风险"的读数，不在二次形式。另外 REV8 指出两个纯工程缺陷（phase B 用了训练模式 dropout；软硬松弛），REV9 会量化它们的份额。

---

## 8. V7 / V7X：文图细粒度对齐（EvalMuse / GenAI-Rank；2026-10-13 → 进行中）

**问题。** 在 FGA-BLIP2 的训练路径上，把元素级软投票 y∈[0,1] 当作 P/Q 的质量，比较三种元素损失：M0 L1（FGA 原生）、M1 平衡 logistic、M2 VCS (1−J)/4；同一接口、同一初始化、同一数据顺序；N0 为原生 FGA 训练器复现。读数：EvalMuse DEV 自然 Brier（主）、GenAI-Rank 人工选图 M9（终末，标签封存未读）。v2 包把主家族改为"M2-E vs FGA / VQAScore / TIFA / DSG"（系统比较），M2 vs M1 降为归因家族。

### 8.1 已有数字
- **Table A（DEV 自然 Brier，`V7_TABLE_A_results/v7_table_a.json`）：** M0 0.1814 / 0.1838 / 0.1830；M1 0.1730 / 0.1734 / 0.1734；M2 0.1735 / 0.1740 / 0.1736。**M2 − M1 = +0.0005 [−0.0007, +0.0021]、+0.0005 [−0.0006, +0.0016]、+0.0003 [−0.0009, +0.0014]**（307 prompt 聚类 bootstrap）。三个 seed 点估计都为正（M2 略差），区间全含 0。限定：每个 M 臂选中的 lr 都是网格上沿 2e-5（协议无扩展规则）。
- **X21 共同特征（768-d 未训练 BLIP-2 初始化，15 个主拟合 + 3 种敏感性，`V7X_X21_results/`）：** N-VCS-F − N-BCE-F +0.0009 / +0.0002 / −0.0001（含 0）；− N-DV、− N-NWJ 含 0；− K-KDE-S **−0.0155 / −0.0153 / −0.0169（VCS 明显更好）**；− K-NYSTROM +0.0020 [+0.0001, +0.0039] / +0.0022 [+0.0000, +0.0045] / +0.0006（Nyström 略好，DEV Brier 0.1789 为全场最低）；− K-RULSIF 含 0。敏感性：PCA-64 时 VCS 略优于核与 BCE（部分区间不含 0）；PCA-256 时 Nyström / RuLSIF 反超（+0.0044 / +0.0030）；N0 训练过的表示（S2）上所有方法改善 ~0.005，排序不变。
- **X11 受控锚点（`V7X_X11_results/x11_summary.md`）：** 经典 CS-KDE 在 d=20 的每个固定带宽都错过真值（δ=2：mult 1.0 给 0.048，真值 2；Scott 7.06）；KL 桥接的 DV/NWJ 在 δ=2 发散（−0.19、−5.6 对真值 2）；学习型 VCS/BCE 后验 MSE 0.018 ≈ 彼此；MMD/HSIC 置换检验水平 0.035–0.060，功效合理，XOR 依赖只有 HSIC 能看到。
- **外部系统（B0 FGA、B1 VQAScore、B2、TIFA、DSG、PN-VQA、HPSv3、CLIPScore）：** GenAI-Rank 与 GenAI-Bench 的预测大部分已生成或在队列；**人工标签未读**，主家族（8 对比）等 owner 的评分批准。

### 8.2 解读
V7 是迄今匹配最严的自然数据实验（同接口、同初始化、同数据顺序、冻结协议）。结论与 §2 一致：**M2 ≈ M1；学习型路线（VCS 与 BCE 一起）明显优于 KDE 后验，但不优于调好的核回归（Nyström / RuLSIF）**。这意味着在"给定表示、估计一个后验"的任务上，二次形式没有可测的精度增益，而且"学习型 vs 核"的优势取决于维度（PCA-64 翻转）。真正能产生"外部竞争力"主张的是完整模型对 FGA / VQAScore / TIFA / DSG 的人工选图对比——这部分尚未读标签，是 V7 剩下的最大未知。

---

## 9. 横向总结：暴露的根本问题

1. **"同一后验"是硬约束，不是失误。** VCS 与 logistic 的总体最优解相同（docs/03 §3），有限样本差异只是优化路径效应（R4-E1：拟合准则本身贡献 +0.0004，不显著）。六条自然数据线 + 受控数据全部印证。**任何以"VCS 比 logistic 更准"为前提的实验设计都不会有结果**——D1、V7、R5、VL3、R6 都是这么设计后得到"无差异"的。
2. **成对二分类后验不适合"从候选中选一个"。** R5（候选数越多差距越大，−0.3 → −3.5）、D1（同一问题内 mask 选择处于随机水平）、R6 GAIN 的小效应，都是同一结构：任务要候选间竞争或排序，而 P/Q 后验只给每个候选一个孤立的分数。补救（R5 的 query 内归一化、D1 的 S3 直接风险）都来自任务侧，不来自 VCS。
3. **精度已有的对手太强，而我们从不占优。** 对传统估计器（KDE、RFF、Nyström 小 N）学习型路线有优势，但那是"学习型 vs 核"的共性（JS 同样享有）；对调好的核（RuLSIF、Nyström 大 N / 高维）持平或略输（P86、X21）。
4. **三条替代论证各有限定（§4.4）。** 鲁棒性：P134 支持的斜率优势被 R4-E0 重读为"局部退化优势"，且理论上不能写成定理；嵌套增量：恒等式精确、P118 有真值链上 VCS 通过全部判据、但自然特征上不可见；认证下界：对任意有界 T 成立，弱依赖在 n ≤ 512 永远无法认证。
5. **工程缺陷反复出现在 VL 线。** D1 phase B 的训练模式 dropout、软硬松弛（REV8）；D1 的选择分数混合了 budget 间与 budget 内的方差（R8-04）；R5 的 63 个预定行未评价、测试标注未封存；V7 的 lr 全部选在网格上沿。SSL 线经过 100+ run 的打磨后没有这类问题——这是成熟度差距的直接表现。
6. **大量结论被后续审查撤回或改写**（R4_02 §0 列表：P41 "within 1 point" 撤回、P108 "下界读数"撤回、P134 "supported"→mixed、P120 CIFAR-100 VCS>JS 翻转、P76 solo-learn REFUTED、P128 撤回……）。好的一面：台账机制有效；坏的一面：说明一次性读数在这个项目里的可靠性低，任何新 claim 都要预设会被重读。

## 10. VL 相对 SSL 的成熟度

| 维度 | SSL（CIFAR） | VL（R5 / D1 / V7） |
|---|---|---|
| 迭代次数 | 155 run + v3–v7 线，100+ 配方 | 每条线 1–2 轮冻结单元（D1：18 选择器；V7：21 单元） |
| 任务与对象的匹配 | 对象（两视图 J）就是任务目标 | 任务是选择/排序/系统分数，对象是成对后验：**结构不匹配**（§9.2） |
| 对照 | 等预算 SimCLR / VICReg，matched JS | 匹配 logistic 到位；外部系统（SparseVLM、FGA、VQAScore…）全部强于我们 |
| 根因诊断 | 几何诊断、逐层 probe、机制§3 已固定 | REV8 刚给出候选，REV9 在跑；R5 根因（候选竞争）已识别但补救不归 VCS |
| 官方测试 | 一次（v2 配方） | R5 一次、R7-50 三相；V7 标签未读 |
| 可写结论 | "CS 型目标可达 SimCLR 同级" | "匹配条件下 VCS ≈ logistic；学习型后验 ≫ KDE；系统比较待读" |

**判断：** VL 线的成熟度差距不是"再跑几轮就能补"的那种——它来自任务结构（§9.2）。SSL 之所以成熟，是因为 J 本身就是 SSL 的目标；VL 任务里 J 只是一个中间量。继续在 VL 上追求"VCS 独有优势"没有数据支撑；VL 能提供的是"学习型后验路线是否有外部竞争力"（V7 主家族）和"候选任务需要什么样的读数"（R5/D1 的结构发现）。

## 11. 下一步（建议，按信息量排序）

**A. 保留 JS 对照，但改变它在论文里的角色（已定）。** 台账角色 = "同一后验、不同评分规则"；所有自然数据表里 VCS 与 JS 并列为"学习型后验路线"的两个实例；不再把 M2>M1 / S1>S2 当作任何表能否成立的门槛（v2 包已这样改）。

**B. 把优越性论证限定到能证明的三条，并补齐它们缺的实验（估计器层，便宜）：**
- 鲁棒性 → **P134 扩展成分离轴实验**：ε 轴 × 污染类型 × 维度 (d 20/64) × N，以"共同污染后验上的绝对误差"和"相对自身干净种子的退化比"两种读数并列报告（R4 重读用的正是后者），预设 falsifier（SOURCE_ANALYSIS §C：MSE 比 ≥ 0.9 即不成立）。这是 V7X-11 锚点框架可直接承接的（加一个 ε 轴，CPU 即可）。
- 嵌套增量 → 在 X11 的 A2 链上做"有真值的嵌套压缩"（P118 的 Gaussian 链换成 X11 的 d=20 joint-vs-product），报告 nested_exact 的 L² 可加性与 JS 的 Bregman 增量并列。
- 认证下界 → 只写保守一侧界的正确算法与覆盖率（R4-T2 已有），明确"不认证弱依赖"。
- 这三条都不需要 GPU 大预算；全部可在两周内给出带真值的表。

**C. VL 侧：收口而不是扩张。**
- D1：等 REV9（LLaVA 10k + FULL）结果——若 I1/I2 显著缩小 S1−S2 / S1−grid 差距，则 D1 的报告以"训练缺陷修复后的匹配对照"收口；若不显著，D1 作为"后验读数不适合候选选择"的负面结构发现收口。不再做新的 D1 变体。
- V7：完成 v2 主家族（M2-E vs FGA / VQAScore / TIFA / DSG，人工选图），这是整个项目**唯一还可能给出"对外部系统有竞争力"结论**的实验；读标签前把 TIFA/DSG 的本地替代标记清楚（V7X_30 的 API 必要性评审可由标签前的保真度试点决定）。
- R5 / R6：已完结；论文里 R5 写"候选归一化（KLIEP 实例）过官方测试"，R6 REG 写"学习型 critic 类效应"。

**D. 论文结构（解读，供 owner 定夺）：** (1) 对象与性质（S、J、恒等式、RPC 定位、嵌套分解、保守界）；(2) 估计器基准：学习型后验（VCS/JS）vs 传统估计器，含分离轴（鲁棒性）表；(3) 应用：SSL（等预算对照，同级）、R6 REG（正面，类效应）、V7（系统比较，待读）；(4) 归因：matched JS 并列，"无可测二次增益"作为明确陈述而不是遗憾；(5) 结构发现：成对后验在候选任务上的局限与 R5/D1 的补救。

---

## 附录：证据索引（按线）
| 线 | 冻结协议 | 结果（results-only commit） | 报告 |
|---|---|---|---|
| SSL v2 配方 | P36/P41/P43 prereg | `reports/P44_ceiling_results_table.md`、P68 官方测试 | `reports/CVPR_EVIDENCE_LEDGER_20260928.md` §A、`reports/SYNTHESIS_20260925.md`、`reports/SSL_TECHNICAL_NOTE_20260926.md` |
| SSL v3–v7 | P104/P105/P107/P114/P120/P129 prereg | 各 P 的 results | `reports/P107_AP3_LAYER2_REPORT_20261003.md`、`reports/P114_V5_JS_AP3_REPORT_20261004.md`、`reports/STATUS_20261008.md` |
| 估计器 | `reports/P85_ESTIMATOR_BENCHMARK_PREREG_FROZEN_20260928.md`、P108/P134/P152 prereg | `outputs/P85_estim_benchmark`（85 JSON）、`reports/R4/P151_r4/` | P86 / P108 / P134 / P151 / P152 报告；**`reports/R4/R4_02_EVIDENCE_LEDGER_20261011.md`（总台账）**；R4_E1、R4_T2 |
| 理论恒等式 | — | `reports/P121_theory_checks.json` | `reports/P121_V6_THEORY_REPORT_20261003.md`；`r4/package_r4_20261010/docs/03_*.md` |
| 第二应用预检 | P45–P68 prereg | — | `reports/SECOND_APP_FINAL_SUMMARY_20260928.md`；ledger §B |
| VL3 / R4-VL | `reports/R4/R4_VL_TRAINING_PROTOCOL_FROZEN_20261011.yaml` | 0ece645、89ceb47 | `reports/VL3/VL3_COMPLETE_12CELL_REPORT_20261010.md`、`reports/R4/R4_VL1_REPORT_20261011.md`、`R4_VL2_REPORT_20261011.md` |
| R5 | `reports/R5/R5_FINAL_EVAL_FROZEN_20261012.yaml` | 337cc98 (`reports/R5/final_eval/`) | `reports/R5/R5_FINAL_EVAL_REPORT_20261012.md`、`R5_PHASE_REPORT_CN_20261011.md` |
| R6 | 各 REG/GAIN 单元 prereg | 各单元 results | `reports/R6/R6_SYNTHESIS_20261012.md`；`reports/OWNER_DECISIONS_20261012.md` |
| R7 D1 | `reports/R7/R7_D1_PROTOCOL_FROZEN_20261012.yaml`、`R7_D1_FULL_STAGED_FREEZE_20261012.yaml`、`R7_50_PROTOCOL_FROZEN_20261012.yaml` | `reports/R7/R7_D1_30_RESULTS.json`、`R7_D1_FULL_results/`、`R7_50_results_{L,Q,F}/` | D1-30 / D1-FULL / 40c / E1 / R7-50 L,Q,F 报告 |
| REV8 / REV9 | `reports/REV8/REV8_DIAG_PROTOCOL_FROZEN_20261013.yaml`、`reports/REV9/REV9_PROTOCOL_FROZEN_20261014.yaml` | `reports/REV8/REV8_DIAG_RESULTS/` | `reports/REV8/REV8_DIAG_REPORT_20261014.md`；REV9 运行中 |
| V7 / V7X（分支 v7/t2i-alignment） | `reports/V7/V7_03_PROTOCOL_FROZEN.yaml`、`V7_03_AMENDMENT_V2_20261014.yaml`、`V7X_20_*`、`V7X_11_*` | `reports/V7/V7_TABLE_A_results/`、`V7X_X21_results/`、`V7X_X11_results/` | `reports/V7/V7X_00_INTAKE_DELTA.md`、`V7X_10_EVIDENCE_COVERAGE.csv`、`V7X_30_EXTERNAL_BASELINES.md` |

---

## Update 2026-10-14 (after publication of this map): P155, official test of A-P3
Owner, 2026-10-14: A-P3 is the final recipe; test it.  `reports/P155_AP3_OFFICIAL_TEST_REPORT_20261014.md` (pre-stated contrasts, frozen in `78085a9`):
- **A-P3 − SimCLR (SimCLR recipe at VCS's view / epoch budget), official CIFAR-10 test, linear: +0.63 [+0.36, +0.90], 5 / 5 seeds → "ahead".**
  A-P3 88.83 ± 0.10 vs SimCLR 88.20 ± 0.13.  kNN: −0.32 [−0.94, +0.30], no detectable difference.
- A-P3 − JS-AP3 (matched JS, same recipe): linear −0.07 [−0.57, +0.43]; kNN −0.38 [−0.78, +0.02] (n = 3); no detectable difference.  The lead
  belongs to the recipe, not to the quadratic loss.
- §3's sentence "R4_02 §6.1: no v3–v7 recipe was ever confirmed on held-out data" no longer holds for A-P3.  §3's interpretation now reads:
  with the final recipe, a CS-type objective leads SimCLR at equal view / epoch budget on the linear probe (not on kNN); matched JS on the same
  recipe is level with it.

## Corrections (2026-10-17; numbers unchanged)
1. §7.1, line 175 — "grid 58.70 > SparseVLM 58.79 > S2 56.57 > S1 55.73 > S3 53.41" inverts the first two rows: 58.70 < 58.79.  At 25 % (144
   tokens) on GQA testdev, R7-50 phase F, the order is SparseVLM (adapted) 58.79 > uniform grid 58.70 > S2 56.57 > S1 55.73 > S3 53.41 (full 61.95;
   rows of `reports/R7/R7_50_results_F/r7_50_results.json`, seed means for S1–S3, printed by `scripts/r7_d1_closeout_tables.py`).  SparseVLM is
   the best row at 25 %, as the phase F report reads (`reports/R7/R7_50_PHASE_F_REPORT_20261014.md:148`).  Line 175 keeps its sentence and
   carries a pointer to this section.

## Update 2026-10-17: D1 closed
D1 is closed by `reports/R7/R7_D1_CLOSEOUT_REPORT_20261017.md`, after REV9, REV9-FULL and REV10 (§7.2 above still describes REV9 as running).  The
closeout rests on D1's pre-registered refutation rule (`reports/R7/R7_D1_PROTOCOL_FROZEN_20261012.yaml:257-260`; the clause "no practical
advantage" fired on both hosts).  The closing rule of §11 C (line 237) is not applied as written and neither of its labels is used (closeout §4);
the D1 interpretations of §7.3 (line 185) and §9.1–9.2 (lines 206–207) are not carried forward (closeout §8).
