# 需要负责人决定的事项（2026-10-12）

截至 2026-10-12 07:40 UTC，所有能自主推进的工作都在运行或已完成；下面各项需要你决定。
回复时写编号和选项即可，例如 "T1 批准；T2 先更新方法列表，边界 1.0；P1 方法；D4 删除"。
没有回复的项不影响当前运行，会维持表中"当前做法"一列。

## 一览

| 编号 | 事项 | 我的建议 | 当前做法（未决定时） |
|---|---|---|---|
| **T1** | r7 官方测试一次性评估（R7-50） | 批准，在 D1-30 读数之后执行 | 不读任何官方测试集 |
| **T2** | r5 最终独立评估（官方测试集） | 先更新方法列表，再批准；边界由你定 | 不读 |
| **T3** | R4-02 问题 3：确认数据用哪一个未读集合 | 与 T1、T2 合并决定 | 不读 |
| **P1** | 查询内归一化的 VCS critic 写进方法还是只写分析 | 方法变体 + 分析 | 报告中只作分析 |
| **P2** | r6 REG 的表述 | 学习 critic 类效应，不写成 VCS 特有 | 按综合报告的表述 |
| **P3** | r6 GAIN 的表述 | 澄清性结果 + CREMA-D 部署演示 | 同左 |
| **P4** | R4-02 问题 1：核函数比较的主面板 | 预注册乘积伙伴为主，多负样本单列 | 同左 |
| **P5** | R4-02 问题 2：SimCLR 的称呼 | 改为 "SimCLR recipe at VCS's view / epoch budget (P41)" | 报告里已这样写 |
| **P6** | R4-02 问题 4：公开编码器的 COCO 预训练暴露 | 作为声明的限制 | 同左 |
| **P7** | AV-MNIST 基准缺陷要不要写进论文 | 脚注或附录 | 只在报告里 |
| **D1** | SP-DocVQA 注册 | 暂不需要 | 缺口已记录 |
| **D2** | Ego4D NLQ 许可 | 暂不需要 | 未下载 |
| **D3** | Tier C 大数据集 | 继续推迟 | 未下载 |
| **D4** | 删除冗余文件（约 77 GB，清单见下） | 删除 A–D，保留 11 GB 易失档案和 REG-P3 用的目录 | 不删（删除由你执行） |
| **D5** | 发表前的数据许可确认 | 见下文逐项 | 内部研究使用 |
| **U1** | 从未授权的 R4-VL2 增补 1（softmax(ν)） | 放弃 | 不运行 |

另有三项**仅供知悉**（I1–I3）：没有异议就按计划进行。

---

## 一、官方测试集与最终评估（必须由你明确批准；我不会自行读取）

### T1 — r7 官方测试一次性评估（R7-50 §A）
- **内容**：用 D1-30 中预先固定的各行，在 **GQA testdev-balanced（12,578 题）** 和 **ChartQA test（2,500 题）** 上各读一次。
  - 不提议 TextVQA test：它的答案在服务器端，要上传，而 r7 规则禁止向第三方上传。
- **程序**：
  - 预测先算 sha256 并提交到 git，然后才打分；标签只由一个 CPU 作业读取一次。
  - 不重新测延迟，沿用 D1-30 的测量。
- **成本**：LLaVA 约 6–10 GPU 小时；如果做 R7-40，Qwen 另需约 4–6 GPU 小时。
- **选项**：(a) 批准，在 D1-30 读数（如做 R7-40 则在其后）之后执行；(b) 只批准其中一个测试集；(c) 不做，论文只报 DEV / val。
- **建议**：(a)。GQA val 的图像来自 Visual Genome，LLaVA 训练时见过，所以 testdev 是唯一干净的 GQA 对照。
- **文件**：`reports/R7/R7_50_60_OUTLINE_DRAFT.md`（文末附审批文字草稿）。

### T2 — r5 最终独立评估（RefCOCO 系列官方 test）
- **现状**：`reports/R5/R5_FINAL_EVALUATION_PROPOSAL_DRAFT_20261011.yaml`，状态为"仅提议，未授权"。
- **它早于 R5-QN / R5-QX1**，方法列表里没有归一化 critic，也没有 RefCOCO。我建议批准前先更新：
  - 加入 VQ / JQ（R5-QN、R5-QX1），以及分布 softmax D；
  - 加入 RefCOCO testA / testB（R5-X1、R5-QX1 已有 CAL 选出的检查点）；
  - 主对比可改为 VQ − set softmax（非劣效）和 VQ − VCS（优效），再按数据集 / 划分做 Holm 校正。
- **非劣效边界由你决定**（提案给出的选项）：
  - 0.5：约等于种子间一个标准差，严格，3 个种子时功效可能不足；
  - 1.0：小于这些基准上方法之间通常的差距，即"同一精度档次"的说法；
  - 2.0：只能支持"可用的替代方案"这种粗略说法。
- 边界必须**依据用途**来定，不能依据我们的 DEV 结果。我已经看过 DEV 结果（VQ − E 为 −0.7 到 −1.3），所以**不推荐具体数值**，以免看起来是受 DEV 影响的选择。
- **选项**：(a) 我按上面三点更新提案，你给出边界后批准；(b) 按原提案批准；(c) 不做。

### T3 — R4-02 问题 3：确认数据
- **问题**：哪个未读集合用来对 v3–v7 SSL 配方或 VL 结果做一次性确认，什么时候做？
- **候选**：CIFAR-100 test、ImageNet val、VL 的 test（即 T1 / T2）。CIFAR-10 test 已经用掉。
- **建议**：VL 部分按 T1、T2 处理。SSL 已按你的指示停止，不再新开确认，除非你另有安排。

---

## 二、论文范围与表述

### P1 — 查询内归一化的 VCS critic（KLIEP 归一化）
- **证据**：在 RefCOCOg / RefCOCO+ / RefCOCO 上，VCS 落后 softmax 的差距分别有 61 % / 69 % / 86 % 被归一化弥补（+1.13 / +2.22 / +7.56 点）。
  - 估计量不变，最优 critic 仍在函数类内。
  - 剩下 0.8–1.7 点的"准则差距"，三个数据集上区间都在 0 以下。
  - 诊断显示，它消除了差距随干扰项数量增长的趋势。
  - 报告：`reports/R5/R5_QNORM_REPORT_20261012.md`、`reports/R5/R5_QX1_REFCOCO_REPORT_20261012.md`。
- **选项**：(a) 作为方法变体（"VCS critic with within-query normalization"）并写入分解分析；(b) 只作为分析，解释差距从何而来；(c) 不写。
- **建议**：(a)，但要如实写明：它仍比 softmax 低约 1 点，并不更好。

### P2 — r6 REG（跨模态配准）的表述
- **证据**：学习 critic 比逐图 MI / NMI 多配准 18–20 个百分点（平移和 ±30° 刚体两种设定）。在 < 42 px 时与 CoMIR 没有可检测差异，在 < 9 px 的精度上更好。VCS、logistic、InfoNCE 三者没有可检测差异。
- **建议**：写成"CS 族学习得到的局部关系 critic 是比互信息更可靠的配准能量"，**不写成 VCS 特有**。这与冻结措辞一致；见 `reports/R6/R6_SYNTHESIS_20261012.md`。

### P3 — r6 GAIN（按需计算额外模态）的表述
- **证据**：回归实际收益 h 的策略优于熵（CREMA-D 上明显，AV-MNIST 上很小）。平方分歧 z 可能显著劣于 h（CREMA-D 的 Brier 族）。CREMA-D 上，25 % 预算只用"总是计算视频"约 29 % 的墙钟时间。
- **建议**：写成对证据增益代理量的澄清性、部分负面结果，加一个 CREMA-D 部署演示。两者都是标准风险回归，不写成 CS 特有方法。

### P4 — R4-02 问题 1：核函数比较
- **建议**：主面板放预注册的乘积伙伴（§2.2），多负样本效应单列为"收益与成本"面板（§2.3）。另一种做法是保留 cyclic-8 为主，并说明它不是预注册伙伴。
- **文件**：`reports/R4/R4_02_EVIDENCE_LEDGER_20261011.md` §8。

### P5 — R4-02 问题 2：SimCLR 的称呼
- **建议**：稿件中凡是写 "tuned SimCLR" 的地方，改为 "SimCLR recipe at VCS's view / epoch budget (P41)"。这符合你"SimCLR 不调参"的规则。

### P6 — R4-02 问题 4：预训练暴露
- **建议**：公开编码器可能见过 COCO 图像，作为声明的限制写明，不做审计。RefCOCO / + / g 共用 COCO 图像，三者不是独立领域，这一点也要写明。

### P7 — AV-MNIST 基准缺陷
- **事实**：MultiBench 的 AV-MNIST 训练集中 60,000 条音频只有 1,601 个唯一片段，每个约重复 37 次。同一片段对应约 8 个不同数字的图像，标签纯度 0.44。
- **建议**：写成脚注或附录。用图像分组划分会把"记住片段"误当成音频证据，这一点对使用该基准的人有用。

---

## 三、数据、许可与磁盘

### D1 — SP-DocVQA（需要你在 rrc.cvc.uab.es 注册）
- 只在 D1 的跨文档验证（可选）中用到。**建议**：暂不需要；只有当你希望 R7-40 / R7-50 覆盖文档领域时再注册。

### D2 — Ego4D NLQ（需要你签署许可，约 48 小时后拿到 AWS 凭证）
- 只用于 D6（时间支持），不在当前路线上。**建议**：暂不需要。

### D3 — Tier C（DataComp-small 约 450 GB、CC3M 130–400 GB、HowTo100M、原始视频）
- 放不进 1.2 TB 的空间下限（当前剩余约 1.5 TB；这个文件系统与其他用户共享）。**建议**：继续推迟。

### D4 — 删除冗余文件（删除由你执行；合计约 77 GB）

**A. 已解压、可从官方重新下载的档案，72.4 GB**
- 位置：`/projects/EEG-foundation-model/yinghao/data/` 下。
- 校验：19 个档案逐一核对过，其中每个文件都已在磁盘上，大小与档案内一致；sha256 仍记在 manifest 里。

| 档案 | GB |
|---|---:|
| `gqa/zips/images.zip` | 21.82 |
| `clevr_v1/zips/CLEVR_v1.0.zip` | 19.02 |
| `coco/zips/train2014.zip` | 13.51 |
| `textvqa/zips/train_val_images.zip` | 7.07 |
| `coco/zips/val2014.zip` | 6.65 |
| `gqa/zips/questions1.2.zip` | 1.50 |
| `cub_200_2011/zips/CUB_200_2011.tgz` | 1.15 |
| `coco/zips/val2017.zip` | 0.82 |
| `coco/zips/annotations_trainval2014.zip`、`annotations_trainval2017.zip` | 0.50 |
| `zurich_summer_5914759/zips/Zurich_dataset_v1.0.zip` | 0.14 |
| `snli_ve/zips/snli_1.0.zip` | 0.09 |
| `pcmepp/zips/coco_annotations.tar.gz` | 0.05 |
| `gqa/zips/sceneGraphs.zip` | 0.04 |
| `coco/zips/caption_datasets.zip` | 0.04 |

注意：删掉 GQA questions、CLEVR、SNLI、PCME++ 的档案后，它们的测试文件只剩 `closed/` 里那一份（已计算哈希，从未读取）。

**B. AV-MNIST 重复解压副本，1.50 GB**
- 路径：`av_mnist_multibench/avmnist/`。
- GAIN 实验读取的是另一份只含训练集的副本 `extracted/avmnist/`。

**C. 已完成 SSL 单元的中间文件（R4-02 问题 5），约 1.3 GB**
- 236 个**未纳入 git** 的 `*.partial.json`：在 `ssl_pilot/reports/` 下（P116、P118、P142–P144 等，10-02 至 10-07），约 55 MB，各单元的最终 JSON 都在。
  - 另有 16 个已提交进 git 的保留不动。R4-02 账本当时写的是 74 个，现在实际数是 236。
- `outputs/P128_*` 四个 SimCLR 部分运行，共 1.23 GB。这组网格已于 10-04 被你撤回。

**D. 已完成单元的冒烟 / 检查目录，约 1.6 GB**
- 都在 `/home/infres/yinwang/CS_QMI/outputs/` 下；对应结果都已写进报告。

| 目录 | 大小 |
|---|---:|
| `R7_D1/d1_30_smoke` | 579 MB |
| `R7_D1/train_smoke` | 230 MB |
| `R7_D1/cache_smoke` | 216 MB |
| `R6_GAIN/smoke` | 182 MB |
| `R6_REG_cpu_smoke`、`R6_REG_cpu_smoke2` | 各 59 MB |
| `R6_REG_P2_comir_check`、`R6_REG_P2_comir_check_reg`、`R6_REG_P2_cpu_smoke` | 约 16 MB |
| `R6_GAIN/superseded_image_roles` | 8.8 MB |
| `R7_D1/d1_30_fix_staging`、`_E2`、`_E3`（补丁已应用并提交） | 各约 4 MB |
| `R7_D1/data_provisional`（已被冻结的 manifest 取代） | 2.2 MB |

**暂不删除**
- `R6_REG/smoke`（50 MB）、`R6_REG_P3_check`（289 MB）、`R6_REG_P3_cpu_smoke`、`R6_REG_P3_gpu_smoke`（各 63 MB）：REG-P3 还在开发，它的测试和冒烟会用到。等 REG-P3 结束再删。
- 应保留的 11.0 GB：托管在 Google Drive 的 `avmnist.tar.gz`、`cvpr2016_cub.tar.gz`、`AMBER_images.zip`（这些 Drive 链接已经失效过），以及 QVHighlights 镜像压缩包（在通过验证前保留）。

**建议**：删除 A–D。`outputs/` 下的部分按你的惯例用 `scripts/cleanup_outputs.sh`（先干跑，再加 `DO_DELETE=1`）；`/projects` 下的档案目前没有脚本覆盖，需要的话我可以写一个默认干跑的删除脚本给你执行。

### D5 — 发表前的数据许可
| 数据 | 情况 | 建议 |
|---|---|---|
| EPFL RGB-NIR（REG-P3 拟用） | 没有许可文本；你此前只为 P60 接受过 | 内部研究可以用；发表前需要你确认 |
| Zurich Summer | Zenodo 写 CC BY 4.0，压缩包内 README 写 CC BY-NC-ND 4.0；**没有运行**（会触顶） | 无需动作 |
| AV-MNIST | ESC-50 噪声为 CC BY-NC 3.0，仅限研究 / 非商业 | 论文中注明 |
| CREMA-D | ODbL 1.0 / DbCL 1.0 | 论文中注明 |
| QVHighlights 特征 | 非官方镜像，标记为未验证；没有使用 | 只有 D6 用到时才需要验证 |

---

## 四、从未授权的草稿

### U1 — R4-VL2 增补 1（set softmax 配检测分数参考 ν）
- **现状**：`reports/R4/R4_VL2_ADD1_SOFTMAX_NU_DRAFT_20261011.yaml`，从未授权，从未运行（代码里的 `set_softmax_q` 分支未使用）。
- **建议**：放弃。r5 已经表明参考分布不是关键（R5-10），关键在查询内竞争（R5-QN / R5-QX1），这一臂已经回答不了新问题。

---

## 五、仅供知悉（没有异议就按计划进行）

- **I1 — R7-40（在 Qwen2.5-VL-7B 上做跨宿主验证）**：属于包内路线任务，约 25–45 GPU 小时。
  - 被测方法（S1）已经在任何 D1-30 结果之前记录在案。
  - 我打算在 D1-30 出结果前冻结 40a；外部基线 IF-Prune 只作单独的宿主列。
- **I2 — REG-P3（EPFL RGB-NIR，第二对模态）**：这是我利用空闲 GPU 加的单元，约 6–10 GPU 小时，正在实现，冻结后作为填充任务运行。
- **I3 — D1 的数据暴露**：
  - LLaVA-1.5 训练时见过全部 GQA train 图像和全部 train-balanced 问题。
  - 我们改用 train_all，并排除了 LLaVA 见过的全部二元问答（331,902 行，分布在 66,852 张图上）；图像仍然是见过的，已在协议和报告中披露。

## 当前运行（供参考）
- **r7 D1-20**：第一阶段剩最后一个单元，之后自动选学习率，再跑第二阶段 12 个单元。
- **r7 D1-30**：不依赖选择器的行已经提前运行；SparseVLM 分片正按 E3 恢复；driver 等 D1-20 完成。
- 所有结果和报告都已推送到 GitHub。
