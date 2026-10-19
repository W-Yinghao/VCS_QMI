# 第一批诊断：先量化影响，再批准修改训练

## 0. 统一约定

这是一轮新的审查，不是原始预注册的补写。记下首次看到当前结果的时间。本包不修改主任务权重、最终测试评价或正在运行作业。所有新增输出写独立目录，如 `outputs/REVIEW_R8/`；仓库报告编号先查重，建议前缀 `REV8-`。

首批自然数据只用现有 FIT/CAL，禁止以官方 GQA testdev、ChartQA test 和 RefCOCO test 选择审查结论或超参数。DEV/val 可用于事先写明的描述性复验，但已经不是全新独立证据；默认先不扩大。

采样单位：image/chart → questions → masks → stochastic repeats。不同 mask、Gumbel 或 dropout 重复不当作独立图片。S1/S2 用同一 image/question/mask 列表；可复用 S3 作任务级控制。缺失、NaN、常数、全同 h 必须保留计数。

## REV8-00 接入与作用范围

**目的**：确认被审查的是正在使用的代码，而不是一个已经被修好的旧版本。

交付：
- HEAD、worktree dirty 状态、两个 critic 模块 blob/SHA256；live job 的 code hash、resume version；不要只记录 main 的名字。
- D1-FULL/E1/E2/phase F 的 completed/running/queued/unknown 与依据，不编造进度百分比。
- 官方测试的模型/规则冻结记录；哪些测试已读；新修改需要何种 owner decision。
- 本轮审查用的阶段 A checkpoint 是否存在。只有最终 gate 权重时，不能声称已经恢复同一 A 起点和 gate 初始化。

辅助命令 `repo_source_audit.py` 只读 AST、git 状态和文件字节；`repo_runtime_probe.py` 仅在显式允许时导入原模块，使用微型合成输入记录训练模式。

## REV8-01 模式、随机性、梯度路径

### 样本与范围

先用合成输入检验定义；再对每宿主每目标 seeds 0–2 的已选阶段 A critic 做自然数据诊断。初始自然子集固定为 CAL 哈希最小的 128 张图、每图最多 4 个现有二元问题；数据不足用全部并记录。这个数是审查预算建议，不是科学效应预测。每个 stochastic condition 重复 32 次；主预算先用 25% 和 50%，全保留作零改动 sentinel。

### 条件

1. `eval / fixed hard mask`：同输入重放，重复输出与梯度（必要时容差，不强求跨 GPU bit-identical）。
2. `train / fixed hard mask`：固定 mask，独立 dropout。
3. `train / same-input B=A`：量化没有观测改变时的自分歧。
4. `train / common RNG`：仅诊断 common randomness 影响；明确 shapes/forward 次序相同，否则不能说 dropout 精确耦合。
5. `eval / soft mask + autograd`：冻结参数、可训练 gate，mask/gate 梯度有限且非零；parameter checksum 不变。清除旧 A.grad 后再判断无新 critic 梯度。

输出 per question 的 mean/std(T_B,T_A)、mean squared gap、covariance、self-gap、grad norm、mask-grad、module modes。重复估计的方差作为诊断，不用 32 次随机前向得到一个“独立样本 n=32”的自然性能 p 值。

**判断**：可以确认 train-state stochasticity 的量级；只有再训练 gate 的配对干预，才能判断它是否造成任务差距。若差异接近数值误差，就如实结束这条解释。

## REV8-02 四格 mask 与预算偏差

保持 critic eval、同输入固定 full output/无 dropout，取相同 gate logits。对每问题预算生成一次 ξ 后比较 HD/SD/HN/SN，另重复 ξ 得到稳定分布。包括 τ=当前最终温度 0.1；τ=1 与接近硬选择的值只作预先指定诊断，不能据任务成绩挑温度。

记录：
- 四个 risk 与 `SN-HD = (SN-HN)+(HN-HD)` 数值检查。
- `sum(m)-k`、Σ m(1−m)、m>0.5 的数量、有效 key 支持；mask floor 总质量约 `(N−k)*eps` 只是输入质量，不等于实际 attention 质量。
- HD/HN 交并比、排序一致性、不同预算 nesting 是否符合当前设计。
- deterministic soft/hard 输入对应的 pooled output、T_A 与 mask 梯度。

**关键**：旧 gap 同时换噪声/松弛，继续原样报告但改注释；只要发现 gap 不能就自动切 hard/ST 训练。硬前向 STE 的偏梯度是一种新求解器，需独立规范和对照。

## REV8-03 Spearman 审查与选择影响

使用现有 CAL predictions 的精确未四舍五入数组。主结果：legacy ordinal Spearman vs average-tie Spearman；constant 返回 `null` 并给 reason。tie 统计按真实存储精度计算；只用于展示的 round 后数值不可用于复算。

顺序：
1. 先对已选模型重新预测原固定 mask，检查读数变化。
2. 对相同数组共同置换 20 次，记录 legacy 和 corrected 的变化；正确 average Spearman 应对共同置换不变（浮点容差）。
3. 有逐 checkpoint 预测才复算历史 argmax；只有 selected/last checkpoint 就报告 incomplete coverage，不从相关性 scalar 反推重选。
4. 记录对 LR 排名、selected step 的潜在改变，**不访问测试结果来选择**。

替代代码必须通过：无 ties 一致；两边常数返回未定义；一边常数返回未定义；ties 使用平均名次；common permutation invariant；NaN/Inf/长度不符显式处理。不能加随机 jitter。

如果调整选择准则，视为新选模规则。原论文原结果与 corrected-sensitivity 并列；仍然能报告旧 frozen 模型的任务结果。

## REV8-04 同问题同预算的决策评估

定义每个 `(image_id, qid, k)` 的有限候选 mask 集合 M_qk，现有三种 mask 是起点。不同 masked inputs 若字节完全相同，记录 duplicate 关系；不把复制项当作新增证据。

```text
r_j = critic 预测的压缩风险
h_j = 该 mask 上实际宿主平方风险相对完整宿主的变化
j_hat = argmin_j r_j
j_oracle = argmin_j h_j                # 用标签的经验参考，不可部署
regret = h[j_hat] − min_j h[j] >= 0
```

预测并列时用预先固定的 mask_id 字典序，另报随机平均 tie 作为敏感性可选；不能用 h 打破预测 ties。

主要读数：
- 图像宏平均 regret、相对 uniform/grid 的实际 h 变化；同时展示各策略 h、d 与 residual，不只给差。
- within-query/within-budget pairwise direction accuracy：对 h 非零差的 pair，预测 tie 记 0.5；h tie 数量单列。
- within-group Spearman（3 masks 很粗，只描述，不用大量小组 Pearson p 值）；no-spread 子集与缺失组单列。
- 按预算、任务类型、宿主、正确/错误 full prediction、视觉质量分层，不挑最好格作核心结论。

`min h` 只是有限 mask 集合、一个真实标签下的经验 oracle；不是 Bayes-optimal compression，也不代表这份收益可从输入预测出来。

### 策略 mask 的外推诊断

在同一 CAL 子集加入已有 gates 的 hard-deployed mask（S1/S2/S3），宿主只补尚未存在的前向。宿主调用数量 = unique missing (question,mask) 对数，预算先测。

比较 `r-h` 的 signed bias、MAE、h−d、实际错误率；h 可为负，不截断。若 r 小但 h 大，才讨论潜在低估；不要因为某一条差很多就宣布整个 critic 被利用。

另在 FIT 留出的审查子集以同规则复核，区分训练数据过拟合与 mask-policy shift。不可把新 CAL 结果用于多轮 gate 调参后继续称它独立确认。

## REV8-05 表示覆盖与接口（次优先）

只有前四项无法解释失败时才扩大：记录 token 坐标覆盖、网格空洞、空间聚集、OCR 区域保留（需独立可用标注/检测，不新增不可核对标签）。先用 grid/random 作等预算参照。

Qwen 的可变 N、padding、原 M-RoPE 位置与 selector 因子位置表应维持已验证接口。检查 PCA 投影在不同任务/位置的误差可以作为诊断，**不可据此自动改为更高维或更大模型**。

## 首轮统一输出

- `intake.json`、`source_audit.json`、`runtime_probe.json`（若运行）。
- `mode_probe.csv`、`mask_fourcell.csv`、`rank_audit.json`、`mask_decisions.csv`。
- `SERVER_RESPONSE_CN.md`：逐条证据、接受/反驳意见、缺口、是否建议新实验。
- `run_manifest.json`：hash、数据 id 范围、种子、dtype、GPU、采样权重、耗时，不存用户访问令牌。

本包参考 CSV 汇总器只覆盖定义与分组，不实现宿主推理或完整统计报告。服务器须对接本地已有模块，不能把脚本样例输出当自然结果。
