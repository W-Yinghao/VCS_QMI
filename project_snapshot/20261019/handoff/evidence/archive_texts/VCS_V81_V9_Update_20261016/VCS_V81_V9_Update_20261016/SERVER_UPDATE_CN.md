# V8.1 / V9 增量审查意见 · 2026-10-16

## 0. 范围与版本

这是上一版 `VCS_V8_1_Server_Diagnostics_v1_20261015.zip` 的补充件，不是 V8.2、V9.1 或 V10。保留原 V8.1 任务、V8 冻结结果与 V9 已批准的运行范围。本件没有向 GitHub 写入，没有提交、取消或修改服务器作业；以下修改建议需要服务器确认现行快照后执行。

本次读取截点：

| 分支 | 提交 | 相对上一包 |
|---|---|---|
| main | `55d47c9fe73375230c86a7ec1b2e575bdaeb19d5` | 无变化 |
| v7/t2i-alignment | `2dc39b5f9892aba828ea6d857e40dfd8120cbc74` | 无变化 |
| v8/utility-aligned-cs | `d2067c4e24407e0c731cfca9c8bad7f8fd5d38ff` | 比 `580781f` 新增 7 个提交；最新时间 2026-10-15 19:38:46 UTC |
| v9/reliable-correspondence | `8a994f3cce6d9c8b414018200e3d2f38e1633903` | 比 `1ce0e84` 新增 7 个提交；最新时间 2026-10-15 18:17:01 UTC |

最新记录是已提交状态，不是实时 SLURM 队列。本次 GitHub 分支列表未包含 `v8.1/utility-diagnostics`，直接读取该 ref 的审查报告返回 404；但 V8 报告已经引用它。应先推送该分支／报告和输出摘要，而不是把“外部暂时读不到”解释成“服务器没有做”。

## 1. 已核对的新结果：只更新结论，不据此改模型

### 1.1 TIFA160 已完成

800 图像、160 提示；TIFA 两人评分与 DSG 另行收集的评分分别报告。它们是同一批图像上的两套标注，不是两个独立图像 benchmark。以下为三个种子的原始分数先平均，再计算 Spearman（不是三个 Spearman 的平均）。[R1]

| 方法 | TIFA 两人均值 | DSG 评分均值 |
|---|---:|---:|
| U-CS | 0.767 | 0.773 |
| U-BCE | 0.776 | 0.786 |
| U-RPS | 0.774 | 0.778 |
| U-CE | 0.782 | 0.790 |
| FGA released | 0.749 | 0.765 |
| VQAScore CLIP-FlanT5-XXL | 0.696 | 0.704 |
| 本项目本地 TIFA 实现 | 0.525 | 0.530 |

U-CS−VQAScore：+0.071 [0.018, 0.124] / +0.068 [0.016, 0.122]。U-CS−FGA：+0.018 [−0.002, 0.039] / +0.008 [−0.014, 0.029]。U-CS−BCE：−0.009 / −0.013。保留原协议“次级评价、报告区间、不据此提出新的主声明”的阅读方式。相关性不等于 best-of-9 选图效果；不能将三套 benchmark 的不同指标直接拼成总排名。

更新后的判断：V8 在已测外部任务上有竞争力，不能再概括成“所有外部任务都落后 VQAScore”；仍没有证据把系统收益主要归于 CS 而非共享图文模型与监督。TIFA 路线有 3 个提示无保留问题、15 张图按原规则为 0；保留，不在事后删除。[R1]

GenAI-Rank、GenAI-Bench、TIFA160 的原模型、聚合方式、标签用途不变。TIFA 两种标注上的检验家族实际实现与协议说明有差异，已在原报告披露；本件不重新调整 p 值来产生新结论。若后续需要新的正式声明，再单独说明比较家族，不静默覆盖旧表。

### 1.2 epoch 0 已更正

V8 报告已确认：`selected_epoch=0` 是第一整轮训练之后的保存点，报告称其包含 2,379 步、全部 26,177 张 FIT 图像；不是未训练 head。[R1,R2]

只更新表述，不因此重训。不再以 `epoch=0` 判定 U-RANK 完全没有学习。其是否学到了排序，应按 V8.1 的原生损失、梯度、同提示指标核对。不要重新开启八损失学习率网格。

## 2. V8.1 要补的只是执行可见性与报告口径

### U01：推送已引用的 V8.1 原始材料（P0，零新训练）

交付 intake、epoch 审查、D02–D06 JSON/CSV/Markdown、任务完成状态。若运行仍未结束，提交明确的部分状态，不把局部审查称为全诊断完成。首先复用已做的工作，禁止再提交同一套诊断。

### U02：维持诊断对象（P1，零新训练）

继续回答：
1. 相近质量是否来自相同选择？区分逐种子与分数集成，正确处理并列。
2. MSE 改善来自提示级偏差还是提示内区分？居中使用真实标签的读数只作诊断。
3. FIT 与 CAL 轨迹是否支持过拟合或适配问题？权重缺失时停在日志分析，不伪造历史 checkpoint。
4. 正常／固定错配输入是否改变输出？错配输入不能沿用原标签作为语义真值。

新增 TIFA160 成绩仅纳入完整外部汇总，不据其生成器子集、seed 或误差调结构。不将其结果当作 C1 的适配退化证据。原 C1（CS/RPS/MSE ×3）仍是条件提案，不能自动启动。

## 3. V9-N2 的紧急接入问题

以下是源码事实和独立小例子核对，不是已经观察到真实训练失败。若服务器已在未推送快照修复，请返回 diff、真实入口检查和实际快照，而不是重复修复。

### U03：HOST_LOCK 机器格式与加载接口不一致（P0）

[R3] 中存在：

```yaml
model (SGR single, SGRAF opts.py defaults):
  img_dim: 2048; word_dim: 300; embed_size: 1024
```

以及同一行 `i2t_rule: "..."; t2i_rule: "..."`。逐字提取的片段经 PyYAML 解析分别抛 ScannerError／ParserError，不能直接供 `yaml.safe_load` 使用。

另一个独立问题：[R4] 的 `cmd_unit` 把 `(lock or {}).get('host', lock)` 交给 `HostConfig.from_host_lock`；[R3] 的 `host` 是字符串 `SGR_single`，而 [R5] 的构造器对传入对象执行 `.get`。修正 YAML 语法后，这里仍会产生字符串无 `.get` 的类型错误。

`precheck` 读取 `training`，真实锁文件却名为 `training (native SGR, train.py)`；即使不报错，也可能默默使用 HOST_DEFAULTS。`native_epochs`、`learning_rate`／optimizer 描述、`num_layers_gru` 与构造器字段还需显式映射。

要求：
- 人类可读说明与机器配置分开，或统一为有类型的 schema。每字段独立写 YAML，不以分号串联。
- 一个共享解析器同时服务 precheck、teacher、weighted unit 与官方基线 wrapper；明确数据集选择。
- 使用真实 `HOST_LOCK.yaml` 跑解析测试，导出最终配置。缺失、未知或类型错误不能静默退回默认值。
- 检查 Flickr：40 epochs、lr_update 30、batch 128、Adam 2e-4、margin 0.2、clip 2；COCO 不属于当前块，不应继承 Flickr 默认后悄悄运行。
- 缓存 teacher／weight 时检查配置、代码、数据、角色和分数哈希；仅判断文件存在不足以保证兼容。

不取消其它实验；只在修复前暂缓使用该入口提交新的 N2 正式块。

### U04：teacher AUROC 应是诊断，不应自动成为整块拒跑条件（P0，需修订草稿）

[R6,R7] 新增 `AUROC>0.999` 或 `<0.55` 即拒绝该 dataset/rho/seed 全部正式运行，并将高 AUROC 描述为 `nothing to weight`。

这不是由理论推出的。反例：干净对得分 1，错配对得分 0，teacher AUROC=1；在 40% 错配训练集中，用正确的 0/1 权重剔除错误正对仍然有意义。较低 AUROC 说明这个 teacher 不可靠，不代表 NCR/L2RM 原生机制没有价值。若分数方向反了，低 AUROC 甚至不是“无信息”。

已读取的负责人记录要求的是预检查表，没有列这两个数值拒跑规则。[R8] 当前代码不是看测试成绩后筛选，但仍然把一个描述性诊断升级成了未经该记录支持的研究范围门槛。

建议：将它们改为 warning／difficulty_stratum，不过滤种子，不为了通过阈值重训 teacher，不因它们取消外部基线。真正必须拒跑的条件保留：非有限值、样本或配置错位、空类、非法权重、未通过基线一致性检查等。若仍希望因问题过易／过难节约预算，另列负责人决定及被排除范围，不能写成统计必然结论。

### U05：precheck 的 Spearman 重现了并列值缺陷（P0，零新训练）

[R6] 的 `spearman` 使用两次 stable argsort，没有平均并列名次。独立复算：

| 输入 | 当前函数 | 平均名次定义 |
|---|---:|---:|
| [1,1,1,1] 对 [0,1,2,3] | 1.0 | 未定义 |
| [0,0,1,1] 对 [0,1,0,1] | 0.8 | 0.0 |
| [1,1,2,3] 对 [4,3,2,1] | −1.0 | −0.948683… |

Unweighted 权重恒定，RankMass 大量 0/1，因此这是实际预检查会遇到的情况。AUROC 函数已处理平均名次；问题特指 Spearman，不是所有排名函数都错。

改用平均名次，常数返回 JSON null + reason；添加上述测试和一般无并列例子。重算已生成预检查表并记录差异。不因此重训 student，不据受影响的相关性决定两方法等价。

## 4. V9-N2 的两个统计／复用限定（P1）

### U06：相同系数＋相同 seed 不足以直接合并正式单元

[R6] 的等价分组只看归一化系数差 <=1e-12 和 seed。当前协议允许每方法选择不同 lr multiplier；相同系数、不同学习率不产生同一训练轨迹。不同 optimizer、scheduler、初始化、batch、精度、剪裁和选模也类似。

只对相同算法路径且完整运行 fingerprint 相同的单元考虑复用。先比较实际训练 dtype 下的系数，再比较模型初值、数据顺序、随机状态、优化器与 schedule、步数、精度和选模。NCR 双模型／L2RM 重新匹配不得用此条件并入普通 weighted SGR。数值接近不是证明；若无法证明，按原计划各自运行。

当前 `pipeline.run_unit` 读到 merged 字段后仍创建 student，因此我没有证据说它已经错误跳过训练；需要在后续 dispatcher 实现跳过之前补全这条规则。[R9]

### U07：“observed image” 的 80/20 划分不能直接宣称独立区间

[R9] 在配对完成后按 observed_image_id 拆 U。错配 caption 可能来自另一个源图像，配对的两个来源会横跨两组；参考 Q 的循环配对同样引入依赖。拆了接收图像 ID 不等于两组基础观测或分数独立。

短期：保留 point estimate 与权重实验，区间标为 held-out-image diagnostic，不称已验证的独立样本保证。输出接收图像、caption 原图来源、donor 与连通成分的重叠检查。

如要理论覆盖声明，应先按基础图像划角色再在角色内配对，或使用覆盖配对依赖的界；不要为这一项要求重建全部 student 训练。不得用标注正确性作为训练过滤条件。以当前管线量化区间的限制与完整检索性能分开报告。

## 5. G1 / G2 接续（P1）

G2 已有 EPFL 双轴草稿、CPU/GPU 检查和计时安排，但本次未读到正式汇总；G1 也未见新完整结果。协议列 48 个学习单元＋4 个核参考，必须明确它们是计划还是实跑。[R10]

继续现有数据可用的负责人决定，不重开许可讨论。完成真实计时与字段校验后按已批准范围推进；BF–SHG 封存测试依旧按原确认触发条件处理，不因 G2 准备好就自动解封。

两点在草稿冻结前核对：
- `<42` 与 `<9` 是换算后的误差单位，EPFL 对应 `<10.5` 和 `<2.25` 原生像素，正文不要混写。
- CAL ceiling rule 若改变区分性读数，必须检查 aggregator 的实际主 family 是否同步；不根据 DEV 更好临时切主指标。这里尚未对 aggregator 逐行复核，不预先认定其实现有错。

## 6. 本轮交付与优先级

| ID | 优先级 | 交付 | 新完整训练 |
|---|---|---|---|
| U01 | P0 | V8.1 分支、原报告、数值输出和完成矩阵 | 0 |
| U02 | P1 | TIFA 已完成状态、统一外部表、原 D02–D06 完成或缺失原因 | 0，必要时补原开发推理 |
| U03 | P0 | 真实 HOST_LOCK 解析、唯一配置解析器、端到端入口检查 | 0，仅原定 smoke |
| U04 | P0 | AUROC 改 warning 的草稿修订／或独立批准证据 | 0 |
| U05 | P0 | 并列排名修复、回归测试、重生成 precheck | 0 |
| U06 | P1 | 全配置 fingerprint 复用规则 | 0 |
| U07 | P1 | 来源依赖审查、区间用途标记 | 0 |
| G2 | P1 | 草稿冻结与计时、指标转换核对 | 原批准范围，不增加单元 |

这不是要求所有 CPU 检查结束前把所有 GPU 暂停。V8 确认、G1/G2、REV10 继续各自已批准流程；只针对具体受影响入口限制新提交。N2 的启动范围依最新负责人记录，COCO不自动由Flickr触发。

## 7. 来源

[R1] https://github.com/W-Yinghao/VCS_QMI/blob/d2067c4e24407e0c731cfca9c8bad7f8fd5d38ff/reports/V8_UTILITY/V8_04_EVAL_REPORT_20261015.md

[R2] https://github.com/W-Yinghao/VCS_QMI/blob/d2067c4e24407e0c731cfca9c8bad7f8fd5d38ff/scripts/v8/v8_train.py

[R3] https://github.com/W-Yinghao/VCS_QMI/blob/8a994f3cce6d9c8b414018200e3d2f38e1633903/reports/V9/HOST_LOCK.yaml

[R4] https://github.com/W-Yinghao/VCS_QMI/blob/8a994f3cce6d9c8b414018200e3d2f38e1633903/scripts/v9/n2_run.py

[R5] https://github.com/W-Yinghao/VCS_QMI/blob/8a994f3cce6d9c8b414018200e3d2f38e1633903/src/v9/n2/teacher.py

[R6] https://github.com/W-Yinghao/VCS_QMI/blob/8a994f3cce6d9c8b414018200e3d2f38e1633903/src/v9/n2/precheck.py

[R7] https://github.com/W-Yinghao/VCS_QMI/blob/8a994f3cce6d9c8b414018200e3d2f38e1633903/reports/V9/V9_N2_PROTOCOL_DRAFT.yaml

[R8] https://github.com/W-Yinghao/VCS_QMI/blob/55d47c9fe73375230c86a7ec1b2e575bdaeb19d5/reports/OWNER_DECISIONS_20261015.md

[R9] https://github.com/W-Yinghao/VCS_QMI/blob/8a994f3cce6d9c8b414018200e3d2f38e1633903/src/v9/n2/pipeline.py

[R10] https://github.com/W-Yinghao/VCS_QMI/blob/8a994f3cce6d9c8b414018200e3d2f38e1633903/reports/V9/V9_G2_PROTOCOL_DRAFT.yaml

原 V8.1 包：01_SERVER_AGENT_PROMPT_CN.md、docs/04_DIAGNOSTICS_CN.md、docs/06_CONDITIONAL_EXPERIMENTS_CN.md，本次按其完整文本核对；未修改原包。

## 8. 本地核验范围

附带 `review_checks.py` 只复现本次读取快照中的小型数值／接口问题，并可只读解析指定 repo 的 HOST_LOCK。它不调用网络，不读取真实训练／测试数据，不修改仓库，不验证自然数据性能，也不是服务器修复已经通过的证明。见 LOCAL_CHECKS.json。
