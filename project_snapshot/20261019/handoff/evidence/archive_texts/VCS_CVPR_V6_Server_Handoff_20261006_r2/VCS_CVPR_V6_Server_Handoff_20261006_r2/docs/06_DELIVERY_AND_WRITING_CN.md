# 06｜结果交付和稿件同步

## 1. 建议目录 [PROPOSED]

以下是接收端**新产物建议路径**，不是声称仓库已存在这些文件：

```text
reports/manuscript_v6_handoff/
  intake_report.md
  reconciliation.json
  fixture_registry.json
  task_to_p_number.json
  results/                  # 逐实验配置/seed/重复记录
  tables/                   # 机器可读结果与呈现表
  figures/                  # 绘图源数据、代码、最终图
  writing_change_log.md
  phase_report.md
```

原reports、raw logs、checkpoint不移动。当前工作树有并行agent时先约定文件ownership，不重写它们正在维护的聚合文件。

## 2. 每次返回必须包含的内容

用 `PHASE_REPORT_TEMPLATE.md` 说明本次做了什么、复用什么、新增什么；实际值、效应大小、样本/seed、区间和预算是什么；哪些数学/统计条件满足、哪些只在简化测试验证；负结果、失败和缺失有哪些；修改稿件的哪一句或哪一张表。

结构化记录从 `RESULT_RECORD_TEMPLATE.json` 派生；不要填0来代替未完成或未知。status包括DRAFT、RUNNING、COMPLETED、FAILED、BLOCKED、REUSED；数据读取成功或job exit0不是自动COMPLETED。`evidence_tier`区分validation、official-test、fixed-fixture、oracle、smoke、未执行。

每个结果至少保留base-image/encoder/measurement重采样三个层次的单位信息；paired区间注明以什么单位配对。单seed不得写mean±sd across seeds；单个fixture不得叫跨模型结论。

## 3. 图表到论文的映射

| 论文位置 | 应更新什么 | 不应混入什么 |
|---|---|---|
| Fig1/2 | 同目标estimator误差、样本与维度轴；raw source追溯 | MI原始值作S误差、人工估计的error bars |
| Table1 | 同oracle链、同condition、同readout的increment误差 | 无oracle视觉数据的假posterior MSE |
| Table2 | 每dataset一个VCS、外部baseline、完整SSL | pilot、旧recipe official-test、标准化probe混成一列 |
| Table3 | 固定外部模型下method × budget × null/detection | 把内部encoder改良当解释工具胜出 |
| Table4／§5.3 表示分析 | 同一测量协议；最终VCS与外部encoder；检测/响应强度及seed对齐 | 旧内部配方比较；跨强度拼接；把检出率解释为信息比例 |
| Supplement F response | colour.1/blur.25逐seed概率、准确率、翻转与null | 将2000检测抽样与8000响应图像称为相同样本量 |
| §3.3/§5.3 compression | 实际map、signed readings和拟合局限 | 强制单调、跨encoder相减、probe当信息损失 |
| Supplement | shared score、strong aug、内部消融、分层readout | 为新方法占位的内部配置名 |

图表首要显示可解释的效应和公平对照；配置、审计日志、P编号只放作者证据记录，不将内部报告口吻原样写入学术正文。

## 4. 表述边界

| 可以写 | 不可以据此推出 |
|---|---|
| 在测试的同S目标kernel实现上posterior error更低 | 任何依赖估计器都不如VCS |
| JS与VCS学习同一balanced posterior | 二次loss新颖性或posterior标定仅VCS有 |
| 在固定模型、指定n下检出属性依赖 | 原因已识别、某像素有因果贡献 |
| 在指定n和预算下未检出 | 属性信息不存在、已彻底删除 |
| 给定图像操作的预测变化 | 有害shortcut已证明 |
| nested在oracle链降低误差 | 图像端小残差必定代表小oracle误差 |
| VCS的linear更高，kNN相反 | 所有readout一致占优 |
| 当前已完成规模化训练及同预算对照 | 数据可读或吞吐预检查已经完成规模验证 |

## 5. 新结果如何进入下一版

保持 `reference_v6` 不变。每个改动写明V6原句／原数值、源报告、HEAD新增原证据、是否相同协议、拟替换文字、变更理由。新条件与旧条件不同则新增行或单独表，不替换历史行来抹掉反向证据。

修订V6的正文与supplement分别提供Markdown，LaTeX/PDF为排版依据。格式修正、来源更正、新增科学结果分项记录。

最终交付可包含原V6基础上的LaTeX patch和编译PDF；本包本身不升级稿件版本。未完成实验保持在作者计划，不能用TBD数据占正文最终结果表，也不能将未来指标写成观察结果。

## 6. 第一批完成的标准

接入盘点、证据复用、方法/fixture映射、已知测试、需要补齐的协议与预算全部可以复核；每个阻塞项明确归属，不丢弃其他可并行任务。科学实验的完成另按各自冻结协议判断，不以“主要假设全部成立”作为交付门槛。


## 7. r2 两本表账 [V6-R]

- `measurement_comparison`：固定encoder，测量方法为行因素，对应Table3。
- `ssl_retention_response`：固定measurement_protocol，encoder为行因素，对应Table4。

使用两个CSV模板；同一条原始测量可以被不同展示引用，但不得被当作两个独立重复。所有表显示encoder身份与estimator身份，列出检测与响应各自n。两个表不合并计算总体显著性。

服务器的新增结果另保存，`reference_v6/`维持Revised原样。已在编辑的论文需三方合并：旧V6基线、服务器改动、新V6 Revised；不要用解压覆盖未提交写作。
