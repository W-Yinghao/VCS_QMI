# 完整训练与代码接入

## 1. 按源码而不是名字复现
已核对FGA来源commit `45c8bb4af5d8c0d1d9f3733156177cc67ad9b7e5`。
- 活跃ITM分支：overall L1 + 0.1 token L1 + 0.1 validity-mask L1；不是BCE。[G02]
- 论文Eq.(1)含exp(prompt variance)，当前活跃分支未乘该项。保留N0-Code行；Paper-var只在明确恢复并另记变体后运行，不能静默声称一样。
- YAML指定5epochs、初始lr1e-5、warmup100、wd0.05、训练batch14、image364、freeze_vit=True。这些是已读配置，不是实测可运行性保证。需展开model_type:coco对应初始模型配置。[G03]
- 原生dataset把mask/token label截成32，并将index0设为overall标签；这不适用于“每个语义元素独立概率”的直接解释。[G04]

## 2. 四配置训练矩阵
| ID | 输入/头 | 元素准则 | 总体/有效性辅助项 | 用途 |
|---|---|---|---|---|
| N0 | 原生FGA-code | 原生token L1 | 活跃原生代码 | 从相同FIT重训的任务复现 |
| M0 | 共享完整提示＋语义跨度池化 | abs(d−y)，自然采样 | 与M1/M2相同 | 检查输入/池化改变与L1任务拟合 |
| M1 | 同M0 | 平衡Logistic | 同M0/M2 | 后验拟合归因对照 |
| M2 | 同M0 | (1−J)/4 | 同M0/M1 | VCS主方法 |

M0不被解释成严格proper后验基线。N0与M2有输入/权重/聚合差异，不能只归因于loss；M1/M2是对应的严格匹配比较。默认所有M共享自然overall L1（1–5尺度）、validity-mask L1，系数遵循源代码0.1并明确元素损失尺度；辅助项强度的对称敏感性只在CAL，不能按test结果改。

## 3. 初始化与可训练参数
发布FGA checkpoint只用于外部推理参考；主训练从同一个BLIP2 ITM/coco初始化，不从看过全部EvalMuse的FGA成品开始。保存每个tensor加载来源、unexpected/missing keys与requires_grad清单。

保留官方视觉冻结选择；训练Q-former、query tokens、ITM与新增/共享头。完整训练是完整多模态任务路径，并非强制解冻十亿参数视觉塔。共同参数初始化按seed完全一致；更换损失不能改变head大小、token长度、数据顺序和可见标签。

## 4. 输入与跨度
M组使用完整图像的确定性resize到364，无水平翻转/随机裁剪，以免左右、数量和边缘对象标签失效；N0原生处理器独立保留。该输入改动是公共接口设计，不计入VCS独有创新。首次冻结前报告原生处理器具体行为。

M组文本最大长度提案128：只用FIT输入长度分布检查覆盖；若尾部仍明显超出，先验证架构支持后固定更长长度。不要按DEV准确率选截断长度。图文模型超过位置表支持时禁止静默截断，需报错并使用预先固定分块/回退。

跨度使用tokenizer offset mapping/确定性的字符跨度转换；不要靠“同字符串token序列的第一个出现”解释所有实体。元素logit从其语义跨度平均后转换概率。重复词/多跨度/重叠/大小写/标点/否定均有测试。GT索引0总体score不作为元素，padding永不加入分母。

## 5. 从图片batch计算正确的期望
D的目标是prompt→image→element均匀。可使用uniform-image DataLoader并给每图权重N_images/(N_prompts*n_images_in_prompt)，再平均其有效元素；所有M组共享权重。π按FIT同一权重估计，不使用裸元素计数代替。

训练类平衡使用soft投票的两个分量权重，固定总体π。归一化分母不得随哪个标签在微批出现而改变。梯度累计/DDP应合并原始加权分子和全局分母；CPU测试对比大batch和分块精确一致（dropout关闭用于数值检查）。

## 6. 配方与选择规则
主epoch=5，global batch初始按N0源码14。资源允许不构成随意提高M2batch的理由。采用AMP，损失、logsumexp及校准计算至少fp32。warmup100steps、wd0.05，其余实际scheduler按锁定原生代码复现。

N0先跑原生lr1e-5；M0/M1/M2对称搜索lr={5e-6,1e-5,2e-5}，专用tuning seed=17，与正式seed={0,1,2}分开。每个搜索跑完整5epochs，所有epoch checkpoint按CAL-SELECT锁定规则选。

主选择规则：M1/M2按**自然Brier**最小选择，M0亦用同一风险选择以检验概率用途；同时保留各自native criterion选中的epoch作为次级诊断，不据test改选。N0按原生任务L1/官方可验证选择规则，若只有固定最后epoch则保持，单列差异。

主部署score=E，备用O与Hα分别登记，Hα的α∈{0,.25,.5,.75,1}仅用于次级表。CAL-SELECT比较总体score也要记录，不将后处理调参预算藏掉。CAL-PROB只做概率校准，不重选checkpoint。最终正式seed全部运行。

## 7. dropout和复现模式
训练期dropout按共同配置运行。所有评价和部署一律model.eval()；校准不更新模型参数。若新方案含冻结模块的可微优化，明确eval与requires_grad区别，不沿用旧r7冻结参数却未关闭dropout的做法。

保存初始、每epoch、选定权重的hash、RNG状态、数据顺序、effective steps与样本暴露。原始标签/输出未改写；NaN、缺图、跳batch按固定策略记录。

## 8. 外部估计路线是正式子实验
原K0泛称拆成X21中的KDE、RuLSIF、Nyström核平方回归，以及共同特征上的VCS/BCE/DV/NWJ。固定初始BLIP2表示、相同P/Q与数据角色，按11–12号规范输出同目标读数；有样本级后验者进入相同选图接口。它们不替代完整多模态训练。

经典CS原生批次散度另保留在A0；不能给一个batch scalar强行加“每图概率”名称。参数与pair预算公平，低维带宽/高维标准化不故意省略。

## 9. 资源和执行器
本包不假设服务器有某种GPU或外部API。先实测一个epoch/推理池，再在既有owner批准资源内调度，保留每GPU型号。完整模型顺序仍为原生复现 → 9次专用tune → 12个正式fit → 完整主评价；外部系统与X10/X21按资源并行，不等待M2先赢M1。调参单元计入成本，不能只报3seed的费用。

runner建议接口：`prepare-data / audit / train / export-predictions / evaluate / report`；每个命令仅凭manifest与frozen config运行。仓库可复用工具，但不要导入旧错误的Spearman或把旧R7输出目录混用。实际GPU训练代码由server实现，本包只提供CPU数学/指标参考。
