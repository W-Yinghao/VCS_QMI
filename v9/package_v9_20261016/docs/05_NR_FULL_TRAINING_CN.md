# 05 · NR-N2：完整噪声图文学习，而不是只做读数器

## 1. 主任务与两个公平性层面
本轮主任务是 **少量可信锚点辅助的噪声图文检索（trust-assisted noisy correspondence learning）**。可信锚点是一项额外信息，不能隐藏。

A表：NCR/L2RM等官方路线、原特征/索引/原生评价，先复现与报告偏差。它不使用新增干净锚点，不能与B表直接宣称同条件SOTA。

B表：所有方法看到相同的可信锚点、相同初始化、相同噪声配对与验证数据；原方法有需要则加入同样anchor-supervised项，标明NCR+A/L2RM+A。这是V9的主要因果与应用表。

C表：固定同一teacher/输入/学习宿主，只替换可靠度估计与使用规则，区分概率读数、rank-only与核方法。

## 2. 数据与可信锚点
Flickr30K先行，COCO第二块；数据保持公开检索train/val/test分区。主协议按原训练图像的10%抽取可信锚点A，90%为未知U；抽取在噪声前，固定seed和图像ID，所有caption随图像分组。

A内按图像再分40% teacher-warmup、30% critic-fit、15% critic-select、15% reference-moments。比例按整数分配后的实际数落盘；每个组不少于可运行最低样本数，否则报数据不足，不移用test。所有角色均可在冻结权重后并入最终student训练，anchor权重为1；这样不会引入测试标签，但性能评价只在官方独立任务集。

teacher：按共同SGR路径，仅在A的teacher-warmup上训练5个完整epoch，冻结其图像投影、文本编码、相似性模块后提取共同score。成本、额外干净监督与所有基线共享。teacher不是VCS先训出来的专用特征；critic-fit等角色不参与teacher选择。

U按固定生成协议注入0或40%**配对机制错配**。为维持A干净且donor只来自U，这一映射不一定等于原论文的全TRAIN索引；故B表明确为trust-assisted构造，原生A表仍读官方索引。所有B表方法使用同一U映射。

生成器保留design_rho、realized_rho、原配对映射、注入位于仅评价侧；critic/student loader只读观测后的pair。真rho不能流进权重函数。实际每个图片组是否错配、caption比例是否同一都落manifest。

## 3. 固定teacher与权重流程
1. 在A-critic-fit上构造P与指定Q，训练/拟合N1各后验方法；Q donor按角色隔离，拒绝同图时明确Q不是无条件乘积。
2. 用A-critic-select作选模；固定模型后用A-reference-moments估计P/Q矩。
3. 在U的**无标签观测**上估计rho。记录这是transductive training-pool估计，不是用EVAL调参；对另一个独立批次报告区间。
4. 对每个U pair算w，写`pair_id,w,score,rho_hat,fit_hash,reference_hash`，锁定后训练student；不在线更新。
5. N1固定的主critic是cos-2；低秩/MLP只有预登记敏感性，不依据任务DEV再挑主函数类。

交叉拟合不是免费证明：首轮采用独立A/U；若以后critic改为读U标签或拟合U，则重新安排out-of-fold，禁止原地切到自训练。

## 4. 完整学习宿主
首选NCR/L2RM共同上游的 **SGRAF中的单模型SGR** 路径。使用原论文公开region特征和词表，训练图像投影、文本编码、相似性推理模块等全部原生可训练部分。

这是该benchmark的完整matching model训练，不是冻结向量上的线性探针；同时不得称为“raw-image视觉骨干端到端训练”。SGR与SGRAF双模型ensemble分别标注。未来CLIP完整微调是另一个现代宿主扩展，不作为本包默认大实验。

server必须读取实际上游配置后填`HOST_LOCK.yaml`：具体commit、feature dims、模块、batch、epoch、优化器、margin、hard-negative规则、warmup、checkpoint选择。不知道的部分标为待核，而不是沿用一个看似合适的猜测。基线能够完整复现后锁定共同recipe，不按VCS的DEV结果优化。

## 5. 权重怎样进入loss
对每个观测正对i，原宿主产生anchor条件下的双向损失ell_i（先得到逐样本row/column项，再聚合）。首轮加权版本：

对合并的 A 与 U，每个 A 样本令 v_i=1，每个 U 样本令 v_i=w_i。主训练目标固定为：

`L = (sum_{i in A} ell_i + sum_{i in U} w_i ell_i) / (n_A + sum_{i in U} w_i)`。

分母在训练前对整个训练池计算。minibatch 用 `mean(v_i*ell_i) / mean_v_full_pool` 估计这一固定目标；所有主方法的anchor角色和自然抽样权重一致。另存U独立的未归一化风险诊断，不将A/U分别归一化后任意相加。

这里mean_w在训练前由全部U冻结，不随minibatch改变；同一recipe所有权重法共用定义。记录这是一种有限样本自归一化加权，而非严格无偏估计。保留未归一化loss的诊断。

w固定且detach，不乘logit，不将低w样本自动从负库删除。原生in-batch hard-negative的潜在错误负例仍存在；加权只能作为**经验方法**，其理论保证不能直接引用独立负库恒等式。

另做一个小但完整的严格接口对照：固定teacher表示，固定独立负库U_neg，逐对loss仅依赖critic可用表示时，核对oracle/posterior权重恢复的风险。它是机制验证，不替代SGR终点。

## 6. 八个正式方法
| 编号 | 方法 | 信息与角色 |
|---|---|---|
| N0 | Unweighted+A | 同锚点teacher/初始化和全部数据，不加可靠度控制 |
| N1 | NCR+A | NCR原纠错机制，同可信锚点；适配写清diff |
| N2 | L2RM+A | 原OT重匹配能力，同锚点；不削成仅过滤 |
| N3 | VCS-Weight | 本包的固定critic → rho → w |
| N4 | BCE-Weight | 同函数类、同teacher、同参考、同rho策略，归因 |
| N5 | RuLSIF-Weight | 传统相对密度比，映射到相同w |
| N6 | Nystrom-Weight | 强核回归，同w接口 |
| N7 | RankMass | 共同原始score排序，保留与N3同等总权重质量的top部分；末端可分数权重 |

RankMass只读取N3训练侧总质量，不读其逐样本排序/测试值；是配对的“数值是否超越同等保留量”诊断，不宣称独立最佳过滤器。另报告自身CAL选择保留率的普通过滤baseline（不额外加入完整主家族）。

ESC/ASL/ReCon是重要外部方法，必须完成论文/代码可用性报告。ReCon本轮官方README仍写coming soon，ESC/ASL代码未核实，不伪造已可运行状态。若实现可用，在结果前追加独立块；若只能按论文复现，清楚命名，报告source fidelity。未补齐时不能声称覆盖所有最新方法或SOTA。

## 7. 调参、种子与单元
正式种子901/902/903；调参种子917不进入正式均值。每个dataset/noise/method完整epoch数据训练，所有种子都运行。各方法获得同样的3个预先固定学习率倍率{0.5,1,2}；原生额外超参数按官方默认，需调时提前写对称可比的预算。

先按Flickr的官方val rSum选任务checkpoint，COCO亦使用同名原生val设置；不可读取test选择。对rho估计器、critic的选择仍只用reference角色。把混合估计与检索选择两层分开。

96正式单元 + 最多96个单种子三点调参单元（8方法×2数据×2噪声×3倍率）是完整规划上限，teacher与原生复现额外计费；执行前实测预算并分块确认。不得把上限当即时提交指令。现有吻合产物可复用并从账单扣除，但必须有相同数据/状态hash。

## 8. 正式评价
每方向R@1/5/10、rSum=六项之和，另给mR=rSum/6，指标单位统一为百分点。COCO 5K gallery为主、5×1K单列；Flickr官方1K为主，不混跨gallery数字。多caption的image-to-text只需任一真实caption命中；text-to-image按真实image ID。

不删除难例/错配误判。每一方法面对同一test candidate全集；两阶段方法/ensemble计全部成本。报告noise=0与0.4的绝对性能和退化，不只画退化百分比。

## 9. 首轮真正要证明什么
主要系统家族：N3 vs NCR+A/L2RM+A，每数据集在rho=.4的rSum，共4个预定主对比；另分传统方法N3 vs RuLSIF/Nystrom共4个；N3 vs N4为归因家族。清洁性能保持和RankMass差异完整报告，不以宽松“非劣效”门槛重新命名失败。

优势可以是任务准确率，或在明确相同质量下减少可信标注/实际成本。仅J/Brier变好而检索无改进，只支持估计结果。下一轮修复/重配对不是默认；先判定权重是否有效以及失败在哪一层。
