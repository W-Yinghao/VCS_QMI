# 修复清单与小型估计确认

## 1. P151：修的是统计对象，不只是列名

本轮读取 `scripts/p151_e0_aggregate.py` 的 111–160 行，确认 SAME 面板仍把 `native.value`、`native.signed_error` 放入 common J 字段，stability 与 resolution 也继续使用 native 数组。[R3]

### 字段字典

| 输出字段 | 允许来源 | 禁止替代 |
|---|---|---|
| `common_J` | `posterior.J_eval`，或有 schema 证明的等价字段 | JS2/InfoNCE/DV 的 native value |
| `common_J_minus_S` | `posterior.J_eval_minus_S` 或 common_J−同目标truth.S | native.signed_error |
| `posterior_mse` | `posterior.posterior_mse` | 平方风险 R；R 含不可约 Bayes 风险 |
| `common_J_bootstrap` | 公共读数的保存数组 | native.boot_values 改名 |
| `native_value` | 原生值＋明确 estimand | 无 estimand 的数字 |
| `nonfinite` | 明确保存的状态 | 缺失时填零 |

缺失应为 null 加 `reason`，而不是 0。metadata 写 `fit_criterion` 与 `evaluation_readout` 两栏；JS→J 是合法路径，JS native 不是同一个 S。

### 修复流程

1. 从当前单元 schema 读取实际 key；保存覆盖率表。不要全靠旧脚本的字段假设。
2. 所有选定行保留 `selected` 来源、种子、配对构造、输入数据 hash。RFF 的 family 合并不得按字母序挑到有利子组。
3. 对可恢复指标生成新聚合；公共 bootstrap 未保存时，先做 point/seed-spread，不伪造 CI。
4. 运行回归测试：故意设置 JS native=1.1/common_J=0.7 的 fixture，检查输出是 0.7。测试缺失值必须失败或显式空缺。
5. 报告“哪些结论改变”“哪些 posterior MSE 不受影响”“哪些图需要重画”。保留旧文件与 supersession 说明。

## 2. P134：原始 posterior 重读先于新训练

每个污染单元分别选择同 method family 与 negative construction 的注册选中行。clean 以同 seed、同构造连接，不跨种子替代。

$$e_{k,\epsilon}=E_{M_\epsilon}(T_{k,\epsilon}-\eta_\epsilon)^2,$$
$$d_{k,\epsilon}=e_{k,\epsilon}-e_{k,0},$$
$$\Delta e_\epsilon=e_{V,\epsilon}-e_{L,\epsilon},\quad
\Delta d_\epsilon=d_{V,\epsilon}-d_{L,\epsilon}.$$

这比较污染任务中的误差如何变化；不同 ε 的目标与测度变了，不能直接叫清洁鲁棒性。

首要报告绝对误差与配对增量差；退化比 $d_V/d_L$ 仅在分母同号、远离零时给出，否则标记不可解释。三种子 t 区间仅作该小样本条件的说明，不能把 EVAL 中成千上万 pair 当作额外训练重复。多 ε 共用 clean，若作整体趋势用同 seed 联合重采样/模型而非独立格拼接。

输出每类污染全表，包括不利于 VCS 的重尾/离群；预先注册的 product 读法与事后其他构造分开。旧论文标题的“supported”只代表它当时规定的 normalized-shift 规则，不能替代本次 common-posterior 结论。

### 真正需要新确认时

只取一个新的生成器/条件，保留原实现与已核 oracle：例如现有非线性生成器，N=4096、原信号/总维度、ε={0,0.1,0.2}；VCS/JS×3 seed=18 个小型 critic 拟合单元。参数是建议，执行前冻结实际 setting 名。

在同一次拟合后分别读取：污染后真值误差、清洁分布上相对于清洁后验的误差、训练/CAL风险、选中预算、logit/参数梯度。如果清洁评价数组原来没保存，就不能从污染 MSE 反推出清洁 MSE。

不要把 18 格与其他 d/N/数据集自动相乘。恢复 clean 的修正损失暂不默认训练，因为它可能仍是 RPC 参数实例，且误差传播必须先处理。

## 3. 求解诊断：从已训练 JS 起点继续优化 J

这一步只是分辨“优化/泛化/选择”的小实验，不宣称新优化器。

主条件使用已存在 oracle 的一个中等依赖条件，固定表示、函数类、FIT、CAL、EVAL。总预算记为 U，并保存中点 U/2：

| 臂 | 前半 | 后半 | 用途 |
|---|---|---|---|
| V→V | VCS | VCS | 直接平方路线 |
| L→L | JS | JS | 直接 Logistic 路线 |
| L→V | 同一个 JS 中点 | J | 检查转换目标能否改善共同风险 |
| L→L 控制 | 同一个 JS 中点 | JS | 若优化器状态重置，必须相同重置的控制 |

如果直接 L→L 与重启控制完全相同，不重复计算；只保留一份轨迹。若学习率因目标缩放需要选取，两个后半臂使用同样的预先有限候选和 CAL 选择，记录所有开销。

现有 checkpoint 缺失才补最小训练。N 扩展只在原条件完成后取一档，不预设差距缩小或反转。关键输出是独立风险曲线，而不是只看梯度饱和比例。P152 较少饱和既不能单独证明优化慢，也不能单独证明抗污染强。

## 4. 核路线的有限加强

优先修预算后重读。需要新增时，选一个低 N 和一个 padding 高维条件：

- product 的同目标 partner 必须保留；cyclic/in-batch 另列。
- 当前带宽最优值位于边界时有限外扩一次，不给任何方法无限搜索。
- padding 场景可以加入预先拟合的 ARD/维度尺度核，但尺度只用 FIT/CAL；记录可训练参数与调参开销。
- 在可实现时比较 N、K-pair 或 all-pair 条件，真实计时；“闭式”不是“零成本”。
- 经典 KDE-CS 的真值属于其固定参考目标，独立报告；不能拿它直接填 common-S RMSE。

默认不为每种神经方法重跑所有 kernel 网格。只回答已经提出的比较缺口。

## 5. 新颖性与安全措辞检查

应删除/限定：JS 理论上至少同样准确；不同损失不可能改变排序；S-J 恒等式只有平方拟合能用；普通 T≥0 等于语义匹配概率一半；增加预算必反转；独立样本 1/n+1/m 是当前首次发现。

另需限定：“线性类闭式解”指对输出 T 的线性/二次目标，可能带约束；若 T=tanh(wᵀφ)，整体一般不是无约束岭回归闭式解。ridge 后 tanh 是两阶段近似，不能称为精确最优 J。

发布前把每句分成：原理论事实、直接代数、现有经验、解释假说、待证新结果。作者最终确定主贡献；服务器不能通过命名替代文献排他性审查。
