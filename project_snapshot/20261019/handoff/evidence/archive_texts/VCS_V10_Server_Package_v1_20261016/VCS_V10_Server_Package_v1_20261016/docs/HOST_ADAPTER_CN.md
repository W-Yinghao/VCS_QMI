# 真实宿主接入契约

## 已知与未知

历史源文件是`src/v9/reg/g1.py`与`src/v9/reg/r6_reg_pilot.py`，但本轮没有取得其bytes。这里的接口名描述server需要提供的行为，不声称它们已经是仓库API。不要按文档猜函数名后把import成功当作接入完成。

## 必须在当前源码记录的映射

| 行为 | 需要核对和记录 |
|---|---|
| 数据加载 | 真实class/function、batch字段、scene/core/parent ID、原始尺寸、划分来源 |
| 图像编码 | 两侧模块、输入通道、归一化、训练/评估模式、感受野、输出shape、checkpoint结构 |
| 配对采样 | 每场景位置数、坐标系、普通permutation是否允许固定点、是否存在禁止边 |
| 训练目标 | P/Q各项的系数、scene权重、loss正负号、AMP、正则和优化器 |
| 几何 | 变换方向、warp顺序、插值、align_corners、支持、padding、搜索器与初值 |
| 评估 | native坐标误差、阈值来源、failure状态、固定模型与固定抽样 |

映射文档必须附真实文件路径、提交和相关函数名称；命令日志与config/hash保存于V10输出目录。T00预检只能生成inventory，不能自动证明映射正确。

## 推荐接入边界

1. 从宿主得到每个场景的正对应稠密descriptor；独立处理各场景，不跨场景拼成乘积。
2. 在descriptor后加两侧可学习projection+softmax，得到`A:[n,r]`、`B:[n,s]`。实际后端API以`core/README_math_CN.md`为准。
3. 用普通优化实现MF-perm1与M0；两者除Q求值方式外共享同一模块/初始化/训练器。
4. 再接M1：本批特征固定内层求W，原tensor图保留供同目标外层反传。不要每次内迭代重跑encoder。
5. 测试接入口只计算固定W的能量，不能调用head solver。测试时使用eval模式并冻结模型参数，但保留对几何变换参数的梯度。

M1的checkpoint按主文规定成对保存encoder与最后训练batch求得的W，不额外执行最终FIT refit。gap只标记inner更新前的特征/数据scope，不能作为保存后模型的全FIT证书。

共同能量表的warp顺序与CoMIR原生流程分开。支持/可见域先继承真实宿主；任何全候选域交集等新支持规则都要对所有方法同步版本化，不能静默改变旧benchmark。

`torch.no_grad()`不能包住整个配准能量：虽然模型权重固定，warp参数仍需梯度。可将参数`requires_grad_(False)`，对几何参数保留autograd。

## 接入验收顺序

- 同一个真实小batch导出A/B/W后，独立dense与矩比较目标和全部梯度。
- 检查场景权重和Q模式；用不同场景n的fixture排除误用按位置加权。
- 在真实warp链上检查正项和负项的几何梯度，测试至少两种非零变换。
- 只进行一个更新时，MF-perm1/M0使用相同正对应和scene顺序；确认W投影、输入特征非恒定、encoder参数实际有梯度和变化。
- 固定W测试时，在不同几何候选前后比较模型state/hash，证明没有在线重拟合。
- 恢复checkpoint后重复同一能量/变换；不通过读取评价真值选择checkpoint。

包内dense/矩数学检查只覆盖数值特征。上述图像、warp、checkpoint验收必须在真实宿主上执行，不能拿本地NumPy结果替代。

## blocked dense与成本

所有K、N与采样规则保持相同。若用checkpoint降低score图内存，要将反向重计算算入时间。若实现仅把for循环分块但仍保存每块的score反向图，表中注明graph retained；这不是证明矩方法胜过最佳分块实现的充分比较。

## 与旧工作的共存

在隔离的V10代码/输出路径实现；不要覆盖旧V9 manifest、checkpoint、实验日志或取消旧任务。当前server已有授权范围内的只读检查、实现和小验证可以继续。未知的大规模总预算先实测与登记，不把配置中的所有设计单元自动提交成网格。
