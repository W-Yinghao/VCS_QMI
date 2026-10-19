# 已读源码：直接影响实验设计的六项发现

## C1. 原生FGA不是Logistic
`fga_blip2.py: forward(match_head='itm')`的活跃分支计算绝对误差。论文也把L_os、L_es、L_mask写成L1。V7必须保留N0/M0-L1，不能把“native FGA”与“matched BCE”当成同一行。[G02,P01]

## C2. variance weighting存在paper–code差异
论文有exp(σ(prompt))权重，下载到的活跃源码没有乘；读入了var，但含exp(var)的代码被注释且在此前return后。N0-Code与N0-Paper-var应独立命名。公开发布权重究竟采用哪条路径，本轮仅凭代码不能确认；若需要复现论文最优表，先查作者说明/具体checkpoint元数据，不擅自补乘再叫原生。[G02,G06]

## C3. 元素标签不是普通固定维向量
原始JSON每条样本的element_score是不同键的投票字典；自动viewer转换可能失败。`process_train.py`将投票均值化，`element2mask.py`把元素映射到token并对重叠位置平均，某元素找不到会过滤样本。不能先运行预处理再忘掉丢了多少数据。[D01,G05,G06]

## C4. 截断和index0会改变概率含义
dataset读取时截成32、padding为0，index0被置为总体评分归一化值。V7元素风险不得把该位当成人工二元元素；M组保留显式语义跨度和missing标记。N0原生复现照旧但说明覆盖率，不把原生32token结果与M组完整输入差异都归给损失。[G04]

## C5. 原生N=3不是全组合
GenAI脚本把9候选按3个连续分组，每组argmax再平均；N=9整组选一张。其并列score遵守argmax首个索引。V7官方复现保持候选顺序；全84个子集和并列期望作为单独敏感性，不能覆盖主值。[G07]

## C6. 模型/代码版本不是可交换的
t2v_metrics README明确原论文应使用legacy v3.0；现代模型环境另行维护。不能升级依赖后仍称完全复现原论文。各系统的输入模板、视觉分辨率、yes/no读取、token归一化和可训练模块均入manifest。[G08]

## 接入回复格式
每项：`fact_confirmed / fact_modified / not_applicable`；附本地commit、函数、实际有效配置、20条以内示例与修正决定。确认源码行为不等于它解释了全部性能；不预写“修了这一项VCS必胜”。
