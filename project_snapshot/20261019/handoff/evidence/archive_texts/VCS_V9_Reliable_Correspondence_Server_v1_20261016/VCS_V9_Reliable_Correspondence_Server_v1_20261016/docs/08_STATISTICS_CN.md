# 08 · 指标、统计与结果解释

## 1. N1主指标
rho MAE/RMSE，95%区间coverage/平均宽度/full-set/empty-set比例，成分后验Brier与log loss；配对机制分类AUROC仅补充。原生J单列，先解释对应P/Q。

比较至少包含各方法原始与校准后概率；ECE只作诊断，不作为唯一证据。相同排序的两种概率可以有不同Brier，但不能在固定保留率的rank-only表中伪造差异。

区间反演条件见03。普通ratio bootstrap可能在弱分离下不可靠；保存raw样本均值和间隔，不只存最后裁剪的rho。空区间比例是压力测试输出的一部分，不删除。

## 2. 检索
I2T R@K：top-K中有任一同image的官方caption；T2I R@K：真实image进入top-K。K={1,5,10}；rSum六项相加，mR=rSum/6。打分并列按固定ID排序；额外报告随机并列敏感性。所有baseline用同gallery和ground-truth映射。

主系统比较在rho=.4：VCS vs NCR+A/L2RM+A × 两数据，共4项，Holm校正。传统路线VCS vs RuLSIF/Nystrom为独立4项家族；VCS vs BCE为2项归因家族。报告原始差值、每seed数值和置信区间，不只写“通过/未通过”。

测试图像为配对bootstrap单位，所有caption跟随；gallery固定，重新采样query图像并不构成新gallery。训练种子的离散程度另列，不能将数万caption当作训练重复。3seed时seed CI不稳定，应给全部seed而非仅小p值。

## 3. REG
以物理图像/切片为bootstrap单位，内部patch与10个扰动全部同组；配对同一随机变换。误差在像素坐标中精确反算；native与resize尺度不同必须转换。失败保留并报告，不能只算成功项median掩盖尾部。

## 4. 无数值预言，区分成功的层级
- 统计优：概率/比例估计更准，只在对应样本条件成立。
- 决策优：数值进入权重/注册后，最终任务优于外部方法。
- 方法归因：匹配critic/teacher/信息/成本后还能解释来源。

统计优而任务平局可保留为估计贡献；系统优而BCE相近不能说VCS独占；单一核进入同水平不等于研究无效。不存在本包预先给定的“必须赢X点否则CVPR关闭”。

## 5. 额外监督与成本
A的确认/清理本来就需要人工或可信标注，报告anchor_fraction及每方法可见信息。计算包括teacher、模型提取、核调参、critic训练、rho重估、student全部训练及失败。相同准确率下更便宜的声明需要质量匹配规则在CAL确定。

REG与VLM不同机器的墙钟不能直接相除得加速倍数。采用固定机型/负载的profiling；资源不可用时报告次数与待补计时，不编GPU小时。
