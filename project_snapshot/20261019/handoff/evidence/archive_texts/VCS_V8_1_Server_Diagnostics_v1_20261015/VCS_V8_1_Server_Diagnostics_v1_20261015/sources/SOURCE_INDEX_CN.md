# 来源与核对范围

核对日：2026-10-15。源码事实、已提交结果、本包推导和待检验假设分开记录。仓库链接均固定到提交；Git blob SHA 不是内容的SHA-256。

## [R1] V8-03 冻结方案

[原始来源](https://github.com/W-Yinghao/VCS_QMI/blob/a87b2d53380783db0932b144dddcd368ac4a7b2d/reports/V8_UTILITY/V8_03_PROTOCOL_FROZEN.yaml)

已读取全文；原训练、模型选择、事件和指标的依据

## [R2] V8-04 正式结果及已附更正

[原始来源](https://github.com/W-Yinghao/VCS_QMI/blob/a87b2d53380783db0932b144dddcd368ac4a7b2d/reports/V8_UTILITY/V8_04_EVAL_REPORT_20261015.md)

已读取全文；数字来自报告，未在本地重新计算自然数据

## [R3] v8_train.py 训练循环与导出

[原始来源](https://github.com/W-Yinghao/VCS_QMI/blob/a87b2d53380783db0932b144dddcd368ac4a7b2d/scripts/v8/v8_train.py)

重点核对185–278行；训练先于epoch末CAL和best.pt保存；实际作业快照待服务器核对

## [R4] v8_core.py 损失及分组

[原始来源](https://github.com/W-Yinghao/VCS_QMI/blob/a87b2d53380783db0932b144dddcd368ac4a7b2d/scripts/v8/v8_core.py)

已读取；八种损失、五档概率、排序分组、批次归一化

## [R5] v8_labels.py 原始投票接口

[原始来源](https://github.com/W-Yinghao/VCS_QMI/blob/a87b2d53380783db0932b144dddcd368ac4a7b2d/scripts/v8/v8_labels.py)

已读取；FIT/CAL/DEV、overall_votes、prompt→image自然权重

## [R6] V8-05 确认集冻结协议

[原始来源](https://github.com/W-Yinghao/VCS_QMI/blob/a87b2d53380783db0932b144dddcd368ac4a7b2d/reports/V8_UTILITY/V8_05_CONFIRMATION_PROTOCOL_FROZEN_20261015.yaml)

已读取主要协议、TIFA160与执行记录；850提示、5100图像；不等同Rank的9图池

## [R7] V8-05 提交记录

[原始来源](https://github.com/W-Yinghao/VCS_QMI/commit/a87b2d53380783db0932b144dddcd368ac4a7b2d)

1042053已提交；本次未读取到确认结果报告，非实时队列

## [R8] 负责人现有数据使用决定

[原始来源](https://github.com/W-Yinghao/VCS_QMI/commit/55d47c9fe73375230c86a7ec1b2e575bdaeb19d5)

沿用已有数据可用于项目的负责人决定；不重新设置许可讨论为阻塞项；不扩展到公开再分发

## [T1] 合作者 Variational CS-QMI Research Plan，September 2026

来源：本次对话提供的研究材料；未重新发布全文。

第4–7节：S/J、精确gap、分类解释；原文件已在对话完整提供，本包不附全文或改变公式

## [T2] 2026-10-10 综合稿，附录4勘误

来源：本次对话提供的研究材料；未重新发布全文。

优先采用附录4，不采用正文已撤回的数学论断；正文为历史状态，不覆盖10月15日结果

## [T3] 2026-09-28 补充意见 §3.7

来源：本次对话提供的研究材料；未重新发布全文。

保留MINE/NWJ/InfoNCE/SMILE等外部横向比较的组织原则；不把其中后被更正的专属性质照单接受

## [E1] SciPy rankdata 官方文档

[原始来源](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rankdata.html)

平均名次处理并列；本包独立实现以SciPy作测试参照，不要求升级服务器版本

## [E2] PyTorch Autograd mechanics 官方文档

[原始来源](https://docs.pytorch.org/docs/stable/notes/autograd.html)

eval模式和梯度控制是两回事；仅支持实现说明，不支持任何性能因果主张

## [E3] Reid & Williamson, Convexity of Proper Composite Binary Losses, AISTATS 2010

[原始来源](https://proceedings.mlr.press/v9/reid10a.html)

正规复合损失与link；仅核对出版页和摘要，不宣称全文逐式复现

## [R9] V8-05确认结果：交付前新提交

[提交与报告](https://github.com/W-Yinghao/VCS_QMI/commit/580781f47cbe2249765bc36af05221f307a2b42c)

已读取汇总；未读取原始确认标签。详情见交付增量。
