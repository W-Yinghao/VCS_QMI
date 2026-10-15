# 07 · 数据、基线和评价接口

## 1. 数据矩阵
| 数据 | 首轮用途 | 独立单位/划分 | 正式评价/限制 |
|---|---|---|---|
| COCO P49缓存 | N1探索与代码接续 | 已有按image隔离FIT/CAL/EVAL | 已反复观察，不能当新确认；不是原始训练模型 |
| Flickr30K / NCR数据版本 | N2第一块完整检索 | image ID，全部5 captions同角色 | 官方1K test gallery；全部数据/词表/噪声hash |
| COCO / NCR数据版本 | N2第二块完整检索 | image ID；2014/2017映射要核查 | 5K主表、5×1K另列；不把其混同 |
| CC152K | 自然噪声条件扩展 | 公开train/val/test，URL/图像实存核验 | 无天然噪声比例真值；不能按名给出rho accuracy |
| BF–SHG | REG首批 | 物理切片/样本先分，再切patch | 共用R6真值/单位；被看过的DEV只作开发 |
| EPFL RGB–NIR | REG次轮 | 场景原图/窗口映射 | 许可待核；旧33排除及2px分辨记录保留 |
| gRefCOCO | 开放对应备选 | 原image划分、zero/single/multi | 只准备评测，首批不训练 |

## 2. 从文献到运行的要求
NCR官方README [L02](https://github.com/XLearning-SCU/2021-NeurIPS-NCR) 指明COCO `split=testall`，`fold5=false`为5K、`true`为五个1K。L2RM [L04](https://github.com/hhc1997/L2RM) 指向同一NCR特征/词表与0.2/0.4/0.6/0.8错配索引。**这两个读取事实并不意味着本地文件已存在。**

先生成`DATA_AUDIT.csv`：source URL、local path、sha256、原许可/用途、image/caption数量、去重数、各role、pretraining exposure、坏文件处理。官方split数目以下载实物核验，不凭论文标题猜。

官方特征版和现代CLIP版是不同系统条件。只能在相同条件表里比较直接胜负；额外宿主/外部预训练/ensemble单独列。

## 3. Benchmark三层
**估计层：**有限真值、已定义自然混合；VCS、rawscore、BBSE、MLE/EM、KDE、RuLSIF、Nyström。MPE单成分方法标信息条件。

**应用层：**NCR、L2RM必做；ESC/ASL/ReCon逐项准备；无代码时做源码/论文复现报告，不编结果。混合比例及独立可信集都给所有方法相同使用机会。

**归因层：**matched BCE、RankMass、真rho权重、真注入标签权重（只作带额外信息的diagnostic）。真标签训练不是可部署基线，也不一定是理论性能上界。

## 4. 当前代码可用性与风险
- NCR、L2RM：本轮查到公开实现及README；完整运行/数据下载尚未做。
- ReCon：官方仓库README仍称“代码即将发布”。不能在任务清单中写“直接运行官方实现”。
- ESC、ASL：论文入口确认，本轮未核实执行仓库；负责agent应先获取正文与补充实现说明。
- RRSITR、ConeSep等新领域工作：相关性已经检索，但任务不同，不为增加方法名强塞同一表。
- CoMIR：已有R6接入；原生注册与feature+shared-optimizer必须分开。
- MIND：原论文主要3D，若用2D适配，说明descriptor/offset/优化器，不能直接引用原论文数值。

## 5. 防止“同一来源导致虚假独立”
图片重复、COCO跨版本重复、同caption多个形式、相同切片多个patch按源ID聚合。预训练见过公开数据与本项目标签泄漏是不同问题：公开已知训练来源披露，未知不等于无暴露。

参考集与被量化批次共享同一图像，会低估rho的不确定性。任何跨角色配对都同时检查image-id和caption-donor-id。多次批次抽样不创造新的训练模型或新数据集。
