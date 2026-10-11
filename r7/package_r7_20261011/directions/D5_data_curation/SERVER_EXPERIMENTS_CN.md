# D5 服务器实验方案｜图文数据选择与再描述

**性质：候选方案，尚无本方案自然数据结果。** 先读本目录综述和共同理论/评价规范。数据条目：DS26, DS27, DS28, DS12, DS21；全部定义见 `../../datasets/DATASET_ATLAS_CN.md`。

## A. 两个可分开的任务

**Selection主任务：**从固定pool选K个图文样本，按同一个标准CLIP/SigLIP训练recipe训练学生，比较下游效果。学生本身用标准任务损失，以免VCS筛选与VCS训练两个因素混在一起。

**Recaption条件扩展：**对同一组选中图片，在原始caption、已有recap、按规则选择/混合caption之间比较。先固定图像，再比较文字，不能同时改样本数量/长度/教师。

## B. 统计对象与原型算法

目标reference A是从FIT允许的目标关系样本中得到，不能由官方测试标签构造。来源pool B和选中子集B_w在共同joint representation z=(image,text,relation features)上比较S(P_A,P_Bw)。主critic使用VCS，matchedLogistic读同J，核相对回归/groupedMMD为对照。

可行的第一实现有两种但首轮只选一种：
1. **ratio reweighting**：拟合target-vs-source后验，转换ratio，按CAL决定clip并归一化采样；记录ESS/maxweight与source coverage。
2. **固定字典子集选择**：在预定语义/质量簇上求权重，反复用独立样本估计分布差异；终点仍是学生质量。

二者都是已有分布匹配/重加权思路的CS实例，不自动是新算法。若要形成贡献，需要有限target样本下更可靠的覆盖/成本表现。高T筛选“最好配对”另作质量baseline，不与target-vs-source ratio混同。

## C. 数据与训练

固定共同可访问pool及安全/许可过滤，记录URL失败、重复图、哈希和下载时间。所有策略面对相同pool。保留比例首轮固定一个实用K；第二比例仅在主机制成立后追加。DataComp正式行采用官方small architecture/exposure，内部scale另名。

同样的unique images、examples seen、augmentation、batch和优化steps；若算法重复采样更多，要记录实际重复分布和有效数据量。目标reference不得用下游test label挑选。目标组外性能也要看，避免过拟合reference。

## D. 主baseline和运行单位

原池random、CLIP相似度过滤、Sieve、可运行的DFN/SDM、核目标匹配、Logistic ratio、VCS selection。先用公开/缓存分数尽量复用生成成本，但所有额外教师标注单列。

资源充足前的最小正式阶段为5策略：random、CLIP、一个已验证强过滤器、matchedLogistic、VCS，各3个学生seed=15次完整学生训练。核和SDM可先在同一子集selection层完成；投稿主张优于这些方法时必须加入其学生训练，而不是只有过滤分数。

Recaption采用2×2固定设计：图像选择baseline/VCS × caption原始/相同recap教师，学生recipe相同；属于追加资源项目，不能在第一轮15次之外自动启动。

## E. 指标与预期

主终点DataComp官方38任务summary；另列ImageNet、retrieval、target-type任务和非目标域。筛选阶段报告分布读数/coverage/ESS，不作为学生质量的替代。总成本包括feature extraction、teacher caption、filter fit、selection及student训练，分别报告一次成本和摊销假设。

H5：相同保留数和训练exposure，VCS选择可能在目标relation稀缺时更有效覆盖，并保持总体迁移。不是理论保证，也不预设抗噪/尾部优势。识别原因需要匹配Logistic、核和强过滤器。

## F. 失败处理

若权重饱和/ESS塌陷，先查support与clip，而不是把clip后的分数称无偏。若只改善target小集合、整体下降，报告取舍并考虑混合uniform quota；quota为新策略，要给对照相同机会。若teacher成本高于学生节省，改用已有recap或冻结策略，不宣称高效。若没有足够数据/训练预算，D5保持综述与方案状态，不拿10k图片toy替代完整DataComp证据。
