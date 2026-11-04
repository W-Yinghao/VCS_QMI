# NR1 首轮：冻结 SigLIP2 上的直接图文检索（2026-11-04 第四十版）

接在第三十九版（`R43_R47_AUDIT_RESULTS_20261104_CN.md`）之后。只发布数字表与代码，不附解读。R49、布局无关 TEST 读数与 R50 留待下一版一起发布。

## NR1（`handoff/nr1_20261104/`，`RESULTS_NR1.md`、`RESULTS_NR1_*.csv`）

- 用户授权：「这个是新的路线，准备相关的实验，立即提交」。独立路线，不继承旧 IN²R encoder/cache/中心化/候选/GMM/peer/null；协议 `NR1_SAMPLEWISE_L2_V1`。
- 基础模型 `google/siglip2-base-patch16-224`（FixRes，revision 锁定，权重冻结）；图像与文本各自独立编码、逐样本 L2，无跨样本中心化。
- 矩阵：G（全局残差评分）/ GL（全局 + patch–token 交互）× J / balanced BCE / InfoNCE × Flickr 0% / 60% / 80%，8000 步，训练 seed 1001；P = 观测联合、Q = 边缘乘积；评分从基础 cosine 的正仿射开始、残差零输出；step 0 与基础模型逐 query 排名一致。
- 数据：NCR f30k_precomp 记录顺序 → Karpathy 标注 → 原始图片的显式映射（150,350 条 caption 核对一致；FIT 29,000 图 / 145,000 caption，DEV 1,014 / 5,070）；60% / 80% 复用既有锁定噪声图。CC152K：服务器只有区域特征、无原图与 URL → 6 项 BLOCKED_DATA（`PUBLIC_BLOCKED_DATA_CC152K.json`）。
- 评价：只用 DEV；基础模型 top-100 重排，全部 query，六项 Recall、rSum、救回/丢失、覆盖；分数先写出并哈希，再由独立评价进程读标注。不读 TEST。
- 验收：真实小批次 31 项检查通过（三次失败尝试记录保留）；提取器关闭 TF32 并固定前向 batch 形状、评价用 fp64 打分（`PUBLIC_TOLERANCE_AMENDMENT_A1.json`，容差未放宽）。
- 代码：站点脚本 `code/site/`、`sbatch/`；设计包本身的参考实现（`src/`、`scripts/`、`configs/`、`tests/`、`docs/`）与相对包的完整 diff 不发布；锁定配置以公开摘录形式发布。
- 18 项全部一次完成；结果表含基础模型（未训练 cosine）与每项 step 8000 读数及曲线。

## NR1B（进行中，本版只含冻结协议）

- 主会话按常设授权（「你之后如果发现GPU有空的话，就沿着NR1路线设计和提交」）设计：Flickr00 六配方 × 训练 seed 1002/1003（12 项）；GL 局部置零消融、实例错误四类、训练信号诊断（只读）。冻结协议 `PUBLIC_nr1b_PROTOCOL_FROZEN_NR1B.json`；结果待下一版。

## 范围说明

单训练 seed；配对差不给显著性结论。DEV 在开发中被反复使用；预训练模型可能见过公开数据，不作独立无污染确认。与旧 IN²R 路线的分数属不同基础资源与协议，不直接比较。设计包与包内参考代码不发布。
