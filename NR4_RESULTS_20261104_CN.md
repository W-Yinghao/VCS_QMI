# NR4：同参数预算的输出适配 × 编码器尾部适配（2026-11-04 第四十四版）

接在第四十三版（`CR1_RESULTS_20261104_CN.md`）之后。只发布数字表与代码，不附解读。

## NR4（`handoff/nr4_20261104/`，`RESULTS_NR4.md` 与 `RESULTS_NR4_*.csv`）

- 用户：「准备这个包的实验」「并提交」（`VCS_NR3_Review_NR4_EncoderAdapt_20261104`，证据快照 11b5af74）。
- 条件：Flickr00、Flickr60、COCO60、CC152K_RECOVERED_20261104 × J / BCE / NCE；seed 1001。各条件共用 NR3-B 分支前的 folded NCE 前缀（固定 8000 步 HREF-NCE 父 checkpoint 经 FIT-only 正仿射校准折入 G），追加 4000 步（总 12000 步），NR3-B 配方（新 AdamW，lr 1e−5→1e−6 cosine，wd 1e−4，clip 2，sampler seed 101001）。
- 三种 scope：FROZEN（只更新 G；12 项为 NR3-B 终点，经新 trainer 全程重跑 4000 步逐位一致后复用，`PUBLIC_PREFIX_AND_REUSE_RECEIPT_NR4.json`）；POST32（缓存全局向量上的图、文 rank-32 无 bias 残差，98,304 个新增参数）；TAIL_LORA8（两塔最后 2 个 block 的 q_proj / v_proj LoRA，r = 8、alpha = 8，98,304 个新增参数，其余预训练权重冻结、梯度经固定 pooler 回传）。新训练 24 项。
- TAIL 从原图 / 原文读取：原图内容哈希与 NR1 / NR2C / CC 恢复快照登记一致；前 10 层固定隐藏态缓存与原图路径逐位一致；每步按当前权重执行最后两层与 pooler。候选与训练 proposal 始终由冻结的原 SigLIP2 定义，部署在相同 base DEV Top100 内按当前 s 排序。
- 验收：包 CPU 测试 13/13；真实双塔：模块清单、零适配 t0 与前缀逐位一致、训练后 base 权重不变且特征改变、ID 改名 / 重排 / 分块不变性（仅字节相同 caption 的精确并列按 ID 打破）、两遍 embedding-gradient 重放与直接反向一致（相对差约 2e−7）、断点恢复逐位一致。冻结协议 `PUBLIC_PROTOCOL_FROZEN_NR4.json`。
- 读数：主终点 stage 4000；六 Recall、rSum、相对基础模型 / FROZEN / POST 之差、救回 / 丢失、Top100 覆盖；次读数为当前图文 cosine 的全库检索。资源台账 `PUBLIC_RESOURCE_LEDGER_NR4.json`（隐藏态缓存构建、训练同步计时、encoder item 前向、pair 评分、峰值显存、硬件）。失败记录 `checks/FAILURES_NR4.md`。不读 TEST，训练不读真值。
- 代码：`code/nr4/`、`sbatch/`。设计包本身（`package/`）、包参考实现的字节副本（`code/nr4/__init__.py`、`ref_adapters.py`、`ref_gradient_replay.py`）、改写自未发布训练器的 `code/nr4/train.py` 及其 diff 不发布。

## 范围说明

单 seed；配对差不给显著性结论。总预算 12000 步，含共同 NCE 前缀。Flickr、COCO、CC152K_RECOVERED 分开报告。DEV 在开发中被反复使用。
