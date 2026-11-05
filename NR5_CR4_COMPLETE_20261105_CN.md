# NR5 完整与 CR4 训练后审计（2026-11-05 第四十八版）

接在第四十七版（`CR4_NR5_RESULTS_20261105_CN.md`）之后。只发布数字表与代码，不附解读。用户：「push到github上吧」。

## NR5 完整（30/30；`handoff/nr5_20261104/`，`RESULTS_NR5.md` 与 CSV）

- 第四十七版时 CC152K_RECOVERED 的 B 组 4 个单元（COS NCE、G_ZERO J / BCE / NCE）尚未完成；本版全部 30 个单元终点与曲线齐全（评价缺失 0）。
- 设计不变：D0 为 NR4 12 个 TAIL 单元固定 stage 4000 的只读分解（旧 / 新 head × 旧 / 新 encoder，以及全库 current-score 读数）；A 组为 Flickr60、CC152K_RECOVERED × J/BCE/NCE × {FROZEN, TAIL_LORA8} 的第二 seed（2002）；B 组为 Flickr00 / Flickr60 / CC152K_RECOVERED × {COS, G_ZERO} × J/BCE/NCE，自原始 SigLIP2 + 零 LoRA，P = 观测联合、Q = 乘积边缘，8000 步。修订 NR5_A1 下每个比较组同一 GPU 型号。不读 TEST。

## CR4 训练后审计（`handoff/cr4_20261104/`）

- 训练后只读审计完成：同 checkpoint、同 query、同 COMPLETE 候选上的 CE 分解（分离项、熵、正例内 KL）、PC 梯度方向、各分量梯度范数与夹角，覆盖训练前（初始化、CR3 COMPLETE-CE step 250 / 8000）与训练后（CR4 PC step 250 / 8000）。`PUBLIC_GRAD_COMPONENT_AUDIT_before.json`、`PUBLIC_GRAD_COMPONENT_AUDIT_after.json`；`RESULTS_CR4.md` 已按完整审计重新生成。审计不读真值。

## 范围说明

单 seed（NR5-A 为第二 seed）；配对差不给显著性结论。Flickr、COCO、CC152K_RECOVERED 分开报告。DEV 在开发中被反复使用。设计包与包内参考代码不发布。
