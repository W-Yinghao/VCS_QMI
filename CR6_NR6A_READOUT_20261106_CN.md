# CR6 与 NR6-A 只读读出补充（2026-11-06 第五十一版）

接在第五十版（`NR6A_RESULTS_20261105_CN.md`）之后。只发布数字表与代码，不附解读。用户：「build并push」。NR6-B（10/12）、NR6A_P1105 / NR6B_P1105 仍在运行，本版不含。

## CR6：来源层复核 × 固定关系图的学生范围（`handoff/cr6_20261105/`，`RESULTS_CR6.md` 与 CSV）

- 包 `VCS_CR5_Review_CR6_Server_20261105`（协议 CR6_SOURCE_REPLICATE_AND_STUDENT_SCOPE_V1，证据 05754e6a）；用户「准备CR6并提交」。冻结协议 `PUBLIC_PROTOCOL_FROZEN_CR6.json`。
- A1（12）：Flickr60 × {J, BCE} × 6 个 CR1 两折排除专家，新 source seed 2002（只改初始化与抽样 RNG，fold / CAL 池 / 路由 / 预算不变），8000 步；同规则重拟合校准、CAP 与 CR2 KEEP_TOP1，得新图 J_KEEP_TOP1 / BCE_KEEP_TOP1 / J_MASS_OBS（src2002）。
- A2（6）：三张新图 × student 1002 / 1003，冻结特征 COMPLETE_PC，8000 步。
- B（12）：Flickr60 与 CC152K_RECOVERED_20261104 × 6 张旧 CR4 图 × student 1002，TAIL_LORA8 学生（原 SigLIP2 + 零 LoRA，两塔最后两层 q/v，r8 / alpha8）+ PC，8000 步；FROZEN 参照为同 seed 的 CR5 COMPLETE_PC，组内 250 步回放逐位一致才复用，否则同组重训小 G。
- 不新增训练的补充：CR4 / CR5 / CR5_P1105 PC 学生的六 Recall、按图像分组重采样的配对差、新旧 source 诊断、成本。
- 验收与 GPU：按 GPU 型号验收的登记 `PUBLIC_ADDENDUM_CR6_ACCEPTANCE_BY_GPU_TYPE.json`（用户「CR6一定要在L40S上做吗」「我觉得你这些检查是不是有点太多余了？真的有用吗」后精简为单个 RTX 验收）；RTX 验收 1090989 全部通过；V100 验收 1090990 的估计器克隆检查因验收脚本自身错误（250 步回放的余弦日程长度设为 250 而非 8000）失败，记录保留，修正后的 CPU 检查一致。实际 GPU：A1 与 B 为 RTX PRO 6000，A2 为 V100S。全 DEV、基础模型 Top100、全部 query；训练与图构建不读真值；不读 TEST。
- 代码：`code/site/`、`code/smoke_cpu.py`、`diag/`、`sbatch/`。设计包本身（`package/`）与包参考代码不发布；单元配置不发布。

## NR6-A 只读读出补充（`handoff/nr6a_20261105/readout_p1106/`，`RESULTS_NR6A_READOUT.md` 与 CSV）

- 包 `VCS_NR6A_Review_Action_20261106`（证据 ecbb9f5e）；用户「按这个分析提交所有需要的实验」。0 个新训练。
- 18 个 TAIL stage-4000 终点（Flickr60 / CC seed 2002 来自 NR5-A，COCO60 seed 2002 与全部 seed 3003 来自 NR6-A）× {E1G0（终点 encoder + 条件共同前缀 head G0）, E0G1（原 SigLIP2 + 终点 head）} = 36 个只读读出，同 NR5 D0 定义；同条件 G0 同一哈希；E1G1 复核 18/18。另含 NR6-A 阶段曲线、成本、按 DEV 图像分组的配对 bootstrap 与 NR6-B 登记说明。CPU 单作业，float64；不读 TEST。
- 代码：`code/nr6a/readout_p1106.py`、`sbatch/cpu_readout.sbatch`。

## 范围说明

配对差不给显著性结论；分组区间只描述固定评价集与预测。Flickr、COCO、CC152K_RECOVERED 分开报告。DEV 在开发中被反复使用。
