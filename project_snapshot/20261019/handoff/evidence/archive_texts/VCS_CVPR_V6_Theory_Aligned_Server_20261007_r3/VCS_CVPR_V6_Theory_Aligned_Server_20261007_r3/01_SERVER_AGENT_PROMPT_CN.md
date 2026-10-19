# 给 server agent：V6 / 理论对齐 r3 增量指令

请按本包 `CVPR-MV6-THEORY-20261007-r3` 接续 VCS_QMI。论文仍是 V6 Revised，不回退 Git、不清空旧实验，不以 r3 覆盖正在编辑的论文。先读 `docs/02`、`docs/04`、`docs/05` 与 `docs/06`。

## 本轮研究重点

合作者原计划的重点是：对明确的 mixture-reference CS 依赖，进行 sample-only neural estimation，验证后验误差、估值偏差与波动、梯度行为、强依赖分辨能力和成本。当前属性检验功效与不同 encoder 的颜色保留，是应用结果，不替代这些验证。

原 J、equal mixture、bounded critic、训练/测量角色和现有完整 SSL 均保留。不要把工作偷偷改成一种新的显著性算法，也不要把依赖增量单独升级为必须成功的新中心。Logistic-fit + common-J 与 direct-VCS 是两种拟合路线；直接 logistic SSL 仍是对照，不改名为 VCS。

## 当前状态与接续

本包核查至 `a6fb342ccbe464f21bbf68d80a1dd361469d33ea`（2026-10-07 20:32:16 UTC）。先与当前 checkout 做增量盘点。P130 R50、P138 K16、P145 C100 正对污染已进入确认；P147/P148尚有未齐行。P139在快照中低优先挂起，完整ImageNet预算待负责人。已有授权与 frozen 日程照旧。

P138 已有配对方差与 encoder 梯度只读实验；P86/P108 已有 oracle、稳定性、旋转和 J/plug-in 对照；P101 已有噪声分辨率负结果；P142/P143 条件审计与外部 encoder 扩展已完成；P144真实压缩不具定量分辨。以上都先复用，不改名字重跑。

## 顺序

1. **R3-00 / E0**：建立“理论主张—已有证据—真正缺口”表，恢复 P86/P108/P138 的读数、gradient、bootstrap和时钟字段；做一版直接对应原计划的精度/稳定性/分辨率图。保留修正后的 SMILE、in-batch MINE、NWJ、InfoNCE、同目标核与 logistic。
2. **R3-E1**：只有存量不能恢复时，才对预定三个 Gaussian 依赖水平补缺。记录同目标 posterior MSE、J误差、plug-in、评价/重拟合/配对抽样三类方差，原生目标各自的误差、尾部、梯度和成本。不能用 bounded S 的小标准差自动宣称优于 MI。
3. **R3-V0/V1**：先复用 P122 的冻结两视图测量；缺口才建共享 fixture。测量同一视觉关系在可逆坐标变化与规定共同随机通道下的重复性。重拟合critic、保留T=0与置换阴性控制，不把高预算模型当真值。噪声只用于测量压力检查，不加入SSL训练，也不承诺提高checkpoint分辨率。
4. **R3-C0**：P144仅补一个同数据、同目标的有限诊断；对齐P143/P144的输入、函数类和读数，补逐样本结果与正确bootstrap。原5次混合重复sd不能改名为bootstrap置信区间。不无限扩critic或强制增量单调。
5. **R3-A0/W0**：P142/P143作为已完成辅助案例重新归位；最终VCS只占一个encoder行。稿件先形成建议patch/claim ledger，由负责人决定并入V6，不自动删表或覆盖源文件。

## 必守的比较约定

同目标数值误差只比同一个S；经典CS按自身参考测度，MI按自身目标。置换功效、准确率、rank不是posterior MSE。任意严格单调统计变换可以保持置换检验不变却破坏数值标定，因此保留检测表，但不让它证明估值准确。

真实图像无oracle，数值真值字段必须null。独立pair pools与共享U-statistic分开记不确定性；K个负对不是K个独立图像。按原checkpoint映射、原底层图像分组；不跨augmentation law、encoder/head、属性强度拼接读数。

## 权限与产物

本包是准备材料，不是自动提交命令。小型只读恢复和协议整理可先做；已有授权覆盖的新单元按原权限执行，新增长训练/预算另确认。近期quota事故后，先确认scratch/工作目录与checkpoint存量；不能为恢复被清理的epoch100权重而重训800epoch。

每单元交付结构化结果、配置/来源hash、独立单位、缺失与失败、效应和不确定性、时钟范围、对应论文段落。较强baseline、零信号、反向结果同样是完整交付。首批返回 `templates/DELTA_INTAKE_TEMPLATE.md` 和任务映射，不新增未协调的P编号。
