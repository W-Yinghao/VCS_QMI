# V9 N2F 收尾附录：分段硬件、完整轨迹、崩塌披露、后期敏感性与成本

本附录由 `v9_n2m/n2f_closeout/addendum.py`（sha256 `4032633022d23fdbabaae835aa7d640b98f75cd6fcac188a453c324dc59f1a1c`）只读记录生成；文中数字均由该脚本计算，引文逐字取自所注文件与行号。它补充收尾记录 `/home/infres/yinwang/CS_QMI/handoff/v9_overlap_v10_20261018/CLOSEOUT.json`（status PASS，2026-10-19T14:14:57Z，CPU作业 1046165），不修改冻结收尾包、V9代码或任何实验产物；不训练、不读TEST、不加载或重算模型。收尾的预注册门槛与结论保持不变，本附录不增加新结论。

补充内容对应独立审查 `/home/infres/yinwang/CS_QMI/handoff/review_claude_20261019/REVIEW_CN.md` §2.2 P1-3 与 §2.3 “V9”所列缺口：拆分/续跑单元只记录了终末段硬件、协议要求的完整40 epoch轨迹未输出、崩塌图景不完整、最佳epoch与后期波动、计划与实测成本。

## 0 来源与核验

- 收尾 `CLOSEOUT.json` sha256 `0f5ba97134826d14e2bad05d047f48afeb2f4684d316c6da32319af29072c978`；`V9_CLOSEOUT_CN.md` 的sha256等于其 `report_sha256`。
- 33/33 个单元：`full_result.json` 的sha256等于CLOSEOUT记录；`train_log.json`、`run_manifest.json`、`identity.json` 等于 `full_result.json` 的 `artifact_sha256`；身份、`test_read=false`、40 epoch、最佳DEV（最早并列）、末轮、续跑边界、训练计时、终末硬件、`n_pool` 与严重崩塌标记全部重算并与CLOSEOUT逐值相等。崩塌判定直接调用冻结的 `summarize.py` 中 `collapse()`（sha256与FROZEN.json一致，不调用会写文件的 `summarize()`）。best.pt/last.pt 已由收尾核验，本附录不使用模型二进制。
- `progress.jsonl` 不在收尾哈希链内：sha256固定于生成器，1320 个epoch的DEV rSum与 `train_log.json` 逐值相等。
- 拆分前归档的 `progress.jsonl` 等于 `MIGRATED.json` 记录的哈希，并且逐行是终末 `progress.jsonl` 的前缀；分段边界等于 `train_log.json` 的 `resumed_at`。61 个作业起止边界（控制器StartTime/EndTime、squeue观察的起点、作业trap或控制器记录的结束）与各段epoch时间戳相容（允许 300 秒时钟偏差）；最小余量：段内首个epoch晚于作业起点 760 秒（Unweighted_A/SUM s903，job 1045066），段内末个epoch早于作业结束 1 秒（OracleDrop/SUM s901，job 1044015）。
- 本附录共读取 219 个记录文件；路径、sha256与核验路线列在 `N2F_FINAL_33_UNITS.json` 的 `inputs`。

## 1 最终33单元表

完整列（含六个Recall、mR、各段硬件与job、n_pool、计时与哈希）见 `N2F_FINAL_33_UNITS.csv` / `.json`；下表为摘要。best为预注册终点；末轮=epoch 39；后期均值=epoch 30–39（epoch 30起学习率降为1/10，生成器已逐单元核验）。

| 方法/尺度 | seed | best (epoch) | 末轮 | 30–39均值 | 30–39 max−中位 | mR@best | n_pool | 各段硬件 (job) |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| VCS_Weight/SUM | 901 | 468.32 (39) | 468.32 | 465.16 | 2.69 | 78.05 | 145000 | e0–39 L40S node50 (1043941) |
| VCS_Weight/SUM | 902 | 467.36 (39) | 467.36 | 458.32 | 8.88 | 77.89 | 145000 | e0–39 L40S node50 (1044068) |
| VCS_Weight/SUM | 903 | 473.93 (38) | 472.56 | 470.37 | 3.04 | 78.99 | 145000 | e0–39 L40S node52 (1046654) |
| BCE_Weight/SUM | 901 | 464.20 (38) | 461.58 | 460.67 | 3.12 | 77.37 | 145000 | e0–39 L40S node57 (1043942) |
| BCE_Weight/SUM | 902 | 470.24 (38) | 444.87 | 458.66 | 7.79 | 78.37 | 145000 | e0–17 L40S node52 (1044282)；e18–39 L40S node52 (1046868) |
| BCE_Weight/SUM | 903 | 467.36 (36) | 459.82 | 462.00 | 6.33 | 77.89 | 145000 | e0–32 L40S node50 (1045064)；e33–39 H100 NVL node53 (1046653) |
| RankMass/SUM | 901 | 466.25 (34) | 450.69 | 452.04 | 16.34 | 77.71 | 145000 | e0–39 RTX PRO 6000 node58 (1043972) |
| RankMass/SUM | 902 | 460.28 (36) | 452.31 | 456.12 | 3.44 | 76.71 | 145000 | e0–39 RTX PRO 6000 node59 (1045058) |
| RankMass/SUM | 903 | 471.89 (38) | 455.98 | 458.44 | 14.14 | 78.65 | 145000 | e0–39 RTX PRO 6000 node61 (1045065) |
| Unweighted_A/SUM | 901 | 445.90 (38) | 435.64 | 427.65 | 19.96 | 74.32 | 145000 | e0–39 RTX PRO 6000 node61 (1043985) |
| Unweighted_A/SUM | 902 | 432.52 (36) | 416.59 | 418.15 | 16.06 | 72.09 | 145000 | e0–39 RTX PRO 6000 node58 (1045059) |
| Unweighted_A/SUM | 903 | 446.67 (38) | 443.57 | 432.46 | 11.97 | 74.44 | 145000 | e0–39 RTX PRO 6000 node59 (1045066) |
| RuLSIF_Weight/SUM | 901 | 467.00 (32) | 451.74 | 459.48 | 5.84 | 77.83 | 145000 | e0–39 L40S node50 (1044050) |
| RuLSIF_Weight/SUM | 902 | 469.15 (33) | 456.75 | 458.96 | 10.78 | 78.19 | 145000 | e0–39 L40S node50 (1045062) |
| RuLSIF_Weight/SUM | 903 | 466.55 (39) | 466.55 | 456.39 | 4.58 | 77.76 | 145000 | e0–39 L40S node50 (1045453) |
| OracleMask/SUM | 901 | 478.54 (36) | 478.46 | 476.78 | 1.00 | 79.76 | 145000 | e0–39 RTX PRO 6000 node59 (1044015) |
| OracleMask/SUM | 902 | 478.68 (38) | 476.02 | 476.86 | 1.82 | 79.78 | 145000 | e0–39 RTX PRO 6000 node61 (1045061) |
| OracleMask/SUM | 903 | 480.65 (32) | 478.72 | 479.15 | 1.54 | 80.11 | 145000 | e0–39 RTX PRO 6000 node58 (1045426) |
| OracleDrop/SUM | 901 | 477.04 (33) | 473.43 | 475.42 | 1.34 | 79.51 | 92801 | e0–39 RTX PRO 6000 node59 (1044015) |
| OracleDrop/SUM | 902 | 481.60 (37) | 479.37 | 479.27 | 2.49 | 80.27 | 92804 | e0–39 RTX PRO 6000 node61 (1045061) |
| OracleDrop/SUM | 903 | 477.91 (35) | 476.69 | 476.75 | 1.22 | 79.65 | 92803 | e0–39 RTX PRO 6000 node58 (1045426) |
| VCS_Weight/MEAN | 901 | 467.53 (37) | 465.98 | 464.11 | 3.91 | 77.92 | 145000 | e0–39 L40S node50 (1043941) |
| VCS_Weight/MEAN | 902 | 466.09 (37) | 465.15 | 464.35 | 1.83 | 77.68 | 145000 | e0–39 L40S node50 (1044068) |
| VCS_Weight/MEAN | 903 | 469.78 (38) | 469.07 | 466.59 | 2.13 | 78.30 | 145000 | e0–33 L40S node50 (1045063)；e34–39 L40S node52 (1047108) |
| BCE_Weight/MEAN | 901 | 466.39 (39) | 466.39 | 462.73 | 3.00 | 77.73 | 145000 | e0–39 L40S node57 (1043942) |
| BCE_Weight/MEAN | 902 | 468.15 (37) | 467.77 | 465.54 | 2.61 | 78.02 | 145000 | e0–39 L40S node52 (1044282) |
| BCE_Weight/MEAN | 903 | 463.00 (38) | 462.94 | 460.84 | 1.74 | 77.17 | 145000 | e0–39 H100 NVL node53 (1046655) |
| RankMass/MEAN | 901 | 457.46 (35) | 450.49 | 451.60 | 5.85 | 76.24 | 145000 | e0–39 RTX PRO 6000 node58 (1043972) |
| RankMass/MEAN | 902 | 464.16 (33) | 459.82 | 445.36 | 23.21 | 77.36 | 145000 | e0–39 RTX PRO 6000 node59 (1045058) |
| RankMass/MEAN | 903 | 470.14 (36) | 467.61 | 464.12 | 3.77 | 78.36 | 145000 | e0–39 RTX PRO 6000 node61 (1045065) |
| Unweighted_A/MEAN | 901 | 437.77 (36) | 427.06 | 429.12 | 8.55 | 72.96 | 145000 | e0–39 RTX PRO 6000 node61 (1043985) |
| Unweighted_A/MEAN | 902 | 452.05 (38) | 423.79 | 432.31 | 20.21 | 75.34 | 145000 | e0–39 RTX PRO 6000 node58 (1045059) |
| Unweighted_A/MEAN | 903 | 450.20 (33) | 442.54 | 435.16 | 12.07 | 75.03 | 145000 | e0–39 RTX PRO 6000 node59 (1045066) |

## 2 分段硬件与“同GPU”尺度配对

> 每个 method/seed 的两个尺度在同一个 Slurm job、同一 GPU 上顺序运行；交替尺度顺序是预先固定的，初始化和 shuffle seed 相同。
>
> —— `/home/infres/yinwang/CS_QMI/handoff/v9_full_20261017/PROTOCOL_CN.md:15`

> SUM为预先固定主尺度，MEAN为预定同方法同GPU敏感性配对；终末GPU/节点和续跑记录见CLOSEOUT.json。若中断续跑更换硬件，相应尺度差保留该混杂，不能声称严格同GPU。方法之间也可能使用不同型号GPU。
>
> —— `/home/infres/yinwang/CS_QMI/handoff/v9_overlap_v10_20261018/V9_CLOSEOUT_CN.md:51`

> SUM—MEAN仍可报告为固定seed敏感性结果，但不能再称严格同卡对照或保证跨硬件逐位一致。
>
> —— `/home/infres/yinwang/CS_QMI/handoff/v9_unit_split_20261018/README_CN.md:16`

> Original allowed partitions RTX6000PRO,L40S,H100,exclude node51,node60 unchanged. Split units may use different physical GPUs/models;903 SUM-MEAN is no longer strict same-GPU pairing. Report original and resumed hardware explicitly; no hardware-invariant/bitwise-equivalence claim.
>
> —— `/home/infres/yinwang/CS_QMI/handoff/v9_unit_split_20261018/SCHEDULING_AMENDMENT.json` 字段 `hardware`

CLOSEOUT.json 每个单元的 `gpu`/`host` 取自 `full_result.json`，即终末段硬件。3 个单元由续跑拼接（`resumed_at` 非空），其中 2 个的早期段与终末段节点或型号不同；拆分后另有 2 个单元整段重新训练。逐段记录如下（“段内最佳”为该段内DEV rSum最大的epoch）：

| 单元 | 段 | epoch | 轮数 | job | 节点 | GPU | 段内最佳 (epoch) | 含全程best | 硬件来源 |
|---|---:|---|---:|---|---|---|---:|---|---|
| BCE_Weight/SUM s902 | 0 | 0–17 | 18 | 1044282 | node52 | L40S | 446.96 (16) |  | 902_2/BCE_Weight_mean/full_result.json hardware（同一job 1044282）; TIMEOUT_GUARD_CONTROLLER.json NodeList/Partition |
| BCE_Weight/SUM s902 | 1 | 18–39 | 22 | 1046868 | node52 | L40S | 470.24 (38) | 是 | full_result.json hardware |
| VCS_Weight/MEAN s903 | 0 | 0–33 | 34 | 1045063 | node50 | L40S | 465.40 (33) |  | transitions/903_1/MIGRATED.json hardware_before = checkpoints_before_split/903_1/hardware.json |
| VCS_Weight/MEAN s903 | 1 | 34–39 | 6 | 1047108 | node52 | L40S | 469.78 (38) | 是 | full_result.json hardware = units/903_1/0/hardware_1047108.json |
| BCE_Weight/SUM s903 | 0 | 0–32 | 33 | 1045064 | node50 | L40S | 467.22 (32) |  | transitions/903_2/MIGRATED.json hardware_before = checkpoints_before_split/903_2/hardware.json |
| BCE_Weight/SUM s903 | 1 | 33–39 | 7 | 1046653 | node53 | H100 NVL | 467.36 (36) | 是 | full_result.json hardware = units/903_2/0/hardware_1046653.json |
| VCS_Weight/SUM s903 | 0 | 0–39 | 40 | 1046654 | node52 | L40S | 473.93 (38) | 是 | full_result.json hardware = units/903_1/1/hardware_1046654.json |
| BCE_Weight/MEAN s903 | 0 | 0–39 | 40 | 1046655 | node53 | H100 NVL | 463.00 (38) | 是 | full_result.json hardware = units/903_2/1/hardware_1046655.json |

- BCE_Weight/MEAN s902 在job 1044282 内完成全部 40 轮；该job随后在训练同组下一单元时结束（退出码 137）。
- BCE_Weight/SUM s902：第一段所在job 1044282 在训练本单元第 18 轮后到达时限（控制器 TIMEOUT，TimeLimit 23:59:00，记录退出码 137），随后按原配置续跑一次。
- VCS_Weight/MEAN s903 另有一次未训练任何epoch的尝试：job 1046652（node52 L40S），退出码 1；`DIAGNOSIS.json` 记为 METADATA_ONLY_FAILURE_BEFORE_TRAINING，字段 ess_U 相差 1 个fp64 ULP，`last_and_best_checkpoints_unchanged=true`。
- VCS_Weight/MEAN s903：拆分时原串行作业 1045063 已完成 34 轮（归档时段内最佳 465.40，epoch 33），控制器记录 CANCELLED/ExitCode 0:9；`MIGRATED.json` 记 `physical_gpu_pairing_preserved=false`。
- BCE_Weight/SUM s903：拆分时原串行作业 1045064 已完成 33 轮（归档时段内最佳 467.22，epoch 32），控制器记录 CANCELLED/ExitCode 0:9；`MIGRATED.json` 记 `physical_gpu_pairing_preserved=false`。

按协议，12 个SUM–MEAN配对都应在同一job、同一GPU上顺序运行；冻结的组作业脚本每个job只申请一张GPU（`#SBATCH --gres=gpu:1`，`/home/infres/yinwang/CS_QMI/handoff/v9_full_20261017/full.sbatch:6`），所以同job即同一分配GPU。实际 9 个满足（两单元均单段且同job，时间上顺序执行，已核验）；3 个不满足：

| 组 | 方法 seed | SUM各段 | MEAN各段 | 配对状态 | best差 SUM−MEAN | best所在硬件 SUM / MEAN |
|---|---|---|---|---|---:|---|
| 901_1 | VCS_Weight s901 | e0–39 L40S node50 (1043941) | e0–39 L40S node50 (1043941) | 同一job、同一分配GPU（顺序运行） | +0.79 | L40S node50 / L40S node50 |
| 901_2 | BCE_Weight s901 | e0–39 L40S node57 (1043942) | e0–39 L40S node57 (1043942) | 同一job、同一分配GPU（顺序运行） | -2.19 | L40S node57 / L40S node57 |
| 901_3 | RankMass s901 | e0–39 RTX PRO 6000 node58 (1043972) | e0–39 RTX PRO 6000 node58 (1043972) | 同一job、同一分配GPU（顺序运行） | +8.80 | RTX PRO 6000 node58 / RTX PRO 6000 node58 |
| 901_4 | Unweighted_A s901 | e0–39 RTX PRO 6000 node61 (1043985) | e0–39 RTX PRO 6000 node61 (1043985) | 同一job、同一分配GPU（顺序运行） | +8.13 | RTX PRO 6000 node61 / RTX PRO 6000 node61 |
| 902_1 | VCS_Weight s902 | e0–39 L40S node50 (1044068) | e0–39 L40S node50 (1044068) | 同一job、同一分配GPU（顺序运行） | +1.26 | L40S node50 / L40S node50 |
| 902_2 | BCE_Weight s902 | e0–17 L40S node52 (1044282)；e18–39 L40S node52 (1046868) | e0–39 L40S node52 (1044282) | 部分epoch同job；其余在另一分配（型号、节点相同，物理GPU编号未记录） | +2.09 | L40S node52 / L40S node52 |
| 902_3 | RankMass s902 | e0–39 RTX PRO 6000 node59 (1045058) | e0–39 RTX PRO 6000 node59 (1045058) | 同一job、同一分配GPU（顺序运行） | -3.89 | RTX PRO 6000 node59 / RTX PRO 6000 node59 |
| 902_4 | Unweighted_A s902 | e0–39 RTX PRO 6000 node58 (1045059) | e0–39 RTX PRO 6000 node58 (1045059) | 同一job、同一分配GPU（顺序运行） | -19.53 | RTX PRO 6000 node58 / RTX PRO 6000 node58 |
| 903_1 | VCS_Weight s903 | e0–39 L40S node52 (1046654) | e0–33 L40S node50 (1045063)；e34–39 L40S node52 (1047108) | 不同job与节点，型号相同 | +4.14 | L40S node52 / L40S node52 |
| 903_2 | BCE_Weight s903 | e0–32 L40S node50 (1045064)；e33–39 H100 NVL node53 (1046653) | e0–39 H100 NVL node53 (1046655) | GPU型号不同 | +4.36 | H100 NVL node53 / H100 NVL node53 |
| 903_3 | RankMass s903 | e0–39 RTX PRO 6000 node61 (1045065) | e0–39 RTX PRO 6000 node61 (1045065) | 同一job、同一分配GPU（顺序运行） | +1.76 | RTX PRO 6000 node61 / RTX PRO 6000 node61 |
| 903_4 | Unweighted_A s903 | e0–39 RTX PRO 6000 node59 (1045066) | e0–39 RTX PRO 6000 node59 (1045066) | 同一job、同一分配GPU（顺序运行） | -3.53 | RTX PRO 6000 node59 / RTX PRO 6000 node59 |

OracleMask/OracleDrop 共 3 组，其中 3 组为同job顺序运行（901_or 是，902_or 是，903_or 是）。

只用严格同job配对时，SUM−MEAN的best差（描述性，不改变任何门槛）：

| 方法 | 严格同job配对 | 严格子集均值 | 非严格配对 | 收尾全部3 seed |
|---|---|---:|---|---:|
| VCS_Weight | s901 +0.79，s902 +1.26 | +1.03 (n=2) | s903 +4.14 | +2.06 (3/3正向) |
| BCE_Weight | s901 -2.19 | -2.19 (n=1) | s902 +2.09，s903 +4.36 | +1.42 (2/3正向) |
| RankMass | s901 +8.80，s902 -3.89，s903 +1.76 | +2.22 (n=3) | 无 | +2.22 (2/3正向) |
| Unweighted_A | s901 +8.13，s902 -19.53，s903 -3.53 | -4.98 (n=3) | 无 | -4.98 (1/3正向) |

方法之间的比较本来就跨job，GPU型号也可能不同；各单元型号与节点（含分段，→表示续跑换段）：

| 方法/尺度 | s901 | s902 | s903 |
|---|---|---|---|
| VCS_Weight/SUM | L40S node50 | L40S node50 | L40S node52 |
| BCE_Weight/SUM | L40S node57 | L40S node52→L40S node52 | L40S node50→H100 NVL node53 |
| RankMass/SUM | RTX PRO 6000 node58 | RTX PRO 6000 node59 | RTX PRO 6000 node61 |
| Unweighted_A/SUM | RTX PRO 6000 node61 | RTX PRO 6000 node58 | RTX PRO 6000 node59 |
| RuLSIF_Weight/SUM | L40S node50 | L40S node50 | L40S node50 |
| OracleMask/SUM | RTX PRO 6000 node59 | RTX PRO 6000 node61 | RTX PRO 6000 node58 |
| OracleDrop/SUM | RTX PRO 6000 node59 | RTX PRO 6000 node61 | RTX PRO 6000 node58 |
| VCS_Weight/MEAN | L40S node50 | L40S node50 | L40S node50→L40S node52 |
| BCE_Weight/MEAN | L40S node57 | L40S node52 | H100 NVL node53 |
| RankMass/MEAN | RTX PRO 6000 node58 | RTX PRO 6000 node59 | RTX PRO 6000 node61 |
| Unweighted_A/MEAN | RTX PRO 6000 node61 | RTX PRO 6000 node58 | RTX PRO 6000 node59 |

## 3 完整40 epoch DEV轨迹

> 主比较在每 seed 内报告 SUM−MEAN、VCS−BCE、VCS−RankMass、VCS−Unweighted 的最佳 DEV rSum 差及完整 40 epoch 轨迹；3 seeds 单独显示，报告均值、范围和方向一致性。
>
> —— `/home/infres/yinwang/CS_QMI/handoff/v9_full_20261017/PROTOCOL_CN.md:25`

全部 33×40 个epoch的DEV rSum、mR、相对epoch4比值、epoch内平均U-risk、hard阶段最后200步抽样的锁定比例与冻结判定标记、所在段/job/节点/GPU、epoch用时与时间戳见 `N2F_TRAJECTORIES.csv`。节选（DEV rSum）：

| 方法/尺度 | seed | e0 | e4 | e5 | e6 | e7 | e10 | e20 | e29 | e30 | e39 | best (epoch) | 30–39均值 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| VCS_Weight/SUM | 901 | 290.16 | 369.92 | 398.70 | 410.85 | 411.03 | 424.62 | 451.97 | 451.10 | 461.16 | 468.32 | 468.32 (39) | 465.16 |
| VCS_Weight/SUM | 902 | 271.95 | 379.51 | 396.23 | 415.09 | 427.97 | 441.08 | 449.90 | 442.78 | 460.53 | 467.36 | 467.36 (39) | 458.32 |
| VCS_Weight/SUM | 903 | 282.92 | 371.91 | 407.22 | 408.95 | 416.41 | 432.43 | 444.71 | 458.80 | 465.62 | 472.56 | 473.93 (38) | 470.37 |
| BCE_Weight/SUM | 901 | 291.03 | 369.27 | 392.58 | 403.81 | 409.11 | 423.85 | 436.98 | 450.63 | 457.55 | 461.58 | 464.20 (38) | 460.67 |
| BCE_Weight/SUM | 902 | 278.11 | 377.89 | 387.51 | 418.30 | 425.42 | 439.92 | 449.39 | 458.90 | 469.84 | 444.87 | 470.24 (38) | 458.66 |
| BCE_Weight/SUM | 903 | 281.01 | 371.66 | 412.60 | 419.70 | 426.69 | 423.35 | 448.03 | 456.82 | 461.26 | 459.82 | 467.36 (36) | 462.00 |
| RankMass/SUM | 901 | 269.31 | 350.67 | 407.46 | 411.10 | 422.92 | 440.32 | 450.20 | 444.62 | 461.76 | 450.69 | 466.25 (34) | 452.04 |
| RankMass/SUM | 902 | 251.79 | 363.33 | 398.36 | 413.75 | 427.14 | 438.93 | 451.12 | 450.06 | 455.94 | 452.31 | 460.28 (36) | 456.12 |
| RankMass/SUM | 903 | 272.05 | 359.03 | 405.76 | 421.79 | 427.44 | 432.27 | 453.43 | 451.54 | 452.31 | 455.98 | 471.89 (38) | 458.44 |
| Unweighted_A/SUM | 901 | 254.26 | 343.08 | 356.11 | 384.30 | 399.61 | 382.52 | 393.79 | 434.24 | 406.82 | 435.64 | 445.90 (38) | 427.65 |
| Unweighted_A/SUM | 902 | 247.53 | 335.80 | 366.07 | 355.82 | 375.68 | 391.79 | 395.94 | 409.09 | 416.35 | 416.59 | 432.52 (36) | 418.15 |
| Unweighted_A/SUM | 903 | 259.07 | 340.57 | 374.42 | 361.85 | 385.94 | 402.33 | 406.45 | 416.19 | 412.96 | 443.57 | 446.67 (38) | 432.46 |
| RuLSIF_Weight/SUM | 901 | 291.87 | 376.82 | 383.55 | 398.86 | 421.85 | 441.22 | 432.60 | 457.57 | 465.58 | 451.74 | 467.00 (32) | 459.48 |
| RuLSIF_Weight/SUM | 902 | 278.13 | 375.48 | 402.98 | 423.94 | 429.53 | 444.60 | 454.46 | 454.04 | 463.20 | 456.75 | 469.15 (33) | 458.96 |
| RuLSIF_Weight/SUM | 903 | 278.99 | 371.74 | 416.69 | 411.36 | 429.82 | 436.65 | 444.81 | 433.73 | 443.31 | 466.55 | 466.55 (39) | 456.39 |
| OracleMask/SUM | 901 | 307.67 | 420.69 | 444.30 | 448.60 | 456.09 | 462.64 | 468.13 | 466.47 | 473.83 | 478.46 | 478.54 (36) | 476.78 |
| OracleMask/SUM | 902 | 308.99 | 420.04 | 445.25 | 451.42 | 460.77 | 465.66 | 471.95 | 466.77 | 476.07 | 476.02 | 478.68 (38) | 476.86 |
| OracleMask/SUM | 903 | 312.37 | 417.53 | 447.06 | 453.47 | 454.52 | 465.31 | 471.56 | 470.08 | 476.90 | 478.72 | 480.65 (32) | 479.15 |
| OracleDrop/SUM | 901 | 289.19 | 406.31 | 437.99 | 444.48 | 451.30 | 455.90 | 470.51 | 469.59 | 473.91 | 473.43 | 477.04 (33) | 475.42 |
| OracleDrop/SUM | 902 | 298.44 | 409.80 | 445.05 | 449.78 | 452.37 | 463.71 | 469.29 | 470.51 | 479.35 | 479.37 | 481.60 (37) | 479.27 |
| OracleDrop/SUM | 903 | 304.24 | 415.86 | 440.95 | 449.27 | 457.40 | 463.49 | 468.84 | 469.03 | 476.86 | 476.69 | 477.91 (35) | 476.75 |
| VCS_Weight/MEAN | 901 | 289.49 | 374.54 | 397.30 | 408.09 | 417.97 | 433.02 | 441.50 | 456.04 | 460.91 | 465.98 | 467.53 (37) | 464.11 |
| VCS_Weight/MEAN | 902 | 278.15 | 371.46 | 398.21 | 410.89 | 421.78 | 433.53 | 450.59 | 455.72 | 462.49 | 465.15 | 466.09 (37) | 464.35 |
| VCS_Weight/MEAN | 903 | 279.25 | 366.88 | 389.61 | 414.91 | 421.56 | 431.60 | 441.30 | 452.82 | 461.16 | 469.07 | 469.78 (38) | 466.59 |
| BCE_Weight/MEAN | 901 | 290.04 | 371.66 | 396.73 | 398.34 | 417.57 | 427.14 | 448.50 | 449.11 | 456.80 | 466.39 | 466.39 (39) | 462.73 |
| BCE_Weight/MEAN | 902 | 282.68 | 373.14 | 400.39 | 409.25 | 414.24 | 430.16 | 445.40 | 454.12 | 462.90 | 467.77 | 468.15 (37) | 465.54 |
| BCE_Weight/MEAN | 903 | 272.29 | 362.03 | 378.11 | 390.55 | 407.34 | 422.07 | 434.20 | 448.88 | 456.65 | 462.94 | 463.00 (38) | 460.84 |
| RankMass/MEAN | 901 | 261.58 | 348.42 | 397.51 | 418.32 | 415.86 | 436.11 | 439.15 | 450.26 | 448.74 | 450.49 | 457.46 (35) | 451.60 |
| RankMass/MEAN | 902 | 245.90 | 347.65 | 384.69 | 417.16 | 429.45 | 441.52 | 454.44 | 449.70 | 440.32 | 459.82 | 464.16 (33) | 445.36 |
| RankMass/MEAN | 903 | 264.85 | 340.75 | 406.15 | 417.28 | 433.87 | 441.95 | 441.26 | 452.27 | 452.90 | 467.61 | 470.14 (36) | 464.12 |
| Unweighted_A/MEAN | 901 | 267.89 | 342.13 | 341.54 | 369.49 | 385.09 | 356.92 | 385.13 | 384.12 | 427.36 | 427.06 | 437.77 (36) | 429.12 |
| Unweighted_A/MEAN | 902 | 251.48 | 341.12 | 307.89 | 334.12 | 330.22 | 349.13 | 380.85 | 432.19 | 415.88 | 423.79 | 452.05 (38) | 432.31 |
| Unweighted_A/MEAN | 903 | 266.25 | 345.84 | 223.23 | 128.15 | 325.42 | 393.89 | 422.11 | 418.34 | 406.94 | 442.54 | 450.20 (33) | 435.16 |

全部 33/33 个单元的best都落在epoch 30–39（学习率衰减之后），最早为epoch 32。

## 4 崩塌与近崩塌的完整披露

> 预先定义“严重训练崩塌”：在 hard 阶段连续两个完整 epoch，DEV rSum < 同一次训练 epoch4 的 50%，同时该两 epoch 最后 200 training steps 的日志抽样点中 U-risk>=0.39 的比例各 >=90%。报告实际抽样点数。
>
> —— `/home/infres/yinwang/CS_QMI/handoff/v9_full_20261017/PROTOCOL_CN.md:27`

> 严重崩塌0/33；未解决失败0。
>
> —— `/home/infres/yinwang/CS_QMI/handoff/v9_overlap_v10_20261018/V9_CLOSEOUT_CN.md:65`

按冻结判定重算：严重崩塌 0/33，与收尾一致；每个hard-epoch窗口的实际抽样点数为 10（训练日志每 20 步记录一次）。下表列出满足以下任一条件的全部单元：hard阶段（epoch 5–39）任一epoch的DEV rSum低于本次epoch4；DEV rSum < 50%·epoch4；锁定比例（U-risk≥0.39的抽样点占比）≥90%。

| 单元 | epoch4 | 最低比值 (epoch) | 低于epoch4的epoch | rSum<50%的epoch | 锁定≥90%的epoch | 两条件同时满足 | 严重崩塌 |
|---|---:|---:|---|---|---|---|---|
| Unweighted_A/MEAN s901 | 342.13 | 0.998 (e5) | 5 | — | — | — | 否 |
| OracleMask/SUM s901 | 420.69 | 1.056 (e5) | — | — | 5,6,7,8 | — | 否 |
| RankMass/SUM s902 | 363.33 | 1.096 (e5) | — | — | 5 | — | 否 |
| Unweighted_A/MEAN s902 | 341.12 | 0.903 (e5) | 5,6,7,8 | — | — | — | 否 |
| OracleMask/SUM s902 | 420.04 | 1.060 (e5) | — | — | 5,6,7 | — | 否 |
| RankMass/SUM s903 | 359.03 | 1.130 (e5) | — | — | 5 | — | 否 |
| Unweighted_A/MEAN s903 | 345.84 | 0.371 (e6) | 5,6,7 | 6 | 5,6 | 6 | 否 |
| OracleMask/SUM s903 | 417.53 | 1.071 (e5) | — | — | 5 | — | 否 |

- Unweighted_A/MEAN s901：epoch 4–7 DEV rSum 342.13 → 341.54 → 369.49 → 385.09（最低 0.998×epoch4，epoch 5）；锁定比例 e5 0.2，e6 0.0，e7 0.0；epoch 6 起DEV rSum始终不低于epoch4，best 437.77（epoch 36）。两条件同时满足的epoch：无；冻结定义要求连续两个这样的epoch，因而不是“严重崩塌”。
- Unweighted_A/MEAN s902：epoch 4–10 DEV rSum 341.12 → 307.89 → 334.12 → 330.22 → 338.88 → 367.26 → 349.13（最低 0.903×epoch4，epoch 5）；锁定比例 e5 0.5，e6 0.0，e7 0.0，e8 0.0，e9 0.0，e10 0.0；epoch 9 起DEV rSum始终不低于epoch4，best 452.05（epoch 38）。两条件同时满足的epoch：无；冻结定义要求连续两个这样的epoch，因而不是“严重崩塌”。
- Unweighted_A/MEAN s903：epoch 4–9 DEV rSum 345.84 → 223.23 → 128.15 → 325.42 → 359.29 → 370.57（最低 0.371×epoch4，epoch 6）；锁定比例 e5 1.0，e6 1.0，e7 0.2，e8 0.0，e9 0.0；epoch 8 起DEV rSum始终不低于epoch4，best 450.20（epoch 33）。两条件同时满足的epoch：6；冻结定义要求连续两个这样的epoch，因而不是“严重崩塌”。
- 另有 5 个单元只有锁定比例达到阈值，DEV rSum在hard阶段始终高于epoch4：OracleMask/SUM s901，RankMass/SUM s902，OracleMask/SUM s902，RankMass/SUM s903，OracleMask/SUM s903；单独的锁定比例不满足冻结判定。
- hard阶段DEV rSum低于自身epoch4的单元共 3 个：Unweighted_A/MEAN s901（最低 0.998×），Unweighted_A/MEAN s902（最低 0.903×），Unweighted_A/MEAN s903（最低 0.371×），分属 Unweighted_A/MEAN。这是描述，不是尺度因果证据（见下文）。

### 与wave-1崩塌的比较

> pause wave 1 after the collapse report (4 of 24 rho-0.4 SGR-student runs collapsed right after the epoch-5 switch)
>
> —— `/home/infres/yinwang/CS_QMI/ssl_pilot_v9/reports/V9/V9_N2_units/wave1_paused.json` 字段 `decision`

> collapse census at the pause (units with >= 7 epochs): rho 0.4 19 healthy, 4 collapsed (BCE_Weight s901 x1 formal, Unweighted_A s902 x1 formal, Unweighted_A s917 x0.5 and VCS_Weight s917 x1 tuning), 1 dipped and recovered (Unweighted_A s917 x1); rho 0 16 healthy, 0 collapsed.  Collapse = U risk locked at 0.40 (2 x margin) within 20-40 steps of the epoch-5 switch, dev rSum falling from 335-370 to 1-100
>
> —— `/home/infres/yinwang/CS_QMI/ssl_pilot_v9/reports/V9/V9_N2_PROTOCOL_FROZEN_20261015.yaml:133`

对普查点名的 5 个wave-1单元（unit目录取自 `V9_N2_units/formal_units.json` 与 `tuning_units.json`，根目录为N2F FROZEN.json的 `old_output` 上级）按N2F同一冻结判定重算，并与N2F中同seed、同方法、同学习率、同MEAN尺度的重跑及A/B诊断（300步）对照：

| wave-1单元 | 普查 | epoch4 | 最低比值 | best | e39 | 冻结判定：严重崩塌 | N2F MEAN重跑 | A/B诊断 |
|---|---|---:|---:|---:|---:|---|---|---|
| BCE_Weight s901 x1（正式） | 崩塌 | 369.84 | 0.003 (e25) | 369.84 (e4) | 1.28 | 是（34个两条件epoch） | 901_2：best 466.39 (e39)，最低 1.067×，严重崩塌否 | C1_sum（SUM）：epoch4 377.8 → 切换后300步 387.0，锁定 0.35，collapse_present=false |
| Unweighted_A s902 x1（正式） | 崩塌 | 335.72 | 0.037 (e11) | 335.72 (e4) | 88.07 | 是（33个两条件epoch） | 902_4：best 452.05 (e38)，最低 0.903×，严重崩塌否 | C2_sum（SUM）：epoch4 339.2 → 切换后300步 338.1，锁定 0.285，collapse_present=false |
| Unweighted_A s917 x0.5（调参） | 崩塌 | 334.60 | 0.034 (e27) | 334.60 (e4) | 76.69 | 是（33个两条件epoch） | N2F不含此seed/学习率 | C3_sum（SUM）：epoch4 327.6 → 切换后300步 332.8，锁定 0.39，collapse_present=false |
| VCS_Weight s917 x1（调参） | 崩塌 | 365.09 | 0.011 (e15) | 365.09 (e4) | 27.20 | 是（25个两条件epoch） | N2F不含此seed/学习率 | AA_C4_mean（MEAN）：epoch4 365.5 → 切换后300步 352.6，锁定 0.495，collapse_present=false；C4_sum（SUM）：epoch4 369.3 → 切换后300步 399.2，锁定 0.385，collapse_present=false |
| Unweighted_A s917 x1（调参） | 下探后恢复 | 337.83 | 0.661 (e5) | 439.90 (e38) | 394.34 | 否（0个两条件epoch） | N2F不含此seed/学习率 | — |

两个正式配置在切换前后的DEV rSum（epoch 4–8），wave-1与N2F MEAN重跑并列：

| 配置 | 运行 | e4 | e5 | e6 | e7 | e8 |
|---|---|---:|---:|---:|---:|---:|
| BCE_Weight s901 | wave-1 | 369.84 | 297.81 | 1.38 | 3.51 | 5.42 |
| BCE_Weight s901 | N2F 901_2 | 371.66 | 396.73 | 398.34 | 417.57 | 424.22 |
| Unweighted_A s902 | wave-1 | 335.72 | 172.39 | 96.37 | 35.78 | 235.46 |
| Unweighted_A s902 | N2F 902_4 | 341.12 | 307.89 | 334.12 | 330.22 | 338.88 |

- BCE_Weight s901：wave-1与N2F的 `run_manifest.json` 只在 `formal` true vs false、`mean_v_full_pool` 0.642470375969464 vs 0.6424703759694641（1 个fp64 ULP） 上不同，权重锁、训练配置与其余字段逐字段相同；N2F在 L40S node57 上运行，deterministic_algorithms=true、tf32=false。
- Unweighted_A s902：wave-1与N2F的 `run_manifest.json` 只在 `formal` true vs false 上不同，权重锁、训练配置与其余字段逐字段相同；N2F在 RTX PRO 6000 node58 上运行，deterministic_algorithms=true、tf32=false。

> 新成对实验启用 deterministic algorithms、关闭 TF32/cuDNN benchmark，记录硬件与环境；不能把它与历史非确定性运行简单拼成一个随机实验。
>
> —— `/home/infres/yinwang/CS_QMI/handoff/v9_full_20261017/PROTOCOL_CN.md:15`

A/B诊断（`/home/infres/yinwang/CS_QMI/ssl_pilot_v9/reports/V9/V9_N2_diag`）的预注册尺度定义、读法与判词：

> sum: batch loss = sum_i v_i l_i / mean_v_full, the official ContrastiveLoss scale for v = 1 (= 128 x the frozen scale); mean: the frozen V9 scale
>
> —— `/home/infres/yinwang/CS_QMI/ssl_pilot_v9/reports/V9/V9_N2_PROTOCOL_FROZEN_20261015.yaml:138`

> locked = U risk >= 0.39 on >= 90 % of steps 101-300 after the switch; dev_fall = dev rSum at step 300 after the switch < that run's own epoch-4 dev rSum - 30; collapse_present = locked or dev_fall
>
> —— `/home/infres/yinwang/CS_QMI/ssl_pilot_v9/reports/V9/V9_N2_PROTOCOL_FROZEN_20261015.yaml:140`

> the mean-scale rerun of C4 shows collapse_present, i.e. the original collapse reproduces from scratch
>
> —— `/home/infres/yinwang/CS_QMI/ssl_pilot_v9/reports/V9/V9_N2_PROTOCOL_FROZEN_20261015.yaml:141`

> scale_fix_supported if the A/A is valid and collapse is absent in >= 3 of the 4 sum-scale configs; not_supported if collapse is present in >= 3 of the 4; inconclusive otherwise (and when the A/A is not valid only per-config outcomes are reported)
>
> —— `/home/infres/yinwang/CS_QMI/ssl_pilot_v9/reports/V9/V9_N2_PROTOCOL_FROZEN_20261015.yaml:142`

记录结果：A/A（MEAN尺度从头重跑C4）collapse_present=false，即A/A不成立；4 个SUM尺度配置中 0 个出现崩塌。按预注册判词，A/A不成立时只报告逐配置结果，不给出尺度结论。

> 原 N2 在 rho=0.4 有多例 warmup→hard-negative 后崩塌；最新 A/A 重跑 C4 在 300 step 未复现崩塌，原尺度诊断按自己的规则不能给出 scale_fix_supported。
>
> —— `/home/infres/yinwang/CS_QMI/handoff/v9_full_20261017/PROTOCOL_CN.md:7`

> 因此不能把完整训练成功归为某个已证明的无偏结论，也不能把崩塌直接归为 CS 的数学性质。
>
> —— `/home/infres/yinwang/CS_QMI/handoff/v9_full_20261017/PROTOCOL_CN.md:9`

**明确说明：mean与sum尺度之间的崩塌因果没有被确立。** wave-1中按冻结判定严重崩塌的 2 个正式MEAN配置，在N2F以相同配置重跑时严重崩塌 0/2；A/A未复现原崩塌，故A/B不给尺度结论；N2F全部33次训练严重崩塌 0/33。上面的近崩塌与低于epoch4的波动只作披露。

## 5 最佳epoch与后期波动（描述性敏感性）

> 每 epoch 使用完整原 DEV 文件（1014 图像，保持原 runner 口径），按 DEV rSum 选 checkpoint；不读取 TEST/COCO。
>
> —— `/home/infres/yinwang/CS_QMI/handoff/v9_full_20261017/PROTOCOL_CN.md:13`

> 投入决策是工程阈值而非统计显著性。SUM 预先定为新的主尺度，MEAN 为配对敏感性；二者冲突时必须报告，不取更有利者作为事先主分析
>
> —— `/home/infres/yinwang/CS_QMI/handoff/v9_full_20261017/PROTOCOL_CN.md:29`

> DEV 的多次选择与 n=3 不支持宣称统计显著，不用图像 bootstrap 假装增加训练重复数。
>
> —— `/home/infres/yinwang/CS_QMI/handoff/v9_full_20261017/PROTOCOL_CN.md:25`

预注册终点是40轮中DEV rSum最大的epoch。epoch 30–39 的“最大值−中位数”：27 个非Oracle单元为 1.74–23.21，6 个Oracle单元为 1.00–2.49；“best−后期均值”：非Oracle 1.74–19.74，Oracle 1.16–2.33。下表把收尾的 17 项配对差分别用best、末轮（epoch 39）与epoch 30–39均值重算（各格为三seed均值，括号内为正方向seed数；†表示均值符号与best相反）。best列与CLOSEOUT逐值相等；末轮与30–39均值是看到轨迹之后的描述口径，不是预注册终点。

| 比较 | best（预注册） | 末轮 e39 | 30–39均值 | 30–39均值的逐seed差（901，902，903） |
|---|---:|---:|---:|---|
| VCS_Weight:SUM-MEAN | +2.06 (3/3) | +2.68 (3/3) | -0.40 (2/3)† | +1.06，-6.04，+3.78 |
| BCE_Weight:SUM-MEAN | +1.42 (2/3) | -10.28 (0/3)† | -2.59 (1/3)† | -2.05，-6.87，+1.15 |
| RankMass:SUM-MEAN | +2.22 (2/3) | -6.32 (1/3)† | +1.84 (2/3) | +0.44，+10.76，-5.68 |
| Unweighted_A:SUM-MEAN | -4.98 (1/3) | +0.80 (2/3)† | -6.11 (0/3) | -1.47，-14.16，-2.71 |
| mean:VCS-BCE_Weight | +1.96 (2/3) | +1.03 (1/3) | +1.98 (2/3) | +1.38，-1.18，+5.74 |
| mean:VCS-RankMass | +3.89 (2/3) | +7.42 (3/3) | +11.32 (3/3) | +12.50，+18.99，+2.47 |
| mean:VCS-Unweighted_A | +21.13 (3/3) | +35.60 (3/3) | +32.82 (3/3) | +34.98，+32.04，+31.42 |
| mean:BCE_Weight-Unweighted_A | +19.17 (3/3) | +34.57 (3/3) | +30.83 (3/3) | +33.60，+33.22，+25.68 |
| mean:RankMass-Unweighted_A | +17.25 (3/3) | +28.18 (3/3) | +21.50 (3/3) | +22.48，+13.05，+28.96 |
| sum:VCS-BCE_Weight | +2.60 (2/3) | +13.99 (3/3) | +4.17 (2/3) | +4.49，-0.35，+8.37 |
| sum:VCS-RankMass | +3.73 (3/3) | +16.42 (3/3) | +9.08 (3/3) | +13.13，+2.20，+11.93 |
| sum:VCS-Unweighted_A | +28.17 (3/3) | +37.48 (3/3) | +38.53 (3/3) | +37.51，+40.17，+37.92 |
| sum:BCE_Weight-Unweighted_A | +25.57 (3/3) | +23.49 (3/3) | +34.36 (3/3) | +33.02，+40.51，+29.54 |
| sum:RankMass-Unweighted_A | +24.44 (3/3) | +21.06 (3/3) | +29.45 (3/3) | +24.38，+37.97，+25.98 |
| sum:OracleDrop-OracleMask | -0.44 (1/3) | -1.24 (1/3) | -0.45 (1/3) | -1.36，+2.41，-2.40 |
| sum:VCS-RuLSIF_Weight | +2.30 (2/3) | +11.07 (3/3) | +6.34 (2/3) | +5.69，-0.64，+13.98 |
| sum:RuLSIF_Weight-Unweighted_A | +25.87 (3/3) | +26.41 (3/3) | +32.19 (3/3) | +31.82，+40.81，+23.93 |

均值符号翻转共 5 处：VCS_Weight:SUM-MEAN（30–39均值），BCE_Weight:SUM-MEAN（末轮），BCE_Weight:SUM-MEAN（30–39均值），RankMass:SUM-MEAN（末轮），Unweighted_A:SUM-MEAN（末轮）。
- 通用加权相对Unweighted_A的 7 项比较，best：均值 +17.25…+28.17，每项正方向seed数至少 3/3。
- 通用加权相对Unweighted_A的 7 项比较，末轮：均值 +21.06…+37.48，每项正方向seed数至少 3/3。
- 通用加权相对Unweighted_A的 7 项比较，30–39均值：均值 +21.50…+38.53，每项正方向seed数至少 3/3。
- 收尾VCS专有门槛涉及的三项SUM比较：sum:VCS-BCE_Weight best +2.60 (2/3)、末轮 +13.99 (3/3)、30–39均值 +4.17 (2/3)；sum:VCS-RankMass best +3.73 (3/3)、末轮 +16.42 (3/3)、30–39均值 +9.08 (3/3)；sum:VCS-RuLSIF_Weight best +2.30 (2/3)、末轮 +11.07 (3/3)、30–39均值 +6.34 (2/3)。在两种事后口径下三项均值都不低于+3；这两种口径是看到轨迹之后的描述，协议的checkpoint选择与主比较都按最佳DEV rSum（见上方引文），VCS专有门槛仍按best判定为未通过。
门槛不因此改变；收尾按预注册best终点得出的结论保持不变。

## 6 计划与实测成本

> 按实测单次成本 RTX 3.60h / L40S 6.47h / H100 NVL 7.21h，本矩阵约 115–230 GPU-h（OracleDrop 按约64%训练池估计，额外启动/验证和确定性运行开销另计）
>
> —— `/home/infres/yinwang/CS_QMI/handoff/v9_full_20261017/PROTOCOL_CN.md:40`

协议区间可由单次成本复原：30 个完整训练池单元 + 3 个OracleDrop×0.64 = 31.92 个单元当量；×3.60 h = 114.9 GPU-h，×7.21 h = 230.1 GPU-h（协议写作 115–230）。

实测训练计时（收尾口径，含续跑前已完成段）合计 293.13 GPU-h；按各段所在GPU：H100 NVL 14.53，L40S 171.65，RTX 106.95。用协议单次成本、按实际分段硬件组合计算的计划值为 158.83 GPU-h，实测为其 1.85 倍。协议注明确定性运行开销另计；33/33 个单元的硬件记录为 deterministic_algorithms=true、tf32=false。逐型号/节点的epoch用时：

| GPU（协议标签） | 节点 | OracleDrop | epoch数 | epoch用时中位数 (s) | 折合40轮 (h) | 协议单次 (h) | 倍数 |
|---|---|---|---:|---:|---:|---:|---:|
| H100 NVL | node53 | 否 | 47 | 1113.5 | 12.37 | 7.21 | 1.72 |
| L40S | node50 | 否 | 347 | 1011.3 | 11.24 | 6.47 | 1.74 |
| L40S | node52 | 否 | 126 | 1459.7 | 16.22 | 6.47 | 2.51 |
| L40S | node57 | 否 | 80 | 1021.4 | 11.35 | 6.47 | 1.75 |
| RTX | node58 | 否 | 200 | 575.7 | 6.40 | 3.60 | 1.78 |
| RTX | node58 | 是 | 40 | 390.8 | 4.34 | 2.30 | 1.88 |
| RTX | node59 | 否 | 200 | 563.7 | 6.26 | 3.60 | 1.74 |
| RTX | node59 | 是 | 40 | 387.9 | 4.31 | 2.30 | 1.87 |
| RTX | node61 | 否 | 200 | 560.3 | 6.23 | 3.60 | 1.73 |
| RTX | node61 | 是 | 40 | 374.4 | 4.16 | 2.30 | 1.81 |

> 实际训练计时累计（含可恢复的已完成段）293.130 GPU小时。它不包含所有初始化、被中断但尚未落盘的epoch及调度开销，不能冒充完整allocated GPU时长。
>
> —— `/home/infres/yinwang/CS_QMI/handoff/v9_overlap_v10_20261018/V9_CLOSEOUT_CN.md:67`

> sacct会计服务当前不可用，完整分配时长存在缺口，明确保留该成本限制，不以估算补造实测值。
>
> —— `/home/infres/yinwang/CS_QMI/handoff/v9_overlap_v10_20261018/V9_CLOSEOUT_CN.md:69`

## 7 收尾门槛（按收尾原样复述，不新增结论）

> VCS 专有路线只有在完整 3 seeds 中 VCS−BCE、VCS−RankMass、VCS−RuLSIF 的平均差均 >=3 rSum、至少 2/3 seeds 同方向、且崩塌率不更高时，才值得设计独立确认。未满足则结束“VCS 独特优势”扩展。
>
> —— `/home/infres/yinwang/CS_QMI/handoff/v9_full_20261017/PROTOCOL_CN.md:31`

> 通用可靠性加权路线只有在 VCS/BCE/RankMass/RuLSIF 中至少一种，相对不加权平均提升 >=5 rSum、至少 2/3 seeds 同方向、且不增加崩塌时，才值得后续独立确认。不可把这里选择的优胜者当预先选定方法。
>
> —— `/home/infres/yinwang/CS_QMI/handoff/v9_full_20261017/PROTOCOL_CN.md:32`

| VCS专有比较（SUM） | 三seed均值 | 正方向 | 达到门槛 | 使均值恰为+3所需的VCS s903 SUM |
|---|---:|---:|---|---:|
| sum:VCS-BCE_Weight | +2.60 | 2/3 | 否 | 475.11 |
| sum:VCS-RankMass | +3.73 | 3/3 | 是 | 471.74 |
| sum:VCS-RuLSIF_Weight | +2.30 | 2/3 | 否 | 476.02 |

其余单元结果固定时，三项同时达到+3需要VCS s903 SUM ≥ 476.02（由 sum:VCS-RuLSIF_Weight 决定；在该值下正方向规则也满足）；实际为 473.93，所以VCS专有门槛未通过，与收尾一致。这只是收尾门槛算术的复述，不是新判据。

| 尺度 | 通用加权比较 | 三seed均值 | 正方向 | 达到门槛 |
|---|---|---:|---:|---|
| SUM | sum:VCS-Unweighted_A | +28.17 | 3/3 | 是 |
| SUM | sum:BCE_Weight-Unweighted_A | +25.57 | 3/3 | 是 |
| SUM | sum:RankMass-Unweighted_A | +24.44 | 3/3 | 是 |
| SUM | sum:RuLSIF_Weight-Unweighted_A | +25.87 | 3/3 | 是 |
| MEAN | mean:VCS-Unweighted_A | +21.13 | 3/3 | 是 |
| MEAN | mean:BCE_Weight-Unweighted_A | +19.17 | 3/3 | 是 |
| MEAN | mean:RankMass-Unweighted_A | +17.25 | 3/3 | 是 |

通用加权门槛：SUM=True，MEAN=True，与收尾一致。

> VCS未同时达到相对BCE、RankMass、RuLSIF的原预设资源阈值。当前不建立稳定的VCS独有优势，不追加本路线网格。
>
> —— `/home/infres/yinwang/CS_QMI/handoff/v9_overlap_v10_20261018/V9_CLOSEOUT_CN.md:53`

> 通用加权资源阈值：SUM=True，MEAN=True。阈值要求相对不加权平均至少+5、至少2/3同方向且崩塌不更多。
>
> —— `/home/infres/yinwang/CS_QMI/handoff/v9_overlap_v10_20261018/V9_CLOSEOUT_CN.md:55`

> 至少一个尺度支持通用加权的实际收益；按用户决定，保留此结论并结束本轮，不自动扩大V9。
>
> —— `/home/infres/yinwang/CS_QMI/handoff/v9_overlap_v10_20261018/V9_CLOSEOUT_CN.md:57`

> 以上三seed差值为描述性结果，不作统计显著性或独立TEST确认主张。
>
> —— `/home/infres/yinwang/CS_QMI/handoff/v9_overlap_v10_20261018/V9_CLOSEOUT_CN.md:51`

## 8 文件与复现

- `N2F_FINAL_33_UNITS.csv`：sha256 `88b055a2df1a1efe904a15f5c222c64249a234746c37819608ce60b6732dfa44`
- `N2F_FINAL_33_UNITS.json`：sha256 `7368c115abd5fdfc65b62050444ba389e4c4053aada8c78d9ed7190a265f48e5`
- `N2F_TRAJECTORIES.csv`：sha256 `a80d5647d78713f3be249992b91175691ff5df81c703ae14e398f86db7ee0e92`
- `V9_CLOSEOUT_ADDENDUM_CN.md`：本文件。

复现：`python3 v9_n2m/n2f_closeout/addendum.py` 重写四个文件（输入不变时逐字节相同）；`python3 v9_n2m/n2f_closeout/addendum.py --check` 只重算并与仓库中的文件逐字节比较。任一输入哈希或重算值与记录不符时，生成器拒绝运行且不写任何文件。可选 `--plot <png>` 另绘33条轨迹（不入库，不参与 --check）。
