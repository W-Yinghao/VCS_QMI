# R22-A：VCS 作为 SGR 学生的检索主损失（PILOT/DEVELOPMENT，只读 DEV）

方案见根目录 `R22_ALTERNATIVES_CN.md` §2.A；理论只取 `Variational_Cauchy_Schwarz_divergence.pdf` 式 (33)–(40)。

## 与 N2F/N2M/N2P-C1 相同的部分

宿主 SGR 学生（`v9.n2.train.WeightedStudent`）、数据（Flickr30k，观测池 145000，A=14500，ρ=0.4，N2F 冻结 token cache）、配方（HOST_LOCK ea22393b：40 epoch、batch 128、Adam 2e-4、第 30 epoch 起 ×0.1、host 参数 clip 2.0、`loss_scale="sum"`、确定性精度设置）、seed 901/902/903、选择规则（最佳全 DEV rSum，最早的并列）。输入与 helper 直接复用 `v9_n2p/c_server/c1_student.py`。

## 只换损失（`losses.py`）

- s 为 B×B 的 SGR 相似度（0–1）；critic T=tanh(softplus(a)s+b)，a、b 是损失模块的两个标量，与学生一起用同一个 Adam（同 lr 与衰减）训练。
- P：对角的 B 个观测对；Q：全部 B(B−1) 个非对角对（批内边缘乘积的 U 统计量，式 39 的全排列平均）。
- 臂：
  - `VCS`：损失 1−Ĵ，Ĵ=E_P T−E_Q T−½E_P T²−½E_Q T²（式 40），无权重；
  - `VCSW`：P 换成按 N2F VCS 权重加权的联合 P_w（A 为 1，U 为锁定权重），用宿主的全池固定归一化 mean_v_full，Q 不加权；
  - `BCE`：同一 critic，logit=2f，平衡 BCE（E_P softplus(−2f)+E_Q softplus(2f)），无权重；
  - `NCE`：对称 InfoNCE，logit=s/τ，τ=0.05 固定，无权重。
- `train.py` 只加了 `objective` 参数：为 None 时走原 hinge（CPU 检查与冻结的 N2F train.py 逐位一致）；否则用 objective 的损失，其参数加入同一 Adam，状态写入 checkpoint，可从最后一个完整 epoch 续跑。

## 设计取舍（实现可变，理由记录在此）

1. **尺度**：所有新损失乘以 B，沿用宿主 `sum` 约定（官方 SGRAF/NCR 对 anchor 求和），使锁定的 clip 2.0 与 hinge 运行处于同一量级；每 20 步记录 clip 前梯度范数。
2. **无 warm-up 切换**：warmup_epochs 只控制 hinge 的 mean→max-violation 切换；新损失没有难负例挖掘，所以从第 0 步起就是同一目标；lr 衰减不变。
3. **critic 初值** softplus(a)=5、b=−2.5：s=0.5 时 f=0，即所有 s 相同时 J 的最优 T=0。
4. **加权 P 的归一化**用宿主的全池固定分母（对全池无偏，与加权 hinge 相同），不按批次自归一化。
5. Q 包含同一图像的其他 caption：独立抽样下它们就是边缘乘积的样本；hinge 同样把它们当负例。
6. `train.py` 续跑时的步数计数改为已完成的更新数（原来按日志条数，续跑后偏小；只影响日志）。

## hinge 基线：复用，不重跑

`r22_student.py identity` 写 `HINGE_IDENTITY.json`：代码（宿主各模块与 N2F 冻结和 N2M runtime 的 sha 一致；train.py 基线 cafac01f）、HOST_LOCK、token cache、数据文件、每个 seed 的观测对与权重文件、配方（sum、40 epoch、lr×1、同 seed）、环境（Python 3.12.12、Torch 2.9.0+cu128、确定性、无 TF32）全部一致，结论 `REUSE_IDENTICAL_EXCEPT_HARDWARE`。
- Constant-U、hinge+VCS 权重：N2M（RTX PRO 6000）；
- hinge 无权重（Unweighted_A sum）、hinge+VCS 权重：N2F（分别为 RTX、L40S）。
- 硬件不同：新臂按三分区调度。同一 hinge+VCS 配方，N2F（L40S）与 N2M（RTX）相差 +1.32/+0.36/+2.96 rSum，这是跨硬件差异的参考量级。

## 检查结果

- CPU 1056085：14 项全部通过（J 与逐元素循环一致到 1e-12、加权 P 归一化、梯度有限且 a/b 有梯度、BCE/InfoNCE 对照、sum=B×mean、hinge 路径与冻结 train.py 逐位一致、各臂续跑逐位一致）。
- GPU 冒烟（seed 901，每臂 200 步，再做一次完整 DEV；每步记录日志，所以步时比正式运行偏慢）：
  - RTX 1056088 PASS：每步 0.62–0.64 s，DEV 77 s；
  - L40S 1056089 PASS：每步 1.14 s，DEV 157 s；
  - H100 1056090 一直在排队（GPU 配额满）；两种卡通过后，为把 normal 名额让给正式运行，未运行就撤下了。H100 的步时由落在 H100 的正式运行记录（attempts.jsonl、progress.jsonl）。
  - 损失（前 20 步均值 → 后 20 步均值，sum 尺度）：VCS 128.0→109.5、VCSW 127.8→86.9、BCE 177.4→155.9、NCE 620.9→542.8、HINGE 50.8→31.1（RTX；L40S 相近）。
  - 冒烟中 HINGE 与新损失每步耗时相同，新损失不增加成本。按 N2M/N2F 的 hinge 实测每 epoch 时间（RTX 577 s、L40S 1028 s、H100 1014 s），正式运行约 RTX 6.4 h、L40S/H100 11.3–11.4 h，都在 23:59 时限内。
  - sum 尺度下 clip 前梯度范数为 240–540，每步都被 clip 到 2.0，与 hinge 的官方 sum 设置相同。200 步内 a、b 几乎不动（α 5.00→5.007），s 的形状主要由学生的 sigmoid 输出层承担。

## 运行

- CPU 检查：`sbatch r22_cpu_test.sbatch`（14 项）。
- 冒烟：`r22_vl.sbatch smoke 901 VCS,VCSW,BCE,NCE,HINGE 200`，每种 GPU 一个作业，作业名 `r22_vl_chk_<gpu>`；结果在 `outputs_v9_r22/vcsloss/smoke/<job>_<t>/SMOKE.json`。
- 正式：`R22_VL_LINES.txt` 共 12 行（每个 (arm, seed) 一个作业，seed 901 在前），由 `handoff/review_claude_20261019/deps_watch/app_lines_feed.sh R22_VL_LINES.txt r22_vl 28 1587` 投放。超时后原样重交同一行即可续跑。
- 输出：`/home/infres/yinwang/CS_QMI/outputs_v9_r22/vcsloss/runs/s<seed>__<ARM>/`（只保留 best.pt 与 last.pt，各约 0.2 GB）。
- 汇总：`/home/infres/yinwang/CS_QMI/env/bin/python -I -B r22_student.py summarize --root /home/infres/yinwang/CS_QMI/outputs_v9_r22/vcsloss`，写 `SUMMARY.json` 和 `SUMMARY_UNITS.csv`，并打印每个 seed 的表与配对差值（VCS−ConstantU、VCS−hinge 无权重、VCSW−hinge+VCS（N2M/N2F）、VCS−BCE、VCS−NCE、VCSW−VCS 等）。

不读 TEST；每次运行的 open trace 在 `traces/`。

## R22-A2：大斜率 critic（`losses_sharp.py`，R22_ALTERNATIVES_CN.md §A2）

- 依据：A 的中间结果里 critic 斜率几乎不动（softplus(a) 5.00→5.007），T 一直偏软；J 的 Q 项对所有负样本等权平均，难负样本得不到强调。
- 只改 critic 参数化，其余（宿主、数据、配方、seed、精度、`sum` 尺度、P/Q 定义）与 A 相同：
  - f=α·s+b，**α=20 固定**（Python 常数，不是参数；与 InfoNCE 的 1/τ=1/0.05 同一尺度），T=tanh(f)，仍在式 (36) 的有界函数类内；
  - **b 可学，初值 −10**（s=0.5 处 f=0），**单独一个 Adam 参数组，基础 lr 1e-3**；学生仍是 Adam 2e-4，host 参数 clip 2.0，b 不参与 clip；
  - **b 的组跟随学生的步进衰减**：第 30 epoch 起 ×0.1（即 1e-4）。理由：A 的 critic 与学生同一调度，A2 只改 b 的基础 lr，差异最小；衰减后每 epoch 仍可移动约 0.1（s 上约 0.006），足以跟踪相似度分布。
  - 远处负样本 T≈−1，tanh 导数 1−T²≈0，梯度自然集中在难负样本上，不需要挖掘步骤。
- 臂：`VCS_SHARP`（1−Ĵ，式 40，P 对角、Q 全部非对角，各自求均值，×B）；`VCSW_SHARP`（P 按 N2F VCS 权重加权，与 A 的 VCSW 完全相同，Q 不加权）；`BCE_SHARP`（同一 f，logit 2f，平衡 BCE，无权重）。InfoNCE 复用 A 的 NCE 臂。
- 每 epoch 记录（`train_log.json` 每个 epoch 的 `objective_epoch`）：b、P/Q 上 T 的均值、Q 中 T>−0.9 的比例（难负样本比例）、P 中 T>−0.9 的比例、批上的 Ĵ（不加权；VCSW 另记加权的训练 J）、P/Q 上 T 的 5/25/50/75/95% 分位（每 20 步取一批，detach）。每 20 步的步日志另有 b 与该批的难负样本比例。
- `train.py` 只加了两个仅在 objective 定义了相应方法时才生效的钩子：`param_groups()`（自带参数组；带 `base_lr` 的组 lr=base_lr×0.1^(epoch//lr_update)）和 `epoch_stats()`（写入 epoch 记录）。A 的四个臂和 hinge 不定义这些方法，行为不变：`test_r22_a2.py` 用 df8a0707（A 开跑时）的 train.py 对 VCS/VCSW/BCE/NCE/HINGE 逐位比较（模型、优化器状态、critic、步日志、epoch 日志、manifest）；`losses.py` 未改动。
- 预先写定的扩展规则（`r22_student.py` 的 `A2_RULE`）：seed 901 上 VCS_SHARP 对 Constant-U（N2M，443.6）或 VCSW_SHARP 对 hinge+VCS 权重（N2M，467.0）的最佳全 DEV rSum 差距小于 10（高于 hinge 也算），就由主会话补 seed 902/903；否则停止并报告。`summarize` 打印 A2 行、A2−hinge 与 A2−A 的配对差值、b 和难负样本比例的轨迹以及该规则的判定。

## R22-A3：用可靠度权重给 InfoNCE 加权（`losses_nce_w.py`，R22_ALTERNATIVES_CN.md §A3）

- 依据：A 里不加权 InfoNCE 在第 7–11 个 epoch 达到 468.4 / 465.9，与 hinge+VCS 权重（467.0）相当，之后跌到约 400；同一套权重让 hinge 比 Constant-U 高约 26。VCS 留在它擅长的位置：给出可靠度后验 w=(1−ρ̂)(1+T)/(1+(1−2ρ̂)T)。
- 损失：A 的对称 InfoNCE（τ=0.05，`sum` 尺度 ×B），第 i 个正对的行、列两项 CE 乘 v_i/mean_v_full（A 为 1，U 为锁定权重；mean_v_full=(n_A+Σ_U w)/n_pool 是加权 hinge 的全池固定分母，写法与 A 的 VCSW 相同）。softmax 分母里的负样本不加权，低权重的对仍是普通负样本。没有 critic 参数。
- 臂：`NCE_VCSW` 用 N2F 的 VCS_Weight 文件（A 的 VCSW、N2F/N2M 的 hinge+VCS 用的同一份）；`NCE_BCEW` 用 N2F 的 BCE_Weight 文件（N2F 的 BCE_Weight_sum hinge 臂用的同一份）。两者都经原读取函数 `read_weight_lock` 和 N2F 冻结 sha 检查；`run` 另要求锁文件的 method 与臂一致。
- 不加权的 NCE 复用 A：`r22_student.py a3-identity` 写 `A3_IDENTITY.json`，检查 A 开跑（df8a0707）以来只有 `v9_r22/vcsloss/` 和 train.py（A2 钩子，NCE 类目标不会触发）有改动、losses.py 未改、A 的 NCE 记录的 losses/train.py/输入/目标配置/环境一致、两种权重文件与对应 N2F hinge 臂的 manifest 一致；A3 跑完后再比初始模型、批次顺序、配方和精度设置。N2F 的 hinge+BCE 权重（BCE_Weight_sum）作为基线并入汇总。
- CPU 检查（`test_r22_a3.py`）：加权损失和梯度与逐元素循环一致；v=1 时损失、梯度和逐 anchor 项与 A 的 NCE 逐位一致，经训练器跑完也逐位一致；归一化等于 hinge 路径的 Σ(v·ℓ)/mean_v_full；w=0 的对仍出现在别人的分母里；续跑逐位一致；三个 seed 的权重文件；扩展规则；losses.py、losses_sharp.py、train.py 与 A2 开跑（ccb5f331）逐字节相同，A/A2 的常量、权重映射和规则函数不变。
- 预先写定（`A3_RULE`）：seed 901 上 NCE_VCSW 最佳全 DEV rSum ≥ 464.0 就由主会话补 seed 902/903；补齐后 NCE_VCSW − hinge+VCS 权重（N2M，467.0 的来源；N2F 并列报告）和 NCE_VCSW − NCE_BCEW 都按三 seed 平均 ≥ +3 且至少 2/3 为正判断。`summarize` 打印 A3 行、配对差值（NCE_VCSW − hinge+VCS（N2M/N2F）、− NCE_BCEW、− NCE，NCE_BCEW − hinge+BCE（N2F）、− NCE）和判定。
- 正式：`R22_VL3_LINES.txt` 2 行（seed 901），由 `app_lines_feed.sh R22_VL3_LINES.txt r22_vl3 28 1587` 投放，作业名 `r22_vl3_*`，nice 60；不另做 GPU 冒烟（坏掉的完整运行几分钟内就会失败）。
