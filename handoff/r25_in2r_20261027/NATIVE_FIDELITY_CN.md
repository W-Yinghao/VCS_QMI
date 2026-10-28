Publication copy (sample names aliased; numbers and conclusions unchanged).

# R25 原生实现忠实度（IN2R-code 与 SPS）

状态标签：**paper-reported**（论文数字）、**reproduced**（本服务器实测）、**not-run**。本文件只写本轮实际执行的内容。

## 1. IN2R-code（主宿主）

作者源码 `liuyyy111/IN2R@909e53293ec3c633823597b8215ce23a71796d69`，只读保存在 `/projects/EEG-foundation-model/yinghao/models/r25_src/IN2R`（逐文件 sha256 见 `RUN_LOCK.json`）。R25 运行副本在 `code/`。

### 1.1 原样执行（未改一行）

`co_train.py` 的 GraphRefiner、MemoryQueue（FIFO、指针、topk 近邻、历史特征 detach）、warmup、eval_train（逐样本 loss→两分量 GMM）、split_prob（含“无噪声时强制 1% unlabeled”）；`vse.py` 的编码器、GPO、批内中心化与再归一化、ContrastiveLoss（clean hard-negative 求和）、`infonce`（实为 CE+RCE 的 SCE，τ=0.05、ε=1e-7）、AdamW(5e-4, wd 1e-4)；`data.py` 加载与增强；`evaluation.py` 的 encode/i2t/t2i；`utils.adjust_learning_rate`（每 15 轮 ×0.1）。原生 noisy 分支调用的就是作者 `infonce`，门禁逐位核对。

### 1.2 所有臂共用的工程改动

| 改动 | 原因 | 是否影响算法 |
|---|---|---|
| `vse.py`：`from torchtext.vocab import GloVe` → `glove_shim.GloVe` | torch 2.9 环境装不了 torchtext；GloVe 840B 由 HF 镜像下载（zip sha256 在 RUN_LOCK），按 torchtext 解析规则（按 b" " 切分、跳过 1 维行、最后一次重复覆盖）抽出 init_weights 可能查到的 16,779 个词；全文件 2,196,018 行中没有维度异常行 | 否：查到的词向量与 torchtext 相同；词表外词与原代码一样保持随机初始化 |
| `r25_co_train.py` 包装 main/train | 噪声修正目标钩子；完整状态保存/恢复；返回 6 项 recall | 原生臂只是同一计算 |
| B<2 或 unlabeled 划分为空时跳过 noisy 修正（所有臂） | 设计包 §5.2；原代码在空划分时会崩溃 | 计数记录（`skip_small_or_empty`） |
| 完整状态：两套编码器、四个 GraphRefiner、两个优化器（含动态加入的 refiner 参数组，按原生顺序重建）、两个 FIFO（内容/指针/full）、avg_clean_prob、all_loss、best_rsum、python/numpy/torch/cuda RNG | 官方 `VSE.state_dict()` 只含编码器，不能恢复 co-training | 门禁：恢复后与连续运行的第 1 轮统计和最终参数逐位相同 |
| 有标签 batch 只有 1 个样本时跳过该步（所有臂，计数 `skip_labeled_lt2`） | **原生缺陷**：`forward_emb` 批内中心化把单样本 batch 变成零向量，l2norm 反传得到 NaN 并被优化器写入参数（`warmup()` 有同类保护，`train()` 没有）。labeled 划分大小 ≡1 (mod 128) 时触发，每个 peer-epoch 约 1/128。首次出现：H0 Flickr40% seed1002 第 11 轮 peer B（划分 88,449=691×128+1），DEV 崩到 2.07、下一轮 GMM 遇 NaN 报错；该尝试存档于 `attempts/H0_f40_s1002_1067789_labeledB1_nan`，从同一前缀重跑 | 只在触发时有区别；没有触发的运行与修复后代码逐位相同（collect 逐 epoch 核对划分大小） |
| 不再每轮保存 `checkpoint_{e}.pth.tar`；只存 `model_best.pth.tar`（原生格式）+ `state_last.pt` + `prefix_state.pt` | 存储 | 否 |
| 训练结束不自动跑 TEST；`tools/test_eval.py` 需 `SELECTION_LOCK.json` 与 `--confirm-locked` | 协议：配置锁定前不读 TEST | 否 |
| 训练数据在作业开始时拷到节点本地 `$TMPDIR`（只拷 FIT/DEV 四个文件，按大小核对） | NFS 随机读使 GPU 空转 | 否（同一文件） |

**保留但记录的原生行为**：梯度裁剪只覆盖 `net.params`（编码器），GraphRefiner 参数不裁剪；本轮所有臂都保持这一原生行为，没有建立 native-fixed 变体。`cudnn.benchmark=True`、fp32、无 AMP（torch 默认：matmul TF32 关、cuDNN TF32 开）。

### 1.3 R25 选定的 code-version 配置（不是作者完整运行配置）

batch128、embed1024、warmup 5、main 40、lr_update 15、K=5、i2t 与 t2i 都开启（`run.py` 默认两者关闭、warmup 1、K 1——默认关闭的运行不是完整方法）、queue 65536、p_threshold 0.5、workers 0。论文与代码的差异按设计包记录：论文的 soft-target SCE 与代码的对角 SCE；clean intra 三项等权（代码）与论文 λ_intra；Adam/AdamW 与调度。本轮一律按代码执行。

### 1.4 数据与噪声

Flickr30K `f30k_precomp`：训练 29,000 图 / 145,000 caption；完整 DEV 1,014 图 / 5,070 caption（`dev_ims.npy` 每条 caption 一行，原生 validate 用全部 DEV，没有截断）。没有额外 clean anchors。40%/60% 全池噪声映射用作者 `data.py` 的同一算法、固定 `RandomState(20261027)` 在任何训练前生成（`NOISE_LOCK.json`）：实际错配率 0.39999 / 0.59998，所有方法与 seed 共用。官方 NCR-data 只有 0.2/0.5 索引，因而与文献的具体噪声抽样不同，需披露。

## 2. SPS（独立完整系统参照 S0）

作者源码 `C-TeaRanger/SPS@b985abe9936df95fff6d7f2745c408b88336c93c`，只读。补丁只有两处（`patches/sps_*.diff`）：①每个 co-training epoch 保存 checkpoint（原生只存 best/阶段/最后一轮）；②去掉训练末尾的自动 TEST。原生内部 validate 只用 DEV 前 1,000 行（200 图）做 warmup 选择与 best 记录——这是 SPS 算法内部步骤，保持不动，**不作为 R25 选择依据**；R25 用 `tools/sps_fulldev_eval.py` 在完整 DEV 上评估每个已存 epoch，按 full-DEV rSum 选最优。SPS 是 SGR 匹配模型，作业固定在 RTX6000PRO（与 NCR B1 同理：L40S/H100 上预计超过 24 h 时限，且无法续跑）。

## 3. 原生 IN2R-code 的 NaN 崩溃：失败记录、根因与修复（2026-10-27）

**失败**：两个原生 H0（Flickr40%）都在 GMM 拟合处报 `ValueError: Input X contains NaN`（co_train.py:767 eval_train）。
- seed1001（作业 1067786）：第 0–28 轮正常，最佳 DEV rSum 512.62；第 29 轮 DEV 掉到 2.07（peer A 505.2，peer B 2.1），第 30 轮 GMM 遇 NaN。
- seed1002（作业 1067789）：最佳 506.84；第 11 轮 DEV 2.07（peer A 498.3，peer B 2.1），随后同样报错。
两次尝试的输出目录完整移到 `outputs_v9/r25_in2r_20261027/attempts/H0_f40_s100{1,2}_*_labeledB1_nan`，日志保留在 `logs/`，计入成本。

**根因（已证实）**：co-training 中 peer B 用 peer A 的 clean 划分训练。两次崩溃的那一轮，该划分大小恰为 128 的倍数加 1（seed1002 第 11 轮 88,449=691×128+1；seed1001 第 29 轮同样 ≡1 mod 128），于是最后一个 labeled batch 只有 1 个样本。`VSE.forward_emb` 做批内中心化 `x − mean(x)`，单样本时嵌入恰为零向量；`l2norm` 在零向量处反传为 inf×0=NaN，`clip_grad_norm_` 与 AdamW 把 NaN 写进参数。该步的 loss 仍是有限值（零向量的相似度为 0），所以训练统计看不出问题，只在随后 DEV 与 GMM 暴露。
- 最小复现（同一 `vse.l2norm`、同一中心化）：B=1 → 嵌入全零、梯度含 NaN；B=2 → 正常。
- 逐 epoch 核对：两次崩溃各只有一个 ≡1 (mod 128) 事件，恰好对应崩溃的那一轮和那一个 peer；其余所有运行（V0/V1/B0/B1 f40_s1001、H0 f60_s1001）截至核对时没有此类事件，也没有崩溃。
- 排除的候选：AMP/溢出（全程 fp32、无 AMP，SCE 中 exp(S/0.05)≤e^20 不溢出）；refiner 不在梯度裁剪内（若是它，NaN 不会恰好出现在 ≡1 的那一轮且训练 loss 全程有限）；空划分/空队列路径（已有跳过逻辑，且两轮都有 688–692 次正常 noisy 更新）；R25 工程补丁（GloVe shim、状态保存不参与该计算；作者 `warmup()` 已对单样本 batch 做 `break` 保护，而作者 `train()` 没有——缺陷在原代码中）。

**修复（所有臂相同，即本轮的 native-fixed 基线）**：`r25_co_train.train()` 在 labeled batch 少于 2 个样本时跳过整步（不做 clean/enqueue/noisy 更新），计数 `skip_labeled_lt2`。触发概率约每 peer-epoch 1/128。该检查发生在任何计算和随机数消耗之前，因此**没有触发的运行与修复后代码逐位相同**；已在跑的 VCS/BCE 臂若触发会以同样方式崩溃（不会被悄悄污染），届时从同一前缀用修复后代码重跑。`tools/collect.py` 对每个完成的运行逐 epoch 列出 ≡1 事件与跳过次数，作为等价性的证据。同一事件随后也击中 V0 f40_s1001（作业 1067808，第 31 轮 peer A，崩溃前最佳 511.79；存档 `attempts/V0_f40_s1001_1067808_labeledB1_nan`），同样从共同前缀用修复后代码重跑；B0 f40_s1001（作业 1067810，第 37 轮 peer A，崩溃前最佳 510.26；存档 attempts/B0_f40_s1001_1067810_labeledB1_nan）亦然。V1、B1 f40_s1001 与 H0 f60_s1001 跑满 40 轮且无 ≡1 事件，与修复后代码逐位等价，不重跑。两个 H0 已从各自的共同前缀用修复后代码重跑（1068672、1068583）；由于恢复是逐位可复现的，重跑会复现崩溃前的全部轨迹，只在触发那一步不同。


### 3.1 第二个原生缺陷：增强把整条 caption 删空（2026-10-27）

**失败**：修复 3 后重跑的 H0 f40_s1002（作业 1068583）在第 12 轮 unlabeled batch 前向时报 `RuntimeError: Length of all samples has to be greater than 0`（vse.py:270 pack_padded_sequence）。存档 `attempts/H0_f40_s1002_1068583_emptycaption`。

**根因（已证实）**：作者 `data.py::process_caption(drop=True)` 对包括 `<start>`/`<end>` 在内的每个 token 以 0.2×0.4=0.08 的概率删除，从不保证留下至少一个 token。Flickr30K 训练集有 11 条 2 词、47 条 3 词 caption，n 个 token 全被删掉的概率为 0.08^n。用原代码对 2 词 caption 抽 200,000 次得到 10 次空 caption（期望 8.2）。`eval_train` 使用同一增强，同样会触发。

**修复（所有臂相同）**：若一条 caption 的全部 token 都被选中删除，则不删除（保留已做的 mask/随机替换），随机数消耗不变。补丁 `patches/in2r_data.py.diff`；修复后同样抽 200,000 次空 caption 为 0。行为只在原代码会崩溃的情形下改变，因此未触发的旧代码运行与修复后代码逐位相同；触发的运行必然崩溃并从同一前缀重跑。H0 f40_s1002 再次从其前缀以修复后代码重跑。

## 4. 原生完整训练结果

见下方（完成后填写）。

### 3.2 SPS 原生缺陷：DataLoader 迭代器 `.next()`（2026-10-28 01:10UTC，主会话）

- 现象：S0 SPS f40_s1001（作业 1067812）warmup 5 轮完成后，进入 co-training 第一步即崩溃：`AttributeError: '_MultiProcessingDataLoaderIter' object has no attribute 'next'`（`sps_code/train.py:424/435`）。
- 原因：作者代码用 Python 2 风格的 `it.next()`，当前 torch 迭代器只支持内置 `next(it)`。另一个后果更隐蔽：`train.py:409` 的 `clean_iter.next()` 包在裸 `try/except` 中，在新版 torch 下总是抛错并被吞掉，导致 `skip_clean_train=True`、clean batch 每步都被静默跳过；noisy 分支的第二次调用不在 try 中，所以才崩溃。
- 修复：三处 `x.next()` → `next(x)`，恢复作者在旧 torch 下的原意（耗尽时 StopIteration 仍走原 except 路径）。diff：`patches/sps_train_iternext.py.diff`。不改任何超参或流程。
- 处理：失败尝试整体移入 `outputs_v9/r25_in2r_20261027/attempts/S0_sps_f40_s1001_1067812_iternext`（SPS 原生不支持续跑），从头重交为 1069504（RTX，24h），成本计入失败尝试。
