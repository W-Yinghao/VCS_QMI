# VCS-QMI：CIFAR-10 自监督学习 Server Agent 执行规范

**版本：1.0｜日期：2026-09-24｜阶段：首轮视觉 SSL 接入**

## 0. 给执行 agent 的任务摘要

你的任务是：**把合作者原始 VCS-QMI 神经依赖目标接入普通双视图图像 SSL，完成可审计实现、必要测试、CIFAR-10 小规模试跑与冻结特征评估。**

第一轮使用 `ResNet-18-CIFAR + projector + tanh MLP critic`。VCS 训练损失只有 `-J`，encoder、projector、critic 通过同一次普通反向传播联合更新。运行一个 SimCLR 对照、一个 matched-projector VICReg 对照。不要修改合作者的数学目标，不要另建新的研究方向。

推荐按 `P0 环境核查 → P1 核心测试 → P2 真数据 smoke → P3 三个20-epoch试跑 → P4 回报` 执行。**本文件不自动启动200-epoch、多种子、大规模调参、CIFAR-100、ImageNet、图文或EEG。** 这些属于审核首轮报告后的下一阶段。

交付物必须包括：代码改动、解析后的配置、数据划分 manifest、29项参考测试结果及服务器集成测试、原始日志、checkpoint、冻结线性评估、计算成本和失败记录。

**优先验证“实现正确、编码器确实在学习”，不是通过改算法使曲线变好。** 结果不佳可以是研究结果；不得擅自加入 VICReg 正则、EMA teacher、hard negatives 或新的梯度算法来掩盖它。

### 0.1 随包文件的状态

| 文件 | 当前状态 | 执行 agent 的责任 |
|---|---|---|
| `reference/ssl_core.py` | 已有可运行的损失、critic、配对与普通联合更新参考 | 集成到服务器训练框架，不另写不一致的损失 |
| `tests/test_ssl_core.py` | 本地CPU已通过29项测试 | 在服务器环境复跑，并补数据/模型/恢复测试 |
| `configs/cifar10_pilot_*.yaml` | 已冻结的三份首轮默认配置 | 实现读取与校验；不得静默忽略字段 |
| 本执行规范 | 操作、评估、验收合同 | 作为实现依据 |
| `vcs_ssl.train/evaluate/preflight` 等命令 | **尚未随包实现**，是后文要求的CLI接口 | 在现有repo中实现或给出等价入口 |

随包代码不是完整的 CIFAR trainer。已有测试只验证公式和梯度路径，不表示已经运行了图像训练或取得了任何准确率。

---

## 1. 来源优先级与本次范围

### 1.1 哪些是原计划，哪些是工程选择？

**[S1 原计划]** `Variational_CS_QMI_Research_Plan (1).pdf`：§3–§7定义，§9实现，§10稳定性范围。其 Eq.(5)、Eq.(7)、Eq.(17) 是本次必须保持一致的依据。

**[S2 已讨论的实验规范]** `VCS_QMI_CVPR_Experimental_Protocol.md`：§5建议普通联合更新、CIFAR ResNet-18、128维投影、K=1循环错配和冻结线性评估。

**[E 本文件新增工程默认]** 具体层宽、数据增强参数、初始化、学习率、优化器、划分seed、20-epoch试跑预算和输出格式。这些不是合作者的理论结论，也不是已经证实的最佳超参数。

若服务器已有合作者的原始代码，先逐项比较数学定义与实现。本文件没有提供的额外设计不能被冒称为“合作者默认”。发现冲突时在报告中列出，不以服务器旧配置或上一条研究支线覆盖本文件。

### 1.2 当前明确不做

- 不添加表示Gaussian平滑目标，不采用两点/零阶梯度，不阻断VCS critic到encoder的正常梯度。
- 不加入CS-Aligner式边缘分布对齐，不做CLIP或EEG—视觉检索。
- 不改变参考测度；不把目标说成Shannon MI。
- 不以 `2 log((1+J)/(1-J))` 训练，不对 `J` 做非负裁剪。
- 不加入方差、协方差、正交、重建、IB、域消除等额外VCS损失。
- 不使用预训练权重、teacher、EMA、队列、样本标签筛选负样本、hard-negative mining、memory bank。
- 不因效果不佳自行把拼接critic换成余弦、双线性、注意力或多层融合。
- 不启动DDP、梯度累计、`torch.compile` 或混合精度优化；先建立单卡FP32参照。

VICReg对照自身的三项损失不属于给VCS加正则；两者必须是不同run。

---

## 2. P0：环境核查，不假设服务器路径与资源

本对话没有指定repo、分区、GPU型号、数据目录、环境名称。不得把ChatGPT sandbox路径当作服务器路径，也不得从旧项目中猜测后直接开始长训练。

在当前授权工作区做只读核查，输出 `reports/P0_preflight.md`：

| 项目 | 必须记录 |
|---|---|
| repo | 绝对路径、当前branch、commit、dirty文件、是否已有SSL trainer |
| 环境 | Python/PyTorch/torchvision版本、CUDA runtime、cuDNN、驱动 |
| 资源 | GPU型号/显存/可用数量、CPU、内存、可用磁盘与inode |
| 数据 | CIFAR-10是否存在、原始文件校验结果、dataset index顺序 |
| 训练框架 | 数据增强、backbone、checkpoint、评估脚本可复用的位置 |
| 调度 | 当前应使用的scheduler/partition；若有scheduler，不在login node训练 |
| 待实现 | 哪些入口缺失；哪些旧实现与本规范不一致 |

允许读取现有配置、目录和模块。不得清理他人进程、删旧checkpoint、重装共享CUDA、批量升级系统包。缺环境依赖时使用项目独立环境，并记录变更。数据默认 `download=false`；找不到数据则先报告确切阻塞，不偷偷改用随机数据或其他数据集。

只在指定工作区新增文件；不覆盖未提交的用户改动。路径由agent探测并回填：

```text
REPO_ROOT=<已确认的项目根目录>
DATA_ROOT=<已确认的CIFAR数据目录>
OUTPUT_ROOT=<有配额的实验结果目录>
MANIFEST_ROOT=<本项目的固定划分目录>
```

`REPO_ROOT` 等仍未解析时，训练CLI必须报错，不能在字符串字面量目录下继续。

---

## 3. 必须保持不变的目标

### 3.1 总体定义 [S1]

对两种视图表示 `Z1,Z2`：

\[
P=P_{Z_1Z_2},\qquad Q=P_{Z_1}P_{Z_2},\qquad M=(P+Q)/2.
\]

\[
T_\phi(z_1,z_2)=\tanh f_\phi(z_1,z_2).
\]

\[
J(T)=\mathbb E_P T-\mathbb E_Q T
-\tfrac12\mathbb E_P T^2-\tfrac12\mathbb E_Q T^2.
\]

\[
I_{\mathrm{VQ}}=S=\sup_TJ(T),\qquad
\mathcal L_{\mathrm{VCS}}=-\widehat J.
\]

总体有 `S-J(T)=E_M[(T-eta)^2]`。实现不需要显式计算密度、M、PMI或总体最优T。

### 3.2 Mini-batch数值定义

设正样本分数 `t_pos` 共B个，负样本分数 `t_neg` 共K×B个：

```python
J = (
    t_pos.mean()
    - t_neg.mean()
    - 0.5 * t_pos.square().mean()
    - 0.5 * t_neg.square().mean()
)
loss = -J
R = 0.5 * (1 - t_pos).square().mean() \
  + 0.5 * (-1 - t_neg).square().mean()
# J == 1 - R
```

正负两项分别求均值。增加K不能把负分布权重乘以K。特别禁止：

```python
# 错误：平方放在均值外面
(t_pos.mean()) ** 2
# 错误：正负数量不等时，直接拼接后平均MSE而不平衡两类权重
F.mse_loss(torch.cat([t_pos, t_neg]), labels)
# 错误：训练非负截断后分数，可能直接切断梯度
loss = -J.clamp_min(0)
```

K=1时，拼接后平衡MSE恰等于R；但R和-J相差常数1。`0.5 * 拼接MSE`还会把梯度减半，不得无记录替换。

### 3.3 日志命名

训练记录叫 `J_raw` / `R_binary`，不要叫 `true_MI`、`Shannon_MI`、`certified_S`。

`S`在总体上位于[0,1]，但任意critic的J不保证非负。有界分数下经验J位于[-3,1]；负数不是错误，也不能自动裁掉。

首轮不报告变换后的CS。以后若确实需要，该派生指标须明确值域、投影规则和裁剪比例，且不参与训练与checkpoint选择。

---

## 4. 数据合同：先划分图像ID，再生成视图

### 4.1 数据与固定划分 [E；原始规模见W4]

首轮仅使用CIFAR-10官方训练部分。开发划分：45,000张 `fit` + 5,000张 `selection`。官方10,000张测试集在首轮**不实例化、不读取、不评估**。

划分算法固定：以官方train dataset index作为稳定UID；使用 `numpy.random.default_rng(20260924)`；按class id从0到9依次打乱该类的全部索引，前500个作为selection，其余作为fit；最终各split按UID排序写入manifest。标签仅允许在创建分层划分和下游评估时读取，SSL Dataset向训练器返回 `view1, view2, uid`，不返回class label。

manifest至少包含：数据源、原始文件哈希、dataset版本、划分算法、seed、fit/selection UID数组、类计数、UID交集检查、manifest SHA-256。

所有run共享同一个manifest。不能每个method/seed重新分割。不能先增强或缓存窗口再随机分割。两个视图来自同一UID时始终属于同一split。

### 4.2 SSL数据加载

- 在fit中随机shuffle，每epoch无放回；`drop_last=true`。
- `batch_size_images=256`：这是256张基础图片和512个增强视图，不是512张独立图片。
- 默认45,000 // 256 = 175个optimizer step/epoch，20 epochs为3,500 steps；实际每epoch处理44,800张基础图片，20epochs为896,000次基础图片呈现。日志按实际batch计数，不写成900,000。
- 同一batch禁止重复UID；没有class-balanced sampler或按类排序采样。
- 每张图片独立调用两次增强函数，不重复设置相同随机seed，不复制同一次增强结果。
- 标签绝不进入critic、negative sampler、augmentation选择或训练loss。

### 4.3 增强与归一化 [E]

按下列顺序，对两个view独立执行：

| 顺序 | 操作 | 参数 |
|---|---|---|
| 1 | RandomResizedCrop | 32×32；scale=(0.2,1.0)；ratio=(3/4,4/3)；bilinear；antialias=True |
| 2 | RandomHorizontalFlip | p=0.5 |
| 3 | RandomApply(ColorJitter) | p=0.8；brightness/contrast/saturation=0.4，hue=0.1 |
| 4 | RandomGrayscale | p=0.2 |
| 5 | ToTensor | float图像 |
| 6 | Normalize | mean=(0.5,0.5,0.5)，std=(0.5,0.5,0.5) |

这是首轮固定CIFAR配方，不冒称精确复现原版ImageNet SimCLR。原版多种增强与projector的作用参见W1。这里不用Gaussian blur、solarize、mixup、cutmix、额外噪声。

固定0.5归一化无需从selection/test估计统计量。清洁评估变换只有 `ToTensor + 同一Normalize`；保持原始32×32，不裁剪、不翻转、不color jitter。

必须保存16张实际fit图像的双视图样例，以人工检查增强。样例文件不得用于改变某个method专属增强。

---

## 5. 模型合同：三块模块的张量维度和数据流

### 5.1 Backbone

`torchvision.models.resnet18(weights=None)`，改成CIFAR stem：

```python
backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
backbone.maxpool = nn.Identity()
backbone.fc = nn.Identity()
```

保留avgpool，输出 `h: [2B,512]`。不存在分类头参与SSL训练。必须在环境日志中记录torchvision版本和最终模型字符串。

### 5.2 Projector

```text
Linear(512,512,bias=False)
BatchNorm1d(512)
ReLU
Linear(512,128,bias=True)
```

输出 `p_raw: [2B,128]`。无最后一层BN。VCS与SimCLR使用 `z = normalize(p_raw, dim=1, eps=1e-8)`；VICReg对照使用未归一化的p_raw。

一次拼接两组view：

```python
x_all = torch.cat([x1, x2], dim=0)
h_all = encoder(x_all)
p_all = projector(h_all)
z_all = F.normalize(p_all, dim=1, eps=1e-8)
z1, z2 = z_all.chunk(2, dim=0)
```

这样encoder/projector的BN政策对所有方法一致。BN使训练表示依赖同batch其他样本；不能把这份训练实现当成完全逐样本iid统计定理的直接数值证明。估计诊断时模型转eval，冻结BN状态。

### 5.3 Critic

`input = concat(z1,z2)`，维度256；**有序拼接**，不添加额外交叉项或sample ID。

```text
Linear(256,512) → ReLU → Linear(512,512) → ReLU → Linear(512,1) → tanh
```

无BN、LayerNorm、dropout、batch attention。每对样本的分数只取决于这对表示与当前参数。

初始化：前两层保持`nn.Linear`默认初始化，最后一层权重用Xavier uniform、gain=0.1，bias=0。最后一层权重不能全零，否则首个step不会向编码器传递有效梯度。这是工程初始化，不是新的方法。

真实与独立配对使用**同一个critic对象**。禁止两个独立critic、在输入中编码“正/负分支”、或利用batch排列位置识别标签。

### 5.4 三种特征必须分开保存名称

- `h`：512维，projector之前，**主线性评估对象**。
- `p_raw`：128维，未L2归一化，仅用于诊断/相应baseline。
- `z_l2`：128维，VCS或SimCLR的配对输入。

不要在实验报告中把critic分类准确率当作h的分类准确率，也不要将projector上的好结果替代encoder主结果。

---

## 6. 正负配对：K=1随机非零循环移位

### 6.1 首轮精确定义 [S1 §9；E选择其中一种]

对随机shuffle后的batch，正配对是 `(z1[i],z2[i])`。每step从 `{1,...,B-1}` 均匀抽一个shift s，负伙伴：

\[
\pi(i)=(i+s)\bmod B.
\]

负配对 `(z1[i],z2[pi(i)])`。默认只构造B个负配对，不构造B²表。

用独立CPU `torch.Generator`抽shift，并将索引搬到GPU。保存该generator的state以支持恢复。`B<2`必须明确报错。

若后续增加K，使用K个不同非零shift，`K<=B-1`；仍然对全部K×B个负分数取平均。参考实现已经支持此接口，但首轮配置只用K=1。

### 6.2 禁止的悄悄变更

- 不能直接用允许fixed point的randperm而不审计identity配对。
- 不对同class样本进行排除；同class、不同UID是当前无标签边缘采样的一部分。
- 不使用单独“困难负样本”筛选，不加入额外标签。
- 不从selection/test池取负样本。
- 不将负分支表示detach；负样本对encoder也应贡献梯度。
- 不因为后续代码方便而把view1-view2两类配对改为全部2B视图混合，未记录地改变采样定义。

### 6.3 正确解释统计含义

在iid基础样本的总体模型下，i≠j的配对可用于边缘乘积期望；同batch的多组配对彼此相关。有限数据集无放回训练又排除了同UID，因此不等于“对经验边缘做有放回抽样的完整乘积”。这是所选的off-diagonal实现，不要称其为每batch有K×B个独立基础样本，更不能声称K增大带来等量独立样本。

若以后需要严格的独立估计实验，应使用独立样本流另做E1，不改变当前SSL实现来迎合某个统计定理。

---

## 7. 训练：普通联合最大化，不是对抗训练

### 7.1 模式

首轮只有 `end_to_end_joint`：encoder、projector、critic共同降低 `-J`。不设置min-max反号，不设critic warmup，不额外更新critic五次，不用stop-gradient teacher。

```python
encoder.train(); projector.train(); critic.train()
optimizer.zero_grad(set_to_none=True)
h = encoder(torch.cat([x1, x2], dim=0))
p = projector(h)
z = F.normalize(p, dim=-1, eps=1e-8)
z1, z2 = z.chunk(2, dim=0)
stats, shifts = vcs_pair_loss(z1, z2, critic, generator=pair_rng)
loss = stats["loss"]
loss.backward()
# 先记录并检查三个模块的原始梯度；不裁剪
optimizer.step()
```

`reference/joint_step_fp32`提供对应可运行实现。正式trainer应额外处理scheduler、日志、非有限数诊断、UID和checkpoint。

### 7.2 必须通过的梯度检查

第一次真实step核查：三个模块的可训练参数全部出现在optimizer中，且不存在重复参数；每个模块至少一组参数有非零有限梯度；两视图和负分支都有梯度。optimizer step后各模块参数至少有一项实际变化。

异常时先区分：参数没注册、`detach/no_grad`、最后一层全零、loss符号、tanh饱和、错误模式。不得靠添加辅助损失绕过错误。

### 7.3 估计器单独训练只是测试模式

可以实现 `estimator_only` 供单元测试：detach表示，仅更新critic。它不属于端到端SSL主结果。冻结encoder上critic能学会配对，不等于encoder得到了改善。

若合作者稍后提供其原生交替训练代码，作为显式命名的独立配置比较；不得改变本轮run中途的训练模式。

---

## 8. 首轮训练超参数 [E]

| 配置项 | 冻结默认 |
|---|---|
| 基础图像batch | 256；单GPU；不累计梯度 |
| 精度 | FP32；TF32关闭；不启用compile |
| 优化器 | AdamW，betas=(0.9,0.999)，eps=1e-8 |
| encoder/projector学习率 | 1e-3 |
| critic学习率 | 1e-3；同一个optimizer的独立参数组 |
| encoder/projector矩阵权重decay | 1e-4 |
| bias与BN参数decay | 0 |
| critic参数decay | 0；避免未声明的额外critic收缩 |
| epoch | 20，seed=0 |
| warmup | 2 epochs，逐step线性 |
| 其后schedule | cosine降至各组base LR的0.01倍 |
| gradient clipping | 无 |
| DataLoader | workers=4，pin_memory=True，persistent_workers=False |
| 同时运行 | 最多一个GPU job |

组内weight decay也是工程优化配置，不要将有weight decay的训练称作完全无参数正则。

令step编号s从0开始，总步数T，warmup步数W。每步在更新前设置：

\[
f(s)=\begin{cases}(s+1)/W,&s<W,\\
0.01+0.99\,[1+\cos(\pi(s-W)/(T-1-W))]/2,&s\ge W.
\end{cases}
\]

各参数组学习率=`base_lr*f(s)`。`W=350,T=3500`用于标准首轮配置。实现须处理W=0和过短smoke horizon，不允许分母0。

**不把20-epoch run直接续成正式200-epoch实验。** 两者学习率horizon/warmup不同。正式200epochs从相同初始化重新开始，warmup10epochs，另外锁定配置。恢复中断run则必须沿原始horizon继续。

OOM时首先保存报错与配置；不自动减batch、开AMP或开梯度累计。资源确实不足需要另建显式变体，并对比较方法对齐。

---

## 9. 首轮对照：只做必要的三种训练

### 9.1 `vcs_qmi`

按本文方法，主loss=-J。额外critic计算量单独记录。

### 9.2 `simclr_matched`

共用encoder、projector、增强、数据划分、优化器基础配方与epoch。对2B个归一化表示计算NT-Xent，temperature=0.2：每个anchor遮蔽自己，保留真正positive在分母中，目标索引为 `(i+B) % (2B)`。[W1]

它使用2B-2个其他负视图/anchor，而VCS首轮K=1。必须记录这个不同；不能据此声称统计采样预算完全相同。这里比较的是各目标的首轮具体实现，不是证明单位负样本效率。

该run不创建未使用的critic。不要把VCS的tanh直接塞进SimCLR再称之为标准SimCLR。

### 9.3 `vicreg_matched_128`

共用encoder和128维projector，但**输入VICReg损失的是p_raw，绝不L2归一化**。损失系数invariance=25、variance=25、covariance=1；std中eps=1e-4，方差/协方差分母B-1。[W2]

该设置是小projector的公共组件对照，名字始终保留`matched_128`。它不是原论文完整native配方；不能把这份首轮结果当作已经充分调优的VICReg上限。

### 9.4 `rpc_111_identity`

只做数值与梯度等价单元测试：在相同T和相同配对分数下，RPC的(1,1,1)与J一致。[W3]

不要消耗一份独立GPU预算重复训练它，再将浮点/seed差异描述为VCS超过RPC。general/tuned RPC、native VICReg、SSL-HSIC等留到下一阶段，首轮报告不能据此宣称已超过这些完整方法。

### 9.5 随机初始化参照

保存相同seed的**epoch0 backbone checkpoint**，对它做一次相同规则的clean frozen linear probe。该probe只有分类头训练，没有SSL。这用来判断encoder是否真的得到改善。

随机卷积特征可能不止10%准确率；不要把10%随机猜测当作随机网络特征的必然结果。

---

## 10. 评估：主对象是冻结的h，不是critic分数

### 10.1 Encoder状态必须彻底冻结

`encoder.eval()`，所有参数`requires_grad=False`，在`torch.no_grad()`下提取h。BN running mean/var和num_batches_tracked也不能改变。评估前后对encoder state_dict做校验。

评估优先使用从checkpoint独立加载的模型副本。不能在训练实例上永久设置`requires_grad=False`后，仅调用`.train()`就以为恢复了训练；`.train()`不会重新开启参数梯度。若复用同一个实例，必须完整恢复每个参数的requires_grad、每个模块的train/eval模式、BN buffer和训练RNG，并测试恢复后的三个模块仍有梯度。

不运行projector/critic训练。feature cache记录checkpoint SHA、split SHA、transform SHA、dtype和UID顺序。缓存键不能只使用run名或epoch整数。

### 10.2 Pilot主线性评估：固定budget、固定末尾checkpoint

- 预训练checkpoint：epoch20；另评估epoch0作参照。
- clean transform，提取fit45k与selection5k的原始512维h，float32缓存。
- probe：`Linear(512,10,bias=True)`，只有线性头训练；不L2归一化h、不额外feature标准化。
- 标签：只用于probe；fit标签训练头，selection标签评分。
- SGD，lr=0.1，momentum=0.9，weight_decay=0，batch=256。
- probe训练100epochs，cosine降到base LR的0.001倍；seed=20260925；头以PyTorch默认初始化。
- probe自身也取第100epoch，不按最高selection accuracy挑epoch。
- 输出 `linear_val_top1_pct` 与交叉熵；首轮无 `test_top1`。

这些数值是开发默认，不是最佳probe宣称。若三种方法都出现probe训练明显不足，单独报告，再用对所有方法共同的验证选择规则处理。不得只为VCS搜索更优probe。

### 10.3 kNN辅助监控

epoch0、5、10、20评估。使用fit作为有标签memory bank，selection作为query；h做L2归一化。k=200，temperature=0.1；按余弦top-k加权投票：

\[
w_j=\exp((s_j-\max_k s_k)/0.1).
\]

逐query块计算，默认256；不要构建巨大的全量query×train矩阵。fit与query UID不重叠，无需通过自邻居制造成绩。输出 `knn_val_top1_pct`，不根据它自动追加训练。

### 10.4 Critic held-out诊断（仅VCS）

在selection上用与训练相同分布的两次随机增强，模型整体eval，冻结所有权重和BN。固定eval seed，做4次不同增强/错配重复，记录每次J/R/score矩，再报告均值与重复SD。

batch=256，最后不足batch保留；若仅剩1张，单独合并到前一批，不丢数据且不做self-pair。按正/负实际计数累计总和后求总体均值，不直接无权平均不同大小batch的均值。

这里的 `heldout_J` 是当前critic在未参与SSL fit的图像上的诊断值，不是refit后的supremum、不是真实S，也不是梯度证书。selection会用于后续调参，因此不能把多轮开发后的结果称为完全 untouched final evaluation。

eval不得消耗训练RNG后改变接下来的训练轨迹：使用隔离评估进程或保存/恢复全部相关RNG、使用独立eval generator。仅给函数传一个seed但仍让worker共享训练RNG，不算完成隔离。

### 10.5 正式测试留待下一阶段

先锁定SSL配方与probe规则，再选择是否在完整50k train上重训并评估官方10k test。该协议须另行统一，不允许只让表现好的方法重训，也不能对test做LR/epoch选择。

---

## 11. 训练诊断：第一天就留下足够信息

### 11.1 每50步记录

`loss`, `J_raw`, `R_binary`, `t_pos_mean`, `t_neg_mean`, `t_pos_second`, `t_neg_second`, `sat_pos_frac`, `sat_neg_frac`；所有模块的LR；encoder/projector/critic的梯度L2范数；批次耗时；基础图片数、正负配对数、当前shift。

非VCS方法没有J/R时用null，不伪造同名MI指标。SimCLR记录NT-Xent，VICReg记录三个分项。

梯度统计在backward之后、optimizer之前采集；首轮不裁剪。未来若开FP16，必须先unscale再记录。[W5]

### 11.2 表征谱与坍塌诊断

用同一selection集合中UID排序前4096张clean图像计算h、p_raw、z_l2，各run使用同一ID列表/哈希。先按样本中心化，协方差分母N-1，记录：

- 平均每维方差、协方差trace、样本范数mean/SD；
- 特征值及最大特征值占比；
- `effective_rank = exp(-sum(pi*log(pi)))`，pi为归一化非负特征值；
- 若trace为0，effective_rank记0并加`zero_covariance=true`，不对空分布造熵；
- 数值负特征值只在容差内置0；超出容差则报告数值问题。

不同batch-size实验不能在各自训练batch上算rank再直接比较，否则会混入样本数上限效应。归一化后的样本范数恒约为1不代表没有坍塌。

操作报警而非理论结论：固定评估集上trace<=1e-8或effective_rank<=2，连续多次出现时标记`COLLAPSE_SUSPECTED`；只报警，不自动加正则或删掉该seed。

### 11.3 训练/验证间隙

保存训练与selection的正负分数直方图及原始汇总。J很高而h的linear/kNN不改善，必须如实报告为“配对判别改善未转化为表示质量”，不能只展示critic accuracy。

### 11.4 成本

记录基础图片数、视图数、正负对数、optimizer steps、参数量、训练秒数、评估秒数、峰值GPU memory allocated/reserved。CUDA计时时正确同步或用CUDA events。

吞吐量区分 `images/s`（基础图像）和 `views/s`（两倍），不混用。不要预估A100 GPU天数当作实测；首轮用实际warmup后步耗时报告。

固定checkpoint的多batch梯度方差研究预留接口，但不强制首轮启动庞大扫描。若执行，固定模型与BN状态，说明eval-mode梯度诊断不同于train-mode minibatch BN。

---

## 12. 精度与分布式政策

首轮FP32、单GPU、TF32关闭；矩统计至少float32，测试使用float64核对恒等式。`torch.amp`等API以服务器实际版本为准，不强制升级。[W5]

下一阶段若引入AMP，必须先比较固定batch上的FP32/AMP loss和梯度误差；critic前向及T²归约建议置FP32，报告精度配置。不能只在最后对已经溢出的分数`.float()`就声称整个critic用了FP32。

下一阶段DDP需要单独规定local/global负样本、all-gather是否保留梯度、重复样本padding、loss缩放及SyncBN政策。当前代码不得自动使用检测到的全部GPU。

**梯度累计不等于增大本目标的配对batch**；它不会自动增加当前critic看到的负配对池，也不改变每个micro-batch的BN统计。

---

## 13. 随机性、恢复与checkpoint

### 13.1 RNG与公平初始化

记录Python、NumPy、PyTorch CPU和所有CUDA RNG；固定DataLoader generator、worker_init_fn、pairing专用CPU generator、probe和eval RNG。[W6]

首轮method之间使用同一encoder/projector初始state_dict和相同数据划分。额外初始化critic不得改变后续数据shuffle/增强RNG。独立初始化并随后显式重置/隔离数据RNG；保存初始权重哈希验证。

默认 `cudnn.benchmark=False`，记录确定性设置与硬件。跨PyTorch版本/硬件不能承诺逐bit一致。[W6]

### 13.2 保存内容

```text
encoder_state, projector_state, critic_state (if applicable)
optimizer_state, scheduler_state
run_id, code_commit, config_hash, manifest_hash
completed_epoch, optimizer_step, intended_total_steps
rng_python, rng_numpy, rng_torch_cpu, rng_torch_cuda
rng_loader_generator, rng_pair_generator
precision_flags, model_hparams, best_metric_policy=None
```

保存`initial.pt`，每epoch原子写`last.pt`，保留epoch5/10/20快照。写临时文件再rename，禁止写入中断的半文件冒充完整checkpoint。处理SIGTERM时尽量保存已完成的epoch；不要夸称正在进行的batch已精确恢复。

默认只承诺epoch边界可恢复。DataLoader worker预取、随机增强和mid-epoch sampler位置若未全部恢复，不得声称bitwise mid-epoch continuation。优先从上一个完整epoch边界重做，并计入额外成本。

resume时拒绝配置/manifest变化；允许变更输出目录，但不能改变总horizon、split、batch、K、loss、模型。想改配置必须新run，不能修改原run历史。

---

## 14. 日志与结果目录

建议结构（可映射到现有repo，须提供映射表）：

```text
src/vcs_ssl/
  data/cifar.py, data/splits.py, data/transforms.py
  models/backbone.py, models/projector.py, models/critic.py
  losses/vcs.py, losses/simclr.py, losses/vicreg.py
  pairing.py, train.py, preflight.py
  evaluate.py, diagnostics.py, checkpoint.py
configs/
tests/
reports/
outputs/<run_id>/
  config.resolved.yaml, environment.json, sources.json
  code_state.txt, manifest.json, manifest.sha256
  checkpoints/initial.pt, last.pt, epoch_005.pt, epoch_010.pt, epoch_020.pt
  logs/steps.jsonl, epochs.jsonl, stderr.log
  evaluations/linear.json, knn.json, critic_holdout.json, spectrum.npz
  artifacts/view_examples.png
  status.json, summary.json, report.md
```

每条epoch结果必须带：

```text
run_id, method, stage, seed, code_commit, config_hash, split_hash,
physical_batch_images, views_per_image, K, pair_sampling, world_size,
encoder_dim, projector_dim, critic_params, objective_target,
optimizer_step, seen_base_images, J_raw, heldout_J, R_binary,
linear_val_top1_pct, knn_val_top1_pct, h_effective_rank, z_effective_rank,
train_seconds, eval_seconds, peak_allocated_mb, peak_reserved_mb,
status, failure_reason
```

状态至少区分：`NOT_RUN, RUNNING, COMPLETED, FAILED_NUMERICAL, FAILED_INFRA, STOPPED_BUDGET`；另设诊断标记`COLLAPSE_SUSPECTED`。缺结果用null，不能用0、空字符串或虚构数值占位。保存所有失败seed和重跑记录。

---

## 15. 单元测试与集成验收

### 15.1 已附29项本地CPU测试

具体测试代码是可执行依据，覆盖：

| 类别 | 覆盖点 |
|---|---|
| 代数 | J=1-R；E[T²]位置；J/R边界；负J不裁剪；重复负样本不改变权重 |
| 梯度 | 分数导数符号；RPC(1,1,1)数值及梯度一致；负分支反传 |
| 真值 | 二状态离散oracle S=0.12与回归gap |
| 采样 | 非零shift、B=2、K边界、无self、可恢复专用RNG、全shift覆盖off-diagonal |
| 模块 | critic pointwise/bounded；三个模块收到梯度并更新；estimator_only隔离 |
| 稳健性 | 非法shape/空scores/integer输入、half归约升FP32 |
| 对照 | SimCLR手算分母和全相同特征值；VICReg常数特征的方差惩罚 |
| 评估 | frozen encoder参数与BN buffer不更新 |

执行：`python -m pytest -q`。本地通过不代替服务器复跑，不代表CUDA/DDP已测。

### 15.2 服务器必须补的集成测试

1. **真实ResNet形状**：32×32输入输出h512/p128/z128，critic输出B与K×B。
2. **真实梯度与optimizer覆盖**：每个可训练模块参数覆盖正确，正负两路梯度存在。
3. **标签不可访问**：SSL训练batch无标签；修改class label不改变固定输入下的loss/配对。
4. **split隔离**：fit/selection/test UID集合无交集；训练日志无selection/test UID。
5. **增强独立**：同一图片的两view不是同一个随机变换对象结果；样例可查看。
6. **eval不污染**：线性probe不更新encoder参数/BN；critic验证不消耗后续训练RNG。
7. **checkpoint恢复**：小型单worker固定随机流的连续N步与中断恢复结果一致；真多worker只声明已实现的恢复级别。
8. **配置严格性**：未知字段报错；`extra_regularizers=[]`；`K=1`；`negative_detach=False`；`weights=None`。
9. **无真实测试集访问**：P2/P3启动参数中official_test=false，训练器不会创建其Dataset。
10. **收益来源**：保存epoch0与epoch20 h评估；不能仅报告critic。

这些测试失败时，不进入P3。若某项只部分完成，报告`PARTIAL`及未覆盖情况，不写“全部通过”。

---

## 16. 阶段任务与停止条件

| 阶段 | 内容 | 预算与进入条件 | 输出 |
|---|---|---|---|
| P0 | 环境、repo、数据核查，创建manifest | 只读核查+新增项目文件 | preflight、差异表 |
| P1 | 集成reference，复跑29项，补集成测试 | CPU/小GPU张量 | junit/log、git diff |
| P2 | 三方法真数据100-step smoke | 单卡；默认B256；不用于排名 | 梯度、张量、内存、恢复与耗时 |
| P3 | VCS/SimCLR/VICReg各20epochs，seed0 | P1/P2通过；单GPU顺序运行 | 固定epoch20 probe、kNN与诊断 |
| P4 | 汇总并停止 | 不增加超参run | 结果、错误归因与下一阶段建议 |

P2是独立smoke run；不能接着作为P3前100步。smoke scheduler以自己的100steps配置并记录，不混入正式loss曲线。

每个阶段允许修复明确实现错误后重跑；保留失败run和补丁原因。不得无限换seed/学习率直到成功，不能把重复试验隐藏成一次运行。

### 16.1 立即停止并报告

非有限loss/梯度；数据污染；错误负样本；optimizer漏模块；错误loss系数；GPU不足导致OOM；checkpoint/磁盘损坏；环境与原始数据无法确认。

保存出错step、UID、shift、随机state、分数范围、当前权重和配置。不要自动skip NaN batch、替换损失或继续覆盖last.pt。

### 16.2 警告但不擅自换方法

J接近上限、tanh大量饱和、表示rank低、linear不优于epoch0、VCS落后对照。这些是要分析的研究现象，不自动说明代码错，也不授权加新模块。

20epochs只用于开发诊断，不能据此宣称最终方法胜负。一个没有NaN但没有有用表示的run，不被总结为“稳定表征学习成功”。

---

## 17. 要实现的命令接口

**下列是服务器需实现的目标CLI，不是本starter已经包含的可直接训练命令。** 若复用现有trainer，提供一一对应命令，不允许只给未验证的伪代码。

```bash
# 0. 解压starter后，先运行已有的核心测试
python -m pytest -q

# 1. 下面四类入口由server agent实现/映射
python -m vcs_ssl.preflight --config configs/cifar10_pilot_vcs.yaml
python -m vcs_ssl.train --config configs/cifar10_pilot_vcs.yaml --smoke-steps 100
python -m vcs_ssl.train --config configs/cifar10_pilot_simclr.yaml --smoke-steps 100
python -m vcs_ssl.train --config configs/cifar10_pilot_vicreg.yaml --smoke-steps 100

# 2. P2通过后执行三份P3配置，按资源政策提交，不在login node长跑
python -m vcs_ssl.train --config configs/cifar10_pilot_vcs.yaml
python -m vcs_ssl.train --config configs/cifar10_pilot_simclr.yaml
python -m vcs_ssl.train --config configs/cifar10_pilot_vicreg.yaml

# 3. 评估入口必须读取run内已冻结的checkpoint和manifest
python -m vcs_ssl.evaluate --run-dir "$RUN_DIR" --checkpoint epoch_020.pt --protocol pilot

# 4. 汇总不能丢弃失败run
python -m vcs_ssl.summarize --stage P3_pilot --output reports/P4_ssl_pilot_report.md
```

命令中的路径由P0实际探测后填入；shell环境变量应在解析后验证目录存在。调度脚本必须保存到repo，记录job id，不仅在交互终端临时运行。

---

## 18. 给用户的回报格式

`reports/P4_ssl_pilot_report.md` 应按以下顺序写：

### A. 实现核对

代码commit、是否使用原始公式、是否普通joint更新、实际网络、负采样、是否有额外损失、哪些配置是工程选择、与合作者原代码有无差异。

### B. 测试

参考29项通过数、服务器新增测试逐项状态、数据无泄漏证据、两视图梯度、checkpoint恢复级别。不能仅说“preflight pass”。

### C. 运行表

| run | method | seed | epochs | linear-val(%) | kNN-val(%) | heldout-J | h-rank | train time | peak GPU MB | status |
|---|---|---|---|---|---|---|---|---|---|---|
| 实际run_id | 实际配置名 | 0 | 实际完成数 | 实测/null | 实测/null | 实测/null | 实测/null | 实测 | 实测 | 状态 |

另报每个seed初始h的linear/kNN和相同协议下的增量。表格只填观测值，不能填论文预期。

### D. 现象解释

分别回答：critic是否学到；encoder是否改善；是否发生训练/验证gap；是否饱和/坍塌；对照是否正常；成本主要在哪里。小样本曲线只能提出假设，不能直接归因于理论性质。

### E. 下一步

只提出一个有优先级的下一轮：例如保持方法做200epochs确认，或先修复某个具体实现问题。不得一次性提交CIFAR100、ImageNet、EEG、图文和十组新正则。

在P4之后等待用户对结果和资源预算的决定，不后台自行扩展实验。

---

## 19. 下一阶段的边界（只做规划，不执行）

在数据/代码/评估正确后，可以将固定配方从头训练200epochs，warmup10，确认多个完整预训练seed。再根据验证集与公平预算讨论LR、critic容量、K、joint/常规alternating。

不要完整笛卡尔积扫描；一次改变一个有明确问题的因素。若需要baseline的native配方和tuned RPC，作为独立命名方法加入，不用matched128的短run充当其最佳表现。

之后才决定CIFAR-100、ImageNet-100和EEG。原理论与训练量不因数据集扩展而自动改变。

---

## 20. 文献与来源定位

此处仅为agent查公式/API，不授权重写研究方向。完整URL和来源哈希也保存在`sources.json`。

- **S1**：合作者 `Variational_CS_QMI_Research_Plan (1).pdf`，§4 Eq.(5)与Eq.(6)，§5 Eq.(7)，§6训练量，§7 Eq.(12)，§9 Eq.(17)，§10稳定性限制。
- **S2**：`VCS_QMI_CVPR_Experimental_Protocol.md`，§5视觉训练与评估、§8记录字段。
- **W1**：Chen et al., *A Simple Framework for Contrastive Learning of Visual Representations*, ICML 2020。双视图、projector、NT-Xent与冻结线性评估。`https://proceedings.mlr.press/v119/chen20j.html`
- **W2**：Bardes et al., *VICReg*, ICLR 2022。方差/不变性/协方差目标。`https://arxiv.org/abs/2105.04906`
- **W3**：Tsai et al., *Self-supervised Representation Learning with Relative Predictive Coding*, 2021。这里只使用目标的代数核对；不是完整算法复现。`https://arxiv.org/html/2103.11275v3`
- **W4**：CIFAR官方页面，核对数据规模和原始下载/校验信息。`https://www.cs.toronto.edu/~kriz/cifar.html`
- **W5**：PyTorch官方AMP recipe，precision与GradScaler记录。`https://docs.pytorch.org/tutorials/recipes/recipes/amp_recipe.html`
- **W6**：PyTorch官方Reproducibility / DataLoader文档，RNG、worker与可复现性限制。`https://docs.pytorch.org/docs/stable/notes/randomness.html`
- **W7**：torchvision RandomResizedCrop API，显式参数而非依赖随版本变化的默认值。`https://docs.pytorch.org/vision/stable/generated/torchvision.transforms.RandomResizedCrop.html`

**首轮完成的判据：一个能审计、能恢复、有真实冻结表征评估的VCS-QMI SSL实现，而不是一条被额外技巧修饰过的高分曲线。**
