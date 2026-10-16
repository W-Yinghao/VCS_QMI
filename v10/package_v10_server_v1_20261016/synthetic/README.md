# V10 T03：可观测离散变量的受控真值生成器

本目录提供两个有限离散联合分布、精确总体 `p/q/eta/S`、彼此独立生成的 FIT/CAL/EVAL 基本观测及实际检查结果。仅依赖 NumPy。它用于估计误差和有限函数类表达能力实验，不包含神经训练器、GPU 计时或自然图像结果。

## 1. 数据对象与两种难度

观测直接为 `X,Y ∈ {0,…,m−1}` 的整数，没有只存在于生成器内部的潜在标签。两个家族的边缘分布均为均匀分布，因此 `Q(x,y)=P_X(x)P_Y(y)=1/m²`。

### `two_block`：平衡的两块关系

`m` 必须为偶数。对前一半状态令 `s(x)=1`，后一半令 `s(x)=−1`：

\[
p(x,y)=\frac{1+\rho s(x)s(y)}{m^2},\qquad
S=\frac{\rho^2}{4-\rho^2}.
\]

`rho>0` 时，`p−q` 的矩阵秩为 1。这里的两块是可观测状态集合的划分，不是另存的训练标签。

### `diagonal_mixture`：多状态对应关系

\[
p(x,y)=\frac{1-\rho}{m^2}+\frac{\rho}{m}\mathbf1[x=y],
\]

\[
S=\frac{\rho^2(m-1)}{(2+\rho(m-1))(2-\rho)}.
\]

`rho>0` 时，`p−q` 的矩阵秩为 `m−1`。因此在相同 `m` 下，它能检验小维度结构化关系类是否受到表达能力限制。

所有配置都使用合作者原来的对象：

\[
M=(P+Q)/2,\quad\eta=(p-q)/(p+q),\quad S=E_M\eta^2,
\]

\[
J(T)=E_PT-E_QT-\tfrac12E_PT^2-\tfrac12E_QT^2,
\qquad S-J(T)=E_M(T-\eta)^2.
\]

`truth.npz` 的真值由完整有限表计算，另存上述闭式 `S_closed_form` 供独立核对。这里的“精确”指总体解析对象，不是无浮点舍入；文件为 float64，没有 Monte Carlo 真值误差。

## 2. 一条命令生成默认 12 个条件

```bash
python src/v10_synthetic.py \
  --output-dir /your/output/t03_discrete_seed20261016 \
  --seed 20261016 \
  --families two_block diagonal_mixture \
  --m 8 32 \
  --rho 0 0.25 0.75 \
  --n-fit 4096 --n-cal 2048 --n-eval 8192
```

`--n-train` 是 `--n-fit` 的别名。全部参数可显式设置；上述 family、m、rho 和样本量也是默认值。新运行需要一个新的或空的输出目录，程序拒绝覆盖非空目录。

## 3. 输出结构与角色

每个 `case_id` 下有：

| 相对路径 | 内容 | 用途 |
|---|---|---|
| `fit/samples.npz` | `x,y,base_id` | 训练；不含 oracle 数组或逐样本 oracle 标签 |
| `fit/metadata.json` | 独立基本观测数、RNG 流、文件哈希 | 训练可复现性 |
| `cal/samples.npz`、`cal/metadata.json` | 新生成的 CAL 观测及元数据 | 按预定的 sample-only 规则选配置 |
| `eval/samples.npz`、`eval/metadata.json` | 新生成的 EVAL 观测及元数据 | 最终独立样本读数 |
| `evaluator_only/truth.npz` | `p,q,px,py,eta,S,S_closed_form` | 仅评估器和真值检查 |
| `evaluator_only/truth.json` | 真值定义、解析公式、哈希 | 仅评估记录 |

输出根目录 `manifest.json` 记录配置、角色文件路径、样本数和 seed 方案。`x,y` 为一维 `int32`，`base_id` 为一维 `int64`；每行对应一次联合分布独立抽样。

### 训练读取接口

将 `src` 加入 Python 导入路径后：

```python
from v10_synthetic import load_samples

fit = load_samples(case_directory, "fit")
x, y = fit["x"], fit["y"]
```

`load_samples` 只打开指定角色的 `samples.npz`，遇到额外字段会报错，不读取真值目录。训练器可以对整数观测使用 one-hot、embedding 或预定特征；所有方法必须使用相同可观测输入与数据预算。

评估器另行使用 `load_evaluation_truth(case_directory)`。例如，对整个 `m×m` 状态表的预测 `T`，精确后验均方误差为 `sum(0.5*(p+q)*(T-eta)**2)`。真值不能参与 FIT、超参选择、早停、函数类选择或 solver 预算选择。这里提供的是文件与 API 角色分隔，不是操作系统访问控制。

## 4. 独立基本样本和负对

本工具只生成来自 P 的独立基本观测 `(X_i,Y_i)`，不预先膨胀负对。对 `n` 条基本观测形成 `n(n−1)` 或 `n²` 个配对，仍然只使用这 `n` 条基本观测；这些配对共享随机变量，不能以配对数代替统计独立样本数。

对固定 critic，独立基本观测的非对角配对 `(X_i,Y_j), i≠j` 边缘服从 `P_XP_Y`。包含对角的经验乘积与不含对角的 U 目标不同，训练器必须显式登记规则。若 critic 在同一数据上拟合，不能把训练后经验最大值称为无偏总体估计。

EVAL 文件与 FIT/CAL 独立生成。精确真值表可用于已冻结模型的总体加权误差；EVAL 样本可用于独立经验读数。两种读数应分开报告。

## 5. 随机流如何固定

每个 `(family,m,rho,role)` 有一个单独的 `SeedSequence`，其 `spawn_key` 由固定版本号、家族代码、`m`、`rho` 的 binary64 两个 uint32 字及角色代码确定，再交给 PCG64。不同角色不共享可变 RNG。

不使用 Python `hash()`，也不使用依赖遍历顺序的 `spawn()` 调用。反转家族、`m`、`rho` 的遍历顺序不会改变某一条件的样本。改变 FIT 的抽样数不消耗 CAL/EVAL 的随机流；同一条件同一角色增大抽样数时保留已有前缀。

元数据保存根 seed、spawn key、四个 uint32 的状态指纹和基本观测 ID namespace。不同角色可以抽到完全相同的整数值；独立随机观测不要求取值集合互斥。精确复现还应保留 NumPy 版本；本轮使用 NumPy 2.3.5、Python 3.12.14。

## 6. 已运行的检查

本包保留小型实例 `validation/example_data`：12 个条件，每个条件 FIT/CAL/EVAL 分别为 257/131/509 条基本观测，总计 **10,764 条独立基本观测**。

```bash
python tests/check_synthetic.py \
  --suite-dir validation/example_data \
  --output validation/CHECK_RESULTS.json
```

实际结果为 **138 项通过**：

| 检查 | 实际结果 |
|---|---:|
| p/q 归一化、非负、边缘一致性 | 最大误差 0 |
| 有限表 S 与独立闭式比较 | 最大误差 1.11e−16 |
| 原 `S−J=E_M(T−eta)^2` 与 `J(eta)=S` | 最大误差 2.22e−16 |
| rho=0 独立性 | 所有默认独立条件通过 |
| `p−q` 的秩 | 两块 1；对角混合 `m−1` |
| 每条件额外 30,000 次抽样的已知事件概率 | 最大标准化误差 3.03；预设容许值 7 |
| 36 个角色流及 ID namespace | 全部不同 |
| 反转条件遍历顺序后的实际文件再生成 | 样本逐位相同 |
| FIT 数量改变与 CAL RNG 分离 | 通过 |
| FIT 读取器拒绝 oracle 字段 | 通过 |
| 不覆盖已有输出 | 通过 |

概率、原 gap 和抽样事件均由实际结果核对；随机流检查验证实现与复现行为，不是用有限检验数学证明 PRNG 的统计独立性。

## 7. 覆盖限制

只实现可观测离散变量。没有加入 Gaussian/cubic 通道，也没有用未经说明的 Monte Carlo 数值当作解析 S。有限表中受限函数类的结果不能直接外推到自然图像、连续表示、训练后泛化或配准性能。现有输出没有估计器优劣结论，也没有计算资源优势结论。

文件：`src/v10_synthetic.py`、`tests/check_synthetic.py`、`validation/CHECK_RESULTS.json`、`validation/example_data/`。
