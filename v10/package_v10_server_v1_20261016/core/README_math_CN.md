# V10 MF-VCS：数学后端与可执行验证

日期：2026-10-16。**这是执行候选后端；本地已经完成 47 项独立 NumPy 数学检查，尚未执行真实 PyTorch 后端。**

本地 primary runtime 为 Python 3.12.14、NumPy 2.3.5、SciPy 1.17.0；`torch` 未安装。没有联网安装、远程训练或 GPU 作业。`verify_torch.py` 和 `benchmark_torch.py` 必须由 server agent 在已授权、已有 PyTorch 的环境中运行；目录中没有伪造的 Torch/GPU 成绩文件。

## 1. 文件与证据

| 文件 | 内容 / 状态 |
|---|---|
| `mf_vcs.py` | 可复用 PyTorch 后端；静态语法检查通过，真实执行待 server 核对 |
| `verify_numpy.py` | 独立 dense pair-feature oracle、矩式、有限差分、SciPy 盒 QP 与 NumPy 求解算法检查 |
| `MATH_NUMPY_AUDIT.json` | 已实际运行：47 / 47 通过；明确不等于 PyTorch 验证 |
| `verify_torch.py` | 真实 Torch 值、特征/W 梯度、HVP、盒求解、M1、checkpoint 分块与输入拒绝测试；尚未运行 |
| `benchmark_torch.py` | 默认小规模 smoke；完整规模需 `--full`；尚未运行 |

不要沿用 v0.1 压缩包中既有 `MATH_CHECK_RESULTS.json`，把它当成当前后端已经执行的证据。当前检查与旧参考的检查分别记录。

## 2. 保持的数学对象

\[
J(W)=E_PT-E_QT-\tfrac12 E_PT^2-\tfrac12 E_QT^2,
\qquad T(x,y)=a(x)^TWb(y).
\]

`a`、`b` 的每行非负且和为一，`W` 每个元素在闭盒 `[-1,1]` 中，因此 `T` 是 `W` 元素的凸组合。浮点实现只存在归一化和乘加舍入的误差；不在输出端再施加 tanh。该有限类不自动包含真实最优 critic。

所有 API 的 `ridge` 默认 **0.0**。显式指定 `ridge=lambda>0` 时：

\[
F(W)=J(W)-\tfrac\lambda2\|W\|_F^2.
\]

返回 `ObjectiveTerms.j`、`.ridge_penalty`、`.f`，以免把正则化后的 F 报成原始 J。经验 J 不强制截断为非负，也不被标为总体 S 的逐批下界。CS 变换和总体估计在此后端之外处理。

## 3. 实际支持的 Q

| 模式 | Q / 分母 | 正对数要求 | 含义 |
|---|---|---|---|
| `q_mode="v"`，默认 | 同一场景内所有 `n²` 对 | `n>=1` | 包含原始对角，等于普通随机 permutation 的条件平均 |
| `q_mode="u"` | 同一场景内所有 `i!=j`，`n(n-1)` | `n>=2` | 明确更换经验 Q；不是普通 permutation 的条件平均 |
| `q_mode="independent"` | 独立指定的两边池全部 `n_x n_y` 对 | 正对至少一条，两边池非空 | 不删除同号索引；统计独立性由数据供应方保证 |

每个 `SceneBatch` 只包含一个图像/场景的 P 和 Q。`objective_terms` 逐场景计算，再按调用方显式提供的固定非负权重加权；权重必须和为一，不按场景大小悄悄改权重，不自动归一化，不允许可训练 Tensor 权重。

`partition_scenes(A, B, graph_ids)` 为 V/U 场景生成本地图内分组，按排序后的 scene ID 返回结果。权重必须按这个顺序传入。调用方若直接把跨图特征塞进同一个 `SceneBatch`，后端无法从特征值判断错误；真实 G1 adapter 必须保留图像 ID 并使用分组接口或显式本地对象。

**不支持：任意 forbidden edges、空间邻域禁止边、每行归一化 eligible-edge Q、均匀 eligible-edge Q、sampled derangement U、跨图负对模式。** `forbidden_edges` 非空时立即报 `NotImplementedError`。v0.1 的一项 NumPy 禁止边核对不能代替完整后端；需要这些 Q 时先完成准确的边集合、归一化和梯度实现，禁止降级为 V/U 后仍沿用原名称。

支持 float32 / float64。为避免无声明的消减误差，不支持半精度/AMP head 计算。GPU 实际行为尚未验证。`check_w=False` 和 `check_permutations=False` 只用于输入已验证的计时/训练循环，不提供任何修复作用。

## 4. 核心矩与 HVP

对 Q 的两个特征池，令 `u=sum(A)`、`v=sum(B)`、`G_A=A.T@A`、`G_B=B.T@B`。V / 独立池的两项为：

\[
q_1=u^TWv/(n_xn_y),\qquad
q_2=\langle W,G_AWG_B\rangle/(n_xn_y).
\]

U 模式分别从分子减去配对 `t_i`、`t_i²` 的和，分母改为 `n(n-1)`。不能先做输出 tanh 再沿用此二阶矩公式。

固定特征下：

\[
F(W)=\langle\Delta,W\rangle-\tfrac12\langle W,H_\lambda(W)\rangle,
\]

正对部分 `H_P(V)=A.T @ (B * diag(A @ V @ B.T)[:,None]) / n`；完整乘积负对部分为 `G_A @ V @ G_B / denominator`，U 再删除同一正对贡献。全部场景按同样固定权重加权，最后增加 `lambda * V`。不构建 `(r*s) × (r*s)` Hessian。

`QuadraticHead` 把当前特征值 detach 后 clone，以便求解时不被调用方后续存储修改影响；其后端只存正对特征和局部 Gram 矩。**这一 snapshot 仅适用于建立它时的特征、Q 和场景权重。** 外层更新特征后必须重新建立，不能把旧 gap 当成新状态的证书。

## 5. 盒求解与 gap

`solve_box_head` 使用投影上升，可带单调重启的 FISTA 加速。步长来自 `trace(H0)+lambda` 的曲率上界；这里不是 `trace(H_lambda)`。有限预算可能达不到容差，返回 `converged=False` 和 `iteration_budget_exhausted`，不会把达到最大迭代数称为已收敛。

对当前 F 的梯度 `g`，返回：

\[
\mathcal G(W)=\sum_{kl}|g_{kl}|-\langle g,W\rangle.
\]

计算时用逐元素非负形式，避免两个大和相减：`g>=0` 时累加 `g*(1-W)`，否则累加 `-g*(1+W)`。

对当前固定特征、当前经验 F、当前盒约束，凹性给出 `F* - F(W) <= gap`。这是数值精度范围内的一阶优化不足检查，**不证明总体 S 估计正确、critic 类充分、泛化改善、几何配准可识别或近似包络梯度足够准确**。无正则时 H 可奇异；不额外承诺唯一解。

## 6. 最小调用示例

### 6.1 M0：完整矩 J、投影 head

```python
import torch
from mf_vcs import SimplexBilinearCritic, partition_scenes, objective_terms

critic = SimplexBilinearCritic(x_dim=64, y_dim=64, r=16, s=16).to(device)
optimizer = torch.optim.Adam(critic.parameters(), lr=1e-3, weight_decay=0.0)

# x/y 是本步配对描述子；graph_ids 保留每行所属图像。
a, b = critic.features(x, y)
scenes = partition_scenes(a, b, graph_ids, q_mode="v")
# 明确采用每图等权的例子；真实协议按冻结的场景权重填。
weights = [1.0 / len(scenes)] * len(scenes)
terms = objective_terms(scenes, critic.W, weights=weights)
optimizer.zero_grad()
(-terms.j).backward()
optimizer.step()
critic.project_head_()
```

`SimplexBilinearCritic` 只是最小 learned map，不是替代真实完整图像编码器的应用模型。其两侧线性 map 可被明确配置的网络替代，但输出必须保持 simplex。W 的零初值不要求两侧特征恒定；禁止把 A/B 初始化为全样本完全相同向量后假定能学出依赖。

### 6.2 独立边缘池

```python
from mf_vcs import SceneBatch, objective_terms

scene = SceneBatch(
    a_pos, b_pos, q_mode="independent",
    a_marginal=a_independent_pool, b_marginal=b_independent_pool,
    scene_id="specimen_001",
)
terms = objective_terms([scene], w, weights=[1.0], ridge=0.0)
```

### 6.3 M1：同 tensors、同 F、固定已解 W 的 outer

```python
from mf_vcs import profiled_objective

# 一次 feature forward；不在求解后重抽 dropout/augmentation 或刷新 BatchNorm。
a, b = critic.features(x, y)
scenes = partition_scenes(a, b, graph_ids, q_mode="v")
profile = profiled_objective(
    scenes, weights=weights, ridge=0.0,
    initial=critic.W.detach(), tolerance=1e-7, max_iterations=5000,
)
feature_optimizer.zero_grad()  # 只管理编码器和两侧 feature maps
(-profile.terms.f).backward()
feature_optimizer.step()

# 仅作为下一步 warm start；当前 gap 不延伸到已更新的特征。
with torch.no_grad():
    critic.W.copy_(profile.solution.w)
```

内层冻结的是当前 A/B 数值的 snapshot；outer 使用传入的同一原始 tensors 及其原有梯度图，固定已解 W，并计算同样的 F。没有另一个可悄悄换数据或换目标的 `outer_scenes` 参数。默认 `require_converged=True`，达不到 gap 时抛错；若协议允许近似 head，可显式关闭并保存实际 gap，但不能继续声称精确包络。

当 `lambda>0`，backward 的目标仍应是返回的 F。固定 W 时，ridge 项对特征的直接导数为零，所以 J 与 F 的**固定 W 偏导**相同；这不代表沿 `W_lambda*(theta)` 的未正则 J 总导数也相同。本地独立核对给出一个实际例子：F 包络偏导约 `-0.0157195`，ridge 最优 W 上的 J profile 导数约 `-0.0277299`。不得以“ridge 项对 feature 没梯度”为由混淆两种 profiled objective。

若 outer 改用验证数据、另一组图像、重新 forward 的随机特征或另一目标，此 API 的包络解释不再成立，需要另行设计对应的微分与评价协议。

## 7. dense / permutation 对照

```python
from mf_vcs import dense_objective_terms, draw_ordinary_permutations, sampled_permutation_terms

full = dense_objective_terms(scenes, w, weights=weights)
blocked = dense_objective_terms(
    scenes, w, weights=weights,
    chunk_size=128, checkpoint_blocks=True,
)
perms = draw_ordinary_permutations(scenes, draws=8)
sampled = sampled_permutation_terms(scenes, w, weights=weights, permutations=perms)
```

普通 permutation 对照只支持 V 场景，允许固定点，且逐图抽样。`permutations` 显式传入，便于同一输入、同一随机状态下核对值和梯度。固定一次 permutation 的 J 不必等于 V；全部 permutation 的条件平均才等于 V。不能把重抽配对的方差减少写成独立样本数增加。

`dense_objective_terms(..., chunk_size=..., checkpoint_blocks=False)` 仅分块前向分数，autograd 可能保留全部块；不能据此声称反向峰值线性。`checkpoint_blocks=True` 对每个块调用 `torch.utils.checkpoint.checkpoint(..., use_reentrant=False)`，反向时重算块分数。U 的行偏移作为显式参数传给 checkpoint，以防循环闭包在反向读到最后一个 block 的偏移。该内存/时间权衡仍需真实服务器测量，当前不报告节省倍数。

## 8. 实际核对结果

本地 `python verify_numpy.py` 已执行，47 / 47 通过。其主要证据如下；完整逐项数值见 JSON。

| 核对 | 观测结果 |
|---|---|
| V/U/独立池矩一二阶 vs 显式 pair-feature oracle | float64 最大绝对误差约 `5.55e-17`；float32 约 `5.96e-8` |
| 矩 HVP vs 显式 Hessian | 最大绝对误差约 `1.11e-16` |
| W、两侧 logits、独立边缘池有限差分 | 最大绝对误差约 `5.62e-11` |
| 不同大小场景的固定权重目标/HVP | 与逐场景 dense 相等；跨图 pooling 反例使目标变化 `0.96` |
| 枚举 120 个普通 permutation / 44 个 derangement | 平均分别等于 V/U；后者只作数学核对，未实现 sampled-U 后端 |
| 无正则 / 显式 ridge，普通和 active-box 例子 | NumPy 投影算法最终 gap 均 `<1e-9`，7–98 次迭代；与 SciPy 最优值差最多约 `1.11e-16` |
| M1、ridge > 0 的 F profile 有限差分 | 与固定 W 包络偏导差约 `1.24e-10`；raw-J profile 明确不同 |
| 原 J 残差恒等式、rank-one 退化情形 | 通过；精确解 head 不意味着特征有表达力 |

**这些是独立 NumPy 数学与算法核对，不能升级表述为 mf_vcs.py 已在 Torch 通过。** Torch 的 checkpoint、autograd、GPU dtype/device、实际 solver 及计时程序仍需以下执行。

## 9. Server 的最小执行顺序

在已有 PyTorch 2.x 的环境中，不需要下载数据。先运行 correctness；运行 GPU 或完整网格的资源须属于外层任务已授权的范围。

```bash
python verify_torch.py --device cpu
python benchmark_torch.py --device cpu
```

正确性通过且 GPU 执行已获当前任务授权后：

```bash
python verify_torch.py --device cuda
python benchmark_torch.py --device cuda --dtype float32
python benchmark_torch.py --device cuda --dtype float64
```

完整网格必须显式选择：

```bash
python benchmark_torch.py --device cuda --dtype float32 --full
```

默认 smoke 仅 `(N,rank)=(32,4),(128,8)`。`--full` 才启用 `N={128,512,2048,8192}`、`rank={8,16,32}` 的 12 格。

每格五种方法：perm1、perm8、dense、checkpointed dense block、moments。A/B/W 完全相同，无编码器/优化器更新；perm1 为 perm8 的第一个 permutation。报告 forward + `autograd.grad(A,B,W)` 的同步墙钟计时、值/梯度误差，CUDA peak allocated 和 peak reserved。CPU 的 CUDA 字段为 null，不提供伪造的 GPU 或 CPU 峰值内存。

GPU 在每次计时前后同步。输入验证在计时前完成；被计时的方法统一关闭重复的 W/permutation 值检查。预热、重复次数、dtype、版本、设备、baseline allocated/reserved 和每次 timing 全部入 JSON。pairing 保持固定，**这份 timing 脚本不测重抽方差**。编码器成本明确排除；后续应用预算需补 encoder、内层 head 次数与 end-to-end 成本。

完整 dense 与 moments 必须在数值容差内相等，否则该格标 `correctness_failed`，不把错误程序的速度当有效结果。permutation 相对 V 的偏差被明确标为抽样偏差。OOM 按格记录并继续，不据此删除该格；程序按实际 CUDA 可用性执行，无 silent CPU fallback。checkpoint 版本不兼容等其他错误同样写入 JSON，最终非零退出。

## 10. 对 v0.1 参考的独立审查结论

核心恒等式、完整产品矩、U 对角删除、固定特征二次头和盒 FW gap 均成立；没有发现需要更换原 J 的数学原因。执行接口需修正的地方是：

1. v0.1 `objective(..., off=True)` 默认 U，与当前 G1 普通 permutation 的 V 目标不一致。当前默认 V，U 必须显式标记。
2. v0.1 dense 的 `torch.eye(n)` 默认在 CPU，会妨碍 GPU 对照。当前 mask 在实际 feature device 创建。
3. 原参考的独立池只有 q 矩函数，缺少完整 J、HVP 和 head 接口；当前支持完整 local independent Q。
4. 原参考未验证 simplex、有限值、固定非负场景权重和图内边界。当前增加契约检查与分组接口，但图 ID 真实性仍由 adapter 保证。
5. M1 需要把“同 tensors、同 F、固定已解 W”实现成接口约束；当前 wrapper 采用这一模式，显式区分 ridge 下的 F 与 J。
6. 任意 forbidden edges 和 row-normalized Q 不能从小型参考算例推出已具备生产支持。当前明确拒绝，保留为后续独立任务。

此外，当前没有提供对实际配准器的变换梯度、warp-then-encode 一致性、真实 BatchNorm/dropout 状态、分布偏移或目标性能的验证。接入真实 G1 adapter 后必须核对这些接口，不能拿头部代数相等替代应用端端到端检验。
