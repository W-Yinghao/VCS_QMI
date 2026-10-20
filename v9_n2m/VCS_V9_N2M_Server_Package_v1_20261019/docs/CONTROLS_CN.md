# 两个控制的输入、生成与核对

`v9n2m.controls.generate_controls` 使用已冻结的 observed 表和生产 VCS CSV/lock，一次生成三臂：

| 臂 | U 权重 | A 权重 | 文件身份 |
|---|---|---|---|
| VCS | 原生产权重 | 1 | CSV 逐字节复制，因此保留原 CSV 的 SHA256；新 lock 记录新路径与原 lock 来源 |
| ConstantU | 原生产权重的 float64 均值 μ | 1 | CSV 使用 17 位有效数字，读回保持 float64 数值 |
| ShuffleVCS | 按 observed image 的五行块置换 | 1 | 同上；`permutation.npy` 与 SHA256 单独保存 |

生成器不读取 teacher、critic、caption/image 内容或机制标签。observed 输入只允许 `pair_id, caption_id, observed_image_id, role` 四个字段；重复 pair/caption ID、跨角色 observed image、缺失/多余 U 权重、非有限/越界权重、CSV 与 lock 不匹配都会报出具体错误。它拒绝 evaluator-only 路径与标签字段。原权重 CSV 中的 `score, rho_hat, fit_hash, reference_hash` 可保留，但不用于选择控制。原 lock 必须标记 `VCS_Weight`，存在 `rho_strategy/deployable` 字段时还须是 estimated/可部署生产权重。

## 唯一的 Shuffle-VCS 定义

1. 只取 role=U，按 `observed_image_id` 分组，每组必须恰有五行。实际结构不符合时停止并报告，不改为逐 pair 置换。
2. image ID 先按可转成整数的数值排序，数值相同时按原字符串排序；其余 ID 按词典序。每块内 caption ID 使用同一排序规则。
3. 使用 `Generator(PCG64(SeedSequence([seed, 20261019, 9101])))` 抽取一次长度为图像块数量的 permutation。目标块 k 获得 `W[permutation[k], :]`。
4. 固定点允许存在。不会重抽到无固定点，不会查看机制标签后挑置换，也不会每个 epoch 再置换。

保存置换数组、NumPy 版本、块数、块顺序哈希、块内 pair 顺序哈希、固定点数量/比例和权重相关系数。恒定向量的相关系数没有定义，输出 `null`。保留的是接收图像的权重块结构，不声称保留所有 caption 原始来源依赖。

## 数学不变量

按 observed 表中的 U 顺序读回三臂权重，验证 ConstantU 每个值精确等于 float64 μ；ShuffleVCS 排序后的 float64 数组与原权重逐元素相同。两种控制的 U 总质量和全池分母 `(n_A + sum(w_U))/(n_A+n_U)` 均与 VCS 相容；ShuffleVCS 的 ESS 也保持相容。求和顺序会带来 float64 舍入差，因此质量、分母和 ESS 用 `rel_tol=1e-12, abs_tol=1e-12` 核对；这不是性能容差。ConstantU 的正质量 ESS 为 n_U，不能要求它等于 VCS ESS。全零向量 ESS 按既有实现惯例记 0，并在 lock 中说明。

所有 CSV 只列 U。A=1 在原学生接口中实现，不能把该 CSV 当作删除低权重负候选的指令。全训练池仍然完整保留。

## 不覆盖与重复执行

控制定义版本、生成器源码哈希、seed、数量/块大小和全部输入路径/SHA256 组成 request identity。生成时先在独立临时目录写完并读回检查，最后发布完整目录。输入生成期间发生变化会拒绝发布。相同输入再次调用会核对现存 CSV/lock/permutation/manifest 后直接返回，保留每个已发布文件的原内容与时间。输入或输出被改动、已有无完整 manifest 的非空目录均不覆盖，使用新的实验输出目录排查即可。

默认数量为 145000 全池、130500 U、每块 5 行。可传入更小计数仅便于清楚标记的合成实现核对；正式 experiment 的准备流程仍须固定生产计数。

## 纯 CPU 验证

在包根目录运行：

```bash
PYTHONPATH=src python -m unittest discover -s tests -p test_controls.py -v
```

这里检查数学不变量、置换方向/排序、允许固定点、无标签路径、浮点序列化、输入/输出哈希、ID 覆盖、重复执行和篡改检测。它不训练学生，也不替代服务器上的 Torch 数值与恢复等价核对。
