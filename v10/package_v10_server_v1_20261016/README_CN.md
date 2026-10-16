# V10 MF-VCS · Server 实施交接包 v1

**从这里开始：**先读`V10_DESIGN_CN.md`，然后把`01_SERVER_AGENT_PROMPT_CN.md`交给server agent。此包接续对话「系统学习变分互信息」中已经接受的V10 v0.1，补全实施设计与数值工具。

## 当前状态

已经完成：旧讨论与附件恢复、T00–T05具体协议、数据/比较/预算/选模/评价定义、MF-VCS候选PyTorch后端、独立数学验证、离散真值生成器、服务器验证与计时脚本、接入契约、配置与结果表结构。

服务器尚需执行：当前仓库/数据预检、真实图像宿主adapter、PyTorch与连续warp梯度验证、GPU计时、完整encoder训练及配准。当前环境没有PyTorch，也未取得当前宿主源码；本包没有已验证的自然图像训练入口。详细状态见`VALIDATION_REPORT_CN.md`。

## 文件导航

| 路径 | 内容 |
|---|---|
| `V10_DESIGN_CN.md` | 完整主文档；固定理论、实现、两个数据协议、基线、T00–T05和结论规则 |
| `01_SERVER_AGENT_PROMPT_CN.md` | 可直接交给server agent的任务说明 |
| `docs/HOST_ADAPTER_CN.md` | 真实宿主接口必须核对的行为与验收 |
| `docs/PROTOCOL_SOURCES_CN.md`、`docs/sources.json` | 数据集与先行工作来源、核验范围和限制 |
| `core/` | MF-VCS候选后端、独立NumPy检查、PyTorch检查和计时脚本 |
| `synthetic/` | T03离散生成器、138项检查、完整小型示例数据 |
| `scripts/` | 不导入训练器的只读预检、计划单元生成器 |
| `configs/` | 主配置、114条设计单元、数据/结果/成本表结构 |
| `results/` | 本包15项语法与manifest接口检查结果 |
| `provenance/` | 恢复的v0.1设计/历史19项检查与V9相关协议 |

114条单元由60条计算格、48条学习方法/数据/seed组合、6条非学习系统组成，全部状态为`designed_not_launched`；它们不是已经提交的网格。所有方法的实际应用接入仍以主文定义和真实宿主为准。

## 可以直接执行的数值入口

以下命令在解压后的包根目录运行。NumPy/SciPy参考环境版本记录在验证报告；PyTorch命令需要服务器已有相应环境。

```bash
python core/verify_numpy.py
python synthetic/tests/check_synthetic.py --suite-dir synthetic/validation/example_data --output synthetic/validation/CHECK_RESULTS.json
python scripts/make_units.py --protocol configs/V10_PROTOCOL.json --output configs/EXPERIMENT_UNITS.csv
```

真实PyTorch后端首先验证CPU，再验证可用CUDA设备：

```bash
python core/verify_torch.py --device cpu
python core/benchmark_torch.py --device cpu
python core/verify_torch.py --device cuda
python core/benchmark_torch.py --device cuda --dtype float32
```

默认计时只有小型smoke；`--full`才执行N×K完整表。命令不会提交队列作业。计时脚本只覆盖数值头，完整encoder/warp/配准成本由宿主adapter另测。

只读预检示例，路径由server agent填写实际值：

```bash
python scripts/preflight.py \
  --repo /actual/path/to/VCS_QMI \
  --manifest /actual/path/to/V10_DATA_MANIFEST.csv \
  --data-root /actual/path/to/data \
  --output /actual/path/to/v10_outputs/PREFLIGHT.json
```

预检记录commit、源码hash/顶层符号、manifest和跨角色冲突；它不会导入宿主训练代码。返回`INVENTORY_WRITTEN_REQUIRES_HOST_REVIEW`只说明清单生成，不表示宿主接入或数据独立性已通过。

生成新的T03训练样本：

```bash
python synthetic/src/v10_synthetic.py \
  --output-dir /actual/path/to/v10_outputs/t03_seed20261016 \
  --seed 20261016 --families two_block diagonal_mixture \
  --m 8 32 --rho 0 0.25 0.75 \
  --n-fit 4096 --n-cal 2048 --n-eval 8192
```

## 四个实现提醒

1. 自然配准主模式是同场景V；普通permutation允许固定点。U必须显式命名。
2. M1训练按配置采用预算内近似head时，显式传`require_converged=False`并记录每批gap；严格数学验收保持默认True。小gap不是外层梯度误差证书。
3. M1最终encoder与W成对选模；主协议没有额外最终FIT refit。CoMIR原生流程与共同能量表分开。
4. 当前没有任意forbidden-edge生产后端、真实图像adapter或Gaussian/cubic生成器；相应实现任务和可替代参考明确写在主文，不作为已交付运行功能。
