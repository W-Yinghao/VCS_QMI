# r2 本地核对记录

本轮对照的是已提供的 r1 服务器包和 V6 Revised 稿件源包；未调用 GitHub、未访问服务器、未提交作业或评价 checkpoint。

## 实際运行

- `check_contract.py`：14 项小型确定性数学/配对约定检查通过。脚本与 r1 逐字节一致。
- `check_r2_sync.py`：10 项交接同步检查通过，涵盖任务依赖、两种比较轴、已报Table4单位转换、样本量区分、模板不是结果，以及31个原样参考文件的hash。
- JSON Schema、自身及r1模板兼容性、工具脚本语法检查通过，详情见 `structure_checks.json`。
- 参考PDF仅做只读可打开/页数检查：完整稿13页、主文加参考文献7页、中文说明3页；没有编辑或重新渲染这些文件，其字节与输入ZIP完全相同。
- 方法合同中的主数学段和SSL实现段与r1完全相同；`intake_repo.py`、`verify_package.py`和`check_contract.py`未改。

## 未验证

原始训练结果、现场checkpoint/head/transform/split是否齐全、真实null水平和置信区间、PyTorch/DDP梯度、执行成本与GPU授权均未在本轮重新验证。已报Table4数值只做转录与单位核对，不能将其称为新实验复现。

`checks_r1_history/` 是旧包执行记录；`reference_v6/evidence/` 是修订稿自带证据，均保留原样，不冒称本轮重新执行。
