# 给server agent的执行指令

请接续VCS_QMI项目V10（MF-VCS）。本包是已接受方法设计的server实施交接，不重启V8/V9主故事。

先读`V10_DESIGN_CN.md`、`docs/HOST_ADAPTER_CN.md`、`VALIDATION_REPORT_CN.md`和`core/README_math_CN.md`。读取本仓库适用的AGENTS/执行约定，按已有会话授权与实际资源继续；不要把未知当前状态写成已完成。

## 固定条件

- 保留合作者S/J；bounded simplex bilinear critic；无默认输出tanh、无默认ridge。
- 第一完整版本M0；M1是条件增强，inner/outer同一batch、同一tensor、同一目标。
- 配准主Q是同场景普通permutation对应的V形式；先核对当前宿主。U是单独实验。
- 测试固定W与encoder，保留对几何参数的梯度；共同能量表warp-then-encode，CoMIR等原生流程单列。
- BF–SHG主数据、EPFL RGB–NIR第二协议；沿用原group/ID/排除与暴露历史。V9已见测试不能重新封为新确认。
- 完成强基线和same-data/same-total-cost比较；不要用只赢BCE或只提高J代替任务结果。

## 第一阶段现在做什么

1. 只读记录当前git HEAD/dirty状态和V9已有任务状态，运行`scripts/preflight.py`。填写真实数据manifest与`HOST_ADAPTER_AUDIT.md`，记录两份历史reg文件在当前源码中的实际映射。
2. 接入`core/mf_vcs.py`，先运行随包PyTorch验证；本包本地没有PyTorch，因此不得将其状态预填PASS。
3. 在真实batch与warp链上执行adapter验收。目标/梯度不匹配先修；有禁止边则补精确实现，不静默略去。
4. 执行T02小表并给出实际GPU/CPU数据；含encoder和完整注册的成本另测。根据当前已允许资源写BUDGET，不取消V9任务。
5. 接受阶段结论后，按包内预定科学协议继续T03–T05；常规实现选择可自行处理，不为每一项小修复反复请求用户确认。新增数据/额外大额资源等超出已有授权的事项，应先把具体方案做成可审查结果。

## 首批回传格式

回传一份阶段报告，开头直接列：当前真实提交、已完成代码/验证、仍缺的输入、实际测量表、下一批具体单元与预计资源。

附件至少包括：PREFLIGHT.json、HOST_ADAPTER_AUDIT.md、manifest及hash、TORCH_VALIDATION.json、GEOMETRY_GRADIENT_CHECK.json、T02长表、BUDGET.json、单元ledger。对每个失败保留原因和原日志。

全量完成后补训练checkpoints索引、CAL选择记录、独立真值误差表、每次配准长表、两种搜索器比较表、完整失败清单和固定规则选择的可视化。不把计划CSV、脚本存在或小矩阵测试当作自然任务完成。
