# V9 · 可靠跨模态对应：量化、加权与有限配准验证
**版本：V9 / server v1 · 2026-10-16。与 V8 并行，不替代 V8。**

主问题：面对不完美的跨模态对应，能否先正确量化其可靠度，再让这个数值改善完整视觉任务，而不是仅改变一个不会影响 argmax 的评分刻度？

主线 **NR（Noisy Correspondence Reliability）**：可信参考 → 批次混合比例 → 样本成分后验 → 完整图文检索训练。辅线 **REG**：复用已有跨模态配准，检验对应监督质量与训练样本量。第三方向“零/多目标对应”只保留问题卡。

## 阅读顺序
1. `01_SERVER_AGENT_PROMPT_CN.md`：接入与并行约束。
2. `docs/02_MASTER_PLAN_CN.md`：任务依赖与本轮范围。
3. `docs/03_THEORY_CONTRACT_CN.md`：原理论、直接推导、假设和反例。
4. `docs/04_NR_QUANTIFICATION_CN.md`、`05_NR_FULL_TRAINING_CN.md`：主线实施。
5. `docs/06_REG_EXTENSION_CN.md`：配准的有限扩展。
6. `docs/07_DATA_AND_BENCHMARKS_CN.md`、`08_STATISTICS_CN.md`、`09_PARALLEL_EXECUTION_CN.md`。
7. `literature/REVIEW_CN.md`：相关工作与核验深度。

完整合并文本见 `FULL_PLAN_CN.md` / `FULL_PLAN_CN.html`；七页左右的执行概览见 `V9_OVERVIEW_CN.pdf`。`INDEX.html` 是离线入口。

## 本包完成与未完成的部分
这是可执行规范、参考数学实现与接入材料，不是已经接入服务器的数据管线。本地测试核对有限分布恒等式、比例区间、权重与分组规则；不包含自然图像训练、原方法复现、服务器作业或新实验结果。自然数据 adapter 和训练脚本由 server 依规范完成并记录差异。

```bash
python -m pytest -q tests
python scripts/demo_finite.py --out validation/demo_finite.json
python scripts/audit_mixture.py examples/independent_scores.json --out validation/example_audit.json
```

无需把第三方数据、模型权重、字体或完整论文打包进仓库。保留来源、版本、文件哈希与许可记录。
