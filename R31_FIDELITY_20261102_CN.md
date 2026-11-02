# R31 IN²R 复现忠实度（2026-11-02 第三十一版）

接在第三十版（`R35_RESULTS_20261102_CN.md`）之后。只发布数字表与代码，不附解读。

## 内容

- `handoff/r31_in2r_fidelity_20261101/R31_FIDELITY.md` 与 `PUBLIC_R31_FIDELITY.json`：15/15 单元的最佳 full-DEV rSum 与一次性 TEST 六项 Recall/rSum（常设 TEST 工具，上游 IN²R evaluation.py；15/15 通过 DEV 复算门禁），三 seed 汇总，并列 IN²R 论文值（arXiv:2606.04061v1 Table 1–2）与此前同条件的原生复现（R25 H0 Flickr 40%、R26 H0 CC152K）：
  - (a) 官方 NCR Flickr 20% 噪声索引 + R25 代码配置：TEST rSum 519.03 ± 0.53（论文 511.6）；
  - (b) 作者 run.py 默认配置：Flickr 40% 514.44 ± 2.16（论文 504.1；R25 H0 514.53）、CC152K 403.70 ± 2.38（论文 380.8；R26 H0 403.10）；
  - (b′) 默认 + --i2t --t2i：Flickr 40% 514.77 ± 4.02、CC152K 404.63 ± 0.51。
- `PUBLIC_DESIGN.json`（三组设计）；汇总脚本 `tools/summarize_r31.py`。
- TEST 汇总表（366 单元，362 过门禁）与实验清单更新（R31 15/15、R33 11/27）。

## 范围说明

每格 3 seed；不给显著性结论。TEST 只做 DEV 选出 checkpoint 的一次性评测，不用于任何选择。作者代码副本不发布。
