# 本地交付验证

- `python -m pytest -q tests`：69项通过（见pytest_output.txt、junit.xml）；含CPU数学和PyTorch梯度核对。
- 两项CLI示例已运行：`demo_finite.json`、`example_audit.json`。人工示例的置信集返回[0,1]，体现样本很少时不输出伪精度。
- 独立有限支持Monte-Carlo：200次、每角色512个iid分数，覆盖200/200；这里只核对实现，不是自然数据实验或区间最优性声明。
- 7个显示公式已渲染为内嵌SVG；无外部MathJax依赖，没有随包分发字体文件。
- 概览PDF为7页，已渲染并逐页查看拼图；无溢出、重叠或缺字。
- 自然数据adapter、GPU训练、官方方法复现、数据许可和服务器作业均未在本次执行。
- 最后完整性检查结果见`package_checks.json`；外部URL有效性没有逐一重新下载确认。
