# 参考代码的范围

`core.py`给出标签分布、CS坐标转换、概率风险、并列与候选完整性的CPU定义。运行 `python -m pytest -q tests`。

`closeout_diagnostics.py`读取服务器导出的CAL/DEV JSONL，输出逐提示指标：
```bash
python reference/closeout_diagnostics.py --input examples/dev_candidates.jsonl --output /tmp/v7c_demo
```
它不是原生V7终末scorer，不会授权读取封存标签。检查路径/角色只是防误用约定，不是安全沙箱。

示例是人为构造数据，不是真实模型结果。完整GPU模型、真实数据adapter和原生benchmark scorer由服务器接入既有仓库。基础单元测试通过不等于训练/任务复现成功。
