# 参考代码，不是服务器训练器

- `vcs_r4_reference.py`：原始 J、RPC、污染恒等式、固定 critic U-statistic、delete-one 方差、图片平均与 proposal-reference 配对。
- `audit_p134.py`：只读原始 P134 单元的 selected posterior 字段，生成逐 seed 误差表；不会把污染后后验 MSE 解释成 clean MSE。先检查实际 schema，与三条手工读取记录核对后使用。

运行本地检查：

```bash
python -m unittest discover -s tests -v
```

服务器原始记录的可选只读审查：

```bash
python reference/audit_p134.py --input /actual/reports/P134 --output /new/output/R4_P134
```

Python 3.10+、NumPy、SciPy；仅自动微分恒等测试另外使用 PyTorch。不包含模型/数据下载、训练和 Slurm 脚本。所有统计区间函数的前提见数学规范：固定表示与 critic、正确的独立单位。通过单元测试不能证明神经网络训练表现或新颖性。
