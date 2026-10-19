# 参考代码的范围

`core.py`为评分分布、CS坐标、先验、RPS、候选评价与分组统计的CPU定义。`ordinal_torch.py`包含可微的五档head读数、六种点式loss和pair/list loss，不是完整VLM训练runner。

运行 `python -m pytest -q tests`。将`loss_per_image`接入完整共享Q-former模型，按冻结权重累积；排名辅助项仅在同提示组内调用。

服务器需要追加数据角色、完整图文forward、参数梯度/冻结模块、AMP、全局分母、所有预测ID与标准evaluator集成测试。这里没有权重、实际训练、Slurm作业或自然数据性能。
