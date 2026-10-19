# CPU参考实现

运行：

```bash
python -m unittest discover -s tests -v
```

需要Python ≥3.10和NumPy。本轮只在本地完成有限概率/平方风险/数据分组/指标fixture测试，没有运行自然图像神经实验。

`vcs_vl_reference.py`验证原始J、精确gap、条件分布平均、有限预测器残差、类先验变化、多正例集合与分布式交叉熵的区别，以及基本分组/成本选择约定。有限模型支持可精确计算不代表自然图像oracle可得。

`fixture_ap_at_r`与`time_iou`仅用于理解和接口测试，不能替换ECCV Caption或QVHighlights官方完整evaluator。本包不提供六条路线的已集成runner，服务器需在固定的正式项目代码上接入与验证。
