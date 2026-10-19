# CPU参考代码

包含数学目标、soft投票的损失、先验修正、并列名次、prompt级聚合、标准GenAI-Rank候选选择与manifest检查。**不包含已集成的FGA训练器、数据下载器、模型权重或GPU作业脚本。**

```bash
python -m pytest reference/tests -q
python reference/validate_manifest.py path/to/manifest.jsonl
python reference/score_rank.py --predictions frozen_prediction.jsonl \
  --labels sealed_human_labels.jsonl --output new_versioned_result.json \
  --allow-scoring
```

预测每行字段：prompt_id, generator_id, image_id, score。一次文件只放一个方法、一个训练seed和一个score口径，不混合。标签字段另有candidate_index(0..8)与human_score(1..5)。不能使用输入行顺序隐式配对。

`score_rank.py`默认保留官方N3分组与首索引平局。`--protocol all_subsets`、`--tie-rule mean`是补充列，不覆盖官方列。工具拒绝缺失候选/重复/非有限输出/覆盖旧结果。

`common_j`从评价标签形成两类条件均值，这仅用于公共统计量的分母。它不把测试类比例用来重新校准概率、修改阈值或选择模型。训练权重与部署prior仍仅来自FIT/CAL规定角色。

bootstrap函数的区间只覆盖提示抽样不确定性；近似双侧p不是精确随机化检验。多seed另外报告。实现集成后需与项目选定的正式统计函数作一致性测试，协议不可事后更换。

依赖：numpy、scipy、pytest；loss梯度测试另需torch。版本以VERIFICATION.json中的本地实测为参照；服务器应锁定兼容自己的视觉模型环境，不直接覆盖已有环境。
