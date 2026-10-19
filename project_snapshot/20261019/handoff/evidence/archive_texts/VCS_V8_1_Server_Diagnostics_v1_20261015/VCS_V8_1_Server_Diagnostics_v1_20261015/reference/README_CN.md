# 参考代码：只读诊断，不替代冻结评价器

## 本地检查

在包根目录执行：

```bash
python -m pytest -q tests
python reference/make_demo.py --out /tmp/v81_demo_rows.jsonl
python reference/diagnose.py --rows /tmp/v81_demo_rows.jsonl \
  --label-mode development --out /tmp/v81_demo_result.json
```

示例是合成数据，不是V8实验结果。输出路径已存在会拒绝覆盖；请使用新的临时路径。`requirements.txt`是最低兼容范围，不要求升级已有环境。

## 接入真实V8开发导出

```bash
python reference/adapt_v8.py \
  --pred /ABS/V8-FIT-U-CS-s0/pred/images_DEV.jsonl \
  --public-train-manifest /ABS/outputs/V7/data/manifest_images.jsonl \
  --role DEV --method U-CS --seed 0 --checkpoint-id best_sha256 \
  --out /ABS/outputs_v81/normalized/U-CS-s0-DEV.jsonl
```

`/ABS/...`须替换成D00核对过的真实路径；该适配器根据已读取的`v8_labels.py`字段构建，不读取图像压缩包或任何外部标签。每个方法/seed生成一份后可顺序合并JSONL，再运行`diagnose.py`。不同数据/角色/候选池不会混合；方法间缺行会报错。

自然数据适配器尚未在真实服务器产物上运行，服务器需先用少量真实行验证字段。若manifest的component字段不是`component_id`，须显式适配并记录，不能猜测连接成分。

## 日志审查

```bash
python reference/audit_logs.py \
  --logs /ABS/unit0/log.json /ABS/unit1/log.json \
  --source /ABS/job_snapshot/scripts/v8/v8_train.py \
  --out /ABS/outputs_v81/epoch_audit.json
```

该工具只读JSON和Python文本，不加载`.pt`。调用行序只是静态证据；真实控制流、快照、成功optimizer更新数仍需人工/服务器核对。`steps`有时只是尝试步数，不能据此证明AMP每次更新成功。

## 原生外部方法

外部评分可以写入标准schema，使用`score_kind: native_score`。它们进入排名和选图比较，未经过独立标定时不进入人工评分MSE。V7/FGA原始接口不保证与V8完全相同，不能机械复用适配器。

本工具只有开发角色的带标签计算和纯无标签模式。无标签模式会拒绝hist/votes等字段，包括外部导出中的dummy标签。它不是安全沙箱；真实数据隔离仍由项目目录、角色和负责人决策保障。

## 尚未自动实现的工作

完整VLM推理、gradient hook、时延benchmark、真正的job snapshot追踪、全局相关系数的cluster bootstrap、每epoch缺失checkpoint恢复等由服务器在现有项目实现。不要将参考脚本通过解释成这些任务已经完成。
