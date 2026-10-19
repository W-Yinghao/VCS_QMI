# V8.1 / V9 增量更新

先读 `SERVER_UPDATE_CN.md`，向服务器发送 `01_AGENT_MESSAGE_CN.md`。`TASKS.yaml` 是任务索引。

本件为审查意见，不是直接应用到仓库的 patch，不包含模型、训练数据或第三方权重。原 V8.1 ZIP 未修改。

## 本地小例子

```bash
python review_checks.py
# 可选：只读检查服务器真实锁文件，不启动训练
python review_checks.py --repo /path/to/VCS_QMI
```

依赖：NumPy、SciPy、PyYAML。

默认八项测试验证的是历史源码片段与反例可复现。它们通过，不表示服务器的当前实现已经修好，更不表示自然数据结果成立。

检查基线：V8 d2067c4；V9 8a994f3。接入时先核对新提交，已经完成的修复给出证据并复用。
