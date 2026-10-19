# VCS-QMI SSL：Server Agent Starter v1

请先读 `VCS_QMI_SSL_Server_Agent_Spec_v1.md`。它是本轮执行规范；不能用此前的critic-value/两点梯度研究支线替代。

## 已提供

- 合作者原始二次损失、tanh pair critic、非零循环负配对、普通联合反向传播的PyTorch参考。
- SimCLR NT-Xent与matched-projector VICReg的损失参照。
- 29项已在本地CPU运行的单元测试。
- VCS、SimCLR、VICReg三份CIFAR-10 20-epoch pilot配置。
- 数据、评估、checkpoint、日志、资源边界和报告合同。

## 尚未提供/运行

完整CIFAR Dataset/ResNet训练器、scheduler CLI、数据manifest实际生成、GPU集成、CIFAR训练、线性probe结果、DDP/AMP验证。需要server agent在实际repo中集成；不要把本文中的目标CLI当成已经存在的模块。

## 立即可执行的检查

在已具备PyTorch和pytest的项目环境中：

```bash
python -m pytest -q
```

`reference/ssl_core.py`不依赖torchvision；真实CIFAR trainer需要服务器已安装且与torch兼容的torchvision。配置生成使用PyYAML。不要为运行此包直接升级共享环境。

## 执行顺序

P0读环境、核对数据与repo → P1核心+集成测试 → P2真数据100-step smoke → P3三个20-epoch pilot → P4完整回报后停止。单卡FP32起步。不自动启动200epochs、多seed、图文或EEG。

`configs/next_stage_plan.yaml`只是一份未启用的下一阶段计划，不可当作可执行训练配置。

## 本地检查边界

真实结果在`reports/local_core_tests.txt`和`reports/local_validation.json`。这些通过测试只支持公式/张量/梯度路径一致，不能支持准确率、收敛速度或GPU性能结论。
