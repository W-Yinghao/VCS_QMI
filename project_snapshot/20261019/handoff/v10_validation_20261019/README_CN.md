# V10新方法验证：2026-10-19

用户明确允许“现在也可以测试了”。当前CPU控制器1047202已在Slurm上完成92项数值和回归测试（全部通过），并提交GPU门禁1047204。GPU结果尚未返回，不宣称新增方法已通过验收。

冻结源为独立只读工作树 `ssl_pilot_v10_validation_20261019`，提交 `5c04719`；全部输入与源文件由MANIFEST.json绑定。原V9的49个文件和BF1的58个文件哈希核验未变。

顺序为CPU数值/梯度/既有实现回归，CUDA数值与七个方法的FIT32驻留、optimizer/RNG/checkpoint续跑、真实完整配准搜索，最后七个方法各自的第一个预定候选做完整训练和所有CAL8×10读出。两种宿主学习方法各3000步、12个CAL checkpoint；CoMIR完整736步；RuLSIF完整FIT32解析拟合；MI/NMI/MIND不虚构训练步数。总计2320次完整CAL配准。这些是验证数据，不进入正式CAL选择或DEV性能结论。

控制器观察实际退出回执和报告哈希；GPU门禁失败时不提交完整验证，单个完整验证失败保留失败记录，其他已就绪方法继续；没有自动重试或自动发布科学release。每次提交都核对唯一账本与共享normal提交限制（最多29项），有可就绪工作时补足最多8个项目GPU pending。共享normal最多8张GPU，新增验证nice500，现有运行作业不取消。

动态状态见LEDGER.json和STATUS.json；CPU_NUMERICS.json/TXT记录CPU结果；runs/<job>/GATE.json、T02.json、RUNNER.json记录真实GPU证据。完整验证各作业最长8h，超时不算PASS。后续正式新增方法实验还需审查验证、冻结方法附件、确认成本预算并生成受校验的正式release。TEST134、EPFL未激活。
