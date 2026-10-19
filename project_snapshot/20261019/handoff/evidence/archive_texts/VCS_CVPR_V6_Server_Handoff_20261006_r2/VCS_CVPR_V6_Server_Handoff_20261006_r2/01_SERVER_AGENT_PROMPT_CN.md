# 给服务器端 agent 的 V6 r2 增量指令

请按 `CVPR-MV6-HANDOFF-20261006-r2` 同步项目。论文版本仍是 **V6**，本包依据最新的 **V6 Revised**；r2 只区分交接修订。先读 `08_R1_TO_R2_DELTA_CN.md` 和 `reference_v6/Paper_Positioning_and_Interpretability_CN.md`。

## 保持的主线与实现

方法主体仍为 VCS estimator，equal-mixture reference、原二次 J、bounded critic、exact regression gap 均不变。完整 SSL 继续推进；可解释部分用独立检查过的测量，理解最终 SSL 表示保留了什么。没有新增通用审计平台、saliency 模块或“内生可解释”架构。

论文只用一个 VCS 名称。当前选定配置对应内部 A-P3：固定 `tanh[a(s-kappa)]`、全视图配对、两端完整梯度、正负分别归一化。旧 recipe、G2 等仍是内部消融。保持原超参数选择规则和不同数据集的已有配置；本次不改 J、不改 trainer、不停掉或重启已有获批作业。

## 两步测量任务必须分开

**A｜检查测量器（MV6-I1；Table 3）。** 固定外部 encoder/checkpoint、图像、特征类、split、N|Y、置换与预算，比较 VCS、匹配 JS、同目标 kernel、条件 HSIC。核/HSIC 补缺沿 r1 接续。不同 native 目标不混进同一 S 误差列。

**B｜理解最终 SSL 表示（新增组织编号 MV6-R0；Table 4）。** 固定测量算法与协议，分别读取最终 VCS 和 SimCLR。critic 权重分别拟合，不是同一组权重跨模型强用；测量器选择不按 encoder 结果方向改变。Table 4 已有 colour 0.1、encoder seeds 1/2、n=2000、100 次检验重复，预测响应来自对应 8000 对图像。先复用 `reference_v6/evidence/retention_source_rows.json` 及仓库原始记录，核对 null、单位和来源；不因新增正文表而重训。

两张表是不同对照轴，不能池化或拼接。Table 3 的 P116、blur0.5/colour0.2 旧实验不能与 Table 4 的 P109/P118、colour0.1 响应伪装为同一实验。Table 4 的检测抽样 n=2000 与响应 n=8000 分别记账，不要求它们是同一个子样本。

## 增量接入，而不是重新开项目

已有 r1 intake、split、配置、checkpoint 和任务映射继续复用。核对当前变化，填 `templates/DELTA_INTAKE_R2_TEMPLATE.md`。新增 `MV6-R0` 仅明确此前未单列的表示分析交付；若已有对应 P 任务，映射过去，不重复分配编号或运行。

后续外部模型扩展继续用 `MV6-I2`：优先已有 provenance 完整的 VICReg 或选定 logistic encoder，并按预定条件扩到 CIFAR-100；库存不足只阻塞对应单元，不预训练新模型凑表。最终 VCS 是表示分析的必含对象，但不替代额外外部模型的覆盖。

真实压缩仍用 `MV6-C1`：仅使用验证过的 h→r→z、h→固定任务头；测量配对中保留同一 Y、N、P/Q 定义。保留原始 signed Delta J、D_T、拟合敏感性，不强制单调。另一 encoder 不是前一 encoder 的压缩。

## 交付

先交 Table 3／Table 4 两本表账、已有行的复用核对、外部模型与数据集缺口、现有任务的接续映射。记录 `comparison_axis`、encoder 身份、测量流程 ID、匹配条件 ID、两种样本量和 head／transform／split 来源，详见 docs/05 与 templates。

正文 §5.3 固定为 **Measuring What SSL Representations Retain**。写明测量器能否可靠检测指定关系，以及各 SSL 表示和预测的具体性质；不把“更容易检测颜色”写成“VCS 更可解释”，不把未拒绝写成信息不存在。两个类型的结果不互相代替。

运行顺序、已有授权与规模化 SSL 保持 r1。只对确有缺行的测量/读出单元拟合；新增长 GPU 训练仍按现有批准流程。论文参考文件使用新版原样副本，不直接覆盖服务器已经编辑的稿件或结果。当前 HEAD 的新科学结果另入 reconciliation，不伪称本包已经核查。

接入检查：`python tools/verify_package.py`、`python tools/check_contract.py`、`python tools/check_r2_sync.py`。三者都不是科学实验或作业提交命令。
