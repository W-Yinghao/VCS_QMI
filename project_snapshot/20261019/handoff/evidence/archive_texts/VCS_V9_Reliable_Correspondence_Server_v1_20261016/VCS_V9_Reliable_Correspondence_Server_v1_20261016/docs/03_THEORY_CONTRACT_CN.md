# 03 · 理论与实施对象：保持S/J，补齐使用条件

## A. 原理论 [R05](local-source://conversation/Variational-CS-QMI-and-supplement)
设P,Q为两个概率分布，M=(P+Q)/2，eta=(p-q)/(p+q)。

\[
S=E_M\eta^2,\qquad J(T)=E_PT-E_QT-\tfrac{1}{2}E_PT^2-\tfrac{1}{2}E_QT^2,\quad S-J(T)=E_M(T-\eta)^2.
\]

这是总体恒等式，不是每次经验读数的下界。基础J与平衡RPC重合；密度比回归、正规评分及CS参考联系分别归属，不认领新散度。[L13,L14]

V9的P/Q可以是“原始标注配对/规定随机重配”；只有严格联合对乘积构造才称QMI。拒绝同图donor的Q是条件化抽样分布，不等同于无条件乘积Q。

## B. 本包直接推导1：任意固定读数的比例估计
令R_rho=(1-rho)P+rho Q。对固定且a!=b的T，a=E_PT,b=E_QT,c=E_RT，则
\[
\rho=(a-c)/(a-b).
\]
不需要T已达到eta，也不需要概率校准。对T=eta，E_P eta=S、E_Q eta=-S，所以a-b=2S。

**关键实施含义：**
- 此公式本质属于矩匹配/label shift思路，BBSE是强对照，不是新定理。[L10](https://proceedings.mlr.press/v80/lipton18a.html)
- T换成uT+v(u!=0)，同样本的比例点估计严格不变。线性尺度优化不能单独创造比例估计优势。
- 单调但非线性变换可能改变矩估计效率；更小后验MSE或更高J不保证该比值方差最小。
- a-b接近0时问题弱识别。报告原始比值、投影到[0,1]的显示值与识别状态；不能仅裁剪后声称准确。
- 已知P/Q参考和一份混合不是单成分PU问题。只有混合样本、没有参考或其他可识别条件时，不能无条件识别rho。[L12](https://proceedings.mlr.press/v48/ramaswamy16.html)

若独立估计三个均值，delta近似方差为
\[
\operatorname{Var}(\hat\rho)\approx[\operatorname{Var}(\hat c)+(1-\rho)^2\operatorname{Var}(\hat a)+\rho^2\operatorname{Var}(\hat b)]/(a-b)^2.
\]
这是诊断近似，不用于弱分离的主置信区间。

## C. 本包直接推导2：弱分离时用集合反演
对互相独立、每个值在[-1,1]内的P/Q/R评价簇均值，各有n_P,n_Q,n_R个独立簇。令
`e_j=sqrt(2 log(6/alpha)/n_j)`。
用并合界至少1-alpha概率同时覆盖三个均值。比例置信集为
\[
\mathcal{C}=\{r\in[0,1]: |\hat c-(1-r)\hat a-r\hat b|\le e_R+(1-r)e_P+r e_Q\}.
\]
这是两个线性不等式的交集；不需要除以小的估计分母。输出可能是宽区间、[0,1]或空集。空集表示该单一矩在规定误差带内不相容，不自动定位原因；非空也不证明没有域移位。

**仅当相应独立簇/有界/混合假设成立才使用该有限样本解释。** 多caption、交叉配对共享图像时，pair数不是独立n。自然数据普通bootstrap另报告为重采样近似；refit bootstrap需在每次重复重做被抽样的拟合/选择步骤。

参考模块的`rho_confidence_set`只实现上述数学。调用者必须提供独立性检查，不能仅传一个`True`替代数据证明。

## D. 本包直接推导3：正确的成分后验
如果T逼近的是**干净参考P对Q**的eta，则可信成分后验为
\[
w_\rho(z)=\frac{(1-\rho)(1+T(z))}{(1-\rho)(1+T(z))+\rho(1-T(z))}.
\]
直接输出sigma(logit(1-rho)+2f)，T=tanh f，避免先除出巨大p/q。rho=0/1的合法支持点单独处理；数值裁剪率要记录。若T在R对Q上训练，公式不能直接套用：必须单独定义其密度比并检查识别/反演误差。

w是“该配对来自规定P成分的概率”，不自动是语义满足概率。随机重配中偶然正确的caption仍来自Q；语义标签另设，不能暗中更换真值。

## E. 重要补充：权重并不无条件恢复任意student的风险
若W是全部观测，C表示来自P，真实w(W)=E[C|W]，对固定可积loss ell(W)：
\[
E_R[w(W)\ell(W)]=(1-\rho)E_P\ell(W).
\]
若额外负样本U与(C,W)独立，固定分布，则同式对ell(W,U)也成立。

但实际critic常只读Z=phi(W)。即使准确学到w_Z=E[C|Z]，对读取更多信息的student也有
\[
E_R[w_Z\ell(W)]-(1-\rho)E_P\ell(W)
=-E_R\operatorname{Cov}(C,\ell(W)\mid Z).
\]
再有估计误差时，另加`E[(w_hat-w_Z)ell]`。如果|ell|<=L，该额外项绝对值不超过`L sqrt(E(w_hat-w_Z)^2)`；**协方差项不会自动消失**。

这说明“teacher特征后验准确 → 原图student无偏学习”不是定理。只有loss对Z可测，或相应充分性条件成立时才可省略该项。参考测试包括Z为常数而C与loss相关的反例。

同样，SGR/CLIP的常规in-batch hard negatives、不同权重改变的采样、阈值删对、EMA重拟合，都可能改变目标。完整任务仍可实证有效，但不能把单样本恒等式作为整个批次损失的精确保证。

## F. 固定权重的优化规范
首轮critic冻结、权重detach；student不通过权重反向更新critic。训练集尺度常数在启动前固定，不按当前minibatch的sum(w)改变有效目标。若为了稳定采用自归一化权重，它是有限样本有偏的ratio estimator，应命名并对所有方法共享。

不要直接乘原生softmax logit：那会改变比较对象。对匹配loss的每个anchor行/列损失乘对应w_i，具体改动写diff。原始权重、加权后总质量、有效样本量(ESS)均记录。

## G. REG中的目标
固定T时优化J(T;P_phi,Q_phi)是一个学习到的能量，不等于每个phi的真实S。监督错配、测试局部缺失分别改变不同分布。成功率、优化预算与真实几何误差才是应用终点。

## H. 新颖性与证据
本包不把矩匹配、BBSE、EM、CS点集配准、L2投影、重要性加权重新称为新发明。候选新增内容是：在明确参考条件下完成可量化的对应可靠性与学习接口，并在强外部方法上验证收益。最终新颖性是否充分，需要实际方法和结果支持。
