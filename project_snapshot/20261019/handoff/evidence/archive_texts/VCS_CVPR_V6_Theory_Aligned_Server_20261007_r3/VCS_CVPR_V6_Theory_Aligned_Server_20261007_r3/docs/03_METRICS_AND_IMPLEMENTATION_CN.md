# 03｜指标、概率对象和实施约定

此文件细化r3建议，不追溯改写已有frozen协议。原理论来源[O1]；“建议/推导”表示本次将其转为可执行检查。包内工具是NumPy参考，不是可替换现有trainer的实现。

## 1. 目标不混淆

$M=(P+Q)/2$，$\eta=(p-q)/(p+q)$，$S=E_M\eta^2$，$J(T)=E_Pa_+(T)+E_Qa_-(T)$，$a_+(t)=t-t^2/2$，$a_-(t)=-t-t^2/2$。S是half triangular discrimination；混合参考CS为$2\log((1+S)/(1-S))$，不是经典固定参考CS，也不是Shannon MI。

对同一T，在独立EVAL记录：`J_common`、`S_plugin=(mean T_P²+mean T_Q²)/2`、T的分位数、接近±1比例；保留原始J，不能截零后报误差。对数转换另字段注明投影/clip比例与规则；主比较用S坐标。接近1时原CS转换导数$4/(1-S²)$放大噪声，bounded T不等于转换值均匀稳定。

**Logistic坐标。** 若logit为u且D=sigmoid(u)，common T=tanh(u/2)；若用半logit f且D=sigmoid(2f)，T=tanh(f)。必须存`logit_coordinate`，不能默默差一倍。平衡指P/Q两类总权重相同，不是每个pair等权后让K决定先验。

## 2. 拟合误差与评价误差

[原理论] $S-J(T)=\epsilon_T=E_M(T-\eta)^2$。

[沿原gap推导] 固定已拟合T，P/Q两个iid评价池彼此及与FIT/VAL独立时，

$$E[(\widehat J-S)^2\mid T]=\epsilon_T^2+\operatorname{Var}_P(a_+(T))/n_P+\operatorname{Var}_Q(a_-(T))/n_Q.$$

这里是平方后验MSE的**平方**成为估值的偏差平方。$\epsilon_T$在真实图像未知；区间不能把这个未知偏差自动消掉。经验$\widehat J$不是逐次下界，P108已有明确修正。

因$a_\pm\in[-1.5,0.5]$，可为固定T的J给保守Hoeffding半径

$$b_\delta=\sqrt{2(1/n_P+1/n_Q)\log(2/\delta)}.$$

它不是S的双侧区间；$\max(0,\widehat J-b_\delta)$可作S的保守下界。若在同一EVAL上选critic/噪声/层，该式不能直接用来忽略选择影响。主实验应独立选择后评价。真实多层/多水平同时区间需预定multiplicity规则，逐格区间只称点态。

## 3. 哪一种方差

令A为base样本/增强，R为配对抽样，F为拟合数据/初始化。分别报告：

- `eval_var_conditional_fit`：固定F和模型，重抽A；
- `pair_var_conditional_batch`：固定F、A，只重抽R；
- `refit_spread_fixed_pool`：固定输入样本，只变优化初始化；
- `refit_spread_resampled_pool`：同时改变FIT样本的端到端拟合差异；
- `encoder_seed_spread`：被测encoder预训练种子变化。

$\operatorname{Var}(\widehat J)=\operatorname{Var}_A E_R[\widehat J\mid A]+E_A\operatorname{Var}_R(\widehat J\mid A)$解释的是相同总体/算法下的方差来源，不承诺改训练后仍保持相同模型。

独立P/Q池：对每个池独立bootstrap，再在方法/变换之间共享该池的索引。成对预测效应：按base image一起重采样clean/modified。共享full-pair U-statistic：按独立原始观测重新构造边或使用独立块；不能对边随机bootstrap而忽略共端点。工具中的iid两池公式只接受第一种设计。

## 4. 分辨能力与跨目标比较

同目标：bias、RMSE、posterior MSE、oracle gap、误差与计算预算。真值MC的标准误单独报告。

不同目标：每种方法按自身定义报告native值、与自身真值/参考的差、相邻水平排序概率与时间。InfoNCE的finite-K population optimum与Shannon MI是不同对象；SMILE带clipping的oracle-logit读数不自动是真正优化极值。无法确定native oracle的字段留null，不能填MI冒充。

`adjacent_order_probability`通过原始独立重拟合/评价重复估计，标明是否coupled。`abs mean difference / pooled sd`可以辅助，但它只对仿射尺度变化不敏感，不是对任意单调变换不变的万能“公平分辨率”。若所有重复的差为零，标不可分；不能因为sd=0返回无限质量。

raw loss梯度范数与原生objective的标度相关。各方法都可报告，但跨目标不按绝对数值选赢家。可另报$\operatorname{tr}\operatorname{Cov}(g)/\|E g\|^2$与方向一致性；当均值与其抽样噪声相当时，归一化值标无信息，并显示分子分母。禁止用随结果挑的epsilon掩盖近零。

## 5. 三种梯度分开

1. `critic_parameter_gradient`：$\nabla_\phi L$，反映拟合过程，受网络参数化影响。
2. `generator_parameter_gradient`：给定生成器$P_\rho$、冻结critic参数，通过样本对$\rho$求$\partial_\rho J$。可与oracle的同目标导数比較，不包含优化器unroll。
3. `encoder_training_gradient`：真实SSL一步对encoder/projector求导。P138的$g_{all}$是固定batch的full-pair梯度，不是总体oracle，已经完成的只读结果直接复用。

对于VCS/JS，半logit f下单pair损失导数分别为$(T-c)(1-T^2)$和$T-c$。这说明更新权重不同，不证明高置信错误一定被正确处理；P145只能报告实际gradient share与最终表现，二者因果关系尚未建立。

原计划不含指数矩指J，不是说神经网络参数梯度必然有界。仍需记录Jacobian/grad tails/非有限步，不能由bounded T直接宣称所有训练稳定。

## 6. 真实视觉fixture的明确默认值（待冻结）

- 数据：CIFAR-10、CIFAR-100；被测模型：最终VCS与SimCLR的R18标准增强800epoch，优先encoder seed1；禁止自动改为分数最好的checkpoint。
- 表示：主块h；z和其他encoder后续按固定范围补，不先展开全笛卡尔积。
- 测量增强：P122的同一个standard law。训练增强与测量增强分别记录，不能把变化一律解释为数据处理。
- FIT/VAL/EVAL：复用20k base-image划分或明确新建；每个base的全部增强和noise replicas留在同一role。
- 拟合主样本量：P/Q各2048，优先训练侧K8/in-batch可用实现；方法间样本曝光和actual pair counts记录。最终EVAL使用主计划的独立三池构造，P与Q各2000pair。
- 拟合候选上限：每方法三项，例如神经lr={1e-4,5e-4,2e-3}，RFF带宽乘子={0.5,1,2}、固定m=1024及已有有界读出。这是r3提出的有限预算，非原计划给定最优；接口不能恢复就先填协议差异，不猜CLI。
- 保存全部候选TUNE分数和选定步；选定权重只对EVAL读一次。逐样本分数可重复bootstrap；不能再根据EVAL改善选择另一候选。
- 注意Nfit含正负各自独立单位与训练中的配对复用不是同一概念。共享正数据训练kernel等方法照原算法，但评价保持独立；统计记录必须保留实际样本ID而不是仅写N。

## 7. 坐标与噪声的正确用法

FIT上对两个view分别拟合固定的$\mu_j$和标量$\gamma_j^2=E\|H^{(j)}-\mu_j\|^2/d$，仅当$\gamma_j>0$使用$(H-\mu_j)/\gamma_j$；零方差表示单列，不能除epsilon后假装正常。预处理为可逆仿射映射，t轴随各模型的固定基准尺度定义；跨模型t是标准化后的噪声耐受，不是原始绝对噪声统一比较。

旋转$R_1,R_2$固定，refit不能只转EVAL。列向量下full-pair MLP第一层W运输为$W'=W\,\operatorname{blockdiag}(R_1^T,R_2^T)$。无需改变bias。共同旋转保持余弦，但两端独立旋转通常不保持余弦；二者不能混同。

Brownian增量：对t递增，用独立增量$\sqrt{t_k-t_{k-1}}\epsilon_{j,k}$；两端和不同base独立，同一base跨t用共同增量耦合。P/Q用同一channel law而不是相同噪声实现强行关联不同pair；不能只给P加噪。为正确重放，把base_id、role、view、channel seed、level和checkpoint哈希写入manifest，按base生成或用可恢复counter。

不在加噪后重新单位化。如果实验确需单位化，应另定义channel、重新证明比较关系，不能沿用本块的Brownian DPI。固定PCA截断不是可逆变换，不能放不变性列。

## 8. 不把必要条件当充分证明

- $T=0$：方差零、可逆漂移零、噪声轴不增加，但非零S的MSE可很大。
- 单调变换$1000\widehat J+7$：置换p值相同，数值目标错误。
- 单个raw $\widehat J$可以大于S；population lower bound不等于经验下界。
- 望远镜和为零可以由代数恒等成立，不能验证所有critic最优；$D_T\simeq0$可能是nested沿用coarse初始化。
- noise通道的所有读数接近0或1，不提供有力分辨证据；既报范围也报稳定性。
- 增量理论、独立评价MSE和JS→CS风险控制都不蕴含SSL accuracy单调改善。

## 9. 本地参考代码的边界

`tools/reference_core.py`只包含小型概率表、J/plug-in、独立池误差、置换排序、采样方差、可重放噪声和配对统计参考。它不读取服务器权重，不实现新的神经优化器，不宣称提供已证明的通用数据驱动S置信上界。server应在其现有runner上对齐这些约定，不替换已有协议。
