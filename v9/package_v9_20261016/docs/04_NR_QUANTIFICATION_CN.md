# 04 · NR-N1：比例与可信度估计

## 1. 三个场景分表
**N1-Exact（主数学验证）：**有限支持P/Q已知，R由iid混合生成；真实eta/w/rho可计算。检查比例、概率、风险及置信集。

**N1-Image（主自然表示验证）：**可信原始图文参考P、明确抽样的Q、未知混合比例R；按图像隔离拟合、校准、参考矩与批次。注入标签只给离线评分器。这里可评价生成机制标签Brier，不把原始caption保证为语义完美。

**N1-Stress（模型假设压力）：**只改变域不改变rho；相似物体hard mismatch；参考锚点有误；重复caption或新类别导致类条件改变。此处不声称混合置信区间必然覆盖。目标是揭示不可识别/错误解释，不隐藏失败。

## 2. 默认输入与critic
优先复用P49的COCO image-disjoint缓存作探索，记录其已被观察，不能将其称新独立确认。Flickr30K的正式角色重新锁定。第一次主量化使用共同teacher的**单一配对相似度s**，以隔离概率刻度，不借更强输入战胜标量基线。

VCS：T=tanh(a s+b)，a=softplus(a_raw)，避免score方向翻转。匹配BCE用同一自然logit 2(a s+b)，不是两种不同函数类。

确定性共同特征敏感性：加入teacher共同池化图文向量、低秩双线性(rank=16)和小MLP；所有方法得到同样输入。主效果只发生在一个受限类时，保留这个范围。不得在看到N1评价后再只留下该类。

## 3. 拟合与选择
- P/Q各1/2权重；源参考不是U噪声训练池；所有标准化仅FIT。
- 3个训练种子{901,902,903}，独立调参种子917；标量类用确定性多启动求解检查加同预算Adam敏感性。
- 学习率/正则/epoch在REF_SELECT上选择，主规则共同J；另保留各方法原生准则的敏感性。不能依据目标批次真实rho挑model。
- 原生BCE、温度/Platt/isotonic校准的BCE分开；所有校准使用相同reference budget。
- 核拟合记录带宽/中心/正则和边界。1D标量主面板不强迫KDE去做不必要的高维密度估计；共同向量敏感性允许PCA与带宽选择。

## 4. 必做量化方法
| 方法 | 使用信息/功能 |
|---|---|
| Raw-score moment | 未训练的共同score，rho=(a-c)/(a-b)；排除只是刻度改变 |
| VCS moment | VCS的固定score接同一矩式 |
| BCE moment | matched score；Logistic不是唯一外部基线 |
| BBSE hard/soft | 同一reference信息、混淆/均值矩阵；检查可逆性 |
| Bias-corrected calibration + MLE/EM | 按label-shift似然估计rho；强概率基线 |
| KDE posterior + moment/EM | 相同s或共同特征；标签与调参预算匹配 |
| RuLSIF(alpha=1/2) + moment/EM | 传统相对密度比；T=r_rel-1，裁剪与原始值均存 |
| Nyström squared regression + moment/EM | 调好核回归，不把RFF名当独立理论方法 |
| Kernel MPE（条件列） | 仅P+R输入和irreducibility条件与双参考不同，单列信息预算 |

先估计rho，再用同一rule计算可信度w；双参考方法可以复用相同rho估计接口以隔离后验拟合。真rho给权重的版本只能是oracle-rho诊断，不能进入部署主表。

## 5. 固定初始网格
主比例{0,.1,.3,.5}，独立批次大小{128,512,2048}。N1-Exact每格200个独立重复，检查区间覆盖；N1-Image先30个随机批次重复/模型种子，明确共享数据会导致重复间相关，不把30当30个新数据集。

画三个图：rho真实值与估计；相同覆盖下区间宽度；概率Brier与ESS。主比较按两个自然来源与4种比例平均RMSE分source报告，不按有利rho挑结果。

## 6. 主指标与失败处理
rho_raw、rho_clipped、absolute_error、RMSE、interval_width、coverage、empty/full interval率、有效P/Q/R簇数；样本成分Brier/logloss、AUROC、分桶校准；读数J作为辅指标。

弱分离时返回[0,1]或明确无法精确量化；数值分母检查不是成功标准。若纯域移位造成高“错配率”，报告模型假设失效，不能以更窄区间补救。增加多个固定score一致性检查可发现部分移位，但不能认证没有移位。

## 7. N1如何进入N2
N1不是“VCS赢BCE才能进入N2”的门。它要求定义/实现正确、权重可计算且无标签泄漏；即使N1方法接近，完整外部比较仍按预算执行。但如果部署接口在当前信息下不可识别，则先修改任务条件，而不是训练一个必然依赖oracle的系统。
