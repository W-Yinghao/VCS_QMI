# D6 服务器实验方案｜叙述视频的时间支持与定位

**性质：候选方案，尚无本方案自然数据结果。** 先读本目录综述和共同理论/评价规范。数据条目：DS29, DS30, DS31, DS32, DS33, DS34, DS35；全部定义见 `../../datasets/DATASET_ATLAS_CN.md`。

## A. 完整任务与宿主

采用Moment-DETR或已能严格复现的QD-DETR作为同架构host，输入官方同版SlowFast/CLIP等features与text tokens，输出若干[start,end,score]及clip saliency。第一轮训练完整融合/定位模型，不只是冻结features上的一个小判别probe。若研究终点是计算节省，最后另做raw-video end-to-end计时与适配，不能拿缓存特征耗时替代。

## B. 两种监督情形分开

**强监督主线：**QVHighlights query的发布相关区间定义support，在其标注协议下建立P(+支持clip|query/video)与Q(已定义背景clip|query/video)。发布标签只代表这个annotation task，不保证所有背景在真实世界绝无关联；边界邻域可标uncertain并在所有方法中统一处理。

**弱监督扩展：**HowTo100M或ASR只能提供正袋，不能将每个时间点当确定标签。必须使用MIL-NCE类袋目标或明示潜在变量模型；不在第一轮强监督recipe里混入弱袋再叫同P/Q。

VCS/Logistic support head同函数类，T在平衡实验上拟合，部署时映射到实际片段support先验再校准。主任务span L1/GIoU/Hungarian或host原损失保留；只有support学习组件替换。新增loss权重在相同CAL预算内选择，不让ours特调而host只用旧默认。

## C. 首轮实验清单

| 单元 | 要做什么 |
|---|---|
| D6-00 | 官方Moment-DETR/QD-DETR training和evaluation fixture；视频切片parent ID、帧率/特征步长/秒单位检查 |
| D6-10 | 同一host标准support loss、matchedLogistic、VCS、VCS+预定义时间结构控制，共4方法×3seeds完整训练 |
| D6-20 | 公共CAL概率校准/时间支持评价，外加MR/HD；时间结构也给Logistic同机会以判断归因 |
| D6-30 | Charades-STA第二数据集，同主要目标3seeds；不自动复制所有ablation |
| D6-40 | 明确弱监督时间噪声下的MIL-NCE对照，或长视频域验证二选一 |

D6-10第四臂只有预算允许且host存在时间结构时运行；若新增结构，必须同时跑JS+同结构，不保持不公平的四臂计数。基础3×3=9为最小，结构成对新增6。自然benchmark结果必须完整长度/多区间，不能只评oracle候选片段。

## D. Benchmark与metrics

QVHighlights：MR mAP@[.5:.05:.95]及R@1@.5/.7；HD mAP/Hit@1按官方saliency阈值；短/中/长事件分层。输入输出使用官方JSONL(qid,vid,pred_relevant_windows,pred_saliency_scores)，先跑sample fixture数值校验。test已公开但保持封存。

Charades-STA：R@1/5 at .3/.5/.7和mIoU，报告原feature类与是否额外预训练。Ego4D仅扩展时使用NLQ R@1/5@.3/.5，不混MQ。自定义support指标包括Brier、frame/clip PR、区间覆盖/长度和calibrated set coverage，均单列custom，不能替代官方MR/HD。

干预分析：固定视频和query，在FIT-only构造时间偏移或背景扩展，DEV用于预定stress。视频反转未必语义不变；无依据不能把它当正对增强。参考负片段的时间距离分布、持续长度和sampler在三方法间相同。

## E. 预期与转向

希望共享平方读数帮助更准确地表达多段支持并改善完整定位/片段选择；无需所有R@1都第一，但必须有实际的risk–localization或cost–quality收益。若改善仅在自制抖动数据，不能宣称自然噪声鲁棒。

若所有收益来自额外时间平滑/更强feature，重新归因；若VCS support校准改善但MR不变，检验它是否服务多区间/选择性定位，不凭内部J自动命名新能力。若问题仍只是softmax竞争，停止在同一视频上继续换参考常数，考虑D2支持核验的完整用途。
