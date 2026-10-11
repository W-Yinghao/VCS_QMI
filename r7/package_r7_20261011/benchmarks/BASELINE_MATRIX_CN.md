# 必做对照与公开系统

每条实物方法/预算见同目录`baseline_matrix.csv`。下表是运行类别，不表示本包已复现。

|方向|方法|类别|实现模式|对照要求|
|---|---|---|---|---|
|D1|VCS,原J|core|matched_training|不是首次二次泛函；有限函数类/优化/数据角色明确|
|D1|Matched Logistic → same J|core|matched_training|相同输入/结构/监督/选择预算；Logistic也能读取原J|
|D1|Same-target kernel/relative regression|estimator_comparison|matched_readout|带宽、核维数与独立样本量/配对计算量分开|
|D2|VCS,原J|core|matched_training|不是首次二次泛函；有限函数类/优化/数据角色明确|
|D2|Matched Logistic → same J|core|matched_training|相同输入/结构/监督/选择预算；Logistic也能读取原J|
|D2|Same-target kernel/relative regression|estimator_comparison|matched_readout|带宽、核维数与独立样本量/配对计算量分开|
|D3|VCS,原J|core|matched_training|不是首次二次泛函；有限函数类/优化/数据角色明确|
|D3|Matched Logistic → same J|core|matched_training|相同输入/结构/监督/选择预算；Logistic也能读取原J|
|D3|Same-target kernel/relative regression|estimator_comparison|matched_readout|带宽、核维数与独立样本量/配对计算量分开|
|D4|VCS,原J|core|matched_training|不是首次二次泛函；有限函数类/优化/数据角色明确|
|D4|Matched Logistic → same J|core|matched_training|相同输入/结构/监督/选择预算；Logistic也能读取原J|
|D4|Same-target kernel/relative regression|estimator_comparison|matched_readout|带宽、核维数与独立样本量/配对计算量分开|
|D5|VCS,原J|core|matched_training|不是首次二次泛函；有限函数类/优化/数据角色明确|
|D5|Matched Logistic → same J|core|matched_training|相同输入/结构/监督/选择预算；Logistic也能读取原J|
|D5|Same-target kernel/relative regression|estimator_comparison|matched_readout|带宽、核维数与独立样本量/配对计算量分开|
|D6|VCS,原J|core|matched_training|不是首次二次泛函；有限函数类/优化/数据角色明确|
|D6|Matched Logistic → same J|core|matched_training|相同输入/结构/监督/选择预算；Logistic也能读取原J|
|D6|Same-target kernel/relative regression|estimator_comparison|matched_readout|带宽、核维数与独立样本量/配对计算量分开|
|D1|Full unpruned host|must_or_direct_prior|inference|完整计算与准确率基线|
|D1|Random / uniform token selection|must_or_direct_prior|inference|相同token数；相同生成参数|
|D1|FastV|must_or_direct_prior|official_inference|位置/attention backend及速度范围一致|
|D1|SparseVLM|must_or_direct_prior|official_inference|文本条件与剪枝层pin|
|D1|FEATHER|must_or_direct_prior|official_inference|报告视觉依赖任务，保留原生策略|
|D1|IF-Prune|must_or_direct_prior|official_training_inference|小模型/teacher额外开销及专有训练数据单列|
|D1|DiffPrune / ET-Prune|must_or_direct_prior|implementation_audit|2026预印本，代码未核实时不得假装复现|
|D1|Direct realized-risk selector|must_or_direct_prior|matched_training|同teacher输出同训练信息；强机制控制|
|D2|Base VLM|core_or_task_anchor|inference|相同prompt/maxnewtokens|
|D2|VCD|core_or_task_anchor|official_inference|增加forward的成本不能忽略|
|D2|M-HalDetect detector/reward|core_or_task_anchor|official_training|同人类标注预算|
|D2|Woodpecker|core_or_task_anchor|official_pipeline|检测/OCR/语言模型调用全部记录|
|D2|SCI|core_or_task_anchor|audit_optional|动态counterfactual题集与静态任务分表|
|D2|VQAScore|core_or_task_anchor|official_inference|用于图文支持/生成图时，不能冒充所有原子事实监督|
|D2|Direct error classifier; text-only classifier|core_or_task_anchor|matched_training|同架构extra监督分开，检查文本捷径|
|D3|Conditional/grouped RBF MMD|core|official_or_declared_adaptation|条件π和feature共同；与S数值不同|
|D3|CMMD|core|official_or_declared_adaptation|无条件CLIP-MMD；条件置乱时是负控制而非弱化主baseline|
|D3|CFID / cFreD|core|official_or_declared_adaptation|最直接条件分布方法；同reference/features|
|D3|FID / KID|core|official_or_declared_adaptation|样本数与编码器预处理一致|
|D3|Density / Coverage|core|official_or_declared_adaptation|多样性与忠实性分离|
|D3|TIFA / VQAScore|core|official_or_declared_adaptation|单图语义工具，不能顶替分布比较|
|D4|Raw CLIP / SigLIP2|core_or_task_anchor|complete_dual_tower_or_official_checkpoint|未适配能力|
|D4|Multi-positive InfoNCE / sigmoid|core_or_task_anchor|complete_dual_tower_or_official_checkpoint|与VCS同标签图、同负例信息|
|D4|PCME++|core_or_task_anchor|complete_dual_tower_or_official_checkpoint|同probabilistichead与train监督|
|D4|ProLIP|core_or_task_anchor|complete_dual_tower_or_official_checkpoint|公开checkpoint锚点与同架构loss控制分表|
|D4|FLAIR|core_or_task_anchor|complete_dual_tower_or_official_checkpoint|保留text-conditionedpooling|
|D4|LongProLIP|core_or_task_anchor|complete_dual_tower_or_official_checkpoint|可选workshop长文本方法，不混主会|
|D5|Random subset|core|selector_plus_full_student|全部策略使用同一pool与exposure|
|D5|CLIP score filtering|core|selector_plus_full_student|图片质量与匹配混合效应要记录|
|D5|DFN|core|selector_plus_full_student|独立filter训练数据与权重|
|D5|Sieve|core|selector_plus_full_student|captioner成本与许可|
|D5|SDM|core|selector_plus_full_student|目标数据对齐最直接先行，不能跳过|
|D5|Recaptioning control|core|selector_plus_full_student|相同captioner相同图片；区分选样与换caption|
|D6|Moment-DETR|core_or_optional|full_temporal_model|同CLIP/SlowFastfeature与官方loss|
|D6|QD-DETR|core_or_optional|full_temporal_model|query结构与监督完整|
|D6|CG-DETR|core_or_optional|full_temporal_model|时间相关/无关校准能力|
|D6|BM-DETR|core_or_optional|full_temporal_model|background-aware直接比较|
|D6|EaTR / Sim-DETR / SG-DETR|core_or_optional|full_temporal_model|至少选择一项新强法，SG额外数据/模型分栏|
|D6|MIL-NCE / NaQ|core_or_optional|full_temporal_model|只在弱叙述/ASR监督路线是直接训练对照|
