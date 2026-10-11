# 数据集与评价接入图谱

本表列出35个数据/协议条目，不等于35套独立图像来源。很多COCO、Flickr派生集共享原图。入口可查不等于数据已下载或所有许可证已接受。

执行前必须提交原始ID去重、许可、内容hash、可访问样本数、官方评估器commit和预训练暴露记录。涉及账号、付费裁判、受限下载，按现有授权处理，不代用户接受条款。

## DS01 — GQA (balanced / original release)
**用途：** D1;D2；train+CAL+official eval。
**独立单位与划分：** original image / scene graph。官方balanced train内按image分FIT/CAL；test-dev按预注册一次读；不得将同图问题跨角色。
**指标：** official answer accuracy; consistency/validity/plausibility等另列。
**获取：** https://cs.stanford.edu/people/dorarad/gqa/index.html
**评价入口：** https://cs.stanford.edu/people/dorarad/gqa/evaluate.html
**风险／权限：** VG预训练region features可能见过部分GQA val；按官方test-dev建议核对；yes/no子集为自定义辅助任务不冒充全GQA。
**来源：** B01（见 ../literature/SOURCE_CATALOG_CN.md）。

## DS02 — TextVQA
**用途：** D1;D2；train/CAL and held-out eval。
**独立单位与划分：** original image。官方train内图像分组；val/test沿官方；图片与TextCaps等共享源排重。
**指标：** 官方10人答案VQA soft accuracy。
**获取：** https://textvqa.org/
**评价入口：** https://github.com/facebookresearch/mmf/blob/main/mmf/utils/m4c_evaluators.py
**风险／权限：** 区分image-only与提供OCR文字/框；图像许可与标注许可分别核验；答案处理器pin commit。
**来源：** B02;T02;T03（见 ../literature/SOURCE_CATALOG_CN.md）。

## DS03 — SP-DocVQA (single-page, 2020/2021)
**用途：** D1；train/CAL and eval。
**独立单位与划分：** source document; page family。官方train/val/test；内部按文档/模板源group，不只问答ID。
**指标：** ANLS; exact match辅助。
**获取：** https://www.docvqa.org/datasets/docvqa
**评价入口：** https://rrc.cvc.uab.es/
**风险／权限：** 下载/提交可能要求注册；本包不接受条款/不上传结果；不能混为DocVQA2026。
**来源：** B03（见 ../literature/SOURCE_CATALOG_CN.md）。

## DS04 — ChartQA
**用途：** D1；train/CAL/eval。
**独立单位与划分：** chart image; source chart/template。官方human/augmented分别记录；同图两类问题同角色。
**指标：** numeric relaxed accuracy within5%; nonnumeric exact; human/augmented分开及官方平均。
**获取：** https://github.com/vis-nlp/ChartQA
**评价入口：** https://github.com/vis-nlp/ChartQA
**风险／权限：** gold table/OCR作为额外输入单列；数值零、百分号、列表/单位以锁定官方代码为准。
**来源：** B04;T05（见 ../literature/SOURCE_CATALOG_CN.md）。

## DS05 — MMBench (pin language/version)
**用途：** D1；evaluation only in first packet。
**独立单位与划分：** image/question family。固定en或cn及v1.1等版本，不混多次排行榜。
**指标：** 官方MCQ/circular evaluation版本。
**获取：** https://github.com/open-compass/MMBench
**评价入口：** https://github.com/open-compass/VLMEvalKit
**风险／权限：** 答案解析可能需judge/API；先固定解析器；不能仅换prompt获取最好分。
**来源：** T02;T07（见 ../literature/SOURCE_CATALOG_CN.md）。

## DS06 — ScienceQA image subset
**用途：** D1；evaluation secondary。
**独立单位与划分：** question/image。明确IMG子集；不以全ScienceQA分母报告IMG成绩。
**指标：** multiple-choice accuracy。
**获取：** https://scienceqa.github.io/
**评价入口：** https://github.com/lupantech/ScienceQA
**风险／权限：** 有无解释/CoT/文本背景相同，图像缺省样本不能混入视觉压缩效果。
**来源：** T02;T06（见 ../literature/SOURCE_CATALOG_CN.md）。

## DS07 — M-HalDetect
**用途：** D2；verifier/reward train and grouped CAL; holdout transfer。
**独立单位与划分：** original image + response generator。沿官方划分；生成模型留出另做；同图回答/片段不跨角色。
**指标：** 官方span/response判别; supported-vs-contradicted Brier/AUPRC自定义补充。
**获取：** https://github.com/hendryx-scale/mhal-detect
**评价入口：** https://github.com/hendryx-scale/mhal-detect
**风险／权限：** 标签可能含分析/不可验证；先审schema，不把所有非支持都当错误；冻结judge，图片COCO重叠审查。
**来源：** H02（见 ../literature/SOURCE_CATALOG_CN.md）。

## DS08 — SNLI-VE
**用途：** D2；optional supervised entailment train/eval。
**独立单位与划分：** Flickr original image。图像级官方train/val/test；Flickr检索同源排重。
**指标：** 3-way accuracy; macro-F1/Brier supplementary。
**获取：** https://github.com/necla-ml/SNLI-VE
**评价入口：** https://github.com/necla-ml/SNLI-VE
**风险／权限：** entailment/neutral/contradiction分开；纯文本可解偏差控制；neutral不是明示false。
**来源：** H11（见 ../literature/SOURCE_CATALOG_CN.md）。

## DS09 — POPE
**用途：** D2;D1；evaluation only。
**独立单位与划分：** original COCO image。random/popular/adversarial三配置分别锁定；不用于训练verifier。
**指标：** accuracy; precision; recall; F1; yes ratio。
**获取：** https://github.com/AoiDragon/POPE
**评价入口：** https://github.com/AoiDragon/POPE
**风险／权限：** 对象存在性范围有限；所有方法相同生成/解析规则；子集图片可能已在其他任务出现。
**来源：** H05（见 ../literature/SOURCE_CATALOG_CN.md）。

## DS10 — HallusionBench
**用途：** D2；evaluation only。
**独立单位与划分：** original/edited-image family and question-pair family。完整官方split及VD/VS；不将image-only镜像951样本自动等同原1129问答。
**指标：** aAcc; fAcc; qAcc分别报告。
**获取：** https://github.com/tianyi-lab/HallusionBench
**评价入口：** https://github.com/tianyi-lab/HallusionBench
**风险／权限：** 官方自动评价和自定义strict yes/no分开；API调用需已授权；配对不能打散bootstrap。
**来源：** H06（见 ../literature/SOURCE_CATALOG_CN.md）。

## DS11 — AMBER
**用途：** D2；evaluation only。
**独立单位与划分：** image。generative与discriminative及existence/attribute/relation分项。
**指标：** CHAIR/Cover/Hal/Cog; no-class P/R/F1; overall Acc。
**获取：** https://github.com/junyangwang0410/AMBER
**评价入口：** https://github.com/junyangwang0410/AMBER
**风险／权限：** 判别P/R/F1正类是幻觉问题(真值no)，不能反用POPE模板；规则库/spaCy版本pin。
**来源：** H07（见 ../literature/SOURCE_CATALOG_CN.md）。

## DS12 — COCO captions / Karpathy split
**用途：** D2;D3;D4；train/secondary eval; D3 grouped reference only after audit。
**独立单位与划分：** COCO original image。Karpathy train/val/test与COCO2014/2017编号对应；跨RefCOCO/POPE/GQA关系图审查。
**指标：** D4 bidirectional R@1/5/10,1k5fold与5k分开; D2 CHAIR/CIDEr。
**获取：** https://cocodataset.org/
**评价入口：** https://github.com/karpathy/neuraltalk2
**风险／权限：** 五条caption不是五个独立图像；未标注对未知；不把单图caption当完整条件生成真值。
**来源：** F05;M05;M06;H08（见 ../literature/SOURCE_CATALOG_CN.md）。

## DS13 — CLEVR / controlled rendered extension
**用途：** D1;D3；controlled checks; not substitute for natural task。
**独立单位与划分：** scene latent/template family。先生成并冻结基础scene split，再渲染/改述/污染；所有同scene变体同组。
**指标：** exact answer; finite S truth; condition/mode error。
**获取：** https://cs.stanford.edu/people/jcjohns/clevr/
**评价入口：** https://github.com/facebookresearch/clevr-dataset-gen
**风险／权限：** 自定义信道须标custom；只有观察映射已知时才能称observed S oracle；renderer版本/许可证核对。
**来源：** B05（见 ../literature/SOURCE_CATALOG_CN.md）。

## DS14 — GenEval
**用途：** D3;D2T2I；generation evaluation only。
**独立单位与划分：** prompt template / concept family。官方prompt list与原生默认采样数固定；自定义难度训练不重用eval prompts。
**指标：** six official subcategory scores and overall。
**获取：** https://github.com/djghosh13/geneval
**评价入口：** https://github.com/djghosh13/geneval
**风险／权限：** detector version阈值pin；语义分不是分布距离真值；有无best-of-N不同预算单列。
**来源：** G04（见 ../literature/SOURCE_CATALOG_CN.md）。

## DS15 — T2I-CompBench original
**用途：** D3；evaluation only。
**独立单位与划分：** prompt/family。original6000/6cat；不要与++混用。
**指标：** 原版attribute/relation/compositional evaluator outputs。
**获取：** https://github.com/Karine-Huang/T2I-CompBench
**评价入口：** https://github.com/Karine-Huang/T2I-CompBench
**风险／权限：** 多种评估器含LLM/VQA；记录缓存题目、权重与调用成本。
**来源：** G05（见 ../literature/SOURCE_CATALOG_CN.md）。

## DS16 — T2I-CompBench++
**用途：** D3；extension evaluation。
**独立单位与划分：** prompt/family。++8000/8cat固定release。
**指标：** ++官方分项与聚合。
**获取：** https://github.com/Karine-Huang/T2I-CompBench
**评价入口：** https://github.com/Karine-Huang/T2I-CompBench
**风险／权限：** 必须独立version字段；原版eval不能沿用++名称。
**来源：** G06（见 ../literature/SOURCE_CATALOG_CN.md）。

## DS17 — ConceptMix
**用途：** D3；evaluation/controlled difficulty。
**独立单位与划分：** prompt/concept tuple。固定概念k、缓存prompts/questions；template families heldout。
**指标：** per-concept and all-constraints success。
**获取：** https://princetonvisualai.github.io/conceptmix/
**评价入口：** https://princetonvisualai.github.io/conceptmix/
**风险／权限：** 生成题目和裁判可能用付费LLM；不得静默自动调用；检查human subset。
**来源：** G07（见 ../literature/SOURCE_CATALOG_CN.md）。

## DS18 — GenAI-Bench / BaiqiL release
**用途：** D2T2I;D3；human evaluation; no verifier training on holdout。
**独立单位与划分：** prompt and generator family。固定HF版本; train/eval by prompt、generator，旧版/新版标注数不混。
**指标：** human alignment correlation; pairwise accuracy; GenAI-Rank bestof。
**获取：** https://huggingface.co/datasets/BaiqiL/GenAI-Bench
**评价入口：** https://github.com/linzhiqiu/t2v_metrics
**风险／权限：** 与GenAI Arena同名数据分开；多版本标注量变化，按本地清单统计，不写固定通用80k。
**来源：** G08;H10（见 ../literature/SOURCE_CATALOG_CN.md）。

## DS19 — HPDv2 / HPSv2
**用途：** D3；external human preference evaluation。
**独立单位与划分：** prompt/grouped generation。使用官方heldout，不作为拟合critic训练数据。
**指标：** human pair preference accuracy/correlation。
**获取：** https://github.com/tgxs002/HPSv2
**评价入口：** https://github.com/tgxs002/HPSv2
**风险／权限：** 偏好不等于支持事实；固定judge/model，授权与素材许可核验。
**来源：** G03（见 ../literature/SOURCE_CATALOG_CN.md）。

## DS20 — PartiPrompts
**用途：** D3；prompt stress evaluation。
**独立单位与划分：** prompt/concept family。发布prompt分组固定；没有真实图集oracle。
**指标：** 官方/预注册human or semantic evaluators。
**获取：** https://github.com/google-research/parti
**评价入口：** https://github.com/google-research/parti
**风险／权限：** prompt-only不能估计未知真实条件图像分布。
**来源：** G03（见 ../literature/SOURCE_CATALOG_CN.md）。

## DS21 — Flickr30k captions
**用途：** D4；training/transfer evaluation。
**独立单位与划分：** original image。官方/Karpathy image split标明；SNLI-VE/FlickrEntities共享排重。
**指标：** i2t/t2i R@1/5/10。
**获取：** https://shannon.cs.illinois.edu/DenotationGraph/
**评价入口：** https://github.com/naver-ai/pcmepp
**风险／权限：** 图像下载许可可能受限；五描述不是全正例图；不得复用已消费DEV当独立确认。
**来源：** M01;M02（见 ../literature/SOURCE_CATALOG_CN.md）。

## DS22 — CUB-200-2011 + Reed captions
**用途：** D4；controlled class-heldout retrieval。
**独立单位与划分：** bird image and class。PCME150train/50heldout class protocol复核；caption/image共同分组。
**指标：** class-relevant retrieval / PMRP per code; query-image matching分开。
**获取：** https://github.com/naver-ai/pcme
**评价入口：** https://github.com/naver-ai/pcme
**风险／权限：** 同类正例协议是类相关而非逐声明视觉真值；文本数据单独许可。
**来源：** M01（见 ../literature/SOURCE_CATALOG_CN.md）。

## DS23 — CxC / Crisscrossed Captions
**用途：** D4；evaluation only。
**独立单位与划分：** COCO image and paraphrase family。官方COCO eval索引；稀疏人工judgment范围。
**指标：** semantic score correlation; expanded retrieval。
**获取：** https://github.com/google-research-datasets/Crisscrossed-Captions
**评价入口：** https://github.com/naver-ai/eccv-caption
**风险／权限：** 267095标注对非独立样本；0–5相似度非匹配后验，未judge不能自己当负。
**来源：** M05（见 ../literature/SOURCE_CATALOG_CN.md）。

## DS24 — ECCV Caption
**用途：** D4；evaluation only。
**独立单位与划分：** COCO image/caption query family。Karpathy test子集1332query images/1261query captions按官方索引。
**指标：** eccv_map_at_r; eccv_rprecision; recalls。
**获取：** https://github.com/naver-ai/eccv-caption
**评价入口：** https://github.com/naver-ai/eccv-caption
**风险／权限：** mAP@R的R是已发布合法相关数；保留官方评价宇宙，另对fully judged子集评风险。
**来源：** M06（见 ../literature/SOURCE_CATALOG_CN.md）。

## DS25 — SUGARCREPE++
**用途：** D4;D2；evaluation only。
**独立单位与划分：** COCO original image/triplet。官方两正一负triplet完整保留；不加入训练伪正例。
**指标：** 官方triplet语义/词汇一致性指标。
**获取：** https://arxiv.org/abs/2406.11171
**评价入口：** https://arxiv.org/abs/2406.11171
**风险／权限：** 服务器从原论文核到author dataset后冻结hash；先不猜数据仓库。
**来源：** M08（见 ../literature/SOURCE_CATALOG_CN.md）。

## DS26 — DataComp-CLIP Small
**用途：** D5；complete downstream-training benchmark。
**独立单位与划分：** source image; domain/duplicate cluster。官方pool+12.8M training exposure设置核验；内部小样本另名。
**指标：** 38 downstream任务官方summary; ImageNet/检索分项。
**获取：** https://www.datacomp.ai/dcclip/getting_started.html
**评价入口：** https://github.com/mlfoundations/datacomp
**风险／权限：** URL失效与安全过滤所有臂同处理；需磁盘/下载授权，代码许可不等于图像许可；不可默下完整medium。
**来源：** D01（见 ../literature/SOURCE_CATALOG_CN.md）。

## DS27 — Conceptual Captions 3M
**用途：** D5；bounded fallback training, not DataComp。
**独立单位与划分：** source image/domain。先共同抓取固定pool，按域/图片hash分target-reference与源。
**指标：** fixed student heldout classification/retrieval。
**获取：** https://ai.google.com/research/ConceptualCaptions/
**评价入口：** https://github.com/mlfoundations/open_clip
**风险／权限：** 下载失效清单、许可审核；不是官方DataComp-small。
**来源：** D01;D03（见 ../literature/SOURCE_CATALOG_CN.md）。

## DS28 — Recap-DataComp-1B matched caption subset
**用途：** D5；recaptioning controlled experiment。
**独立单位与划分：** same image across all captions。同图original/recap不能跨角色；只用预先允许的子集。
**指标：** student quality at same images/exposure; factuality audit。
**获取：** https://github.com/UCSC-VLAA/Recap-DataComp-1B
**评价入口：** https://github.com/UCSC-VLAA/Recap-DataComp-1B
**风险／权限：** 不要下载1B全量；synthetic caption teacher污染/成本记账；caption长度独立控制。
**来源：** D04（见 ../literature/SOURCE_CATALOG_CN.md）。

## DS29 — QVHighlights
**用途：** D6；train/CAL/full task eval。
**独立单位与划分：** parent YouTube video incl sliced IDs。固定官方train/val/test；内部CAL按parent vid；2sec clip索引和duration锁定。
**指标：** MR mAP(0.5:0.05:0.95), R@1(0.5/0.7); HD mAP/Hit@1。
**获取：** https://github.com/jayleicn/moment_detr
**评价入口：** https://github.com/jayleicn/moment_detr/tree/main/standalone_eval
**风险／权限：** annotations CC BY-NC-SA4; code MIT; 当前官方称Codalab关闭、test GT公开，公开不等于训练许可。
**来源：** V02;V03（见 ../literature/SOURCE_CATALOG_CN.md）。

## DS30 — Charades-STA
**用途：** D6；secondary full temporal grounding。
**独立单位与划分：** original video / recording family。官方STA train/test；没有官方CAL则只从train按视频切。
**指标：** R@1/5 at tIoU0.3/0.5/0.7 and mIoU where official。
**获取：** https://github.com/jiyanggao/TALL
**评价入口：** https://github.com/jiyanggao/TALL
**风险／权限：** 使用STA而非动作分类annotation；缓存C3D/I3D/CLIP features不同不可混排。
**来源：** V03;V04（见 ../literature/SOURCE_CATALOG_CN.md）。

## DS31 — ActivityNet Captions
**用途：** D6；optional broader videos。
**独立单位与划分：** original video。train/val1/val2 protocol明写；不把val2随口称官方test。
**指标：** temporal R@K; dense caption指标分别。
**获取：** https://cs.stanford.edu/people/ranjaykrishna/densevid/
**评价入口：** https://github.com/ranjaykrishna/densevid_eval
**风险／权限：** YouTube URL attrition；单/多描述、弱/强监督分表；基础视频排重。
**来源：** V04;V05（见 ../literature/SOURCE_CATALOG_CN.md）。

## DS32 — HiREST
**用途：** D6；hierarchical extension。
**独立单位与划分：** original video and step family。官方split按video；不将同视频各step打散。
**指标：** video retrieval; moment retrieval; step captioning。
**获取：** https://openaccess.thecvf.com/content/CVPR2023/html/Zala_Hierarchical_Video-Moment_Retrieval_and_Step-Captioning_CVPR_2023_paper.html
**评价入口：** 主论文/项目中核验后再执行，当前未给出已验证接口。
**风险／权限：** 官方数据/评测入口从paper核验；当前摘要级，不自动执行。
**来源：** V09（见 ../literature/SOURCE_CATALOG_CN.md）。

## DS33 — Ego4D NLQ
**用途：** D6；long-video extension。
**独立单位与划分：** original recording and participant where available。官方NLQ train/val/test与license流程；不混MQ/GoalStep。
**指标：** R@1/5 at tIoU0.3/0.5。
**获取：** https://ego4d-data.org/
**评价入口：** https://github.com/EGO4D/episodic-memory
**风险／权限：** 需数据许可/下载凭证；不能以冻结特征免去原视频成本声称部署加速。
**来源：** V11（见 ../literature/SOURCE_CATALOG_CN.md）。

## DS34 — HowTo100M
**用途：** D6；optional weak pretraining only。
**独立单位与划分：** parent video/creator。叙述配对与下游数据去重；只对预先授权子集。
**指标：** zero-shot retrieval / downstream training。
**获取：** https://www.di.ens.fr/willow/research/howto100m/
**评价入口：** https://www.di.ens.fr/willow/research/mil-nce/
**风险／权限：** 网页许可和视频授权分开；没有标注不等于随机负例都真实错；不列首批全量。
**来源：** V01（见 ../literature/SOURCE_CATALOG_CN.md）。

## DS35 — MomentSeeker
**用途：** D6；recent long-video external evaluation。
**独立单位与划分：** long original video。官方release/split待服务器接入核验。
**指标：** official long-video retrieval metrics。
**获取：** https://proceedings.neurips.cc/paper_files/paper/2025/hash/281e0b9142763f2b6c944fedb8550ba9-Abstract-Datasets_and_Benchmarks_Track.html
**评价入口：** 主论文/项目中核验后再执行，当前未给出已验证接口。
**风险／权限：** 本輪主源摘要已核，数据与完整evaluator未下载；不能虚构参数/运行成本。
**来源：** V10（见 ../literature/SOURCE_CATALOG_CN.md）。
