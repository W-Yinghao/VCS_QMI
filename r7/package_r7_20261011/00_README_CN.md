# VCS · CVPR V6 · r7：Vision–Language 研究图谱与服务器方案

**日期：2026-10-11。论文仍为V6；r7是研究交接包，不是论文版本。**

本包把VL从“RefCOCO候选排序”重新展开为六种用途：视觉证据压缩、事实支持、条件生成分布、多对多匹配、数据策展、视频时间支持。每条路线各有一份文献综述、一份实施方案。首轮不是同时训练六条线，而是审查数据和基线、明确统计对象，再选择一条主线及至多一条轻量验证线。

## 阅读顺序

1. `01_SERVER_AGENT_PROMPT_CN.md`：直接交给服务器agent的任务。
2. `docs/02_EXECUTIVE_PLAN_CN.md`：全局决策；同名概要PDF适合阅读。
3. `docs/03_THEORY_AND_CLAIM_MAP_CN.md`、`docs/04_SHARED_EVALUATION_CN.md`：共同定义与评价边界。
4. `directions/D1_token_evidence` 至 `D6_temporal_support`：每个文件夹先读 `LITERATURE_REVIEW_CN.md`，再读 `SERVER_EXPERIMENTS_CN.md`。
5. `datasets/DATASET_ATLAS_CN.md` 与 `literature/SOURCE_CATALOG_CN.md`：35个数据/协议条目与67项文献注册。
6. `benchmarks/`、`machine/`、`templates/`：对照清单、阶段任务与交付格式。

## 这不是另一个“都先跑一遍”的包

主问题是：**VCS的统计对象和误差结构，能够让哪一种VL决策更可靠或更有效？**

D1/D2优先做数据、完整应用路径和直接竞争工作的核验；D3先完成有限条件分布验证。D4/D6保留为可复用双塔/视频基础设施的替代；D5依赖下游重新训练，不能因为筛选器便宜就低估总代价。路线一旦选定，应完成预定方法和种子，不按seed0胜负筛选。

## 来源标记

- `[U01]` 合作者原研究计划，保留S/J与混合参考定义；`sources/`提供许可范围内的用户原文件副本。
- `[U02]` 合作者补充意见；本包仅引用其用途和经后续更正的结论。
- `[Fxx/Txx/Hxx/Gxx/Mxx/Dxx/Vxx/Bxx]` 原始论文或官方项目，详见文献注册表。
- **“本方案提出／待检验”** 是这次设计，不是原文已经给出的算法、定理或性能。

本包没有新的自然图像实验结果，没有下载新数据/模型，没有修改GitHub或提交作业。参考代码仅核对数学/指标约定，不是六套已移植的神经训练runner。

## 便捷阅读

解压后打开 `INDEX.html`；完整离线阅读用 `FULL_RESEARCH_AND_SERVER_CN.html`，便于复制给agent用同名Markdown。6页总览为 `VCS_VL_Research_Atlas_r7_Overview_CN.pdf`。详细原文实验约定见 `benchmarks/LITERATURE_PROTOCOL_CARDS_CN.md`。
