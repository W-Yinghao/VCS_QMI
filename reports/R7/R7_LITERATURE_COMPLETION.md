# R7-10 (literature part) — closest works, availability and fixture needs per direction — 2026-10-12

- **Sources:** primary paper pages (arXiv, CVF, NeurIPS, ICLR), official repositories (GitHub REST API: licence, last push, archived flag, file
  listing) and Hugging Face model / dataset APIs.
- **Checked:** 2026-10-11/12 UTC.
- **Scope:** availability and pre-emption only.  Nothing was downloaded, installed or run.  No result table is copied as an expected
  number.
- **Verification marks:**
  - **[V]** read on the primary page or repo;
  - **[V-api]** from the GitHub / HF API;
  - **[U]** unverified (reason given).
- Package ids (T05, H04, ...) refer to `r7/package_r7_20261011/literature/papers.csv`.

## D1 — question-conditioned visual evidence compression (token pruning)
| work | venue / status | full text | code (licence, last push) | weights | evaluator / data in the work | pre-empts |
|---|---|---|---|---|---|---|
| **IF-Prune (T05)** | CVPR 2026 [V: CVF poster page; Snap publication page; no arXiv version found] | **CVF PDF and supplement return HTTP 403 to automated fetch [U]**.  The poster abstract says "eight benchmarks" without naming them [V]. | snap-research/EVLM-IF-Prune.  README states Apache 2.0, though the API sees no licence file.  Last push 2026-03-17.  Training (LoRA) and inference for Qwen2-VL and InternVL2.5; a `llava_kl` folder without instructions [V]. | one HF checkpoint `ZachSun/internvl2_5_1b_pruning_kl_lora32` (InternVL2.5-1B small-model LoRA; no licence on card) [V-api] | ChartQA script `eval_chartqa.sh`; lmms-eval custom model `internvl_kl.py` [V].  Full benchmark list [U]. | A small VLM detects query-non-informative tokens with a **variational information bottleneck** (per-token entropy as pruning signal).  So "information-theoretic, query-conditioned token importance" is not new. |
| **DiffPrune (T06)** | arXiv only: 2605.28051 (27 May 2026, titled "Beyond Surrogate Gradients: Fully Differentiable Token Pruning for VLMs") and 2608.01985 (3 Aug 2026, titled "DiffPrune: differentiable information throttling…"); same four authors; no venue [V] | arXiv [V] | **none**: the abstract says "Code will be publicly available"; no URL [V] | none | ten VLM benchmarks (not named in the abstract) [V] | Trains token scores by **throttling information with variance-preserving noise**, then hard top-K at inference.  It is a learned, information-based score. |
| **ET-Prune (T07)** | arXiv 2608.01979, v1 3 Aug 2026, v2 27 Aug 2026; no venue [V] | arXiv [V] | Labyrinth0419/ET-Prune (MIT), but the repo holds only figures and README ("TBD — the code will be released soon") [V] | none | Qwen3-VL-8B and InternVL3.5-8B; OCRBench-v2, MMBench v1.1, TextVQA; single deterministic pass [V] | **Training-free, evidence-aware dynamic per-sample budget** (evidence uncertainty and density set the token floor).  So "evidence-driven dynamic budgets" is not new. |
| FastV (T01) | ECCV 2024 | — | pkunlp-icler/FastV; **no licence file** [V-api]; last push 2025-01-04 | host weights only | LLaVA-1.5-7B/13B; lmms-eval [V] | attention-based early-layer pruning; the standard baseline |
| SparseVLM (T02) | ICML 2025 | — | Gumpest/SparseVLMs (Apache-2.0, active 2026-07-30) [V-api] | host weights | TextVQA, MME, GQA named in the README [V] | text-guided sparsification |
| PyramidDrop (T04) | CVPR 2025 (final title "Conical Visual Concentration…") | — | Cooperx521/PyramidDrop (MIT, 2025-03-06) [V-api] | host weights | LLaVA-NeXT and LLaVA-1.5; VLMEvalKit [V] | staged dropping |
| FEATHER (T03) | ICCV 2025 [V: project page] | arXiv 2412.13180 | markendo/FEATHER (MIT, 2025-07-31) [V-api] | — | includes RefCOCO / + / g and TextVQA [V] | shows that average QA hides lost localisation evidence |

- Also relevant (not registered in the package; [U] depth): "A Stitch in Time Saves Nine: Small VLM is a Precise Guidance for
  Accelerating Large VLMs" (arXiv 2412.03324).  It is the small-VLM-attention guidance that IF-Prune explicitly revises.

**Fixture needs (D1).**
- **Host:** LLaVA-1.5-7B (`liuhaotian/llava-v1.5-7b` or `llava-hf/llava-1.5-7b-hf`, llama2 licence) for FastV / SparseVLM / PyramidDrop
  parity, or Qwen2.5-VL-7B-Instruct (Apache-2.0).
- **Evaluator:** lmms-eval or VLMEvalKit, both active on 2026-10-10.
- **Data:** TextVQA val and GQA testdev-balanced (public); ChartQA test (repo GPL-3.0).  DocVQA needs RRC registration.
- **Strong method:** FastV or SparseVLM (code runnable).  IF-Prune needs its InternVL2.5 / Qwen2-VL environment (CUDA 12.4, torch 2.6,
  flash-attn 2.7.4).  DiffPrune and ET-Prune **cannot be reproduced** (no code).

## D2 — visual factual support / faithfulness
| work | venue / status | code (licence, last push) | data / weights | evaluator | pre-empts |
|---|---|---|---|---|---|
| **SCI (H04)** | CVPR 2026 (arXiv 2603.07659 v1 8 Mar, v2 25 Mar 2026: "Accepted to CVPR 2026") [V] | KaihuaTang/Self-Critical-Inference-Framework (Apache-2.0, last push 2026-09-19).  VLMEvalKit-based; step scripts 0–4 [V-api] | DRBench is **model-specific / dynamic** [V], so it cannot be pooled with static benchmarks | VLMEvalKit | Multi-round textual and visual counterfactual reasoning extending VCD.  Test-time robustness against language bias. |
| VCD (H01) | CVPR 2024 | DAMO-NLP-SG/VCD (Apache-2.0, 2024-10-07) [V-api] | host weights | POPE, MME scripts in the repo | Contrastive decoding with distorted images, one extra forward pass |
| M-HalDetect (H02) | AAAI 2024 | hendryx-scale/mhal-detect: **annotations only** (`train_raw.json` 17 MB, `val_raw.json`).  **No detector / reward weights and no training code.** [V] | annotations CC BY-NC 4.0; images COCO 2014 val (Flickr terms) [V] | span / response labels | fine-grained human hallucination labels for training verifiers |
| Woodpecker (H03) | arXiv 2023.  **A journal version could not be verified [U].** | repo moved to VITA-MLLM/Woodpecker (no licence file; last push 2024-12-23) [V-api] | pipeline uses external detectors / LLM calls | — | training-free post-hoc correction |
| VQAScore (H10) | ECCV 2024 | linzhiqiu/t2v_metrics (Apache-2.0, active 2026-06) [V-api] | — | — | image–text support score from a VQA model |

**Fixture needs (D2).**
- **Host:** LLaVA-1.5-7B with official VCD code.
- **Data:** POPE (MIT; the three COCO configurations) on COCO val2014.  Run VCD vs regular decoding on a fixed subset per configuration,
  then the full 3 × 3 000 questions as the anchor.
- **Verifier training:** needs M-HalDetect annotations (non-commercial) plus COCO val2014.
- **Benchmarks:** HallusionBench (BSD-3) and AMBER (Apache-2.0) evaluators are public.

## D3 — conditional generation distributions
| work | venue / status | code (licence, last push) | evaluator needs | pre-empts |
|---|---|---|---|---|
| **cFreD (G03)** | WACV 2026 (arXiv 2503.21721 v2 says the toolkit is open-sourced) [V] | JaywonKoo17/cFreD: `cfred/`, `requirements.txt`.  **No licence file** [V-api]; last push 2026-03-07. | features + per-prompt conditioning | Conditional Fréchet distance with prompt conditioning, the most direct conditional-distribution comparator |
| CFID (G02) | arXiv 2021/2022 | Michael-Soloveitchik/CFID (BSD-2, 2021-09-29) [V-api] | — | conditional FID (image-to-image origin) |
| CMMD (G01) | CVPR 2024 | google-research/cmmd (Apache-2.0; `distance.py`, `embedding.py`, `main.py`) [V-api] | CLIP embeddings | unconditional CLIP-MMD; the "C" is CLIP |
| GenEval (G04) | NeurIPS 2023 D&B | djghosh13/geneval (MIT, 2025-03-03) [V-api] | **mmdetection 2.x + Mask2Former** via `download_models.sh` [V] | object-level alignment, not a distribution truth |
| T2I-CompBench / ++ (G05 / G06) | NeurIPS 2023 / TPAMI 2025, **same repo** [V] | Karine-Huang/T2I-CompBench (MIT, 2026-09-04) [V-api] | BLIP-VQA, UniDet (wget from HF + gdown), CLIPScore, 3-in-1.  Optional ShareGPT4V-7B; optional **paid GPT-4V / MiniMax API** [V] | category-wise compositional scores |
| GenAI-Bench (G08) | HF `BaiqiL/GenAI-Bench` (Apache-2.0, 2026-01-30) [V-api] | — | human ratings | human alignment |

**Fixture needs (D3).**
- **Finite truth:** the package's controlled CLEVR / finite constructions on CPU.
- **Official evaluator fixture:** GenEval on a few fixed images.  This needs its pinned mmdet 2.x environment, which is separate from the
  main env.  Or CMMD, which is lighter.
- **cFreD:** licence unclear.  Ask the authors or keep it as an internal comparison only.

## D4 — many-to-many / incomplete correspondence
| work | venue | code / weights (licence) | evaluator | pre-empts |
|---|---|---|---|---|
| **PCME++ (M02)** | ICLR 2024 | naver-ai/pcmepp (MIT-style permission text); HF `SanghyukChun/PCMEPP-ViT-B-16-CC3M-12M-RedCaps(-256M)` [V-api] | ECCV Caption / CxC | probabilistic embeddings for plausible matches |
| **ProLIP (M03)** | ICLR 2025 | naver-ai/prolip (MIT-style); HF `SanghyukChun/ProLIP-ViT-{B-16,L-16,SO400M-14,H-14}-…` [V-api]; the package's id `naver-ai/ProLIP-…` does not exist | — | probabilistic VLM pretraining |
| FLAIR (M07) | CVPR 2025 | ExplainableML/flair (**no licence stated**); HF `xiaorui638/flair` [V-api] | — | text-conditioned pooling |
| ECCV Caption (M06) | ECCV 2022 | naver-ai/eccv-caption (MIT-style; evaluator; active 2026-07) [V-api] | mAP@R, R-Precision | corrected false negatives |
| CxC (M05) | EACL 2021 | archived repo; "may be freely used for any purpose" with acknowledgement [V] | semantic similarity | — |
| SugarCrepe++ (M08) | **NeurIPS 2024 Datasets & Benchmarks** [V]; code / data Sri-Harsha/scpp (MIT, 2024-06-18) [V-api] | triplets | — | — |

**Fixture needs (D4).**
- Dual-tower fine-tuning, which the repo already does (VL3 / R4).
- The ECCV Caption pip evaluator on COCO Karpathy test.
- A PCME++ or ProLIP public checkpoint as the anchor.

## D5 — data selection / recaptioning
| work | venue | code / weights | cost note |
|---|---|---|---|
| DataComp (D01) | NeurIPS 2023 D&B | mlfoundations/datacomp (MIT) [V] | the small pool means 12.8M images fetched by URL (img2dataset) before any student training |
| DFN (D02) | ICLR 2024 | **`apple/ml-dfn` does not exist** [V-api].  Models on HF: `apple/DFN-public`, `apple/DFN5B-CLIP-ViT-H-14` (licence `apple-amlr`) [V-api]. | filter network trained on extra data |
| Sieve (D03) | CVPR 2024 | facebookresearch/SIEVE (MIT, 2024-04-28) [V-api] | captioner inference over the pool |
| SDM (D05) | ICLR 2026 (OpenReview forum cgmo3v18sx) [V] | **no code found** [U] | DataComp-**medium** scale (128M pool) |
| Recap-DataComp-1B (D04) | ICML 2025 | HF dataset `UCSC-VLAA/Recap-DataComp-1B` (CC BY 4.0) [V-api] | captions only; images via DataComp |

## D6 — temporal support (moment retrieval)
| work | venue | code (licence) | features / data | pre-empts |
|---|---|---|---|---|
| Moment-DETR / QVHighlights (V02) | NeurIPS 2021 | jayleicn/moment_detr (MIT, 2026-03-09) + `standalone_eval` [V-api] | official SlowFast + CLIP features `moment_detr_features.tar.gz` (8 GB, Google Drive; **QD-DETR's README says "link expired"** [V]); raw videos URL on nlp.cs.unc.edu [V] | DETR moment retrieval |
| QD-DETR (V03) / CG-DETR (V05) | CVPR 2023 / arXiv (venue [U]) | MIT (both) [V] | Charades features via VSLNet's box.com link [V] | query-dependent representations / calibration |
| Sim-DETR (V07) | ICCV 2025 (pp. 22760–22771) [V] | **SooLab/Sim-DETR** (MIT; v2 link; the v1 link nagara214/Sim-DETR is 404) [V-api] | `data/`, `standalone_eval/` | DETR decoder fixes for grounding |
| SG-DETR (V08) | WACV 2026 [V] | **no repository found**; arXiv v2 says "will be publicly available"; project page gracikk-ds.github.io/sg-detr [U] | extra InternVid-style pretraining | saliency guidance |
| MomentSeeker (V10) | NeurIPS 2025 D&B [V] | probably yhy-2000/MomentSeeker (no licence; author initials match, **official status [U]**); HF datasets incl. `avery00/MomentSeeker`, `VLM2Vec/MomentSeeker` [V-api] | long videos (> 1 200 s average) | long-video MR benchmark |

## Recommendation (availability only; not chosen by any VCS result)
- **Most complete reproducible path now: D1.**
  - Several strong baselines with runnable code and permissive licences (FastV, SparseVLM, PyramidDrop, FEATHER).
  - Active evaluators (lmms-eval, VLMEvalKit).
  - Public hosts (LLaVA-1.5-7B, Qwen2.5-VL-7B) and public data (GQA, TextVQA, ChartQA).
  - **Caveat:** the conceptual neighbourhood is crowded — IF-Prune (information bottleneck), DiffPrune (information throttling), ET-Prune
    (evidence-aware budgets).  Two of those have no code, so a D1 claim must beat runnable baselines plus a direct risk-regression selector,
    and the novelty claim must be narrow.
- **Runner-up for reuse: D4.**
  - It reuses this repo's dual-tower fine-tuning (VL3 / R4).
  - Public evaluator (ECCV Caption) and public checkpoints (PCME++, ProLIP).
  - Licences are permissive, except FLAIR (none stated).
- **D2 is complete for the verifier stage:** VCD code and POPE / HallusionBench / AMBER evaluators.  But its training labels (M-HalDetect)
  are CC BY-NC and ship without the authors' detector or training code.
- **D3:** the controlled finite checks are CPU-ready; the official evaluators need separate pinned environments (mmdet 2.x; UniDet).
- **D6:** features hosting is fragile (an expired Drive link).
- **D5:** least ready — a 12.8M-image URL fetch plus student training, and no SDM code.
