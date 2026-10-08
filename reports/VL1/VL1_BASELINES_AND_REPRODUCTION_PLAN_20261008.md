# VL1 baselines and reproduction plan — 2026-10-08

Owner 2026-10-08: add the public region-matching / grounding methods to the analysis and reproduce them ("你把这几个也列入分析和复现"), organised
as three tables.  Facts below were checked against primary sources on 2026-10-08 (papers, official repositories / model cards); items marked [U]
could not be verified.  Primary dataset: RefCOCOg (UMD) with COCO-2014 images (local); Flickr30k Entities follows when the images arrive.

## Table A — same features, same candidates, same data: does the estimator help? (VL1-10)
Frozen open_clip **ViT-B-16-quickgelu (OpenAI)** region-crop and expression features; candidate set = the image's referred objects (secondary:
+ COCO distractors); FIT N ∈ {1 000, 4 000, all 10 674 eligible} images; 3 init seeds; CAL selects the estimator checkpoint (common J), DEV the
task checkpoint (image-macro Top-1).  Identical scorer class F2 = MLP(concat(u, v)) 256-256-1, GELU, T = tanh f, for every learned row.

| row | objective | role |
|---|---|---|
| raw CLIP cosine | none | base ability / drift reference |
| candidate softmax (multi-positive CE over the image's candidates) | task loss | strongest direct task control |
| balanced logistic (JS), same P/Q | mean_P softplus(−2f) + mean_Q softplus(2f) | isolates the quadratic objective from the shared learnable critic |
| SigLIP-style task control | per-pair sigmoid with the actual positive / negative counts (prior recorded, de-biased before reading common J) | is the gain just pair-wise sigmoid learning? |
| same-target RFF ridge-tanh | P116 two-stage solver on [ψ, 1] | kernel CS route on the same target |
| **VCS (ours)** | 1 − mean_G J_G | — |

Columns: DEV / val Top-1 (query and image-macro), sample curve over N, common J / balanced squared risk (CAL), fit time.  Softmax is a task row only
(no common-J column); real images carry no oracle (no posterior MSE).  **Second shared feature set (pre-selected): FG-CLIP 2 Base region features**
(official RoIAlign interface) — re-run VCS / logistic / softmax there to test whether any gain depends on the weaker CLIP start.
Note (verified): open_clip's plain "ViT-B-16" + "openai" loads GELU instead of QuickGELU (warning only) — the baseline script uses
`ViT-B-16-quickgelu` and asserts the activation.

## Table B — public fine-grained VL models on the SAME given candidate boxes (no task training unless stated)
| method | official setting to reproduce | given-box interface | contamination / training data | environment | priority |
|---|---|---|---|---|---|
| CLIP ViT-B/16 (OpenAI) | crop → official preprocess, cosine | crop (ours) | WIT-400M, not public; Flickr-sourced YFCC subsets possible [U] | main env | done in VL1-00 |
| **ReCLIP** (ACL 2022; allenai/reclip, Apache-2.0, last commit 2022-04) | RN50x16 + ViT-B/32 ensemble; crop + blur (GaussianBlur 100, box pasted back); square squash (no centre crop); text "a photo of …"; logits 100·cos summed over models and isolations; spaCy relation parsing (left/right/above/below by centres, bigger/smaller by area, superlatives) | crop + blur, official | no training; **σ, prompt and relation heuristics were tuned on RefCOCOg val by the authors** (disclose; their val numbers are not held-out) | separate venv (spaCy + en_core_web_sm, openai `clip`; repo pins py3.6-era versions → relax pins / patch the ALBEF import) | **1st** |
| SigLIP 2 Base (Google 2025, Apache-2.0) | global sigmoid(s·cos + b); lower-case text, max_length 64; squash resize, mean/std 0.5 | none official → crop and ReCLIP-style crop + blur, labelled *adaptation* | WebLI 10B; no de-dup statement | needs `transformers` tokenizer (separate venv) | 2nd |
| **FG-CLIP 2 Base** (ICML 2026; 360CVGroup/FG-CLIP, Apache-2.0) | `get_image_region_features(..., region_infos=[[x1,y1,x2,y2]])` (RoIAlign 1×1 on dense features) + `get_text_features(walk_type="box")`, lower-case, max_length 64 | **official region API** | SigLIP 2 init + LAION-2B recaptioned, Wukong, Zero, in-house CN, FineHARD (GRIT + YOLO-World boxes); no COCO / RefCOCO stated | transformers ≈ 4.57 with trust_remote_code + torchvision roi_align (separate venv) | **2nd** |
| FG-CLIP Base (ICML 2025, branch v1.0) | `get_image_box_roi_features(pixel_values, box_info)` (14×14 grid units, 224 squash) | official region API | OpenAI CLIP init + LAION recaptioned + FineHARD | transformers 4.41 (separate venv) | optional |
| FLAIR (CVPR 2025; weights MIT, **code repo has no licence file**) | `get_logits` = text-conditioned attention pooling (the paper's mode), not `get_logits_as_clip` | none → crop / blur, text-conditioned scoring kept; labelled *adaptation* | DreamLIP-recaptioned CC3M / CC12M / YFCC15M (Flickr-sourced; possible Flickr30k overlap [U]) | open_clip only (main env works) | when DCI / DOCCI starts |

Rows are labelled "no task training"; our VCS row uses the task's annotations — stated in the table, and beating a zero-shot model is not read as
a loss-vs-pretraining claim.  **Reproduction checks:** ReCLIP GT-box RefCOCOg-UMD val 68.08 / test 67.05 (paper App. Table 7; an independent user
reproduced 67.91 / 67.07), IPS-only 65.32 / 65.59.  FG-CLIP 2 / SigLIP 2 have no published given-box RefCOCOg number; their published COCO-val2017
GT-box region classification (FG-CLIP 2 B/16 74.9, SigLIP 2 B/16 53.4, CLIP B/16 44.2) is the implementation check for the region path.
Official val is used only for these reproduction checks of external methods and the final validation of frozen configurations; development
uses DEV; the UMD test split opens once, at the end, for all rows.

## Table C — full grounding (no given boxes)
| row | what | contamination | status / risk |
|---|---|---|---|
| fixed proposer + VCS (and + the main matched controls) | query-independent detector proposals, scored by the Table-A scorers; report proposal recall, Acc@IoU 0.5, total time | the proposer must not have seen RefCOCOg val/test images: off-the-shelf COCO detectors (train2017 ⊃ train2014) **leak**; MAttNet's clean detector / detections are offline (host DNS dead); ReCLIP's detection file is 403 | **rebuild** a torchvision detector on COCO-2014-train minus RefCOCO/+/g val+test (`refer/tools/make_coco_minus_refer_instances.py` logic) — a budget item (≈ 1 GPU-day [U]) |
| **MDETR** (ICCV 2021; Apache-2.0; Zenodo weights CC-BY-4.0) | RefCOCOg-fine-tuned R101 / EB3 (paper umd val/test 81.64/80.89, 83.35/83.31) and the pretrained-only R101 | pretraining used RefCOCO/+/g **train** expressions and Flickr30k Entities train (val/test images excluded) — not zero-shot | py3.8 env (transformers 4.5.1 pin); **its refexp metric counts GIoU ≥ 0.5** → report GIoU and IoU |
| **Grounding DINO** (ECCV 2024; Apache-2.0) | **Swin-T OGC** (O365, GoldG, Cap4M — GoldG includes Flickr30k Entities train, no COCO / RefCOCO) as the clean zero-shot row; argmax over 900 queries | Swin-B "cogcoor" trained on COCO + **RefCOCO** → contaminated for RefCOCOg (reported only with the flag, if at all) | custom CUDA op; Blackwell build needs patches → use L40S / H100 nodes or the HF-transformers port; third-party measured Swin-T OGC RefCOCOg val/test 60.4 / 59.7 [S] (paper's zero-shot GD-T 67.46 / 67.13 is a non-released model) |
| TransVG (ICCV 2021) | optional; per-dataset fine-tunes on Google Drive; **no licence file** | DETR init on COCO with val/test images excluded (use the `-gref` files, not the standard COCO DETR) | modern-torchvision patch needed |
| GLIP (CVPR 2022) | optional; Flickr numbers only | GoldG (Flickr30k train) | maskrcnn_benchmark does not build on torch ≥ 1.11 without porting → skip unless via mmdetection |

Two tables are never merged: given-box Top-1 (Tables A / B) and from-image Acc@IoU (Table C) answer different questions.

## Related-work positioning (no duplicate rows)
- **RPC** with the balanced parameter is algebraically the same quadratic functional → stated in the text; a separate row only for a genuinely
  different configuration.
- **CS-Aligner** targets marginal image–text alignment, not this conditional joint-vs-product object → related work for the region task; a row
  only if the programme moves to global image–text adaptation.
- **MINE / DV, NWJ, InfoNCE, SMILE** stay in the estimator benchmark (P85 / P86 / P151 / P152) and one representative real relation (P149); they
  are not re-run on every grounding dataset.

## Execution order
1. **Now (VL1-00 / 01, running):** raw CLIP ViT-B/16 DEV table (job queued); data audit done (`VL1_DATA_AUDIT.json`).
2. **Batch 1:** Table A on CLIP features (VL1-10; fitters for the six rows), and ReCLIP reproduction (GT boxes; rebuilt inputs from REFER + COCO;
   official ensemble and the IPS-only variant; check against 68.08 / 65.32 on UMD val).
3. **Batch 2:** FG-CLIP 2 Base (official region API) and SigLIP 2 Base (crop / blur adaptation) on the same candidates; Table A re-run on FG-CLIP 2
   region features.
4. **Batch 3:** Table C — MDETR (fine-tuned + pretrained, GIoU and IoU), Grounding DINO Swin-T OGC, and the proposer + scorer rows after the clean
   proposer is rebuilt (budget confirmation needed).
5. Flickr30k Entities (given-box matching, then MERGED / ANY-box grounding with MDETR / Grounding DINO) when the images arrive; RefCOCO+ as the
   conditional extension.
Environments: `/projects/EEG-foundation-model/yinghao/envs/vl_baselines` (py3.12: transformers ≈ 4.57, tokenizers, sentencepiece, spaCy +
en_core_web_sm, openai clip) and `.../envs/mdetr_py38` (MDETR pins); the main env stays unchanged.  All weights / data under
`/projects/EEG-foundation-model/yinghao/` with provenance files.
