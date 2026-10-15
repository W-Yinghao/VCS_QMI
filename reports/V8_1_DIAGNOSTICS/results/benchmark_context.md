# D07 external context table (generated 2026-10-15T18:51:31Z from `benchmark_context.csv` / `d07_context.json`; frozen results only, no new scoring)

Sources: v8_eval.json sha256 1949fb32dbf3…; confirm.json sha256 f4094a8aba03…; x40_dev_external.json sha256 f02f1bfe53d1…; V7C-04 primary.json sha256 796bd6f6328c….

| system | host / backbone | trained on | route status | GenAI-Rank N9 selected (regret) DALLE_3 | SDXL_Base | GenAI-Bench 850 pooled Spearman | pairwise | EvalMuse DEV SRCC (descr.) | DEV pairwise |
|---|---|---|---|---|---|---|---|---|---|
| U-CS | FGA-BLIP2 construction (V7 build_model: BLIP-2 COCO-finetuned init, EVA ViT-g frozen in eval mode; Q-Former + ln_vision + query_tokens trained) + Linear(768->5) head | EvalMuse FIT 26,177 images (overall votes), lr / epoch by CAL_SELECT | V8 formal, 3 seeds, frozen V8-03 | 4.138 (0.476) | 3.516 (0.437) | 0.575 | 0.726 | NA | NA |
| U-BCE | FGA-BLIP2 construction (V7 build_model: BLIP-2 COCO-finetuned init, EVA ViT-g frozen in eval mode; Q-Former + ln_vision + query_tokens trained) + Linear(768->5) head | EvalMuse FIT 26,177 images (overall votes), lr / epoch by CAL_SELECT | V8 formal, 3 seeds, frozen V8-03 | 4.148 (0.466) | 3.513 (0.439) | 0.590 | 0.728 | NA | NA |
| U-RPS | FGA-BLIP2 construction (V7 build_model: BLIP-2 COCO-finetuned init, EVA ViT-g frozen in eval mode; Q-Former + ln_vision + query_tokens trained) + Linear(768->5) head | EvalMuse FIT 26,177 images (overall votes), lr / epoch by CAL_SELECT | V8 formal, 3 seeds, frozen V8-03 | 4.148 (0.466) | 3.512 (0.440) | 0.563 | 0.725 | NA | NA |
| U-CE | FGA-BLIP2 construction (V7 build_model: BLIP-2 COCO-finetuned init, EVA ViT-g frozen in eval mode; Q-Former + ln_vision + query_tokens trained) + Linear(768->5) head | EvalMuse FIT 26,177 images (overall votes), lr / epoch by CAL_SELECT | V8 formal, 3 seeds, frozen V8-03 | 4.142 (0.472) | 3.511 (0.442) | 0.566 | 0.726 | NA | NA |
| U-MSE | FGA-BLIP2 construction (V7 build_model: BLIP-2 COCO-finetuned init, EVA ViT-g frozen in eval mode; Q-Former + ln_vision + query_tokens trained) + Linear(768->5) head | EvalMuse FIT 26,177 images (overall votes), lr / epoch by CAL_SELECT | V8 formal, 3 seeds, frozen V8-03 | 4.156 (0.457) | 3.518 (0.435) | 0.568 | 0.724 | NA | NA |
| U-L1 | FGA-BLIP2 construction (V7 build_model: BLIP-2 COCO-finetuned init, EVA ViT-g frozen in eval mode; Q-Former + ln_vision + query_tokens trained) + Linear(768->5) head | EvalMuse FIT 26,177 images (overall votes), lr / epoch by CAL_SELECT | V8 formal, 3 seeds, frozen V8-03 | 4.144 (0.469) | 3.507 (0.446) | 0.571 | 0.725 | NA | NA |
| U-RANK | FGA-BLIP2 construction (V7 build_model: BLIP-2 COCO-finetuned init, EVA ViT-g frozen in eval mode; Q-Former + ln_vision + query_tokens trained) + Linear(768->5) head | EvalMuse FIT 26,177 images (overall votes), lr / epoch by CAL_SELECT | V8 formal, 3 seeds, frozen V8-03 | 4.131 (0.483) | 3.505 (0.448) | 0.523 | 0.729 | NA | NA |
| U-LIST | FGA-BLIP2 construction (V7 build_model: BLIP-2 COCO-finetuned init, EVA ViT-g frozen in eval mode; Q-Former + ln_vision + query_tokens trained) + Linear(768->5) head | EvalMuse FIT 26,177 images (overall votes), lr / epoch by CAL_SELECT | V8 formal, 3 seeds, frozen V8-03 | 4.151 (0.462) | 3.534 (0.419) | 0.559 | 0.728 | NA | NA |
| B0_FGA_released_O | FGA-BLIP2 released checkpoint (same BLIP-2 host family) | all of EvalMuse TRAIN (includes DEV -> DEV exposed) | released model, frozen V7 freeze v3 predictions | 4.163 (0.451) | 3.476 (0.477) | 0.572 | 0.728 | 0.905 | 0.794 |
| B1_VQAScore | CLIP-FlanT5-XXL (t2v-metrics 3.0) | none on EvalMuse (zero-shot VQA probability) | frozen V7 freeze v3 predictions | 4.259 (0.355) | 3.562 (0.391) | 0.583 | 0.682 | 0.470 | 0.624 |
| B4T_TIFA | LLaMA-2 question generation (released local generator) + UnifiedQA filter + mPLUG-large VQA | none on EvalMuse | official route (V7C-03 fidelity: 91 % element precision, 78 % recall vs released GPT-3.5 questions) | 4.095 (0.519) | 3.386 (0.567) | 0.262 | 0.581 | 0.274 | 0.558 |
| B4D_DSG | DSG prompts with Qwen2.5-32B-Instruct (local adaptation) + mPLUG-large VQA | none on EvalMuse | LOCAL ADAPTATION (not the released PaLM-2 graphs; V7C-03 fidelity 81-86 % tuple content) | 4.112 (0.502) | 3.369 (0.584) | 0.381 | 0.583 | 0.399 | 0.573 |
| V7 M2_E (VCS element route, context) | same FGA-BLIP2 construction, element-level heads | EvalMuse FIT (element labels) | V7 closeout, first GenAI-Rank read (V7C-04) | 4.140 (0.474) | 3.477 (0.476) | NA | NA | NA | NA |
| B2_VQAScore_Qwen25VL7B | VQAScore with Qwen2.5-VL-7B | none on EvalMuse | present in V7 freeze v3; not scored in any frozen V8 family (not_scored) | NA (NA) | NA (NA) | NA | NA | 0.485 | 0.632 |
| B5_HPSv3 | HPSv3 preference model | none on EvalMuse | present in V7 freeze v3; not scored in any frozen V8 family (not_scored) | NA (NA) | NA (NA) | NA | NA | 0.483 | 0.639 |
| B3_PNVQA_own | PN-VQA | none on EvalMuse | present in V7 freeze v3; not scored in any frozen V8 family (not_scored) | NA (NA) | NA (NA) | NA | NA | 0.344 | 0.593 |
| CLIPScore_sanity | CLIP similarity | none on EvalMuse | present in V7 freeze v3; not scored in any frozen V8 family (not_scored) | NA (NA) | NA (NA) | NA | NA | NA | NA |

## Frozen contrasts kept as published

| contrast | GenAI-Rank N9 Δ [95 %] Holm (V8-04) |
|---|---|
| U-CS - B0_FGA_released_O @ DALLE_3 N9 | -0.025 [-0.055, +0.006] Holm 0.247 |
| U-CS - B0_FGA_released_O @ SDXL_Base N9 | +0.039 [+0.010, +0.069] Holm 0.046 |
| U-CS - B1_VQAScore @ DALLE_3 N9 | -0.121 [-0.168, -0.072] Holm 0.003 |
| U-CS - B1_VQAScore @ SDXL_Base N9 | -0.046 [-0.088, -0.003] Holm 0.146 |
| U-CS - B4T_TIFA @ DALLE_3 N9 | +0.043 [-0.005, +0.092] Holm 0.247 |
| U-CS - B4T_TIFA @ SDXL_Base N9 | +0.130 [+0.084, +0.177] Holm 0.003 |
| U-CS - B4D_DSG @ DALLE_3 N9 | +0.026 [-0.021, +0.073] Holm 0.297 |
| U-CS - B4D_DSG @ SDXL_Base N9 | +0.147 [+0.103, +0.190] Holm 0.003 |

| family (V8-05, GenAI-Bench 850 × 6) | contrast | Δ [95 %] | Holm |
|---|---|---|---|
| primary_spearman | U-CS − B0_FGA_released_O (spearman) | +0.003 [-0.008, +0.014] | 1.0000 |
| primary_spearman | U-CS − B1_VQAScore (spearman) | -0.008 [-0.041, +0.024] | 1.0000 |
| primary_spearman | U-CS − B4T_TIFA (spearman) | +0.312 [+0.267, +0.358] | 0.0008 |
| primary_spearman | U-CS − B4D_DSG (spearman) | +0.193 [+0.156, +0.229] | 0.0008 |
| primary_pairwise | U-CS − B0_FGA_released_O (pairwise) | -0.001 [-0.010, +0.007] | 0.7680 |
| primary_pairwise | U-CS − B1_VQAScore (pairwise) | +0.044 [+0.025, +0.063] | 0.0008 |
| primary_pairwise | U-CS − B4T_TIFA (pairwise) | +0.146 [+0.129, +0.163] | 0.0008 |
| primary_pairwise | U-CS − B4D_DSG (pairwise) | +0.143 [+0.126, +0.160] | 0.0008 |
| attribution | U-CS − U-BCE (spearman) | -0.016 [-0.021, -0.011] | 0.0002 |
| attribution | U-CS − U-BCE (pairwise) | -0.001 [-0.005, +0.003] | 0.5880 |
| same_data_descriptive | U-CS − U-RPS (spearman) | +0.011 [+0.005, +0.018] | descriptive |
| same_data_descriptive | U-CS − U-RPS (pairwise) | +0.001 [-0.004, +0.006] | descriptive |
| same_data_descriptive | U-CS − U-CE (spearman) | +0.008 [+0.000, +0.016] | descriptive |
| same_data_descriptive | U-CS − U-CE (pairwise) | +0.000 [-0.005, +0.005] | descriptive |
| same_data_descriptive | U-CS − U-MSE (spearman) | +0.006 [+0.001, +0.012] | descriptive |
| same_data_descriptive | U-CS − U-MSE (pairwise) | +0.003 [-0.002, +0.008] | descriptive |
| same_data_descriptive | U-CS − U-L1 (spearman) | +0.004 [-0.002, +0.009] | descriptive |
| same_data_descriptive | U-CS − U-L1 (pairwise) | +0.002 [-0.003, +0.007] | descriptive |
| same_data_descriptive | U-CS − U-RANK (spearman) | +0.052 [+0.036, +0.067] | descriptive |
| same_data_descriptive | U-CS − U-RANK (pairwise) | -0.002 [-0.009, +0.005] | descriptive |
| same_data_descriptive | U-CS − U-LIST (spearman) | +0.016 [+0.006, +0.026] | descriptive |
| same_data_descriptive | U-CS − U-LIST (pairwise) | -0.002 [-0.008, +0.005] | descriptive |

## Cost (historical job timings — not a standard workload; devices differ, see note)

- V8 arm on NVIDIA L40S (12 formal units): training 63-71 min per unit (5 epochs over 26,177 images + 5 CAL_SELECT evaluations), CAL_SELECT evaluation 17-19 images / s, export of 30,540 images 24-42 min (12-21 images / s, batch 32, 364 px, fp16 autocast), peak training memory 7.8 GB.
- V8 arm on NVIDIA H100 NVL (3 formal units): training 34-34 min per unit (5 epochs over 26,177 images + 5 CAL_SELECT evaluations), CAL_SELECT evaluation 25-25 images / s, export of 30,540 images 21-51 min (10-24 images / s, batch 32, 364 px, fp16 autocast), peak training memory 7.8 GB.
- V8 arm on NVIDIA RTX PRO 6000 Blackwell Server Edition (9 formal units): training 33-53 min per unit (5 epochs over 26,177 images + 5 CAL_SELECT evaluations), CAL_SELECT evaluation 29-31 images / s, export of 30,540 images 14-26 min (20-37 images / s, batch 32, 364 px, fp16 autocast), peak training memory 7.8 GB.
- B0_FGA_released_O: native eval in the V7 N0 harness; not timed separately.
- B1_VQAScore: GenAI-Rank 14,400 images: 39 min wall clock on one L40S (V7C-03).
- B4T_TIFA: GenAI-Rank 14,400 images: questions 3.5 h + VQA 2.3 h on L40S (V7C-03).
- B4D_DSG: GenAI-Rank 14,400 images: generation ~2.5 GPU-h (H100 NVL / RTX) + VQA ~2.4 GPU-h (3 x L40S) (V7C-03).
- V7 M2_E (VCS element route, context): V7 training (not re-timed here).

## Table C kernel routes (untrained representation; separate class, not a pure estimator contest)

| route | CAL_PROB RPS (natural) | CAL_PROB mean-MSE macro | DEV RPS | DEV mean-MSE macro |
|---|---|---|---|---|
| pca0/K-KDE | 0.141 | 0.723 | 0.147 | 0.789 |
| pca0/K-NYSTROM | 0.121 | 0.486 | 0.126 | 0.534 |
| pca0/K-RULSIF | 0.121 | 0.484 | 0.127 | 0.533 |
| pca256/K-KDE | 0.157 | 0.923 | 0.167 | 1.050 |
| pca256/K-NYSTROM | 0.123 | 0.504 | 0.127 | 0.538 |
| pca256/K-RULSIF | 0.123 | 0.510 | 0.128 | 0.549 |
| pca64/K-KDE | 0.147 | 0.797 | 0.153 | 0.870 |
| pca64/K-NYSTROM | 0.125 | 0.541 | 0.132 | 0.611 |
| pca64/K-RULSIF | 0.127 | 0.555 | 0.134 | 0.621 |
