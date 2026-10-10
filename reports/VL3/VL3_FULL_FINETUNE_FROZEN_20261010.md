# VL3 — full end-to-end fine-tuning with VCS / matched JS / candidate softmax on RefCOCOg, RefCOCO, RefCOCO+ — FROZEN 2026-10-10

Owner 2026-10-10: "还是之前的纪律，多跑全量实验，不要probe" → chose full fine-tuning; "数据集也跑全一点，我更倾向both" → full-data frozen tables
**and** full fine-tuning, on every dataset.  Same discipline as SSL: one full run per arm first, three seeds only where promising.

## Design (`scripts/vl3_finetune.py`, `slurm/vl3_finetune.sbatch`)
- **Data:** each dataset's FIT eligible images (≥ 2 referred objects; RefCOCOg 10 674, RefCOCO 12 653, RefCOCO+ 12 616) for training; CAL for
  selection; DEV for the reading.  Per-image pair law of Table A (region-uniform, candidates = the referred objects).  UNC / UMD val and test closed.
- **Backbones:** CLIP ViT-B/16 (OpenAI, QuickGELU asserted; tight crop → official preprocess) and SigLIP 2 B/16-224 (tight crop → its processor;
  lower-cased text, length 64).  **All encoder parameters are trained** (image and text towers).
- **Critic:** f = a·cos(u, v) + b, a = softplus(α), init (a, b) = (10, −2.5) for every objective, so the step-0 ranking is the raw zero-shot ranking.
- **Objectives:** VCS (1 − mean_G J_G, T = tanh f), matched JS (balanced logistic), candidate softmax (multi-positive CE over the image's candidates).
- **Recipe (shared, not tuned per objective):** AdamW lr 1e-5 (weight decay 0.05 on matrices), critic scalars lr 1e-3; 200-step warm-up then cosine
  to 0; batch 32 images; 10 epochs; bf16 autocast; grad-norm clip 1.0; no flips (left / right words).
- **Readout every epoch (and at step 0):** CAL and DEV Top-1 (query, image-macro) over the referred candidates; common J with the model's own critic
  (J_own) and with an affine critic refitted on CAL (J_recal, comparable across objectives).
- **Selection:** epoch with max CAL image-macro Top-1 (primary; unbiased for DEV); also max CAL J_own (estimator view).

## Gates
1. Smoke (job 1032516): every objective and both backbones run end-to-end on 64 images (finite loss, evaluation, outputs).
2. In every full run, the step-0 DEV image-macro Top-1 must equal the frozen Table A raw number of that dataset × backbone within 0.3 point
   (CLIP: RefCOCOg 65.99, RefCOCO 58.14, RefCOCO+ 64.27; SigLIP 2: 75.48, 66.39, 73.85).  Otherwise the run is invalid and the pipeline is fixed
   before any reading.

## Units
Stage 1 (now): seed 0 for 3 datasets × 2 backbones × 3 objectives = 18 full runs (one GPU each).
Stage 2 (rule, fixed now): seeds 1–2 for all three objectives of a dataset × backbone cell if at seed 0 VCS is within 0.5 point of the best objective
on DEV Top-1 (CAL-selected) or better.  Otherwise the cell is reported at one seed, as "VCS behind by Δ at seed 0".

## Readings (descriptive at seed 0; paired-by-seed 95 % t intervals where three seeds exist)
1. DEV Top-1 per objective; VCS − JS, VCS − softmax; fine-tuned − zero-shot (step 0).
2. Estimator view: CAL / DEV J_recal per objective; does VCS fine-tuning expose more critic-fittable dependence than JS / softmax fine-tuning?
3. Fine-tuned vs frozen-feature Table A (the same objective's critic on frozen features), per dataset × backbone.
4. Consistency across the three datasets (same sign of VCS − JS / VCS − softmax?).
Not claimed: official val / test numbers; anything about the published fine-tuned grounding models (MDETR etc. are full detectors).

## Decisions at the freeze
- Stage-1 runs (18) submitted with `--dependency=afterok` on the smoke job 1032516 and `--kill-on-invalid-dep`, so none starts unless gate 1 passes.
  Gate 2 (step-0 equality with Table A raw) is read from each run's first log line before any of its numbers is used.
