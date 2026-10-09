# VL1-13 — Table A on ReCLIP's isolation features (same input as the strongest public zero-shot given-box row) — DRAFT 2026-10-09

## Why
Table B on DEV (`VL1_TABLEB_DEV_NOTE_20261009.md`): ReCLIP IPS-only (two CLIP backbones × crop / blur isolation, no training) ranks the referred
objects at 70.78 image-macro.  The VCS critic trained on single-backbone ViT-B/16 crops (Table A) reaches 68.80.  The comparison mixes input and
learning.  VL1-13 holds the input fixed at ReCLIP's own isolation features and asks whether the estimator (and its controls) adds value over that
ranking.

## Features (built with ReCLIP's code, not re-implemented)
`scripts/vl1_13_reclip_features.py`: ReCLIP's `ClipExecutor.tensorize_inputs` (crop and blur isolations, GaussianBlur 100 with the box pasted back,
square resize, CLIP normalisation; raw COCO bbox) and its two OpenAI CLIP models (RN50x16, ViT-B/32, fp16).  Per box u = ½ [crop_RN, blur_RN,
crop_B32, blur_B32], per expression v = ½ [t_RN, t_RN, t_B32, t_B32] on "a photo of " + lower-cased expression; every block L2-normalised.  Then
|u| = |v| = 1 and cos(u, v) = mean of ReCLIP's four cosines.  ReCLIP IPS-only ranks by Σ 100·cos, so **raw cosine on this cache = ReCLIP
IPS-only** and the residual scorer F2r starts exactly at ReCLIP's ranking.  Width 2 560 (MLP input 7 680).

**Gate (job 1030334):** a 20-image smoke run, then the full train-side cache, then the DEV agreement check.  Per-query argmax from the cache
against ReCLIP's own IPS-only output on the referred candidates must reach ≥ 99 % agreement (fp16 encodes, different batch shapes).
If it fails: stop and report, no fits.

## Units (CPU, `slurm/vl1_10_run.sbatch`, VL1_OUT=VL1_13)
raw, VCS, JS, softmax, RFF at N all × 3 seeds; roles / law / scorer F2r / lr grid / selection / 3 000-update cap as `VL1_FROZEN_PROTOCOL.md`.

## Reading (to freeze; descriptive)
1. **Learning value on the best isolation input:** each learned route − raw (= ReCLIP IPS-only) on DEV image-macro Top-1, paired by seed
   (estimator-selected for VCS / JS / RFF; softmax DEV-selected, optimistic).  "Adds value" if the interval excludes 0 above.  The package rule
   (not worse than raw by > 1 point) is checked.
2. VCS − JS (Top-1, CAL J): same-posterior check.
3. Estimator panel: CAL common J on these features beside CLIP / FG-CLIP 2 / SigLIP 2 (W11 caveat: fitted J across inputs is not a DPI
   statement).
4. Placement vs ReCLIP official (with relation parsing, 73.18 on DEV), descriptive.  The learned rows use the RefCOCOg FIT annotations; ReCLIP uses
   none.
Not claimed: anything on official val / test before the final pass.
