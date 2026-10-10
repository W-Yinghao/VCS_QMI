# VL3 addendum 1 — full fine-tuning of the Large backbones (CLIP ViT-L/14@336, SigLIP 2 L/16-256) on RefCOCOg, RefCOCO, RefCOCO+ — FROZEN 2026-10-10

Owner 2026-10-10: full experiments, both, all datasets; keep 5–10 GPU jobs pending.  Same protocol as `VL3_FULL_FINETUNE_FROZEN_20261010.md`
(objectives VCS / matched JS / candidate softmax; shared recipe AdamW 1e-5, 10 epochs, batch 32 images, critic init (10, −2.5); CAL-Top-1 selection;
seeds rule: seeds 1–2 for a cell if VCS is within 0.5 point of the best objective at seed 0).  Only the backbone changes, and activation
checkpointing (`--grad-ckpt`) is used to fit ViT-L at batch 32.  It is a memory technique; the objective and gradients are unchanged.

## Gates
1. Smoke on 64 images for both Large backbones (job below).
2. Step-0 DEV image-macro Top-1 equals the frozen Large raw value within 0.3: CLIP-L RefCOCOg 67.10 / RefCOCO 59.12 / RefCOCO+ 64.38; SigLIP 2-L
   74.87 / 65.90 / 73.07.

## Units
Stage 1: seed 0 for 3 datasets × 2 Large backbones × 3 objectives = 18 runs (after the smoke).  Stage 2 by the frozen rule.

## Readings
As VL3, plus Large − Base per objective and dataset (fine-tuning gain with scale), and the Large fine-tuned vs frozen-feature Large Table A.
