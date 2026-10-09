# VL1-15 — model-scale probe: the same estimator on Base vs Large of each feature family (RefCOCOg) — FROZEN 2026-10-09

Owner 2026-10-09: keep GPUs busy with complete VL experiments.  VL1-12 found that the fitted critic's CAL J orders feature sets by the ranking
accuracy the same critic reaches.  VL1-15 asks the within-family version: **does J rise from Base to Large in every family, and does it move with
the learned Top-1?**  Same protocol as `VL1_FROZEN_PROTOCOL.md` / the VL1-12 addenda (RefCOCOg UMD roles, region-uniform law, F2r, routes,
selection; VL1-14: no critic search needed).

## Feature sets (Large tier; Base results exist)
| family | Base (done) | Large (this unit) | interface |
|---|---|---|---|
| CLIP (OpenAI) | ViT-B/16 | **ViT-L/14 @ 336** (`open_clip` ViT-L-14-336-quickgelu, QuickGELU asserted) | tight crop |
| SigLIP 2 | B/16-224 | **L/16-256** (`google/siglip2-large-patch16-256`) | tight crop (adaptation) |
| FG-CLIP v1 | Base 224 | **Large 336** (`qihoo360/fg-clip-large`, 24 × 24 grid) | RoIAlign region API |
| FG-CLIP 2 | Base | **Large** (`qihoo360/fg-clip2-large`) | official region API |
Extractors unchanged except the model selector (env `CLIP_ARCH`, `SIGLIP2_REPO`, `FGCLIP1_REPO`, `FGCLIP2_REPO`; defaults = the Base models).

## Gates (implementation checks, before any fit)
SigLIP 2 L: ImageNet zero-shot check (as Base) ≥ the Base value (75.63).  FG-CLIP v1 L / FG-CLIP 2 L: COCO val2017 GT-box classification (as Base) ≥ the
Base value (52.28 / 64.77).  CLIP L: QuickGELU assertion and the fp16-cache precision check of the CLIP extractor.  A failing gate stops that model's fits
and is reported.

## Units (CPU, after the gates)
raw, VCS, JS, softmax, RFF at N all × 3 seeds per Large feature set (outputs `outputs/VL1_15_<family>_large`).

## Readings (descriptive)
1. Large − Base per family and route (paired by seed): CAL J and DEV image-macro Top-1.
2. Across all nine feature sets (5 Base + 4 Large): does the VCS critic's CAL J order equal its learned-Top-1 order (Spearman ρ, descriptive)?
3. VCS − JS on each Large set (same posterior); learned − raw; softmax − VCS.
W11 caveat: J across feature sets is a fit-limited reading, not a dependence comparison in the S sense.
