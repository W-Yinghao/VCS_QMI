# VL1 Table B on DEV — public given-box rows on Table A's candidates — note — 2026-10-09

Results-only commit `d0ed119` (`VL1_tableB_dev.json`, `scripts/vl1_tableB_aggregate.py`).  Same 1 356 eligible DEV images and 6 442 expressions as
Table A; candidates = the image's referred objects (ReCLIP input file `refcocog_umd_dev_referred.jsonl`, built from REFER with the ReCLIP format).
ReCLIP reproduction on UMD val: 68.20 (paper 68.08), IPS-only 65.26 (65.32) — `VL1_RECLIP_REPRODUCTION_20261009.md`.

| row | task training | query Top-1 (exact target) | image-macro Top-1 (exact) | image-macro (IoU ≥ 0.5) |
|---|---|---|---|---|
| CLIP ViT-B/16 crop, cosine | none | 64.22 | 65.99 | — |
| FG-CLIP 2 Base, official region API, cosine | none | 71.11 | 72.29 | — |
| ReCLIP IPS only (RN50x16 + ViT-B/32, crop + blur) | none | 69.62 | 70.78 | 71.96 |
| ReCLIP official (IPS + spatial-relation parsing) | none; heuristics tuned on UMD **val** by the authors | **71.81** | **73.18** | 74.30 |
| *Table A, for reference:* VCS on CLIP crops, N all (CAL-selected) | RefCOCOg FIT | — | 68.80 | — |
| *Table A:* VCS on FG-CLIP 2, N all (CAL-selected) | RefCOCOg FIT | — | 72.51 | — |
| *Table A:* softmax on FG-CLIP 2, N all (DEV-selected, optimistic) | RefCOCOg FIT | — | 76.14 | — |

The two ReCLIP scorings differ because "exact target" = predicted index is the target object (Table A's definition), while ReCLIP's own `correct` =
IoU ≥ 0.5, where overlapping candidates can both count.  On all COCO objects of the eligible images (ReCLIP's candidate set, crowd boxes
included), ReCLIP official scores 62.74 exact / 64.90 IoU and IPS-only 59.90 / 62.06.  Table A's all-objects metric uses non-crowd distractors
(CLIP 52.95, FG-CLIP 2 59.44), so those are reported, not compared.

## Reading (descriptive placement)
- **Input isolation beats critic training on these features.** ReCLIP's isolation proposals alone (crop + blur, two CLIP backbones; IPS-only
  70.78) rank better than the VCS critic trained on 10 674 images of single-backbone ViT-B/16 crops (68.80).  FG-CLIP 2's region interface
  (72.29 raw) does better still.  Table A's learning value (+2.8 over its own raw features) is therefore real but smaller than what a better region
  input gives.
- ReCLIP's relation parsing adds +2.4 (73.18 vs 70.78) on DEV.  Its heuristics were tuned on UMD val, and DEV comes from the UMD train split,
  which the authors never used, so these DEV numbers are held out for ReCLIP.
- **Same-input follow-up (proposed as VL1-13, not submitted):** Table A on ReCLIP's own isolation features.  Concatenating the four normalised
  per-box embeddings (RN50x16 / ViT-B/32 × crop / blur) against the matching text embeddings makes cos(u, v) equal to the mean of ReCLIP's
  four cosines.  ReCLIP sums `logit_scale · cos` over models and isolations (`executor.py`), and both released CLIP models use logit scale 100.  The residual scorer's initial ranking (2·cos − 1) then **equals ReCLIP IPS-only exactly**, and learning starts from the strongest
  zero-shot given-box ranking.  This is the clean test of whether the estimator adds value on top of the best public isolation input.
