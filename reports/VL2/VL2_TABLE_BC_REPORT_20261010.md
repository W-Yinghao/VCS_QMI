# VL2 — RefCOCO / RefCOCO+ Tables B and C (public baselines, zero-shot rows, reproduction checks) — report — 2026-10-10

Protocol `VL2_REFCOCO_PLUS_FROZEN_20261009.md`; results-only commits `66da2ec`, `cd77e55`, `0e4e9ae`, `55f4004` (`reports/VL2/VL2_tableBC.json`,
`scripts/vl2_tableB.py`).  DEV = eligible DEV images of each UNC train split with the referred-object candidates (RefCOCO 1 580 images / 11 698
expressions, RefCOCO+ 1 587 / 11 666); UNC val = reproduction checks; testA / testB closed.  Captions for RefCOCO/+ are REFER's lower-cased `sent`
form (MDETR's files; no raw text exists here).

## 1. Table B — given boxes (DEV, image-macro Top-1, exact target; no task training)
| row | RefCOCO | RefCOCO+ | (RefCOCOg, VL1) |
|---|---|---|---|
| CLIP ViT-B/16 crop | 58.14 | 64.27 | 65.99 |
| ReCLIP IPS only (crop + blur, two CLIPs) | 62.88 | 69.02 | 70.78 |
| FG-CLIP v1 Base region API | 65.55 | *pending (bundled job)* | 70.57 |
| FG-CLIP 2 Base region API | 65.95 | 70.65 | 72.29 |
| **SigLIP 2 Base crop** | **66.39** | **73.85** | **75.48** |
| ReCLIP official (+ spatial-relation parsing) | **72.63** | 69.78 | 73.18 |
The ReCLIP-feature cache (Table A raw row) agrees with ReCLIP's own IPS-only output on both datasets (62.87 vs 62.88; 68.88 vs 69.02), as on
RefCOCOg.

## 2. Table C — full grounding (Acc@IoU 0.5; MDETR's GIoU criterion in brackets)
| row | RefCOCO val | RefCOCO DEV | RefCOCO+ val | RefCOCO+ DEV |
|---|---|---|---|---|
| Grounding DINO Swin-T OGC (zero-shot) | **50.69** (50.57); paper 50.41 ✓ | 51.76 | **51.65** (51.52); paper 51.40 ✓ | 52.73 |
| MDETR R101 fine-tuned on that dataset (val only) | **86.89** (86.53); paper 86.75 ✓ | — | **76.00** (75.63); paper 79.52 ✗ | — |
ReCLIP on UNC val with all COCO objects as candidates (its own protocol): RefCOCO 50.95 (parse, IoU) / 45.53 (IPS); RefCOCO+ 52.04 / 51.42.  No
verified given-box reference exists for these two, so they are reported, not used as checks.

## 3. Reading
1. **Location words drive ReCLIP's relation module, as the dataset design predicts.**  Relation parsing adds +9.75 image-macro on RefCOCO DEV (72.63
   vs 62.88) and only +0.76 on RefCOCO+ (location words banned).  On RefCOCO, ReCLIP official is the best zero-shot given-box row, above every
   frozen feature set; on RefCOCO+ (appearance only) SigLIP 2 crops lead by 4.1.
2. **Feature ordering is stable across the three datasets:** SigLIP 2 crops > FG-CLIP 2 ≳ FG-CLIP v1 > ReCLIP IPS > CLIP crops on raw cosine
   (FG-CLIP v1 on RefCOCO+ pending).  RefCOCO+ is easier than RefCOCO for every frozen feature set (+4 to +7 points).
3. **Reproduction checks:** Grounding DINO Swin-T matches its paper on RefCOCO / RefCOCO+ (+0.3) but not on RefCOCOg (60.5 vs 67.46, VL1-20).
   MDETR R101 reproduces RefCOCO (86.89 vs 86.75) and RefCOCOg (81.94 vs 81.64) but **not RefCOCO+ (76.00 vs 79.52)** with the same code,
   captions and preprocessing.  Cause not identified (the released RefCOCO+ checkpoint, or a paper-table number from a different model); the
   RefCOCO+ MDETR row is labelled "released checkpoint, 3.5 below the paper".
4. Table A (the critic on these features, VCS ≈ JS, learning value, probe consistency, RefCOCO+ − RefCOCO) follows when the VL2 fits finish.
