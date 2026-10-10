# VL1-21 — clean detector proposer — report — 2026-10-10

Protocol `VL1_21_CLEAN_PROPOSER_FROZEN_20261009.md` (with the 1 GPU × 16 procedural note); training job 1033183 (12 epochs, H100 NVL, ≈ 7.6 h), eval job 1033362;
results-only commit `9327cc2` (`reports/VL1/VL1_21_detector_eval.json`).  torchvision Faster R-CNN R50-FPN from an ImageNet-1k backbone.  It is trained on
COCO-2017 train minus every CAL / DEV / val / test image of RefCOCOg, RefCOCO and RefCOCO+ (102 485 training images; 14 781 excluded).

## Gate D
**COCO val2017 bbox AP 36.4 (gate ≥ 33: pass); AP50 57.5.**  torchvision's reference for this architecture is 37.0, with a 26-epoch schedule and all
images.  This run used a 12-epoch (1×) schedule without the referring images.

## Proposals (top-100, score ≥ 0.05, per-class NMS) — recall of the referred targets at IoU 0.5
| dataset / split | images | targets | R@10 | R@20 | R@50 | R@100 | detections / image |
|---|---|---|---|---|---|---|---|
| RefCOCOg DEV | 2 190 | 4 219 | 88.1 | 93.2 | 96.5 | 97.2 | 45.8 |
| RefCOCO DEV | 1 699 | 4 221 | 91.3 | 95.9 | 98.1 | 98.6 | 50.7 |
| RefCOCO+ DEV | 1 699 | 4 201 | 91.6 | 96.1 | 98.3 | 98.7 | 52.0 |
CAL and official-val recall are in the JSON (official val: box recall only; no grounding result on val).

## Use
Table C (`VL1_22_TABLE_C_FROZEN_20261010.md`) ranks these DEV proposals with the 108 VL3 checkpoints.  The DEV recall@100 (97–99 %) is the ceiling
for any ranker on these proposals.  The chain (smoke → 12 jobs) starts automatically after this gate.
