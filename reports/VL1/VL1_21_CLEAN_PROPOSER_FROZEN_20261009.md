# VL1-21 — clean detector proposer for Table C (proposer + critic) — FROZEN 2026-10-09

Owner 2026-10-09: "不要浪费GPU时间" (VL priority).  This is budget item D2 of the plan ("clean proposer ≈ 1 GPU-day"), started now because the
GPUs would otherwise idle; it can be stopped by the owner at any time (`scancel` of the vl1_21 jobs).

## Why rebuild
Table C's "proposer + critic" row needs query-independent box proposals.  Every off-the-shelf COCO detector was trained on COCO train2014 / 2017,
which contains all RefCOCO / RefCOCO+ / RefCOCOg images, including their val / test splits.  MAttNet rebuilt its detector without the
referring val / test images (its files are offline).  We do the same, and also exclude our own CAL / DEV images (development readouts).

## Procedure (`scripts/vl1_21_detector.py`, `slurm/vl1_21_detector.sbatch`)
- **Exclusions:** every image in CAL / DEV / VAL_OFFICIAL / TEST_CLOSED of RefCOCOg-UMD, RefCOCO-UNC and RefCOCO+-UNC
  (`outputs/VL1_21/exclusions.json`).
- **Model:** torchvision Faster R-CNN R50-FPN (v1: FrozenBN backbone, 3 trainable stages), ImageNet-1k V1 backbone (`resnet50-0676ba61`); no
  COCO-pretrained weights.
- **Training:** COCO-2017 train minus exclusions (non-crowd boxes > 1 px); 4 GPUs × 4 images, SGD lr 0.02, momentum 0.9, wd 1e-4, 1 000-iter
  linear warm-up, ×0.1 at epochs 8 and 11, 12 epochs (1× schedule), horizontal flip, fp16 autocast; the model's own resize (800 / 1 333).
  Resumable each epoch.  A 1-GPU, 30-iteration smoke run precedes the training.
- **Gate:** COCO val2017 bbox AP ≥ 33 (torchvision's 26-epoch reference for this model is 37.0; ours is 1× with ≈ 10 % fewer images).  If it
  fails: stop and report; no Table C proposer row.
- **Proposals:** per image of CAL / DEV / val for each dataset: the detector's top-100 detections (score ≥ 0.05, per-class NMS), sorted by
  score; recall@{10, 20, 50, 100} at IoU 0.5 of the referred targets.

## Use (next unit, to be pre-registered after the gate)
Proposal features through the Table A feature extractors, scored by the frozen F2r critics (and raw cosine / softmax), top-1 box → Acc@IoU 0.5 on
DEV; reported beside Grounding DINO (zero-shot) and MDETR (supervised, val).  Never merged with the given-box tables.
