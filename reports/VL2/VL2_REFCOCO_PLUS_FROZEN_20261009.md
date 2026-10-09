# VL2 — RefCOCO and RefCOCO+ (UNC splits): Tables A / B / C under the VL1 protocol — FROZEN 2026-10-09

Owner 2026-10-09: "没有实验要提交了吗？不要浪费GPU时间" (VL priority; no new SSL).  The planned "second VL dataset" (Flickr30k images still missing)
is replaced for now by RefCOCO and RefCOCO+, whose COCO-2014-train images are local and whose expressions come with MDETR's annotation archive.
RefCOCO+ forbids location words, so it is the appearance-only (conditional) extension planned in the baselines document.

## Data (audit `VL2_DATA_AUDIT_{refcoco,refcocoplus}.json`, job 1031945)
Source: MDETR `finetune_<dataset>_<split>.json` (COCO image id, COCO annotation id of the target, caption in REFER's lower-cased tokenised
`sent` form; no `raw` text) + COCO-2017 instances (same ids) for distractors.  Counts equal the published UNC statistics (RefCOCO train / val /
testA / testB expressions 120 624 / 10 834 / 5 657 / 5 095; RefCOCO+ 120 191 / 10 758 / 5 726 / 4 889).  Roles by image within each train split,
seed 20261008: FIT / CAL / DEV = 13 596 / 1 699 / 1 699 (RefCOCO) and 13 594 / 1 699 / 1 699 (RefCOCO+); eligible (≥ 2 referred) FIT 12 653 /
12 616, CAL 1 589 / 1 591, DEV 1 580 / 1 587 images; DEV queries 11 698 / 11 666; same-category share 99.3 %.  UNC val = VAL_OFFICIAL
(reproduction checks only), testA / testB closed.
**Contamination bookkeeping:** RefCOCO / RefCOCO+ FIT images include RefCOCOg UMD val (≈ 520) and test (≈ 960–1 000) images, and their val / test
include RefCOCOg train images.  Each dataset is analysed on its own (critics fitted and read within one dataset).  No cross-dataset training,
and the detector proposer (if built) must exclude every referring val / test image of all three datasets.

## Units
- **Features (GPU, jobs 1031948–57):** CLIP ViT-B/16 crops, SigLIP 2 Base crops, FG-CLIP 2 Base region API, FG-CLIP v1 Base region API,
  ReCLIP isolation features — the five VL1 extractors, unchanged, with the dataset switch (`VL_DATASET`; RefCOCOg paths unchanged).
- **Table A (CPU):** raw, VCS, JS, softmax, RFF at N all × 3 seeds per dataset × feature set; F2r, lr {1e-4, 5e-4, 2e-3}, 3 000 updates,
  selection as `VL1_FROZEN_PROTOCOL.md` (VL1-14 found no critic search needed).  Outputs `outputs/VL2_<dataset>_<features>`.
- **Table B:** the raw rows of the five feature sets (ReCLIP IPS-only = raw on the ReCLIP features, as in VL1-13).
- **Table C (GPU):** Grounding DINO Swin-T OGC on UNC val (reproduction; paper zero-shot RefCOCO / RefCOCO+ val 50.41 / 51.40 for a model
  not verified to be the released one) and on DEV; MDETR R101 RefCOCO / RefCOCO+ on UNC val only (paper 86.75 / 79.52).

## Readings (descriptive; as VL1)
1. VCS − JS per dataset × feature set (Top-1, CAL J; expected close).
2. Learned − raw per route (paired by seed); softmax − VCS (task-loss margin).
3. Estimator probe: does CAL J order the five feature sets as the critic's learned Top-1 does (the VL1 consistency property), per dataset?
4. RefCOCO+ − RefCOCO (descriptive, same feature set): learning value and J without location words.
5. Table B / C placement; reproduction checks against the published numbers.
