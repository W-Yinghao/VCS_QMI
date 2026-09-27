# Second application — property pre-checks: execution plan (2026-09-27)

Source: `CS_QMI/VCS_QMI_Second_Application_Server_Brief_v1.md` (exploration brief, not an execution spec).  Four property pre-checks decide which
families of second applications survive; none of them selects a task.  Frozen items and the "discuss first" list are inherited unchanged
(`PLAN_CONSTRAINTS_20260925.md`); every pre-check below stays inside the movable set (encoder/projector, z vs h, batch composition, optimiser,
critic form with tanh output, pairing scheme incl. cross-batch/independent-pool negatives, evaluation protocol).

## Order and status
All four done 2026-09-27 — hand-back: `SECOND_APP_PRECHECKS_SUMMARY_20260927.md` (surviving families, engineering, data risk; no task selected).

| pre-check | property | data | status |
|---|---|---|---|
| D  leakage-detection power | bounded objective, distribution-free bound, one-sided lower confidence bound | existing frozen CIFAR-10 h + planted nuisance (no new data) | **done — does not hold by the frozen false-alarm clause** (permutation test's level 0.10–0.12 in 3 of 16 R = 100 null cells; every power / exactness / monotonicity clause held; framing letter-vs-intent flagged for the owner) (`P46_PRECHECK_D_REPORT_20260927.md` §3.1) |
| B1 closed-form vs neural critic | quadratic objective ⇒ closed-form critic in a fixed feature class | existing checkpoints, two-view features of the selection images | **done — property holds** (`P48_PRECHECK_B1_REPORT_20260927.md`) |
| A  calibration & threshold transfer | pairwise scores have absolute meaning | COCO-2017 captions, no-animal → animal shift; frozen CLIP ViT-B/32 + identity adapters | **done — holds conditionally** (calibration without a calibration set only in the mid-dependence regime; no threshold-transfer advantage: monotone critic) (`P50_PRECHECK_A_REPORT_20260927.md`) |
| C  batch decoupling | positives and negatives averaged separately | A's setting 2 | **done — does not hold** on frozen-tower adapters (all methods flat in batch size within 1 %) (`P54_PRECHECK_C_REPORT_20260927.md`) |
| B2 registration energy surface | closed-form J* as a similarity for rigid/affine alignment | 60 constructed tent-map modality pairs from COCO val2017 | **done — does not hold** (rougher energy than MI/NMI: more local maxima, narrower basin, lower success from large offsets) (`P52_PRECHECK_B2_REPORT_20260927.md`) |

Compute: D and B1 ran on the CPU partition (they are small; the P44 ceiling runs were still on the GPUs).  Owner 2026-09-27: GPUs may be used
for the pre-checks — CLIP feature extraction, adapter grids (A/C), registration energy surfaces (B2) and any full-size sweeps go to the GPU
partitions (A100,H100,L40S,RTX6000PRO) as short jobs; CPU stays for tests of O(n²) size and summaries.

## Data and third-party assets (recorded before use)
- CIFAR-10 (local copy, frozen split hash c35d7cd3…): pre-checks D and B1.
- COCO 2017 (local copy at `/projects/EEG-foundation-model/yinghao/FMCA-AV/coco`: train2017 118 287 images, val2017 5 000, captions_*2017.json;
  annotations CC BY 4.0, images under their Flickr terms — research use): source pairs for A/C.  Identity = image id; a caption is paired
  only with its own image.  Target distribution for A: a covariate-shift split constructed from the instance annotations (e.g. images with
  vs without a given supercategory), identity-disjoint from the source; the exact rule is fixed in A's prereg.  A second, external target
  (Flickr8k/30k) would need a download and licence check — proposed, not done.
- Frozen dual towers for A/C: open CLIP ViT-B/32 `laion2b_s34b_b79k` — owner approved the download 2026-09-27; installed `open_clip_torch` 3.3.0 and
  `scipy` 1.18.1 into the env (uv); weights (605 143 316 bytes, sha256 ac4f8c4b88af6d96…) under `/home/infres/yinwang/CS_QMI/models/open_clip/` with
  `PROVENANCE_ViT-B-32_laion2b_s34b_b79k.json` (source HF `laion/CLIP-ViT-B-32-laion2B-s34B-b79K`).
- B2 real modality pairs (RGB-NIR scene dataset, multi-sequence MRI): licence check pending; the constructed-modality stage does not need them.

## Common discipline (from the brief §3)
Prereg before running; evidence categories completed / report-only / pending / proposed; no official test sets; identity splits with hashes;
competitors get the same tuning budget (budgets tabled); failed configurations kept; mechanism vs observation separated; single-seed marked.
Each pre-check ends in exactly one of: property holds / holds conditionally / does not hold — with the family table of the brief's appendix A.
