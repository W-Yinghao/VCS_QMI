# Second application — property pre-checks: execution plan (2026-09-27)

Source: `CS_QMI/VCS_QMI_Second_Application_Server_Brief_v1.md` (exploration brief, not an execution spec).  Four property pre-checks decide which
families of second applications survive; none of them selects a task.  Frozen items and the "discuss first" list are inherited unchanged
(`PLAN_CONSTRAINTS_20260925.md`); every pre-check below stays inside the movable set (encoder/projector, z vs h, batch composition, optimiser,
critic form with tanh output, pairing scheme incl. cross-batch/independent-pool negatives, evaluation protocol).

## Order and status
| pre-check | property | data | status |
|---|---|---|---|
| D  leakage-detection power | bounded objective, distribution-free bound, one-sided lower confidence bound | existing frozen CIFAR-10 h + planted nuisance (no new data) | **prereg frozen, running** (`P45_PRECHECK_D_PREREG_FROZEN_20260927.md`) |
| B1 closed-form vs neural critic | quadratic objective ⇒ closed-form critic in a fixed feature class | existing checkpoints, two-view features of the selection images | **prereg frozen, running** (`P47_PRECHECK_B1_PREREG_FROZEN_20260927.md`) |
| A  calibration & threshold transfer | pairwise scores have absolute meaning | cross-modal pairs: local COCO-2017 captions (source) with a constructed distribution shift (target); frozen dual towers | **draft** (data + towers plan below; prereg to freeze before any run) |
| C  batch decoupling | positives and negatives averaged separately | shares A's data and towers | **draft** (after A's pipeline exists) |
| B2 registration energy surface | closed-form J* as a similarity for rigid/affine alignment | constructed modality pairs from COCO images first (transform truth known by construction); real RGB-NIR / MRI pairs only if a licensed public source is confirmed | **draft** |

Compute: CPU partition only (the owner's other project uses the GPU quota; the brief says not to pre-empt).  Feature extraction on CPU
(≈ 2 min per 50 k CIFAR images at 16 threads); all tests are O(n²) at most with n ≤ 5 000.

## Data and third-party assets (record before use; nothing downloaded yet)
- CIFAR-10 (local copy, frozen split hash c35d7cd3…): pre-checks D and B1.
- COCO 2017 (local copy at `/projects/EEG-foundation-model/yinghao/FMCA-AV/coco`: train2017 118 287 images, val2017 5 000, captions_*2017.json;
  annotations CC BY 4.0, images under their Flickr terms — research use): source pairs for A/C.  Identity = image id; a caption is paired
  only with its own image.  Target distribution for A: a covariate-shift split constructed from the instance annotations (e.g. images with
  vs without a given supercategory), identity-disjoint from the source; the exact rule is fixed in A's prereg.  A second, external target
  (Flickr8k/30k) would need a download and licence check — proposed, not done.
- Frozen dual towers for A/C: an open CLIP checkpoint (weights would be downloaded; source, licence and sha256 to be recorded in A's prereg
  before the download).  `open_clip` is not installed in the env; installation is part of A's setup.
- B2 real modality pairs (RGB-NIR scene dataset, multi-sequence MRI): licence check pending; the constructed-modality stage does not need them.

## Common discipline (from the brief §3)
Prereg before running; evidence categories completed / report-only / pending / proposed; no official test sets; identity splits with hashes;
competitors get the same tuning budget (budgets tabled); failed configurations kept; mechanism vs observation separated; single-seed marked.
Each pre-check ends in exactly one of: property holds / holds conditionally / does not hold — with the family table of the brief's appendix A.
