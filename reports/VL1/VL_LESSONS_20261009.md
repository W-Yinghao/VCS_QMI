# VL lessons so far — RefCOCOg (UMD) given-box matching and full grounding — 2026-10-09

Owner 2026-10-09: VL first; run the VL baselines, then summarise the lessons; decide experimentally whether the VL critic needs an SSL-style search.
All numbers are DEV (UMD train split, image-level FIT / CAL / DEV roles) unless marked val; official val is used only for reproduction checks of
external methods, and the UMD test split stays closed.  Sources: the VL1-xx reports in this folder (VL1-10/11, VL1-12 + add. 2 / 3, VL1-13, VL1-20, Table B note, ReCLIP reproduction).

## 1. The baseline landscape (what the numbers are)
**Table B — given candidate boxes (the image's referred objects; 1 356 DEV images, 6 442 expressions; image-macro Top-1, no task training):**
| row | Top-1 | note |
|---|---|---|
| CLIP ViT-B/16, crop + cosine | 65.99 | Table A start |
| FG-CLIP v1 Base, RoIAlign region API | 70.57 | CLIP-initialised; COCO box 52.3 (CLIP 44.2) |
| ReCLIP IPS only (RN50x16 + ViT-B/32, crop + blur) | 70.78 | val reproduction 65.26 (paper 65.32) |
| FG-CLIP 2 Base, official region API | 72.29 | released checkpoint 10 points below its paper on COCO box (64.8 vs 74.9) |
| ReCLIP official (IPS + relation heuristics) | 73.18 | val reproduction 68.20 (paper 68.08); heuristics tuned on val |
| **SigLIP 2 Base, crop + cosine** | **75.48** | best zero-shot row; crop is an adaptation |
**Table C — full grounding (no boxes; Acc@IoU 0.5):** Grounding DINO Swin-T OGC (zero-shot) val **60.52** (third-party 60.4), DEV 56.04; MDETR
R101 RefCOCOg-fine-tuned val **81.94** (paper 81.64; with REFER raw captions 81.21), val only because it was trained on RefCOCOg train;
MDETR EB3 not reproducible under timm 1.0 (NaN outputs).
**Table A — learned critics on frozen features (N all, CAL-selected; DEV image-macro / CAL J ×100):**
| features (raw) | VCS | matched JS | RFF | softmax (DEV-selected, optimistic) |
|---|---|---|---|---|
| CLIP crops (65.99) | 68.80 / 7.20 | 68.34 / 7.05 | 67.40 / 5.23 | 70.75 |
| FG-CLIP 2 (72.29) | 72.51 / 8.76 | 72.12 / 8.59 | 72.23 / 4.28 | 76.14 |
| FG-CLIP v1 (70.57) | 73.78 / 9.79 | 73.38 / 9.53 | 71.86 / 6.36 | 76.13 |
| ReCLIP isolation features (70.81 = ReCLIP IPS) | 74.19 / 10.07 | 73.17 / 9.58 | 70.82 / 6.40 | 75.53 |
| SigLIP 2 crops (75.48) | 76.00 / 11.63 | 76.43 / 11.41 | 75.06 / 9.25 | 78.22 |

## 2. Lessons about the estimator / critic (the paper's subject)
1. **Same posterior, nearly the same answer.** VCS and matched JS agree within 0.5 Top-1 on four of the five feature sets and at every N on CLIP.
   VCS's CAL J is 0.1–0.5 (×100) higher, as expected when J is optimised directly.  The exception is ReCLIP's isolation features: VCS − JS +1.02
   [+0.08, +1.96].  Both routes sit at the lr-grid edge there, so it is not read as an objective effect until VL1-14 has screened the optimisation.
2. **The fitted critic is a calibrated estimator that usually also ranks better than raw cosine:** +2.8 (CLIP), +3.2 (FG-CLIP v1), **+3.4 on
   ReCLIP's own isolation features** (74.19 vs ReCLIP IPS 70.81, and above ReCLIP official 73.18), +0.2 (FG-CLIP 2), +0.5 (SigLIP 2).  The gain
   is not monotone in feature strength.
3. **The task loss is the better ranker.**  Candidate softmax beats the CAL-selected VCS critic by 1.3–4 points on every feature set (selected on DEV,
   so optimistic, but the gap exceeds VCS's own selection bias of ≤ 0.8).  The estimator is not the best grounding objective, and the paper should
   not claim it is.
4. **Theory transfers:** ranking by the calibrated critic plus the known region prior beats the critic alone under a non-uniform law (VL1-11:
   +0.56 to +0.66, three routes).  This is a prediction CIFAR cannot test.
5. **J as a probe of frozen features is internally consistent, not externally validated.**  VCS's CAL J orders the five feature sets SigLIP 2
   (11.6) > ReCLIP features (10.1) > FG-CLIP v1 (9.8) > FG-CLIP 2 (8.8) > CLIP (7.2), exactly the order of the same critic's learned Top-1 (76.0 >
   74.2 > 73.8 > 72.5 > 68.8).  It is not the order of zero-shot accuracy (FG-CLIP 2 and FG-CLIP v1 swap).  The kernel route agrees on most pairs but not all (grid-limited).  Cross-feature J is a fit-limited reading (W11).
6. **Same-initialisation contrasts:** fine-grained training on a CLIP init (FG-CLIP v1 − CLIP) raises J for all three estimators (+1.1 to +2.6) and
   the learned Top-1 by about 5 points.  The released FG-CLIP 2 is below SigLIP 2 crops on every estimator, which is a property of that checkpoint
   and interface, not of fine-grained training.
7. **Pair-wise sigmoid with the actual positive rate (SigLIP-style) is a poor estimator of the balanced target** (worst J; −2 Top-1 at N 1 000).

## 3. Lessons about what limits the current numbers (→ VL1-14)
1. **Scorer form matters a lot:** the package's concat MLP never beat raw CLIP (65.9 vs 68.8 for the residual scorer at N all).  Identity
   initialisation at the raw ranking is the working design.
2. **The optimisation is at the edge of its grid:** every estimator-selected VCS / JS fit on every feature set (30 fits) picked lr 2e-3, the top of the grid.  The 3 000-update
   cap binds on CLIP, FG-CLIP 2 and FG-CLIP v1, but not on SigLIP 2 or the ReCLIP features.  The RFF route always picked its grid corner.
3. **Input representation dominates the critic:** changing the frozen features / region interface moves Top-1 by up to 9.5 points (66.0 → 75.5);
   training the critic moves it by at most 3.4.  Isolation and model choice should be fixed before critic work, and every critic comparison
   should be within one feature set.
4. **Answer to "does the VL critic need a search?" — no (VL1-14, frozen rule).**  Four alternatives (optimisation past the lr edge with 10 000
   updates, a 512 × 3 MLP, a low-rank bilinear critic, an affine lower bound) on CLIP and SigLIP 2: the best gains only +0.20 / +0.29 J (×100) over
   the frozen F2r recipe, below the pre-set +0.50, so F2r and the VL tables stand.  The lr edge was real (interior optimum 5e-3, cap no longer
   binding), but cheap.  Calibration of cosine alone gives ~45 % / 62 % of the fitted J.  Ranking is a different matter: the bilinear critic and the
   larger MLP rank better (+0.7 to +1.75 Top-1) at equal or lower J.  Estimation and ranking are separable, and the estimator criterion does not
   select the best ranker (`VL1_14_CRITIC_SCREEN_REPORT_20261009.md`).

## 4. Process lessons (cost us time; now checked by default)
- **Reproduce every external number before using a model:** open_clip "ViT-B-16" silently runs GELU instead of QuickGELU; ReCLIP's inputs had to
  be rebuilt (official data gone); FG-CLIP 2's released checkpoint does not reach its paper; MDETR evaluates on REFER's tokenised `sent` captions
  (raw captions cost 0.7); timm 1.0 breaks MDETR EB3; HF image processors misread crops 1–3 pixels high as channels-first (now always
  `input_data_format="channels_last"`).
- **Contamination is checked per row:** MDETR (RefCOCOg train) is val-only; Grounding DINO Swin-T OGC is the clean zero-shot detector (the Swin-B
  "cogcoor" model saw RefCOCO); ReCLIP's heuristics were tuned on val, so DEV is held out for it.
- **Primary candidate set and metric:** 90 % of DEV queries are same-category (the target shares its category with another referred object), so
  the task is mostly within-category discrimination.  Report query-level and image-macro, exact-target and IoU (ReCLIP).
- Matching a frozen interface exactly is the fair way to test the critic on a strong baseline: VL1-13 reproduces ReCLIP IPS-only inside the
  fitter (99.71 % argmax agreement) and starts the critic from that ranking.

## 5. Still open
Flickr30k Entities (annotations here, images missing); Table C "proposer + critic" needs a clean proposer (COCO-trained detectors leak val / test
images; rebuild ≈ 1 GPU-day, budget decision); final official-test pass (owner decision).
