# VL1-22 — Table C, detection setting: clean detector proposals ranked by the VL3 fine-tuned encoders — report — 2026-10-10

Protocol `VL1_22_TABLE_C_FROZEN_20261010.md`; results-only commit `3f01781` (`reports/VL1/VL1_22_results.json`, `scripts/vl1_22_aggregate.py`, job 1033764).
Proposals come from the clean VL1-21 detector (COCO AP 36.4, gate D passed; trained without any RefCOCO / + / g CAL / DEV / val / test image).
Each of the 108 VL3 CAL-Top-1 checkpoints ranks the DEV proposals of its own dataset by cosine (= its critic's ranking).  The top-1 box counts
as correct at IoU ≥ 0.5.  **Gate T1 (reloaded checkpoint reproduces its given-box DEV Top-1 within 0.3): 108 / 108 pass.**  Smoke passed
(`vl1_22_smoke`, ceiling 100 % on 32 images).

## 1. Results (DEV Acc@IoU 0.5, image-macro, mean ± sd over 3 seeds; paired 95 % t)
| cell | VCS | matched JS | candidate softmax | VCS − softmax | given-box Top-1 (VCS) |
|---|---|---|---|---|---|
| RefCOCOg × CLIP B/16 | 46.27 ± 0.16 | 47.36 ± 0.95 | 47.07 ± 0.49 | −0.80 [−2.42, +0.83] | 80.95 |
| RefCOCOg × SigLIP 2 B/16 | 56.22 ± 0.79 | 57.59 ± 0.46 | 56.86 ± 0.21 | −0.64 [−2.15, +0.86] | 85.01 |
| RefCOCOg × CLIP L/14@336 | 50.75 ± 0.38 | 52.28 ± 0.15 | 51.91 ± 1.07 | −1.16 [−2.88, +0.57] | 83.62 |
| RefCOCOg × SigLIP 2 L/16 | 60.45 ± 0.54 | **61.25 ± 0.46** | 60.69 ± 0.75 | −0.24 [−2.04, +1.56] | 87.27 |
| RefCOCO × CLIP B/16 | 44.23 ± 1.33 | 44.93 ± 0.43 | 48.39 ± 0.60 | −4.16 [−8.65, +0.33] | 84.85 |
| RefCOCO × SigLIP 2 B/16 | 56.56 ± 0.81 | 55.82 ± 0.17 | 57.76 ± 0.54 | −1.20 [−2.75, +0.36] | 88.27 |
| RefCOCO × CLIP L/14@336 | 50.99 ± 0.74 | 50.13 ± 0.77 | 53.58 ± 0.26 | **−2.59 [−4.53, −0.66]** | 89.22 |
| RefCOCO × SigLIP 2 L/16 | 60.12 ± 0.50 | 59.92 ± 1.25 | **61.07 ± 0.78** | −0.95 [−3.98, +2.09] | 90.95 |
| RefCOCO+ × CLIP B/16 | 42.08 ± 0.94 | 42.43 ± 0.70 | 46.20 ± 1.52 | −4.12 [−10.11, +1.86] | 81.76 |
| RefCOCO+ × SigLIP 2 B/16 | 53.46 ± 0.66 | 53.43 ± 0.28 | 55.66 ± 0.37 | **−2.20 [−3.91, −0.50]** | 86.93 |
| RefCOCO+ × CLIP L/14@336 | 46.07 ± 0.21 | 46.04 ± 0.07 | 49.11 ± 1.18 | −3.04 [−6.48, +0.40] | 84.17 |
| RefCOCO+ × SigLIP 2 L/16 | 56.84 ± 1.56 | 57.62 ± 1.67 | **60.28 ± 1.09** | **−3.44 [−6.30, −0.58]** | 89.40 |
| **Grounding DINO-T zero-shot (same DEV expressions)** | RefCOCOg 56.75 · RefCOCO 53.86 · RefCOCO+ 54.41 | | | | |
| Proposal ceiling (any proposal IoU ≥ 0.5) | RefCOCOg 97.3 · RefCOCO 98.5 · RefCOCO+ 98.8 | | | | |

## 2. Pre-registered readings (12 cells as units, 95 % t)
| quantity | pooled |
|---|---|
| **VCS − softmax** | **−2.05 [−2.93, −1.16]** (softmax ahead in 12 / 12 cells; 3 intervals below 0, none above) |
| VCS − JS | −0.39 [−0.89, +0.10] |
| given-box → detection drop | VCS −34.0 [−36.8, −31.3]; JS −33.6 [−36.5, −30.6]; softmax −32.0 [−34.1, −29.8] |

**Pre-fixed wording (interval below 0): "with detector proposals, softmax leads by 2.1 points."**

## 3. Readings
1. **The given-box tie does not survive the detection setting.**  With given referred boxes VCS and softmax tie (VL3: +0.01 [−0.35, +0.36]).
   With about 44–52 query-independent proposals per image, softmax-trained encoders rank better by 2.1 points.  Every objective loses about a
   third of its accuracy; softmax loses ≈ 2 points less.
2. **Same dataset ordering as the given-box table, magnified.**  Mean VCS − softmax is −0.71 on RefCOCOg, −2.23 on RefCOCO and −3.20 on RefCOCO+.
   On given boxes it was +0.60 / −0.11 / −0.47.
3. **Same posterior:** VCS ≈ JS (−0.39 [−0.89, +0.10]).  The detection gap is between the estimator objectives as a group and the task loss.
4. **Against the clean external row:** the SigLIP 2 L/16 rankers beat Grounding DINO-T zero-shot on all three datasets (+4.5 to +7.2 for the best
   objective per dataset).  CLIP B/16 rankers fall about 10 points below it.  Grounding DINO-T was trained for grounding on other data; ours are
   two-stage (clean detector + fine-tuned crop ranker) and trained only on given referred boxes.
5. **Why the drop is large (hypothesis, not tested):**
   - Training candidates were the image's referred objects only.  Proposals add unreferred objects, object parts and loose crops, which no
     training pair contained.
   - Softmax trains the encoder to separate the target from *all* candidates of an image.  VCS / JS score each pair against a fixed calibration
     (P vs Q laws).  That may leave cosine margins less robust to unseen distractors.
   - Proposal-aware training (detected boxes as negatives) or fusing the detector score would be the natural next step.  Either needs a new
     pre-registration.  The frozen unit ranks by cosine alone.

## 4. Caveats
- DEV split, not official val / test (closed).  MDETR has no DEV row (it was trained on these images' expressions); its official-val numbers stay as
  a supervised reference only.
- One detector (Faster R-CNN R50-FPN, 1×).  A stronger proposer would raise all rows; whether it changes the objective gap is untested.
