# VL1-15 (+ add. 1) — model scale on RefCOCOg: the same frozen-feature critic on Base / Large / So400m — report — 2026-10-10

Protocols `VL1_15_SCALE_PROBE_FROZEN_20261009.md`, `VL1_15_ADDENDUM1_SO400M_AND_VL2_LARGE_FROZEN_20261010.md`; results-only commit `cb488e2`
(`reports/VL1/VL1_15_results.json`, `scripts/vl1_15_aggregate.py`, job 1032552).  RefCOCOg DEV, image-macro Top-1; CAL common J × 100; N all × 3 seeds;
F2r critic (VL1-14: no critic search needed).  Note: this is a frozen-feature unit frozen before the owner's 2026-10-10 "full experiments, not probes"
instruction; it is completed and reported, and no further frozen-feature extensions are planned (VL3 = full fine-tuning).

## 1. Gates
| model | check | result |
|---|---|---|
| CLIP ViT-L/14@336 | QuickGELU + fp16 precision | pass (argmax agreement 99.4 %) |
| SigLIP 2 L/16-256 | ImageNet zero-shot ≥ Base 75.63 | 78.20 pass |
| SigLIP 2 So400m/14-384 | same | 79.18 pass |
| FG-CLIP v1 Large | COCO box ≥ Base 52.28 | 63.18 pass |
| FG-CLIP 2 Large | COCO box ≥ Base 64.77 | **55.06 FAIL → excluded** |
| FG-CLIP 2 So400m | same | 66.12 pass |

## 2. Per feature set (DEV Top-1 / CAL J of the VCS critic)
| family | Base | Large | So400m |
|---|---|---|---|
| CLIP (OpenAI) | raw 65.99 · VCS 68.80 · J 7.20 | raw 67.10 · VCS 69.23 · J 7.92 | — |
| SigLIP 2 (crop) | 75.48 · 76.00 · 11.63 | 74.87 · 76.09 · 10.84 | 75.97 · 76.80 · 11.36 |
| FG-CLIP v1 (RoIAlign) | 70.57 · 73.78 · 9.79 | 72.39 · 74.78 · 11.13 | — |
| FG-CLIP 2 (region API) | 72.29 · 72.51 · 8.76 | gate failed | 74.49 · 74.42 · 9.75 |
| ReCLIP isolation features (VL1-13) | 70.81 · 74.19 · 10.07 | | |

## 3. Frozen readings
1. **Scale, paired by seed (larger − Base):**
   - CLIP L: raw +1.11; VCS Top-1 +0.43 [−0.45, +1.31]; **J +0.72 [+0.53, +0.91]**.
   - FG-CLIP v1 L: raw +1.82; VCS +1.00 [+0.19, +1.81]; **J +1.34 [+1.25, +1.42]**.
   - FG-CLIP 2 So400m: raw +2.20; VCS +1.90 [−0.10, +3.91]; **J +0.98 [+0.35, +1.62]**.
   - **SigLIP 2 L: raw −0.61; VCS +0.08; J −0.79 [−1.26, −0.32]**; SigLIP 2 So400m: raw +0.48; VCS +0.80 [+0.13, +1.46]; J −0.27 [−1.14, +0.60].
   In three of the four families the larger model exposes more critic-fittable dependence and is ranked better.  In the SigLIP 2 family (crop
   adaptation at 256 / 384 px), scale does not raise J (Large is lower than Base), and raw / learned Top-1 move by < 1 point.
   Matched JS gives the same signs in every family (J +0.59 / +1.30 / +0.69 / −0.82 / −0.36).
2. **Probe consistency over the 10 feature sets with a passed gate:** Spearman ρ(VCS CAL J, VCS learned DEV Top-1) = **0.90**; ρ(J, raw zero-shot
   Top-1) = 0.83.  The order by J differs from the learned order mainly inside the SigLIP 2 family (B highest by J, So400m highest by Top-1).
3. The task loss keeps its margin at every scale: softmax − VCS (mean DEV Top-1; softmax DEV-selected, so optimistic) is +1.4 to +4.1 across the ten sets (larger models: CLIP L +2.7, SigLIP 2 L +1.4 / So400m +2.1, FG-CLIP v1 L +4.0, FG-CLIP 2 So400m +4.1).
4. W11 caveat unchanged: J across feature sets is a fit-limited reading.  Section 2 shows it ordering feature sets close to (not exactly in) the
   order of the accuracy the same critic reaches.

## 4. For the paper
"The fitted VCS critic's dependence reading rises with model scale in three of four families and ranks ten frozen feature sets close to the
order of the critic's own grounding accuracy (ρ = 0.90)."  The SigLIP 2 exception is stated.  These are frozen-feature results; the learning
claims move to VL3 (full fine-tuning).
