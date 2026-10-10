# VL2 — RefCOCO / RefCOCO+ Table A (frozen features, 10 feature sets) — report — 2026-10-10

Protocols `VL2_REFCOCO_PLUS_FROZEN_20261009.md`, `VL1_15_ADDENDUM1_SO400M_AND_VL2_LARGE_FROZEN_20261010.md` (Large / So400m tier); results-only commit
`3a50b26` (`reports/VL2/VL2_results.json`, `scripts/vl2_aggregate.py`).  DEV image-macro Top-1 (referred candidates), CAL common J × 100; N all × 3
seeds; F2r critic; VCS / JS / RFF CAL-J-selected, softmax DEV-selected (optimistic).  Part of the fits ran on the GPU fitter (identical results,
`vl_gpu_fit_check.json`).  Frozen-feature results; the learning claims are in VL3 (full fine-tuning).

## 1. Base feature sets (raw → VCS / JS / softmax / RFF; J of VCS)
| features | RefCOCO | RefCOCO+ |
|---|---|---|
| CLIP B/16 crop | 58.14 → 64.06 / 63.66 / 66.81 / 60.66; J 6.65 | 64.27 → 70.92 / 70.63 / 72.42 / 66.97; J 8.35 |
| SigLIP 2 B crop | 66.39 → 70.92 / 70.78 / 73.61 / 67.47; J 9.66 | 73.85 → 77.86 / 77.57 / 79.66 / 74.41; J 12.58 |
| FG-CLIP 2 region API | 65.95 → 72.22 / 71.80 / 78.15 / 66.42; J 9.59 | 70.65 → 73.96 / 73.70 / 78.01 / 71.56; J 10.78 |
| FG-CLIP v1 RoIAlign | 65.55 → **86.88** / 86.47 / 89.18 / 67.52; J **18.64** | 70.88 → 75.76 / 75.47 / 79.39 / 71.41; J 11.85 |
| ReCLIP isolation (crop + blur) | 62.87 → **83.10** / 81.84 / 84.33 / 64.47; J **14.37** | 68.88 → 75.69 / 74.99 / 78.47 / 70.76; J 11.75 |
Large / So400m tier: see `VL2_results.json` (CLIP-L, SigLIP 2 L / So400m, FG-CLIP v1 L, FG-CLIP 2 So400m on both datasets).

## 2. Frozen readings
1. **Same posterior:** VCS − JS Top-1 within ±0.5 in 14 of 20 dataset × feature cells.  In the other 6, VCS is ahead (+0.53 to +1.26), and only
   RefCOCO × ReCLIP features excludes 0 (+1.26 [+0.59, +1.94]).  JS is never ahead by more than 0.5.  **VCS's CAL J is higher in all 20 cells**
   (+0.10 to +1.19 × 100), as expected when J is optimised directly.
2. **Learning value:** every learned route beats raw in every cell (VCS +2.6 to +21.3).  **The task loss is the better ranker on frozen features
   in all 20 cells** (softmax − VCS +1.2 to +10.5; DEV-selected, optimistic).
3. **Probe consistency:** Spearman ρ(VCS CAL J, VCS learned Top-1) over the 10 feature sets = **0.96 (RefCOCO), 0.90 (RefCOCO+)** (RefCOCOg:
   0.90); ρ(J, raw zero-shot) = 0.53 / 0.73.  J tracks what the critic can learn, not what the features give zero-shot.
4. **Scale ladders** repeat RefCOCOg: J and Top-1 rise from Base to Large for CLIP and FG-CLIP v1, and from Base to So400m for FG-CLIP 2.
   SigLIP 2 Large reads lower than Base on both datasets (J −0.67 / −0.82).
5. **RefCOCO+ − RefCOCO:** crop-based features gain +5 to +8 (raw and learned) on RefCOCO+.  The region interfaces that see the whole image
   (FG-CLIP v1 RoIAlign, ReCLIP blur isolation) lose 6–11 points of learned Top-1 on RefCOCO+.

## 3. A finding to state carefully — position-bearing region interfaces
On RefCOCO, the critic trained on FG-CLIP v1 or ReCLIP features gains **+20 to +21 points** over raw (to 86.9 / 83.1), against +4 to +7 for every
crop-based feature set.  On RefCOCO+ the same features gain an ordinary +5 to +7.  **Explanation (hypothesis, not tested here):** FG-CLIP v1 pools
the box from a full-image feature map, and ReCLIP's blur isolation keeps the global layout visible, so these region features carry the box's
position in the image; crops discard it.  RefCOCO's expressions use location words heavily ("left man", "right"), RefCOCO+ forbids them, and the
gap appears exactly there.  The same direction appears in ReCLIP's own relation module (+9.8 on RefCOCO, +0.8 on RefCOCO+; Table B), and the
task loss exploits it too (FG-CLIP 2 So400m on RefCOCO: softmax 85.5 vs VCS 74.9).  Roles are image-disjoint, so this is not a split leak.
A direct test (strata by location words, or box-coordinate features added to crops) is not run; the claim stays a hypothesis.
