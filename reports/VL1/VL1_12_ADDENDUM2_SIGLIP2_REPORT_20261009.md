# VL1-12 addendum 2 — Table A on SigLIP 2 Base crop features — report — 2026-10-09

Addendum `VL1_12_ADDENDUM2_SIGLIP2_FROZEN_20261009.md`; results-only commit `f27509c` (`VL1_12s_table.json`, `VL1_12s_siglip2_minus_clip.json`,
`VL1_12s_fgclip2_minus_siglip2.json`, `VL1_12s_strata.json`; job 1030844, whose regression check reproduces the committed VL1-12 reading
exactly).  SigLIP 2 Base (patch16-224 @ 75de2d5), tight crops through its own processor (adaptation), ImageNet check 75.63 vs 78.2.  N all × 3
seeds; same roles / law / scorer / selection as VL1-10.

## 1. Results (DEV image-macro Top-1 %, CAL common J × 100)
| route | CLIP ViT-B/16 crops | FG-CLIP 2 region API | **SigLIP 2 crops** |
|---|---|---|---|
| raw cosine (no training) | 65.99 | 72.29 | **75.48** |
| VCS (estimator-selected) | 68.80 / J 7.20 | 72.51 / J 8.76 | 76.00 ± 0.06 / **J 11.63** |
| matched JS (estimator-selected) | 68.34 / J 7.05 | 72.12 / J 8.59 | 76.43 ± 0.38 / J 11.41 |
| RFF ridge-tanh | 67.40 / J 5.23 | 72.23 / J 4.28 | 75.06 ± 0.20 / J 9.25 |
| candidate softmax (DEV-selected, optimistic) | 70.75 | 76.14 | 78.22 ± 0.09 |

On SigLIP 2: learned − raw = VCS **+0.52 [+0.37, +0.67]**, JS +0.94 [+0.00, +1.88], RFF −0.43 [−0.92, +0.06], softmax +2.73 [+2.52, +2.95]
(optimistic).  VCS − JS: Top-1 −0.42 [−1.51, +0.67] (task-selected +0.01), CAL J +0.22 [−0.14, +0.59] → close.  At N all the CAL-J optimum
falls at 1 850–2 600 updates, so the 3 000-update cap does not bind here, unlike CLIP and FG-CLIP 2.

## 2. Frozen readings
1. **SigLIP 2 − CLIP (paired):** Top-1 VCS +7.20 [+5.87, +8.53], JS +8.09, RFF +7.66, softmax +7.46; **CAL J VCS +4.43 [+3.94, +4.92], JS
   +4.35, RFF +4.02**.  All three estimators agree in sign and size: SigLIP 2 crops carry far more region–phrase dependence than CLIP crops.
2. **FG-CLIP 2 − SigLIP 2 (paired; confounded by construction, since FG-CLIP 2 = SigLIP 2 init + fine-grained training + RoIAlign region
   interface instead of a crop):** Top-1 VCS −3.49 [−3.74, −3.24], JS −4.31, RFF −2.82, softmax −2.08; **CAL J VCS −2.87, JS −2.81, RFF
   −4.97**.  Every estimator reads less dependence through the released FG-CLIP 2 region path than through SigLIP 2 crops.  The released
   checkpoint is also 10 points below its paper on COCO box classification (64.8 vs 74.9), so this says nothing about FG-CLIP 2 as published.
   The crop-vs-RoIAlign interface and the fine-grained training are not separated.
3. **VCS − JS on SigLIP 2:** close (same posterior), as on CLIP and FG-CLIP 2.
4. **Table B row:** raw SigLIP 2 cosine on the referred candidates = 75.48 image-macro (query 74.54), the best zero-shot given-box row on DEV,
   above ReCLIP official (73.18) and FG-CLIP 2 (72.29).
5. Strata (secondary): VCS − raw is positive in every stratum (same-category +0.48, other-category +0.96, long +0.75, short +0.35); softmax −
   VCS +1.89 [+1.58, +2.21] (query-weighted).

## 3. The probe statement across feature sets  (**superseded** by `VL1_12_ADDENDUM3_FGCLIP1_REPORT_20261009.md` §3: with five feature sets J follows the learned, not the zero-shot, accuracy)
With all three feature sets in, the neural estimators' CAL J orders them **SigLIP 2 crops (11.6) > FG-CLIP 2 region API (8.8) > CLIP crops
(7.2)**.  That is the same order as their zero-shot ranking (75.5 > 72.3 > 66.0) and their learned Top-1.  The kernel route agrees on SigLIP 2
being highest and on FG-CLIP 2 < SigLIP 2, and disagrees only on CLIP vs FG-CLIP 2 (grid-limited, corner-selected on every feature set).  The paper
can state: "the same fitted estimator ranks frozen feature sets by how much region–phrase dependence they expose, in the order of their
zero-shot grounding accuracy".  The W11 wording stays: fitted J is a fit-limited lower reading, and a cross-feature difference compares
estimation difficulty as well as dependence.

The estimator's ranking gain over raw shrinks as the features improve (VCS +2.8 on CLIP, +0.2 on FG-CLIP 2, +0.5 on SigLIP 2), while the task loss
keeps +2.7 to +4.8.  On strong features the critic's value is the dependence measurement, not the argmax.
