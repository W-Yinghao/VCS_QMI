# P68 — the one-shot official CIFAR-10 test-set evaluation: report (2026-09-28)

Prereg: `P67_FINAL_OFFICIAL_TEST_PREREG_FROZEN_20260928.md` (frozen after every checkpoint existed; nothing tuned or selected after this evaluation).
Tables: `P68_FINAL_OFFICIAL_TEST.md` / `.json` (results-only commit 8e90de4; GPU job 1012926, 52 checkpoints).  Protocol: the frozen pilot
protocol — linear head trained on the 45 000 fit images with the frozen probe settings, kNN (k = 200) with the fit bank, same clean transform —
scoring the 10 000 official test images instead of the 5 000 selection images; the test batch (sha256 recorded) was opened for the first and only
time by this job.  Every decision in the programme was made on the selection split; the selection numbers are printed next to the test numbers.

## 1. Test vs selection agreement
Across the 22 cells the test linear accuracy differs from the selection accuracy by −0.03 points on average (max |Δ| = 0.66, the single-seed
16-view run); kNN is +0.35 on average (the test set is slightly easier for kNN).  No cell moves by more than its own seed spread plus 0.3 points, so
the selection split was a faithful proxy and the ranking of every comparison below is unchanged.

## 2. Equal-compute comparison on the official test set (linear % / kNN %, mean ± sd over 3 seeds unless marked)
| compute | VCS (frozen recipe) | SimCLR (tuned, P41) | VICReg (tuned, P41) | frozen P5 controls |
|---|---|---|---|---|
| 1× | 4v/B128/100ep **82.98 ± 0.45** / 78.57 ± 0.09 | 4v/B128/100ep **86.94 ± 0.12** / 84.85 ± 0.05 | 4v/B128/100ep **87.00 ± 0.25** / 84.45 ± 0.17 | SimCLR 86.29 ± 0.12 / 84.46; VICReg 85.07 ± 0.27 / 82.27 |
| 2× | 4v/B256/200ep **84.52 ± 0.18** / 81.31 ± 0.29 | 4v/200ep **87.99 ± 0.07** / 86.87 ± 0.09 | 4v/200ep **87.11 ± 0.31** / 84.71 ± 0.17 | |
| 4× | 8v/200ep **85.87 ± 0.07** / 83.69 ± 0.15 (2v/800ep 85.28 ± 0.40 / 82.40; 4v/400ep 85.63; 16v/200ep 86.08 singles) | 2v/800ep **88.21 ± 0.10** / 87.70 ± 0.51 | 2v/800ep **86.91 ± 0.15** / 84.94 ± 0.16 | |
| 8× | 4v/800ep **86.65 ± 0.26** / 85.30 ± 0.10 (strong-aug 88.26 / 85.79; 4v/B128/800ep 87.09; 8v/400ep 86.86 — singles) | 4v/800ep **88.13 ± 0.07** / 87.98 ± 0.53 | 4v/800ep **86.87 ± 0.06** / 84.78 ± 0.30 | |
| 16× | 4v/1600ep 87.60 / 86.43; 8v/800ep 87.42 / 86.58 (singles) | — | — | |

Gaps to VCS on the test set (linear): SimCLR +3.96 / +3.47 / +2.34 / +1.48 at 1× / 2× / 4× / 8×; VICReg +4.02 / +2.59 / +1.04 / +0.22.  On kNN: SimCLR
+6.3 / +5.6 / +4.0 / +2.7; VICReg +5.9 / +3.4 / +1.3 / −0.5 (VCS ahead of VICReg on kNN at 8×).

## 3. Statements the test set supports (pre-committed reporting; no new claims)
1. Under equal tuning budgets, tuned SimCLR exceeds the frozen VCS recipe at every compute point tested, by 4.0 points at 1× narrowing to 1.5 at 8×;
   tuned VICReg exceeds it by 4.0 → 0.2, i.e. the two are level at 8× on the linear probe and VCS is ahead on kNN there.
2. The frozen VCS recipe gains ≈ 1.2 points per compute doubling on the test set (82.98 → 84.52 → 85.87 → 86.65) and reaches 87.4–87.6 at 16× with one
   seed; SimCLR is flat above 2× (87.99 → 88.21 → 88.13).  SimCLR at 2× is above VCS at 16×.
3. The single strong-augmentation VCS run at 8× scores 88.26 on the test set, level with SimCLR's 3-seed 8× mean; it is one seed and was not part of
   the frozen recipe, so it is reported as a single-seed observation, not as the VCS number.
4. The selection split and the official test set agree to within seed noise for every cell, which is the only thing the test set was allowed to
   say about the *process*.

## 4. What is not claimed
Nothing about other datasets, other backbones, or longer schedules for the controls; no VCS accuracy advantage over either control at any
budget; the mechanism results of the technical note (bounded calibrated critic, h-uniformity ordering, no batch statistics) are unchanged and
remain non-accuracy claims.
