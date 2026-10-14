# P155 — official CIFAR-10 test of the final recipe A-P3 — report — 2026-10-14

**Protocol and results.**
- Protocol: `P155_AP3_OFFICIAL_TEST_PROTOCOL_FROZEN_20261014.md` (`78085a9`), frozen before the read.
- Job: 1040187, exit 0.
- Results-only commit: `P155_AP3_OFFICIAL_TEST.{json,md}` and `P155_AP3_OFFICIAL_TEST_contrasts.{json,md}`.
- All numbers below come from those files.
- Owner, 2026-10-14: "我们现在已经决定A-P3是最终配方，所以直接用测试集测啊".

**Gate.** Before the test partition was opened, the stand-in run of A-P3 seed 0 on the selection split reproduced its selection value exactly: 89.06 vs 89.06.

## 1. Official test, 10 000 images (linear probe and kNN; mean ± sd over seeds)
| cell | seeds | test linear | test kNN | selection linear | selection kNN |
|---|---|---|---|---|---|
| **A-P3** (fixed tanh(2s − 1), all-view tokens, full gradient; 4 views, 800 ep) | 0–4 | **88.83 ± 0.10** | 87.75 ± 0.23 | 89.02 ± 0.04 | 87.41 ± 0.08 |
| SimCLR recipe at VCS's view / epoch budget (P41 / P111; 4 views, 800 ep) | 0–4 | 88.20 ± 0.13 | 88.07 ± 0.40 | 88.31 ± 0.21 | 87.86 ± 0.43 |
| JS-AP3 (matched JS on the A-P3 recipe, P114) | 0–2 | 88.92 ± 0.29 | 88.04 ± 0.06 | 88.81 ± 0.45 | 87.67 ± 0.27 |
| v2 recipe, for reference only (P68) | 0–2 | 86.65 ± 0.26 | 85.30 ± 0.10 | 87.01 ± 0.53 | 85.46 ± 0.12 |

## 2. Pre-stated contrasts (paired by seed, 95 % t)
| family | contrast | metric | mean | 95 % interval | per seed | reading |
|---|---|---|---|---|---|---|
| **primary** | A-P3 − SimCLR | linear | **+0.63** | **[+0.36, +0.90]** | +0.76 / +0.86 / +0.54 / +0.68 / +0.30 | **ahead** |
| secondary | A-P3 − SimCLR | kNN | −0.32 | [−0.94, +0.30] | +0.40 / −0.44 / −0.94 / −0.10 / −0.50 | no detectable difference |
| secondary | A-P3 − JS-AP3 | linear | −0.07 | [−0.57, +0.43] | +0.12 / −0.28 / −0.04 | no detectable difference (n = 3) |
| secondary | A-P3 − JS-AP3 | kNN | −0.38 | [−0.78, +0.02] | −0.20 / −0.50 / −0.45 | no detectable difference (n = 3) |

## 3. Reading
1. **The final recipe leads SimCLR on the official test with a linear probe.** The pre-stated wording: "A-P3 ahead of SimCLR on the official test", +0.63 [+0.36, +0.90], positive on 5 of 5 seeds. The comparator is the SimCLR recipe at VCS's view / epoch budget (R4_02 §4.1 label), not a separately tuned SimCLR. The selection-split lead was +0.72 [+0.48, +0.95] (P107). The test lead is slightly smaller, because A-P3 loses 0.19 points from selection to test and SimCLR loses 0.10. This closes R4_02 §6.1 for A-P3: the recipe is now confirmed on held-out data.
2. **kNN does not separate.** −0.32 [−0.94, +0.30]; SimCLR is ahead on 4 of 5 seeds, as on the selection split.
3. **The lead belongs to the recipe, not to the quadratic loss.** Matched JS on the same recipe is level with A-P3 on the linear probe (−0.07). It is ahead on kNN on all 3 seeds (−0.38), but that interval contains 0 at n = 3. This agrees with P114 on the selection split and with every other matched comparison in the programme (`PROJECT_MAP_20261014.md` §2).
4. **Development metric vs test.**
   - Linear: test − selection is within −0.45 to +0.37 for every run.
   - kNN: every run except one SimCLR seed scores higher on the test.
   - The selection split is therefore a faithful development metric for this comparison.

## 4. Disclosures and limits
- **A-P3 was chosen on the selection split** after the v3–v7 screens; seed 0 was the screen-selection seed. This read confirms that choice; it does not select a recipe.
- **Part of the comparator was already known:** SimCLR seeds 0–2 on the test were read in P68. This is the partition's second use, owner-approved, for the final recipe and its paired comparators only.
- **Scope:** one dataset (CIFAR-10), ResNet-18 and one probe protocol. The external solo-learn protocol (P76: SimCLR 91.24 vs VCS 87.33) and the ImageNet-100 pilot (P154: SimCLR +3.08) are unchanged by this read. The strong-augmentation and CIFAR-100 A-P3 runs were not tested.
