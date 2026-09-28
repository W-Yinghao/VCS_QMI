# P72 report — R2 (noisy pairing) and R3 (mismatch identification), frozen CLIP + identity adapters (2026-09-28)

Pre-registration: `P71_ROBUSTNESS_R2R3_PREREG_FROZEN_20260928.md` (frozen 14:47:32Z before compute).  Run: GPU job 1013019 (exit 0, 17 min),
results committed alone in 106d94d (`P72_robust_r2r3.{md,json}`).  Diagnostic added after the results (not pre-registered, CPU job 1013073,
`scripts/r2_raw_clip_retrieval.py`): own-caption retrieval of raw CLIP on the same evaluation splits.

## Verdicts
| line | letter reading | what it rests on |
|---|---|---|
| **R3** mismatch identification | **does not hold** (refuted at every cell) | JS has the highest AUROC in all 4 cells; VCS is below JS by 0.016 / 0.031 (animal, m = 0.2 / 0.4) and 0.020 / 0.043 (indoor → outdoor), and below InfoNCE in 3 of 4 cells |
| **R2** noisy pairing | **not interpretable as robustness** (letter: refuted on animal, holds on indoor → outdoor) | every method's clean retrieval *rises* with the mismatch fraction — see §1; the pre-registered "loss" is negative for all methods |

## 1. Surprise, disclosed: R2's metric measures distance from raw CLIP, not robustness
R@1 on the clean evaluation splits *increases* with m for every method (e.g. animal SRC-EVAL, VCS 0.080 → 0.191 from m = 0 to 0.4).  Cause:

| shift | split | raw CLIP R@1 (no adapter training) | best adapter at m = 0 | best adapter at m = 0.6 |
|---|---|---|---|---|
| animal | SRC-EVAL | **0.368** | 0.080 (vcs) | 0.234 (vcs) |
| animal | TGT-EVAL | **0.334** | 0.069 (vcs) | 0.205 (vcs) |
| indoor → outdoor | SRC-EVAL | **0.284** | 0.031 (infonce / logistic) | 0.212 (vcs) |
| indoor → outdoor | TGT-EVAL | **0.233** | 0.039 (vcs) | 0.188 (vcs) |

The training relation is *topic pairing* (P50 setting 2: the positive caption belongs to another image with the same supercategory set), while R2
evaluates *own-caption* retrieval.  Fitting the topic relation discards CLIP's instance-level alignment (identity-initialised adapters start at
raw CLIP; one epoch in the smoke probe still gave 0.44 on 2 000 images).  Injected random pairs dilute the topic signal, and the selection on each
method's clean CAL loss then picks less training at high m (animal m = 0.4: vcs and infonce 5 epochs; indoor: vcs lr 1e-4) — so the adapters stay
closer to CLIP and own-caption R@1 goes up.  The pre-registered "retrieval loss from m = 0 to 0.4" therefore ranks methods by how much they
move away from CLIP, not by how well they resist noise.  JS sits at a floor (R@1 0.01–0.03 at every m, always selecting lr 1e-3 × 40 epochs), so its
near-zero "loss" is also not robustness.  The design error is mine (R2 should have evaluated the training relation — topic-pair AUROC / J on clean
held-out topic pairs — or used exact pairing); it is recorded here rather than repaired post hoc.

Letter reading, for the record (R@1 loss m = 0 → 0.4, points, negative = gain; VCS minus competitor):
- animal: SRC vcs −11.0, infonce −12.1, logistic −5.9, js −0.7 (VCS − infonce = +1.1); TGT −11.4 / −12.6 / −6.3 / −0.2 (+1.2) → VCS's loss is larger
  than InfoNCE's → **refuted** by the letter.
- indoor → outdoor: SRC −17.9 / −10.9 / −4.5 / −0.5; TGT −14.6 / −12.3 / −3.9 / −0.4 → VCS smallest by ≥ 2.4 points → **holds** by the letter.

## 2. R3 (valid as designed: native score of the training pairs, no clean calibration)
| shift | m | raw CLIP | vcs | infonce | logistic (= +log K) | js |
|---|---|---|---|---|---|---|
| animal | 0.2 | 0.783 | 0.880 | 0.885 | 0.879 | **0.896** |
| animal | 0.4 | 0.780 | 0.856 | 0.862 | 0.843 | **0.887** |
| indoor → outdoor | 0.2 | 0.727 | 0.822 | 0.816 | 0.812 | **0.842** |
| indoor → outdoor | 0.4 | 0.723 | 0.790 | 0.797 | 0.756 | **0.833** |

All adapters clear the raw-CLIP floor.  JS — the balanced logistic on the *same* positive / K = 8 negative construction — is best everywhere, and
its margin over VCS grows with m.  So the mismatch-identification advantage that exists over pairwise logistic belongs to the balanced-mixture
family (and JS does it better), not to VCS.  Seed sd ≤ 0.002, so the ordering is not noise.

## 3. QC sentinels
- Injected counts = round(m · n_fit) at every m (sha256 of each index list in the JSON).  ✔
- "m = 0 R@1 within 1 point of the P58 topic-pairing value": **not checkable** — P58 reported calibration and AUROC, not own-caption R@1.
- Grid interior: at m = 0 every selection has lr = 1e-3 (grid edge) and all but animal-vcs have 40 epochs (edge); the grid is narrow for the
  topic relation (noted; it does not affect R3, which scores the model each method selected).

## 4. What is and is not claimed
- Claimed: under topic pairing on frozen CLIP ViT-B/32 with linear adapters, VCS's native score does **not** identify mismatched training pairs
  better than InfoNCE or JS; JS is the best of the four.
- Not claimed: anything about noise robustness from R2 (§1).  A valid R2 would need exact pairing (own caption as the positive) so that the
  clean metric and the training relation coincide; that is a new unit and is **not** launched without the owner's go.
