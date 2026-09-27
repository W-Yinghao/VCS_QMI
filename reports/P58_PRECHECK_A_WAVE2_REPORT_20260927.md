# Row A, wave 2 — report (P58): A-S1 calibration-vs-dependence curve; A-T mismatch detection without a target calibration set

Prereg: `P57_PRECHECK_A_WAVE2_PREREG_FROZEN_20260927.md` (+ the indoor → outdoor split hashes appended after feature extraction).  Tables:
`P58_precheck_A_wave2.md` (main run, nominal 0.8; results-only commit d322a00) and `P58_precheck_A_wave2_nom0.7.md` (secondary run, nominal 0.7).
Data: COCO-2017 captions, frozen CLIP ViT-B/32 + identity-initialised 512×512 adapters, two shifts — *animal* (P49 splits) and *indoor → outdoor*
(source pool 23 327 → SRC-FIT 13 327 / CAL 5 000 / EVAL 5 000; target pool 10 325 → TGT-EVAL 5 000); four pairing difficulties (exact / topic /
coarse / random); methods vcs / infonce / logistic with the P49 grid (lr × epochs selected on SRC-CAL by each method's own loss), 3 seeds.
Evidence category: completed (GPU jobs 1011090–1011092).

## 1. A-S1 — native calibration of (1+T)/2 vs Platt-on-cosine as a function of held-out J (VCS adapter; ECE, 15 equal-mass bins)
| J | shift | pairing | split | native ECE | cosine + Platt (SRC-CAL) ECE | native − Platt |
|---|---|---|---|---|---|---|
| 0.923 | animal | exact | SRC-EVAL | 0.046 | 0.004 | +0.042 |
| 0.883 | indoor → outdoor | exact | SRC-EVAL | 0.058 | 0.006 | +0.053 |
| 0.797 | animal | exact | TGT-EVAL | 0.089 | 0.036 | +0.052 |
| **0.633** | **indoor → outdoor** | **exact** | **TGT-EVAL** | **0.131** | **0.089** | **+0.043** |
| 0.572 | animal | topic | SRC-EVAL | 0.023 | 0.022 | +0.001 |
| 0.441 | indoor → outdoor | topic | SRC-EVAL | 0.029 | 0.032 | −0.003 |
| 0.157 | animal | topic | TGT-EVAL | 0.088 | 0.089 | −0.001 |
| 0.132 | animal | coarse | SRC-EVAL | 0.023 | 0.024 | −0.000 |
| 0.095 | indoor → outdoor | coarse | SRC-EVAL | 0.022 | 0.023 | −0.000 |
| −0.09 … −0.44 | both | coarse / topic | TGT-EVAL | 0.135 / 0.192 / 0.308 | 0.124 / 0.179 / 0.311 | +0.011 / +0.013 / −0.003 |
| ≈ 0 | both | random | both | 0.020–0.044 | 0.315–0.317 | (estimator artefact, see note) |

*Note on the random-pairing rows.* There the cosine carries no signal and the Platt fit correctly outputs a constant 0.5 (Brier exactly 0.25);
equal-mass binning of tied scores then assigns arbitrary label chunks to the bins and prints an "ECE" of 14/15 × 0.5 = 0.467 for a perfectly
calibrated constant predictor.  Those rows say nothing about calibration and are outside the rule's regime (J < 0.1).  Negative J on the
target (topic / coarse under shift) means the source-trained adapter is anti-aligned there; both native and Platt are badly calibrated then.

**Reading (frozen).**  Settings with J ∈ [0.1, 0.7] on both shifts: five (0.132, 0.157, 0.441, 0.572, 0.633).  Four are within ±0.003 of Platt.
The fifth — indoor → outdoor, exact pairing, target split, J = 0.633 — has native ECE 0.043 above Platt, which exceeds the +0.02 threshold of the
*does-not-hold* clause.  **A-S1: does not hold** by the letter.  What the curve shows: native calibration is as good as a fitted Platt
whenever the *pairing relation* is the mid-dependence topic/coarse one (J 0.1–0.6, on either split), and worse than Platt in every
*exact-pairing* cell (J 0.63–0.92) — the excess there is 0.04–0.05 regardless of J.  So the P50 "mid-dependence regime" reading survives as a
description of the *relation*, not of a J-interval: an exact-pair adapter evaluated under shift lands at J = 0.63 and is still miscalibrated.
The native rule's calibration is set by what the critic was trained on, not by the dependence level where it is evaluated.

## 2. A-T — mismatch detection on the target without a target calibration set (adapters trained on the topic pairing; hard mismatches = coarse-topic captions)
| shift | nominal | rule (VCS embedding) | accept | realized precision | \|prec − nominal\| | AUROC |
|---|---|---|---|---|---|---|
| animal | 0.8 | native (1+T)/2 ≥ 0.8 | 0.163 | 0.871 | 0.071 | 0.816 |
| animal | 0.8 | cosine + Platt fitted on SRC-CAL | 0.074 | 0.911 | 0.111 | 0.816 |
| animal | 0.8 | cosine + Platt fitted on target (oracle, not deployable) | 0.179 | 0.867 | 0.067 | 0.816 |
| animal | 0.8 | raw CLIP, cosine + Platt (SRC-CAL) | 0.082 | 0.873 | 0.073 | 0.713 |
| animal | 0.8 | InfoNCE adapter, cosine + Platt (SRC-CAL) | 0.139 | 0.898 | 0.098 | 0.837 |
| animal | 0.7 | native | 0.250 | 0.838 | 0.138 | 0.816 |
| animal | 0.7 | cosine + Platt (SRC-CAL) | 0.142 | 0.880 | 0.180 | 0.816 |
| animal | 0.7 | target oracle | 0.293 | 0.823 | 0.123 | 0.816 |
| indoor → outdoor | 0.8 | native | 0.241 | 0.534 | 0.266 | 0.531 |
| indoor → outdoor | 0.8 | cosine + Platt (SRC-CAL) | 0.087 | 0.543 | 0.257 | 0.531 |
| indoor → outdoor | 0.8 | target oracle | 0.000 | — (accepts nothing) | — | 0.531 |

**Reading (frozen: holds if native is within 0.05 of nominal and not worse than source-Platt by > 0.02; does not hold if worse by ≥ 0.05).**
*Animal shift:* the native rule is the closest deployable rule to the nominal precision at both nominal levels (0.071 and 0.138 off, vs 0.111
and 0.180 for the source-fitted Platt transfer) and accepts about twice as many pairs — but it is not within 0.05 of nominal, so the *holds*
clause fails; the *does-not-hold* clause is not met.  → **conditional.**  Every rule over-shoots (realized precision above nominal), the
target-fitted oracle included (0.867 at 0.8, 0.823 at 0.7): the hard coarse-topic mismatches are easier to reject than the calibration pairs
were, so no score calibrated elsewhere hits the nominal level here.  *Indoor → outdoor:* AUROC 0.53 for every embedding — the source-trained
adapters do not separate matched from coarse-topic-mismatched pairs in the outdoor target at all; precision is ≈ 0.53 for every rule and the
oracle accepts nothing.  The task is undefined on this shift (nothing to calibrate), which the reading records as *conditional* by the letter
(native is 0.009 worse than source-Platt) and as *no evidence* in substance.  **A-T overall: holds conditionally**, with a small margin of
practical value: on the shift where the task exists, native scoring beats the only deployable competitor by 0.04 in precision-to-nominal and
by 2× in acceptance, while everything stays conservative.

## 3. Consequences for the family table (brief appendix A, row A)
A-S1 closes the claim "native calibration in the mid-dependence regime" as stated by P50 — calibration tracks the training relation, not the
J-interval.  What remains for mismatch detection / open-set rejection / pointwise PMI maps: a *conditional* candidate where (i) the adapter
is trained on the same kind of relation it will score (topic-level pairs → topic-level decisions), and (ii) the shift keeps the pair problem
solvable (AUROC well above chance).  Under those conditions the native rule is the best deployable rule seen here, but never within 0.05 of a
nominal precision; anyone needing the nominal level must calibrate on the target, which the oracle row shows also over-shoots on hard pairs.
Data re-weighting by T stays unsupported.  Not claimed: other towers, non-linear adapters, external targets, retrieval quality.
