# P130 addendum 1 — ResNet-50 CIFAR-100, seeds 0–2, A-P3 / JS-AP3 / SimCLR — report — 2026-10-09

Pre-registration `P130_ADDENDUM1_SEEDS_FROZEN_20261007.md`; results-only commit `6a41cd6` (`reports/P130A1_results.json`,
`scripts/p130a1_aggregate.py`, job 1030725).  Development split; frozen-h linear readout (recipe_raw, P124 rule), kNN alongside; 95 % t intervals
over 3 seeds; P114 labels.  ResNet-18 references: P107 A-P3, P120 JS-AP3, P91 SimCLR (seeds 0–2).  SimCLR keeps its untuned recipe.

## 1. Accuracy (linear top-1 %, per seed 0 / 1 / 2)
| method | ResNet-50 linear | mean ± sd | ResNet-50 kNN | R50 − R18 (linear, paired) |
|---|---|---|---|---|
| VCS (A-P3) | 64.66 / 65.04 / 63.46 | **64.39 ± 0.82** | 57.86 / 57.64 / 58.26 | +4.39 [+2.86, +5.91] |
| matched JS (JS-AP3) | 62.56 / 63.08 / 64.04 | 63.23 ± 0.75 | 57.16 / 57.74 / 58.18 | +4.03 [+2.12, +5.93] |
| SimCLR | 58.54 / 59.76 / 59.34 | 59.21 ± 0.62 | 59.44 / 59.90 / 59.18 | +0.96 [−1.44, +3.36] |

## 2. Frozen reading (paired by seed)
| contrast | linear | label | kNN | label |
|---|---|---|---|---|
| A-P3 − SimCLR | **+5.17 [+2.68, +7.67]** | **clear** | −1.59 [−3.25, +0.08] | inconclusive |
| A-P3 − JS-AP3 | +1.16 [−2.59, +4.91] | inconclusive | +0.23 [−0.82, +1.27] | close |

- **Linear: the balanced-posterior objective beats SimCLR by about 5 points at ResNet-50 (three seeds, interval excludes 0).**  The wider
  backbone adds ≈ 4 linear points for both balanced-posterior objectives and about 1 point (interval includes 0) for SimCLR.  At ResNet-18 the
  3-seed means differ by +1.75 (60.00 vs 58.25).
- **kNN points the other way:** SimCLR's kNN is about 1.6 above A-P3 (inconclusive at n = 3), and SimCLR is the only method whose kNN exceeds its
  linear accuracy.  The linear gap is not an artefact of one linear recipe.  At seed 0 the second linear readout of P141 (standardised features,
  lr / wd chosen by inner CV) gives A-P3 61.42 vs SimCLR 54.58 (+6.84).  The two readout families still disagree, so the paper states the result
  as **"+5.2 linear (clear), kNN not better (−1.6, inconclusive)"** and never as a uniform representation-quality gain.
- **VCS vs matched JS:** inconclusive on linear (+1.16, interval ±3.8; the seed-0 +2.10 shrank) and close on kNN.  This is consistent with the
  same-posterior expectation and is not claimed as a difference.

## 3. Use in the paper
The strongest SSL result of the programme, stated with its readout caveat: on CIFAR-100 at ResNet-50, the VCS objective (and its matched JS
control, which shares the posterior) gives a frozen-backbone linear accuracy ≈ 5 points above untuned SimCLR over three seeds.  The kNN readout
does not show this.  Cost: ≈ 26–29 GPU-h per run (P130 seed 0).
