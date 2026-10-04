# P115 — strong-augmentation factors (crop vs colour jitter) for A-P3 and SimCLR, CIFAR-10 seed 0 — report — 2026-10-04

Pre-registration `P115_V5_AUG_FACTORS_PREREG_FROZEN_20261003.md`; results-only commit `c14486d` (`reports/P115_factors/`, `P115_factor_table.json`,
`scripts/p115_factor_table.py`).  One seed per cell — differences are described, not tested (frozen).  Crop-strong = scale min 0.08 (standard 0.20);
jitter-strong = 0.8/0.8/0.8/0.2 (standard 0.4/0.4/0.4/0.1); everything else per method unchanged.

## 1. The 2 × 2 (linear / kNN %, development validation split)

| cell | A-P3 | SimCLR |
|---|---|---|
| standard | 89.06 / 87.30 | 88.20 / 87.20 |
| crop-only strong | 88.80 / 86.42 | **89.86 / 88.68** |
| jitter-only strong | 88.78 / 87.78 | 88.24 / 88.48 |
| both strong | 88.66 / 86.20 | 89.74 / 88.86 |

| effect (linear; kNN in brackets) | A-P3 | SimCLR | A-P3 − SimCLR |
|---|---|---|---|
| crop main effect | −0.19 (−1.23) | **+1.58** (+0.93) | **−1.77** (−2.16) |
| jitter main effect | −0.21 (+0.13) | −0.04 (+0.73) | −0.17 (−0.60) |
| interaction | +0.14 (−0.70) | −0.16 (−1.10) | +0.30 (+0.40) |

## 2. Reading (frozen rules)
- **No factor is named "the factor carrying A-P3's strong-augmentation loss":** the rule needs an A-P3 effect ≤ −0.5 with SimCLR's ≥ 0; A-P3's
  seed-0 loss (−0.40 linear) is split about evenly between the two factors (single-factor cells −0.26 crop, −0.28 jitter), each below the threshold.
- **What separates the methods is the crop factor:** SimCLR gains +1.66 linear / +1.48 kNN from strong crops alone (crop-only is SimCLR's best cell),
  A-P3 gains nothing from it (−0.26 / −0.88); method contrast −1.77 linear / −2.16 kNN.  Jitter barely moves either method on linear.  So the
  "A-P3 ahead under standard, behind under strong augmentation" reversal (P107 layer 2) is, at seed 0, a **SimCLR-benefits-from-small-crops** effect
  that A-P3 does not share — not a large A-P3 loss.
- **Follow-up rule not triggered:** no A-P3 single-factor cell reaches its standard cell (88.80, 88.78 < 89.06) and no method contrast is ≥ +0.5 in
  A-P3's favour → no seeds 1–2.
- Descriptive link to the diagnostics (P115 diag interim, P123): small crops raise the low-IoU share of positive pairs and that share's gradient weight in
  all methods; the training-outcome difference sits in how the two objectives use those pairs, which the read-only diagnostics job 2 (crop-only /
  jitter-only checkpoints 20 / 100 / 400 / 800) describes next.  No reweighting / filtering is proposed.
