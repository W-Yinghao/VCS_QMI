# Pre-check A — calibration and threshold transfer on frozen CLIP features (COCO no-animal → animal) — 2026-09-27T08:57:57Z

Adapters Linear(512→512, identity init)+L2 per tower; batch 256; captions 0–3 train, caption 4 evaluation; balanced joint/product pairs (product = caption 4 of another image). ECE: 15 equal-mass bins. Thresholds set on SRC-CAL joint pairs at the stated FNR and applied unchanged. Mean over seeds (SD in JSON).

| method | selected (lr, ep) | split | R@1 | probability | ECE | max dev | Brier |
|---|---|---|---|---|---|---|---|
| vcs | 0.001, 40 | SRC-EVAL | 0.269 | native_(1+T)/2 | 0.0464 | 0.1254 | 0.0177 |
| vcs | 0.001, 40 | SRC-EVAL | 0.269 | cosine+Platt(CAL) | 0.0042 | 0.0338 | 0.0133 |
| vcs | 0.001, 40 | SRC-EVAL | 0.269 | score+Platt(CAL) | 0.0069 | 0.0506 | 0.0138 |
| vcs | 0.001, 40 | TGT-EVAL | 0.212 | native_(1+T)/2 | 0.0888 | 0.2259 | 0.0516 |
| vcs | 0.001, 40 | TGT-EVAL | 0.212 | cosine+Platt(CAL) | 0.0363 | 0.1812 | 0.0450 |
| vcs | 0.001, 40 | TGT-EVAL | 0.212 | score+Platt(CAL) | 0.0318 | 0.2033 | 0.0441 |
| infonce | 0.001, 15 | SRC-EVAL | 0.362 | cosine+Platt(CAL) | 0.0028 | 0.0240 | 0.0133 |
| infonce | 0.001, 15 | SRC-EVAL | 0.362 | score+Platt(CAL) | 0.0028 | 0.0240 | 0.0133 |
| infonce | 0.001, 15 | TGT-EVAL | 0.284 | cosine+Platt(CAL) | 0.0643 | 0.3519 | 0.0540 |
| infonce | 0.001, 15 | TGT-EVAL | 0.284 | score+Platt(CAL) | 0.0643 | 0.3518 | 0.0540 |
| logistic | 0.001, 40 | SRC-EVAL | 0.300 | native_sigmoid | 0.3370 | 0.8801 | 0.2483 |
| logistic | 0.001, 40 | SRC-EVAL | 0.300 | cosine+Platt(CAL) | 0.0035 | 0.0262 | 0.0152 |
| logistic | 0.001, 40 | SRC-EVAL | 0.300 | score+Platt(CAL) | 0.0035 | 0.0262 | 0.0152 |
| logistic | 0.001, 40 | TGT-EVAL | 0.204 | native_sigmoid | 0.3571 | 0.7596 | 0.2793 |
| logistic | 0.001, 40 | TGT-EVAL | 0.204 | cosine+Platt(CAL) | 0.0636 | 0.2729 | 0.0661 |
| logistic | 0.001, 40 | TGT-EVAL | 0.204 | score+Platt(CAL) | 0.0636 | 0.2729 | 0.0661 |

| method | score | FNR target | θ (CAL) | SRC-EVAL FNR / FPR | TGT-EVAL FNR / FPR | drift |FNR| / |FPR| |
|---|---|---|---|---|---|---|
| vcs | score_quantile@FNR0.05 | | 0.393 | 0.047 / 0.009 | 0.100 / 0.032 | 0.053 / 0.023 |
| vcs | score_quantile@FNR0.1 | | 0.607 | 0.097 / 0.007 | 0.230 / 0.011 | 0.133 / 0.005 |
| vcs | absolute_score>=0 | | 0.000 | 0.018 / 0.017 | 0.027 / 0.085 | 0.009 / 0.068 |
| vcs | cosine_matched_to_absolute | | 0.082 | 0.019 / 0.017 | 0.027 / 0.084 | 0.009 / 0.067 |
| infonce | score_quantile@FNR0.05 | | 6.870 | 0.048 / 0.008 | 0.034 / 0.061 | 0.014 / 0.054 |
| infonce | score_quantile@FNR0.1 | | 7.819 | 0.097 / 0.004 | 0.069 / 0.036 | 0.028 / 0.032 |
| logistic | score_quantile@FNR0.05 | | -3.445 | 0.048 / 0.010 | 0.065 / 0.076 | 0.018 / 0.067 |
| logistic | score_quantile@FNR0.1 | | -2.710 | 0.097 / 0.006 | 0.134 / 0.045 | 0.036 / 0.039 |
| logistic | absolute_score>=0 | | 0.000 | 0.778 / 0.000 | 0.869 / 0.000 | 0.092 / 0.000 |
| logistic | cosine_matched_to_absolute | | 0.787 | 0.778 / 0.000 | 0.869 / 0.000 | 0.092 / 0.000 |

Raw CLIP reference (no adapter): SRC-EVAL: R@1 0.368, ECE(cos+Platt) 0.0037, FNR/FPR at CAL-FNR 5 % = 0.052/0.010; TGT-EVAL: R@1 0.334, ECE(cos+Platt) 0.0341, FNR/FPR at CAL-FNR 5 % = 0.024/0.038

VCS held-out J (SRC-EVAL / TGT-EVAL, mean over seeds): 0.923 / 0.797; learned (a, b): seed 0: a 6.15 b -0.50, seed 1: a 6.16 b -0.50, seed 2: a 6.15 b -0.50
