# Pre-check A — calibration and threshold transfer on frozen CLIP features (COCO no-animal → animal) — 2026-09-27T08:56:46Z

Adapters Linear(512→512, identity init)+L2 per tower; batch 256; captions 0–3 train, caption 4 evaluation; balanced joint/product pairs (product = caption 4 of another image). ECE: 15 equal-mass bins. Thresholds set on SRC-CAL joint pairs at the stated FNR and applied unchanged. Mean over seeds (SD in JSON).

| method | selected (lr, ep) | split | R@1 | probability | ECE | max dev | Brier |
|---|---|---|---|---|---|---|---|
| vcs | 0.001, 1 | SRC-EVAL | 0.196 | native_(1+T)/2 | 0.1973 | 0.3370 | 0.0792 |
| vcs | 0.001, 1 | SRC-EVAL | 0.196 | cosine+Platt(CAL) | 0.0109 | 0.0497 | 0.0330 |
| vcs | 0.001, 1 | SRC-EVAL | 0.196 | score+Platt(CAL) | 0.0120 | 0.0611 | 0.0330 |
| vcs | 0.001, 1 | TGT-EVAL | 0.233 | native_(1+T)/2 | 0.1865 | 0.4028 | 0.0953 |
| vcs | 0.001, 1 | TGT-EVAL | 0.233 | cosine+Platt(CAL) | 0.1152 | 0.5431 | 0.0876 |
| vcs | 0.001, 1 | TGT-EVAL | 0.233 | score+Platt(CAL) | 0.1136 | 0.5457 | 0.0870 |
| infonce | 0.001, 1 | SRC-EVAL | 0.463 | cosine+Platt(CAL) | 0.0038 | 0.0122 | 0.0205 |
| infonce | 0.001, 1 | SRC-EVAL | 0.463 | score+Platt(CAL) | 0.0038 | 0.0122 | 0.0205 |
| infonce | 0.001, 1 | TGT-EVAL | 0.423 | cosine+Platt(CAL) | 0.0324 | 0.1810 | 0.0335 |
| infonce | 0.001, 1 | TGT-EVAL | 0.423 | score+Platt(CAL) | 0.0324 | 0.1810 | 0.0335 |
| logistic | 0.001, 1 | SRC-EVAL | 0.391 | native_sigmoid | 0.4934 | 0.9740 | 0.4876 |
| logistic | 0.001, 1 | SRC-EVAL | 0.391 | cosine+Platt(CAL) | 0.0081 | 0.0212 | 0.0395 |
| logistic | 0.001, 1 | SRC-EVAL | 0.391 | score+Platt(CAL) | 0.0081 | 0.0212 | 0.0395 |
| logistic | 0.001, 1 | TGT-EVAL | 0.375 | native_sigmoid | 0.4962 | 0.9903 | 0.4930 |
| logistic | 0.001, 1 | TGT-EVAL | 0.375 | cosine+Platt(CAL) | 0.0444 | 0.1815 | 0.0401 |
| logistic | 0.001, 1 | TGT-EVAL | 0.375 | score+Platt(CAL) | 0.0444 | 0.1815 | 0.0401 |

| method | score | FNR target | θ (CAL) | SRC-EVAL FNR / FPR | TGT-EVAL FNR / FPR | drift |FNR| / |FPR| |
|---|---|---|---|---|---|---|
| vcs | score_quantile@FNR0.05 | | -0.022 | 0.055 / 0.034 | 0.006 / 0.187 | 0.049 / 0.153 |
| vcs | score_quantile@FNR0.1 | | 0.099 | 0.115 / 0.019 | 0.021 / 0.124 | 0.095 / 0.105 |
| vcs | absolute_score>=0 | | 0.000 | 0.064 / 0.030 | 0.007 / 0.175 | 0.056 / 0.145 |
| vcs | cosine_matched_to_absolute | | 0.001 | 0.064 / 0.030 | 0.007 / 0.175 | 0.057 / 0.145 |
| infonce | score_quantile@FNR0.05 | | 3.951 | 0.056 / 0.015 | 0.032 / 0.041 | 0.023 / 0.025 |
| infonce | score_quantile@FNR0.1 | | 4.382 | 0.102 / 0.009 | 0.076 / 0.026 | 0.025 / 0.017 |
| logistic | score_quantile@FNR0.05 | | -5.838 | 0.052 / 0.049 | 0.086 / 0.030 | 0.034 / 0.020 |
| logistic | score_quantile@FNR0.1 | | -5.587 | 0.091 / 0.036 | 0.184 / 0.012 | 0.093 / 0.024 |
| logistic | absolute_score>=0 | | 0.000 | 1.000 / 0.000 | 1.000 / 0.000 | 0.000 / 0.000 |
| logistic | cosine_matched_to_absolute | | 0.705 | 0.998 / 0.000 | 1.000 / 0.000 | 0.002 / 0.000 |

Raw CLIP reference (no adapter): SRC-EVAL: R@1 0.492, ECE(cos+Platt) 0.0048, FNR/FPR at CAL-FNR 5 % = 0.062/0.010; TGT-EVAL: R@1 0.454, ECE(cos+Platt) 0.0329, FNR/FPR at CAL-FNR 5 % = 0.022/0.044

VCS held-out J (SRC-EVAL / TGT-EVAL, mean over seeds): 0.680 / 0.619; learned (a, b): seed 0: a 5.00 b -0.00
