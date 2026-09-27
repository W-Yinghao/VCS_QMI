# Pre-check A — calibration and threshold transfer on frozen CLIP features (COCO no-animal → animal; pairing = topic) — 2026-09-27T10:51:48Z

Adapters Linear(512→512, identity init)+L2 per tower; batch 256; captions 0–3 train, caption 4 evaluation; balanced joint/product pairs (product = caption 4 of another image). ECE: 15 equal-mass bins. Thresholds set on SRC-CAL joint pairs at the stated FNR and applied unchanged. Mean over seeds (SD in JSON).

| method | selected (lr, ep) | split | R@1 | probability | ECE | max dev | Brier |
|---|---|---|---|---|---|---|---|
| vcs | 0.001, 15 | SRC-EVAL | 0.080 | native_(1+T)/2 | 0.0231 | 0.0587 | 0.1038 |
| vcs | 0.001, 15 | SRC-EVAL | 0.080 | cosine+Platt(CAL) | 0.0220 | 0.0679 | 0.1039 |
| vcs | 0.001, 15 | SRC-EVAL | 0.080 | score+Platt(CAL) | 0.0293 | 0.0544 | 0.1043 |
| vcs | 0.001, 15 | TGT-EVAL | 0.069 | native_(1+T)/2 | 0.0883 | 0.1511 | 0.2083 |
| vcs | 0.001, 15 | TGT-EVAL | 0.069 | cosine+Platt(CAL) | 0.0893 | 0.1516 | 0.2087 |
| vcs | 0.001, 15 | TGT-EVAL | 0.069 | score+Platt(CAL) | 0.1184 | 0.2164 | 0.2173 |
| infonce | 0.001, 40 | SRC-EVAL | 0.059 | cosine+Platt(CAL) | 0.0127 | 0.0350 | 0.0985 |
| infonce | 0.001, 40 | SRC-EVAL | 0.059 | score+Platt(CAL) | nan | 0.0000 | nan |
| infonce | 0.001, 40 | TGT-EVAL | 0.053 | cosine+Platt(CAL) | 0.0915 | 0.1817 | 0.2084 |
| infonce | 0.001, 40 | TGT-EVAL | 0.053 | score+Platt(CAL) | nan | 0.0000 | nan |
| logistic | 0.001, 40 | SRC-EVAL | 0.059 | native_sigmoid | 0.4828 | 0.9008 | 0.4700 |
| logistic | 0.001, 40 | SRC-EVAL | 0.059 | cosine+Platt(CAL) | 0.0110 | 0.0256 | 0.0982 |
| logistic | 0.001, 40 | SRC-EVAL | 0.059 | score+Platt(CAL) | 0.0110 | 0.0256 | 0.0982 |
| logistic | 0.001, 40 | TGT-EVAL | 0.045 | native_sigmoid | 0.4880 | 0.7784 | 0.4826 |
| logistic | 0.001, 40 | TGT-EVAL | 0.045 | cosine+Platt(CAL) | 0.0787 | 0.1565 | 0.2035 |
| logistic | 0.001, 40 | TGT-EVAL | 0.045 | score+Platt(CAL) | 0.0787 | 0.1565 | 0.2035 |

| method | score | FNR target | θ (CAL) | SRC-EVAL FNR / FPR | TGT-EVAL FNR / FPR | drift |FNR| / |FPR| |
|---|---|---|---|---|---|---|
| vcs | score_quantile@FNR0.05 | | -0.426 | 0.041 / 0.261 | 0.160 / 0.474 | 0.119 / 0.213 |
| vcs | score_quantile@FNR0.1 | | -0.075 | 0.081 / 0.191 | 0.311 / 0.312 | 0.230 / 0.121 |
| vcs | absolute_score>=0 | | 0.000 | 0.091 / 0.178 | 0.350 / 0.283 | 0.259 / 0.105 |
| vcs | cosine_matched_to_absolute | | 0.025 | 0.091 / 0.178 | 0.351 / 0.282 | 0.260 / 0.104 |
| infonce | score_quantile@FNR0.05 | | 5.819 | 0.039 / 0.247 | 0.091 / 0.564 | 0.052 / 0.317 |
| infonce | score_quantile@FNR0.1 | | 6.709 | 0.085 / 0.175 | 0.198 / 0.409 | 0.113 / 0.234 |
| logistic | score_quantile@FNR0.05 | | -6.418 | 0.040 / 0.248 | 0.107 / 0.531 | 0.068 / 0.283 |
| logistic | score_quantile@FNR0.1 | | -5.548 | 0.086 / 0.175 | 0.234 / 0.365 | 0.149 / 0.190 |
| logistic | absolute_score>=0 | | 0.000 | 1.000 / 0.000 | 1.000 / 0.000 | 0.000 / 0.000 |
| logistic | cosine_matched_to_absolute | | 0.844 | 0.999 / 0.000 | 1.000 / 0.000 | 0.001 / 0.000 |

Raw CLIP reference (no adapter): SRC-EVAL: R@1 0.368, ECE(cos+Platt) 0.0120, FNR/FPR at CAL-FNR 5 % = 0.052/0.740; TGT-EVAL: R@1 0.334, ECE(cos+Platt) 0.0874, FNR/FPR at CAL-FNR 5 % = 0.065/0.826

VCS held-out J (SRC-EVAL / TGT-EVAL, mean over seeds): 0.572 / 0.157; learned (a, b): seed 0: a 5.26 b -0.13, seed 1: a 5.26 b -0.13, seed 2: a 5.25 b -0.13
