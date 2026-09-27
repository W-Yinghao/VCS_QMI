# Pre-check A — calibration and threshold transfer on frozen CLIP features (COCO no-animal → animal; pairing = topic) — 2026-09-27T10:49:25Z

Adapters Linear(512→512, identity init)+L2 per tower; batch 256; captions 0–3 train, caption 4 evaluation; balanced joint/product pairs (product = caption 4 of another image). ECE: 15 equal-mass bins. Thresholds set on SRC-CAL joint pairs at the stated FNR and applied unchanged. Mean over seeds (SD in JSON).

| method | selected (lr, ep) | split | R@1 | probability | ECE | max dev | Brier |
|---|---|---|---|---|---|---|---|
| vcs | 0.001, 1 | SRC-EVAL | 0.409 | native_(1+T)/2 | 0.1053 | 0.1700 | 0.1696 |
| vcs | 0.001, 1 | SRC-EVAL | 0.409 | cosine+Platt(CAL) | 0.0524 | 0.0971 | 0.1597 |
| vcs | 0.001, 1 | SRC-EVAL | 0.409 | score+Platt(CAL) | 0.0516 | 0.0912 | 0.1597 |
| vcs | 0.001, 1 | TGT-EVAL | 0.396 | native_(1+T)/2 | 0.0434 | 0.1617 | 0.2275 |
| vcs | 0.001, 1 | TGT-EVAL | 0.396 | cosine+Platt(CAL) | 0.1049 | 0.2437 | 0.2405 |
| vcs | 0.001, 1 | TGT-EVAL | 0.396 | score+Platt(CAL) | 0.1106 | 0.2264 | 0.2415 |
| infonce | 0.001, 1 | SRC-EVAL | 0.465 | cosine+Platt(CAL) | 0.0606 | 0.1330 | 0.1481 |
| infonce | 0.001, 1 | SRC-EVAL | 0.465 | score+Platt(CAL) | 0.0606 | 0.1330 | 0.1481 |
| infonce | 0.001, 1 | TGT-EVAL | 0.433 | cosine+Platt(CAL) | 0.0885 | 0.2194 | 0.2293 |
| infonce | 0.001, 1 | TGT-EVAL | 0.433 | score+Platt(CAL) | 0.0885 | 0.2194 | 0.2293 |
| logistic | 0.001, 1 | SRC-EVAL | 0.380 | native_sigmoid | 0.4976 | 0.8411 | 0.4965 |
| logistic | 0.001, 1 | SRC-EVAL | 0.380 | cosine+Platt(CAL) | 0.0473 | 0.0926 | 0.1853 |
| logistic | 0.001, 1 | SRC-EVAL | 0.380 | score+Platt(CAL) | 0.0473 | 0.0926 | 0.1853 |
| logistic | 0.001, 1 | TGT-EVAL | 0.368 | native_sigmoid | 0.4987 | 0.7972 | 0.4983 |
| logistic | 0.001, 1 | TGT-EVAL | 0.368 | cosine+Platt(CAL) | 0.1045 | 0.1929 | 0.2408 |
| logistic | 0.001, 1 | TGT-EVAL | 0.368 | score+Platt(CAL) | 0.1045 | 0.1929 | 0.2408 |

| method | score | FNR target | θ (CAL) | SRC-EVAL FNR / FPR | TGT-EVAL FNR / FPR | drift |FNR| / |FPR| |
|---|---|---|---|---|---|---|
| vcs | score_quantile@FNR0.05 | | -0.402 | 0.038 / 0.538 | 0.047 / 0.840 | 0.009 / 0.301 |
| vcs | score_quantile@FNR0.1 | | -0.278 | 0.079 / 0.400 | 0.097 / 0.724 | 0.018 / 0.324 |
| vcs | absolute_score>=0 | | 0.000 | 0.280 / 0.198 | 0.303 / 0.429 | 0.023 / 0.231 |
| vcs | cosine_matched_to_absolute | | 0.001 | 0.281 / 0.197 | 0.304 / 0.428 | 0.023 / 0.232 |
| infonce | score_quantile@FNR0.05 | | 0.449 | 0.026 / 0.501 | 0.078 / 0.737 | 0.053 / 0.236 |
| infonce | score_quantile@FNR0.1 | | 0.878 | 0.081 / 0.354 | 0.168 / 0.581 | 0.087 / 0.226 |
| logistic | score_quantile@FNR0.05 | | -7.776 | 0.035 / 0.762 | 0.146 / 0.686 | 0.112 / 0.075 |
| logistic | score_quantile@FNR0.1 | | -7.383 | 0.093 / 0.577 | 0.284 / 0.480 | 0.192 / 0.097 |
| logistic | absolute_score>=0 | | 0.000 | 1.000 / 0.000 | 1.000 / 0.000 | 0.000 / 0.000 |
| logistic | cosine_matched_to_absolute | | 0.642 | 0.996 / 0.000 | 0.999 / 0.000 | 0.003 / 0.000 |

Raw CLIP reference (no adapter): SRC-EVAL: R@1 0.492, ECE(cos+Platt) 0.0279, FNR/FPR at CAL-FNR 5 % = 0.043/0.790; TGT-EVAL: R@1 0.454, ECE(cos+Platt) 0.0764, FNR/FPR at CAL-FNR 5 % = 0.053/0.864

VCS held-out J (SRC-EVAL / TGT-EVAL, mean over seeds): 0.323 / 0.089; learned (a, b): seed 0: a 5.00 b -0.00
