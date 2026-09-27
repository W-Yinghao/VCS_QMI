# Pre-check A — calibration and threshold transfer on frozen CLIP features (COCO no-animal → animal) — 2026-09-27T08:53:09Z

Adapters Linear(512→256)+L2 per tower; batch 256; captions 0–3 train, caption 4 evaluation; balanced joint/product pairs (product = caption 4 of another image). ECE: 15 equal-mass bins. Thresholds set on SRC-CAL joint pairs at the stated FNR and applied unchanged. Mean over seeds (SD in JSON).

| method | selected (lr, ep) | split | R@1 | probability | ECE | max dev | Brier |
|---|---|---|---|---|---|---|---|
| vcs | 0.001, 1 | SRC-EVAL | 0.022 | native_(1+T)/2 | 0.1387 | 0.2059 | 0.1401 |
| vcs | 0.001, 1 | SRC-EVAL | 0.022 | cosine+Platt(CAL) | 0.0266 | 0.1018 | 0.1195 |
| vcs | 0.001, 1 | SRC-EVAL | 0.022 | score+Platt(CAL) | 0.0314 | 0.1244 | 0.1199 |
| vcs | 0.001, 1 | TGT-EVAL | 0.011 | native_(1+T)/2 | 0.1296 | 0.1979 | 0.2251 |
| vcs | 0.001, 1 | TGT-EVAL | 0.011 | cosine+Platt(CAL) | 0.1623 | 0.3036 | 0.2387 |
| vcs | 0.001, 1 | TGT-EVAL | 0.011 | score+Platt(CAL) | 0.1708 | 0.3123 | 0.2422 |
| infonce | 0.001, 1 | SRC-EVAL | 0.036 | cosine+Platt(CAL) | 0.0288 | 0.0871 | 0.0971 |
| infonce | 0.001, 1 | SRC-EVAL | 0.036 | score+Platt(CAL) | 0.0288 | 0.0871 | 0.0971 |
| infonce | 0.001, 1 | TGT-EVAL | 0.020 | cosine+Platt(CAL) | 0.2248 | 0.3434 | 0.2399 |
| infonce | 0.001, 1 | TGT-EVAL | 0.020 | score+Platt(CAL) | 0.2248 | 0.3434 | 0.2399 |
| logistic | 0.001, 1 | SRC-EVAL | 0.018 | native_sigmoid | 0.4885 | 0.8970 | 0.4835 |
| logistic | 0.001, 1 | SRC-EVAL | 0.018 | cosine+Platt(CAL) | 0.0517 | 0.0974 | 0.1804 |
| logistic | 0.001, 1 | SRC-EVAL | 0.018 | score+Platt(CAL) | 0.0517 | 0.0974 | 0.1804 |
| logistic | 0.001, 1 | TGT-EVAL | 0.012 | native_sigmoid | 0.4956 | 0.8383 | 0.4946 |
| logistic | 0.001, 1 | TGT-EVAL | 0.012 | cosine+Platt(CAL) | 0.2135 | 0.3108 | 0.2773 |
| logistic | 0.001, 1 | TGT-EVAL | 0.012 | score+Platt(CAL) | 0.2135 | 0.3108 | 0.2773 |

| method | score | FNR target | θ (CAL) | SRC-EVAL FNR / FPR | TGT-EVAL FNR / FPR | drift |FNR| / |FPR| |
|---|---|---|---|---|---|---|
| vcs | score@FNR0.05 | | -0.305 | 0.052 / 0.325 | 0.328 / 0.329 | 0.276 / 0.004 |
| vcs | cosine@FNR0.05 | | -0.063 | 0.052 / 0.325 | 0.328 / 0.329 | 0.276 / 0.004 |
| vcs | score@FNR0.1 | | -0.175 | 0.113 / 0.227 | 0.488 / 0.188 | 0.375 / 0.039 |
| vcs | cosine@FNR0.1 | | -0.035 | 0.113 / 0.227 | 0.488 / 0.188 | 0.375 / 0.039 |
| infonce | score@FNR0.05 | | 1.095 | 0.058 / 0.228 | 0.433 / 0.145 | 0.374 / 0.083 |
| infonce | cosine@FNR0.05 | | 0.077 | 0.058 / 0.228 | 0.433 / 0.145 | 0.374 / 0.083 |
| infonce | score@FNR0.1 | | 1.555 | 0.113 / 0.146 | 0.606 / 0.063 | 0.493 / 0.083 |
| infonce | cosine@FNR0.1 | | 0.109 | 0.113 / 0.146 | 0.606 / 0.063 | 0.493 / 0.083 |
| logistic | score@FNR0.05 | | -6.016 | 0.049 / 0.788 | 0.269 / 0.555 | 0.220 / 0.233 |
| logistic | cosine@FNR0.05 | | 0.398 | 0.049 / 0.788 | 0.269 / 0.555 | 0.220 / 0.233 |
| logistic | score@FNR0.1 | | -5.593 | 0.100 / 0.615 | 0.472 / 0.317 | 0.372 / 0.298 |
| logistic | cosine@FNR0.1 | | 0.440 | 0.100 / 0.615 | 0.472 / 0.317 | 0.372 / 0.298 |

VCS held-out J (SRC-EVAL / TGT-EVAL, mean over seeds): 0.443 / 0.102; learned (a, b): seed 0: a 5.00 b -0.00
