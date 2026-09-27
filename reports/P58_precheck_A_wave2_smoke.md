# Pre-check A wave 2 — calibration vs dependence (A-S1) and mismatch detection without a target calibration set (A-T) — 2026-09-27T14:45:53Z

Frozen CLIP ViT-B/32, identity 512x512 adapters, grid lr x epochs selected on SRC-CAL by the method's own loss, 3 seeds (1 in smoke).  ECE: 15 equal-mass bins on balanced joint/product pairs (held-out caption 4).  Mean over seeds.

## A-S1 — per (shift, pairing, method)

| shift | pairing | method | selected | split | J (vcs) | native ECE | cosine+Platt(CAL) ECE | native − Platt | R@1 |
|---|---|---|---|---|---|---|---|---|---|
| animal | exact | vcs | 0.001, 1 | SRC-EVAL | 0.680 | 0.1973 | 0.0109 | +0.1864 | 0.196 |
| animal | exact | vcs | 0.001, 1 | TGT-EVAL | 0.619 | 0.1865 | 0.1152 | +0.0713 | 0.233 |
| animal | exact | infonce | 0.001, 1 | SRC-EVAL | nan | nan | 0.0038 | +nan | 0.463 |
| animal | exact | infonce | 0.001, 1 | TGT-EVAL | nan | nan | 0.0324 | +nan | 0.423 |
| animal | exact | logistic | 0.001, 1 | SRC-EVAL | nan | 0.4934 | 0.0081 | +0.4853 | 0.391 |
| animal | exact | logistic | 0.001, 1 | TGT-EVAL | nan | 0.4962 | 0.0444 | +0.4517 | 0.375 |
| animal | topic | vcs | 0.001, 1 | SRC-EVAL | 0.323 | 0.1053 | 0.0524 | +0.0529 | 0.409 |
| animal | topic | vcs | 0.001, 1 | TGT-EVAL | 0.089 | 0.0434 | 0.1049 | -0.0615 | 0.396 |
| animal | topic | infonce | 0.001, 1 | SRC-EVAL | nan | nan | 0.0606 | +nan | 0.465 |
| animal | topic | infonce | 0.001, 1 | TGT-EVAL | nan | nan | 0.0885 | +nan | 0.433 |
| animal | topic | logistic | 0.001, 1 | SRC-EVAL | nan | 0.4976 | 0.0473 | +0.4503 | 0.380 |
| animal | topic | logistic | 0.001, 1 | TGT-EVAL | nan | 0.4987 | 0.1045 | +0.3942 | 0.368 |

## A-S1 — curve: VCS native − Platt gap vs held-out J (sorted by J; competitors' Platt-on-cosine ECE alongside)

| J | shift | pairing | split | VCS native ECE | VCS cosine+Platt | gap | infonce cosine+Platt | logistic cosine+Platt |
|---|---|---|---|---|---|---|---|---|
| 0.089 | animal | topic | TGT-EVAL | 0.0434 | 0.1049 | -0.0615 | 0.0885 | 0.1045 |
| 0.323 | animal | topic | SRC-EVAL | 0.1053 | 0.0524 | +0.0529 | 0.0606 | 0.0473 |
| 0.619 | animal | exact | TGT-EVAL | 0.1865 | 0.1152 | +0.0713 | 0.0324 | 0.0444 |
| 0.680 | animal | exact | SRC-EVAL | 0.1973 | 0.0109 | +0.1864 | 0.0038 | 0.0081 |

## A-T — mismatch detection (adapters trained on the topic pairing; nominal precision 0.8; hard = coarse-topic caption, easy = random caption)

| shift | neg | split | method | rule | accept | precision | \|prec − nominal\| | mean p̂ acc. | FNR | FPR | bal.err | AUROC | ECE |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| animal | hard | SRC-EVAL | raw_clip | cosine+Platt(SRC-CAL) | 0.066 | 0.857 | 0.057 | 0.847 | 0.887 | 0.019 | 0.453 | 0.728 | 0.0316 |
| animal | hard | SRC-EVAL | raw_clip | cosine+Platt(TGT oracle, cross-fitted) | 0.087 | 0.831 | 0.031 | 0.860 | 0.856 | 0.029 | 0.442 | 0.728 | 0.0182 |
| animal | hard | TGT-EVAL | raw_clip | cosine+Platt(SRC-CAL) | 0.076 | 0.855 | 0.055 | 0.855 | 0.870 | 0.022 | 0.446 | 0.709 | 0.0224 |
| animal | hard | TGT-EVAL | raw_clip | cosine+Platt(TGT oracle, cross-fitted) | 0.076 | 0.861 | 0.061 | 0.855 | 0.869 | 0.021 | 0.445 | 0.709 | 0.0232 |
| animal | easy | SRC-EVAL | raw_clip | cosine+Platt(SRC-CAL) | 0.137 | 0.901 | 0.101 | 0.879 | 0.753 | 0.027 | 0.390 | 0.784 | 0.0199 |
| animal | easy | SRC-EVAL | raw_clip | cosine+Platt(TGT oracle, cross-fitted) | 0.152 | 0.882 | 0.082 | 0.886 | 0.731 | 0.036 | 0.384 | 0.784 | 0.0206 |
| animal | easy | TGT-EVAL | raw_clip | cosine+Platt(SRC-CAL) | 0.171 | 0.720 | 0.080 | 0.888 | 0.753 | 0.096 | 0.425 | 0.671 | 0.0841 |
| animal | easy | TGT-EVAL | raw_clip | cosine+Platt(TGT oracle, cross-fitted) | 0.025 | 0.798 | 0.002 | 0.834 | 0.959 | 0.010 | 0.485 | 0.671 | 0.0268 |
| animal | hard | SRC-EVAL | vcs | native_(1+T)/2 | 0.077 | 0.854 | 0.054 | 0.855 | 0.869 | 0.022 | 0.446 | 0.791 | 0.0542 |
| animal | hard | SRC-EVAL | vcs | cosine+Platt(SRC-CAL) | 0.122 | 0.822 | 0.022 | 0.867 | 0.800 | 0.043 | 0.422 | 0.791 | 0.0441 |
| animal | hard | SRC-EVAL | vcs | score+Platt(SRC-CAL) | 0.130 | 0.818 | 0.018 | 0.853 | 0.787 | 0.048 | 0.417 | 0.791 | 0.0397 |
| animal | hard | SRC-EVAL | vcs | cosine+Platt(TGT oracle, cross-fitted) | 0.149 | 0.813 | 0.013 | 0.876 | 0.757 | 0.056 | 0.407 | 0.791 | 0.0392 |
| animal | hard | TGT-EVAL | vcs | native_(1+T)/2 | 0.092 | 0.846 | 0.046 | 0.858 | 0.844 | 0.028 | 0.436 | 0.730 | 0.0361 |
| animal | hard | TGT-EVAL | vcs | cosine+Platt(SRC-CAL) | 0.135 | 0.833 | 0.033 | 0.874 | 0.774 | 0.045 | 0.410 | 0.730 | 0.0385 |
| animal | hard | TGT-EVAL | vcs | score+Platt(SRC-CAL) | 0.143 | 0.829 | 0.029 | 0.858 | 0.763 | 0.049 | 0.406 | 0.730 | 0.0389 |
| animal | hard | TGT-EVAL | vcs | cosine+Platt(TGT oracle, cross-fitted) | 0.087 | 0.855 | 0.055 | 0.864 | 0.851 | 0.025 | 0.438 | 0.730 | 0.0179 |
| animal | easy | SRC-EVAL | vcs | native_(1+T)/2 | 0.072 | 0.906 | 0.106 | 0.857 | 0.869 | 0.014 | 0.441 | 0.854 | 0.1055 |
| animal | easy | SRC-EVAL | vcs | cosine+Platt(SRC-CAL) | 0.211 | 0.886 | 0.086 | 0.892 | 0.625 | 0.048 | 0.337 | 0.854 | 0.0396 |
| animal | easy | SRC-EVAL | vcs | score+Platt(SRC-CAL) | 0.230 | 0.877 | 0.077 | 0.887 | 0.596 | 0.056 | 0.326 | 0.854 | 0.0393 |
| animal | easy | SRC-EVAL | vcs | cosine+Platt(TGT oracle, cross-fitted) | 0.234 | 0.876 | 0.076 | 0.900 | 0.589 | 0.058 | 0.324 | 0.854 | 0.0335 |
| animal | easy | TGT-EVAL | vcs | native_(1+T)/2 | 0.109 | 0.715 | 0.085 | 0.859 | 0.844 | 0.062 | 0.453 | 0.685 | 0.0471 |
| animal | easy | TGT-EVAL | vcs | cosine+Platt(SRC-CAL) | 0.275 | 0.691 | 0.109 | 0.899 | 0.620 | 0.170 | 0.395 | 0.685 | 0.1106 |
| animal | easy | TGT-EVAL | vcs | score+Platt(SRC-CAL) | 0.299 | 0.686 | 0.114 | 0.892 | 0.590 | 0.187 | 0.389 | 0.685 | 0.1171 |
| animal | easy | TGT-EVAL | vcs | cosine+Platt(TGT oracle, cross-fitted) | 0.035 | 0.748 | 0.052 | 0.833 | 0.948 | 0.018 | 0.483 | 0.685 | 0.0301 |
| animal | hard | SRC-EVAL | infonce | cosine+Platt(SRC-CAL) | 0.146 | 0.846 | 0.046 | 0.876 | 0.753 | 0.045 | 0.399 | 0.810 | 0.0510 |
| animal | hard | SRC-EVAL | infonce | cosine+Platt(TGT oracle, cross-fitted) | 0.174 | 0.826 | 0.026 | 0.886 | 0.713 | 0.061 | 0.387 | 0.810 | 0.0496 |
| animal | hard | TGT-EVAL | infonce | cosine+Platt(SRC-CAL) | 0.112 | 0.857 | 0.057 | 0.878 | 0.808 | 0.032 | 0.420 | 0.748 | 0.0396 |
| animal | hard | TGT-EVAL | infonce | cosine+Platt(TGT oracle, cross-fitted) | 0.115 | 0.853 | 0.053 | 0.875 | 0.803 | 0.034 | 0.419 | 0.748 | 0.0200 |
| animal | easy | SRC-EVAL | infonce | cosine+Platt(SRC-CAL) | 0.244 | 0.877 | 0.077 | 0.904 | 0.573 | 0.060 | 0.317 | 0.873 | 0.0422 |
| animal | easy | SRC-EVAL | infonce | cosine+Platt(TGT oracle, cross-fitted) | 0.260 | 0.873 | 0.073 | 0.910 | 0.545 | 0.066 | 0.306 | 0.873 | 0.0432 |
| animal | easy | TGT-EVAL | infonce | cosine+Platt(SRC-CAL) | 0.224 | 0.733 | 0.067 | 0.905 | 0.672 | 0.119 | 0.396 | 0.702 | 0.0917 |
| animal | easy | TGT-EVAL | infonce | cosine+Platt(TGT oracle, cross-fitted) | 0.061 | 0.781 | 0.019 | 0.847 | 0.905 | 0.027 | 0.466 | 0.702 | 0.0334 |
| animal | hard | SRC-EVAL | logistic | native_sigmoid | 0.000 | nan | nan | nan | 1.000 | 0.000 | 0.500 | 0.738 | 0.4974 |
| animal | hard | SRC-EVAL | logistic | cosine+Platt(SRC-CAL) | 0.057 | 0.820 | 0.020 | 0.844 | 0.907 | 0.020 | 0.464 | 0.738 | 0.0308 |
| animal | hard | SRC-EVAL | logistic | score+Platt(SRC-CAL) | 0.057 | 0.820 | 0.020 | 0.844 | 0.907 | 0.020 | 0.464 | 0.738 | 0.0308 |
| animal | hard | SRC-EVAL | logistic | cosine+Platt(TGT oracle, cross-fitted) | 0.085 | 0.810 | 0.010 | 0.853 | 0.862 | 0.032 | 0.447 | 0.738 | 0.0233 |
| animal | hard | TGT-EVAL | logistic | native_sigmoid | 0.000 | nan | nan | nan | 1.000 | 0.000 | 0.500 | 0.700 | 0.4988 |
| animal | hard | TGT-EVAL | logistic | cosine+Platt(SRC-CAL) | 0.008 | 0.969 | 0.169 | 0.843 | 0.984 | 0.001 | 0.492 | 0.700 | 0.1170 |
| animal | hard | TGT-EVAL | logistic | score+Platt(SRC-CAL) | 0.008 | 0.969 | 0.169 | 0.843 | 0.984 | 0.001 | 0.492 | 0.700 | 0.1170 |
| animal | hard | TGT-EVAL | logistic | cosine+Platt(TGT oracle, cross-fitted) | 0.058 | 0.881 | 0.081 | 0.849 | 0.898 | 0.014 | 0.456 | 0.700 | 0.0293 |
| animal | easy | SRC-EVAL | logistic | native_sigmoid | 0.000 | nan | nan | nan | 1.000 | 0.000 | 0.500 | 0.799 | 0.4976 |
| animal | easy | SRC-EVAL | logistic | cosine+Platt(SRC-CAL) | 0.153 | 0.869 | 0.069 | 0.872 | 0.733 | 0.040 | 0.387 | 0.799 | 0.0295 |
| animal | easy | SRC-EVAL | logistic | score+Platt(SRC-CAL) | 0.153 | 0.869 | 0.069 | 0.872 | 0.733 | 0.040 | 0.387 | 0.799 | 0.0295 |
| animal | easy | SRC-EVAL | logistic | cosine+Platt(TGT oracle, cross-fitted) | 0.168 | 0.868 | 0.068 | 0.879 | 0.708 | 0.044 | 0.376 | 0.799 | 0.0206 |
| animal | easy | TGT-EVAL | logistic | native_sigmoid | 0.000 | nan | nan | nan | 1.000 | 0.000 | 0.500 | 0.671 | 0.4987 |
| animal | easy | TGT-EVAL | logistic | cosine+Platt(SRC-CAL) | 0.050 | 0.825 | 0.025 | 0.854 | 0.918 | 0.018 | 0.468 | 0.671 | 0.1072 |
| animal | easy | TGT-EVAL | logistic | score+Platt(SRC-CAL) | 0.050 | 0.825 | 0.025 | 0.854 | 0.918 | 0.018 | 0.468 | 0.671 | 0.1072 |
| animal | easy | TGT-EVAL | logistic | cosine+Platt(TGT oracle, cross-fitted) | 0.026 | 0.840 | 0.040 | 0.836 | 0.957 | 0.008 | 0.482 | 0.671 | 0.0290 |
