# Pre-check A wave 2 — calibration vs dependence (A-S1) and mismatch detection without a target calibration set (A-T) — 2026-09-27T14:59:32Z

Frozen CLIP ViT-B/32, identity 512x512 adapters, grid lr x epochs selected on SRC-CAL by the method's own loss, 3 seeds (1 in smoke).  ECE: 15 equal-mass bins on balanced joint/product pairs (held-out caption 4).  Mean over seeds.

## A-S1 — per (shift, pairing, method)

| shift | pairing | method | selected | split | J (vcs) | native ECE | cosine+Platt(CAL) ECE | native − Platt | R@1 |
|---|---|---|---|---|---|---|---|---|---|
| animal | exact | vcs | 0.001, 40 | SRC-EVAL | 0.923 | 0.0464 | 0.0042 | +0.0422 | 0.269 |
| animal | exact | vcs | 0.001, 40 | TGT-EVAL | 0.797 | 0.0888 | 0.0363 | +0.0524 | 0.212 |
| animal | exact | infonce | 0.001, 15 | SRC-EVAL | nan | nan | 0.0028 | +nan | 0.362 |
| animal | exact | infonce | 0.001, 15 | TGT-EVAL | nan | nan | 0.0643 | +nan | 0.284 |
| animal | exact | logistic | 0.001, 40 | SRC-EVAL | nan | 0.3370 | 0.0035 | +0.3335 | 0.300 |
| animal | exact | logistic | 0.001, 40 | TGT-EVAL | nan | 0.3571 | 0.0636 | +0.2935 | 0.204 |
| animal | topic | vcs | 0.001, 15 | SRC-EVAL | 0.572 | 0.0231 | 0.0220 | +0.0010 | 0.080 |
| animal | topic | vcs | 0.001, 15 | TGT-EVAL | 0.157 | 0.0883 | 0.0893 | -0.0011 | 0.069 |
| animal | topic | infonce | 0.001, 40 | SRC-EVAL | nan | nan | 0.0127 | +nan | 0.059 |
| animal | topic | infonce | 0.001, 40 | TGT-EVAL | nan | nan | 0.0915 | +nan | 0.053 |
| animal | topic | logistic | 0.001, 40 | SRC-EVAL | nan | 0.4828 | 0.0110 | +0.4718 | 0.059 |
| animal | topic | logistic | 0.001, 40 | TGT-EVAL | nan | 0.4880 | 0.0787 | +0.4093 | 0.045 |
| animal | coarse | vcs | 0.001, 40 | SRC-EVAL | 0.132 | 0.0232 | 0.0235 | -0.0003 | 0.011 |
| animal | coarse | vcs | 0.001, 40 | TGT-EVAL | -0.091 | 0.1350 | 0.1244 | +0.0106 | 0.009 |
| animal | coarse | infonce | 0.001, 40 | SRC-EVAL | nan | nan | 0.0283 | +nan | 0.015 |
| animal | coarse | infonce | 0.001, 40 | TGT-EVAL | nan | nan | 0.0896 | +nan | 0.019 |
| animal | coarse | logistic | 0.001, 40 | SRC-EVAL | nan | 0.4952 | 0.0268 | +0.4684 | 0.022 |
| animal | coarse | logistic | 0.001, 40 | TGT-EVAL | nan | 0.4953 | 0.1074 | +0.3879 | 0.026 |
| animal | random | vcs | 0.001, 15 | SRC-EVAL | -0.001 | 0.0197 | 0.3167 | -0.2970 | 0.077 |
| animal | random | vcs | 0.001, 15 | TGT-EVAL | -0.005 | 0.0307 | 0.3154 | -0.2847 | 0.084 |
| animal | random | infonce | 0.001, 40 | SRC-EVAL | nan | nan | 0.4667 | +nan | 0.039 |
| animal | random | infonce | 0.001, 40 | TGT-EVAL | nan | nan | 0.4667 | +nan | 0.049 |
| animal | random | logistic | 0.001, 40 | SRC-EVAL | nan | 0.4961 | 0.4667 | +0.0294 | 0.048 |
| animal | random | logistic | 0.001, 40 | TGT-EVAL | nan | 0.4959 | 0.4667 | +0.0292 | 0.062 |
| indoor_outdoor | exact | vcs | 0.001, 40 | SRC-EVAL | 0.883 | 0.0583 | 0.0055 | +0.0528 | 0.190 |
| indoor_outdoor | exact | vcs | 0.001, 40 | TGT-EVAL | 0.633 | 0.1314 | 0.0888 | +0.0426 | 0.107 |
| indoor_outdoor | exact | infonce | 0.001, 15 | SRC-EVAL | nan | nan | 0.0038 | +nan | 0.282 |
| indoor_outdoor | exact | infonce | 0.001, 15 | TGT-EVAL | nan | nan | 0.1553 | +nan | 0.167 |
| indoor_outdoor | exact | logistic | 0.001, 40 | SRC-EVAL | nan | 0.3876 | 0.0035 | +0.3841 | 0.220 |
| indoor_outdoor | exact | logistic | 0.001, 40 | TGT-EVAL | nan | 0.3900 | 0.2280 | +0.1619 | 0.094 |
| indoor_outdoor | topic | vcs | 0.001, 40 | SRC-EVAL | 0.441 | 0.0289 | 0.0323 | -0.0034 | 0.026 |
| indoor_outdoor | topic | vcs | 0.001, 40 | TGT-EVAL | -0.188 | 0.1922 | 0.1794 | +0.0128 | 0.039 |
| indoor_outdoor | topic | infonce | 0.001, 40 | SRC-EVAL | nan | nan | 0.0137 | +nan | 0.031 |
| indoor_outdoor | topic | infonce | 0.001, 40 | TGT-EVAL | nan | nan | 0.2122 | +nan | 0.034 |
| indoor_outdoor | topic | logistic | 0.001, 40 | SRC-EVAL | nan | 0.4900 | 0.0134 | +0.4765 | 0.031 |
| indoor_outdoor | topic | logistic | 0.001, 40 | TGT-EVAL | nan | 0.4895 | 0.1930 | +0.2964 | 0.030 |
| indoor_outdoor | coarse | vcs | 0.001, 40 | SRC-EVAL | 0.095 | 0.0223 | 0.0225 | -0.0002 | 0.005 |
| indoor_outdoor | coarse | vcs | 0.001, 40 | TGT-EVAL | -0.441 | 0.3080 | 0.3107 | -0.0027 | 0.009 |
| indoor_outdoor | coarse | infonce | 0.001, 40 | SRC-EVAL | nan | nan | 0.0194 | +nan | 0.012 |
| indoor_outdoor | coarse | infonce | 0.001, 40 | TGT-EVAL | nan | nan | 0.1641 | +nan | 0.040 |
| indoor_outdoor | coarse | logistic | 0.001, 40 | SRC-EVAL | nan | 0.4955 | 0.0159 | +0.4795 | 0.017 |
| indoor_outdoor | coarse | logistic | 0.001, 40 | TGT-EVAL | nan | 0.4974 | 0.1641 | +0.3333 | 0.045 |
| indoor_outdoor | random | vcs | 0.001, 40 | SRC-EVAL | -0.001 | 0.0225 | 0.3174 | -0.2949 | 0.037 |
| indoor_outdoor | random | vcs | 0.001, 40 | TGT-EVAL | -0.011 | 0.0443 | 0.3165 | -0.2722 | 0.064 |
| indoor_outdoor | random | infonce | 0.001, 40 | SRC-EVAL | nan | nan | 0.4667 | +nan | 0.049 |
| indoor_outdoor | random | infonce | 0.001, 40 | TGT-EVAL | nan | nan | 0.4667 | +nan | 0.086 |
| indoor_outdoor | random | logistic | 0.001, 40 | SRC-EVAL | nan | 0.4961 | 0.3612 | +0.1349 | 0.070 |
| indoor_outdoor | random | logistic | 0.001, 40 | TGT-EVAL | nan | 0.4960 | 0.3364 | +0.1596 | 0.123 |

## A-S1 — curve: VCS native − Platt gap vs held-out J (sorted by J; competitors' Platt-on-cosine ECE alongside)

| J | shift | pairing | split | VCS native ECE | VCS cosine+Platt | gap | infonce cosine+Platt | logistic cosine+Platt |
|---|---|---|---|---|---|---|---|---|
| -0.441 | indoor_outdoor | coarse | TGT-EVAL | 0.3080 | 0.3107 | -0.0027 | 0.1641 | 0.1641 |
| -0.188 | indoor_outdoor | topic | TGT-EVAL | 0.1922 | 0.1794 | +0.0128 | 0.2122 | 0.1930 |
| -0.091 | animal | coarse | TGT-EVAL | 0.1350 | 0.1244 | +0.0106 | 0.0896 | 0.1074 |
| -0.011 | indoor_outdoor | random | TGT-EVAL | 0.0443 | 0.3165 | -0.2722 | 0.4667 | 0.3364 |
| -0.005 | animal | random | TGT-EVAL | 0.0307 | 0.3154 | -0.2847 | 0.4667 | 0.4667 |
| -0.001 | indoor_outdoor | random | SRC-EVAL | 0.0225 | 0.3174 | -0.2949 | 0.4667 | 0.3612 |
| -0.001 | animal | random | SRC-EVAL | 0.0197 | 0.3167 | -0.2970 | 0.4667 | 0.4667 |
| 0.095 | indoor_outdoor | coarse | SRC-EVAL | 0.0223 | 0.0225 | -0.0002 | 0.0194 | 0.0159 |
| 0.132 | animal | coarse | SRC-EVAL | 0.0232 | 0.0235 | -0.0003 | 0.0283 | 0.0268 |
| 0.157 | animal | topic | TGT-EVAL | 0.0883 | 0.0893 | -0.0011 | 0.0915 | 0.0787 |
| 0.441 | indoor_outdoor | topic | SRC-EVAL | 0.0289 | 0.0323 | -0.0034 | 0.0137 | 0.0134 |
| 0.572 | animal | topic | SRC-EVAL | 0.0231 | 0.0220 | +0.0010 | 0.0127 | 0.0110 |
| 0.633 | indoor_outdoor | exact | TGT-EVAL | 0.1314 | 0.0888 | +0.0426 | 0.1553 | 0.2280 |
| 0.797 | animal | exact | TGT-EVAL | 0.0888 | 0.0363 | +0.0524 | 0.0643 | 0.0636 |
| 0.883 | indoor_outdoor | exact | SRC-EVAL | 0.0583 | 0.0055 | +0.0528 | 0.0038 | 0.0035 |
| 0.923 | animal | exact | SRC-EVAL | 0.0464 | 0.0042 | +0.0422 | 0.0028 | 0.0035 |

## A-T — mismatch detection (adapters trained on the topic pairing; nominal precision 0.8; hard = coarse-topic caption, easy = random caption)

| shift | neg | split | method | rule | accept | precision | \|prec − nominal\| | mean p̂ acc. | FNR | FPR | bal.err | AUROC | ECE |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| animal | hard | SRC-EVAL | raw_clip | cosine+Platt(SRC-CAL) | 0.076 | 0.857 | 0.057 | 0.851 | 0.869 | 0.022 | 0.445 | 0.727 | 0.0188 |
| animal | hard | SRC-EVAL | raw_clip | cosine+Platt(TGT oracle, cross-fitted) | 0.089 | 0.857 | 0.057 | 0.857 | 0.848 | 0.025 | 0.437 | 0.727 | 0.0180 |
| animal | hard | TGT-EVAL | raw_clip | cosine+Platt(SRC-CAL) | 0.082 | 0.873 | 0.073 | 0.862 | 0.857 | 0.021 | 0.439 | 0.713 | 0.0159 |
| animal | hard | TGT-EVAL | raw_clip | cosine+Platt(TGT oracle, cross-fitted) | 0.073 | 0.875 | 0.075 | 0.858 | 0.873 | 0.018 | 0.445 | 0.713 | 0.0138 |
| animal | easy | SRC-EVAL | raw_clip | cosine+Platt(SRC-CAL) | 0.151 | 0.871 | 0.071 | 0.883 | 0.737 | 0.039 | 0.388 | 0.779 | 0.0159 |
| animal | easy | SRC-EVAL | raw_clip | cosine+Platt(TGT oracle, cross-fitted) | 0.147 | 0.875 | 0.075 | 0.882 | 0.743 | 0.037 | 0.390 | 0.779 | 0.0163 |
| animal | easy | TGT-EVAL | raw_clip | cosine+Platt(SRC-CAL) | 0.182 | 0.734 | 0.066 | 0.886 | 0.733 | 0.097 | 0.415 | 0.678 | 0.0830 |
| animal | easy | TGT-EVAL | raw_clip | cosine+Platt(TGT oracle, cross-fitted) | 0.036 | 0.844 | 0.044 | 0.836 | 0.940 | 0.011 | 0.475 | 0.678 | 0.0155 |
| animal | hard | SRC-EVAL | vcs | native_(1+T)/2 | 0.411 | 0.821 | 0.021 | 0.904 | 0.325 | 0.147 | 0.236 | 0.868 | 0.0655 |
| animal | hard | SRC-EVAL | vcs | cosine+Platt(SRC-CAL) | 0.218 | 0.849 | 0.049 | 0.870 | 0.629 | 0.066 | 0.348 | 0.868 | 0.0374 |
| animal | hard | SRC-EVAL | vcs | score+Platt(SRC-CAL) | 0.249 | 0.838 | 0.038 | 0.833 | 0.582 | 0.081 | 0.331 | 0.868 | 0.0223 |
| animal | hard | SRC-EVAL | vcs | cosine+Platt(TGT oracle, cross-fitted) | 0.255 | 0.838 | 0.038 | 0.879 | 0.573 | 0.082 | 0.328 | 0.868 | 0.0322 |
| animal | hard | TGT-EVAL | vcs | native_(1+T)/2 | 0.163 | 0.871 | 0.071 | 0.894 | 0.717 | 0.042 | 0.379 | 0.816 | 0.0708 |
| animal | hard | TGT-EVAL | vcs | cosine+Platt(SRC-CAL) | 0.074 | 0.911 | 0.111 | 0.870 | 0.865 | 0.013 | 0.439 | 0.816 | 0.1193 |
| animal | hard | TGT-EVAL | vcs | score+Platt(SRC-CAL) | 0.085 | 0.908 | 0.108 | 0.833 | 0.846 | 0.016 | 0.431 | 0.816 | 0.1314 |
| animal | hard | TGT-EVAL | vcs | cosine+Platt(TGT oracle, cross-fitted) | 0.179 | 0.867 | 0.067 | 0.885 | 0.690 | 0.047 | 0.369 | 0.816 | 0.0178 |
| animal | easy | SRC-EVAL | vcs | native_(1+T)/2 | 0.384 | 0.879 | 0.079 | 0.904 | 0.325 | 0.093 | 0.209 | 0.917 | 0.0243 |
| animal | easy | SRC-EVAL | vcs | cosine+Platt(SRC-CAL) | 0.370 | 0.881 | 0.081 | 0.900 | 0.348 | 0.088 | 0.218 | 0.917 | 0.0240 |
| animal | easy | SRC-EVAL | vcs | score+Platt(SRC-CAL) | 0.419 | 0.871 | 0.071 | 0.888 | 0.270 | 0.108 | 0.189 | 0.917 | 0.0306 |
| animal | easy | SRC-EVAL | vcs | cosine+Platt(TGT oracle, cross-fitted) | 0.360 | 0.883 | 0.083 | 0.901 | 0.363 | 0.084 | 0.224 | 0.917 | 0.0233 |
| animal | easy | TGT-EVAL | vcs | native_(1+T)/2 | 0.180 | 0.785 | 0.015 | 0.892 | 0.717 | 0.078 | 0.397 | 0.755 | 0.0856 |
| animal | easy | TGT-EVAL | vcs | cosine+Platt(SRC-CAL) | 0.170 | 0.788 | 0.012 | 0.890 | 0.733 | 0.072 | 0.402 | 0.755 | 0.0850 |
| animal | easy | TGT-EVAL | vcs | score+Platt(SRC-CAL) | 0.215 | 0.771 | 0.029 | 0.877 | 0.669 | 0.099 | 0.384 | 0.755 | 0.1152 |
| animal | easy | TGT-EVAL | vcs | cosine+Platt(TGT oracle, cross-fitted) | 0.100 | 0.836 | 0.036 | 0.861 | 0.832 | 0.033 | 0.433 | 0.755 | 0.0256 |
| animal | hard | SRC-EVAL | infonce | cosine+Platt(SRC-CAL) | 0.207 | 0.865 | 0.065 | 0.869 | 0.642 | 0.056 | 0.349 | 0.880 | 0.0275 |
| animal | hard | SRC-EVAL | infonce | cosine+Platt(TGT oracle, cross-fitted) | 0.252 | 0.857 | 0.057 | 0.879 | 0.569 | 0.072 | 0.320 | 0.880 | 0.0183 |
| animal | hard | TGT-EVAL | infonce | cosine+Platt(SRC-CAL) | 0.139 | 0.898 | 0.098 | 0.865 | 0.751 | 0.028 | 0.389 | 0.837 | 0.0409 |
| animal | hard | TGT-EVAL | infonce | cosine+Platt(TGT oracle, cross-fitted) | 0.201 | 0.879 | 0.079 | 0.874 | 0.647 | 0.049 | 0.348 | 0.837 | 0.0120 |
| animal | easy | SRC-EVAL | infonce | cosine+Platt(SRC-CAL) | 0.381 | 0.883 | 0.083 | 0.890 | 0.328 | 0.089 | 0.209 | 0.924 | 0.0119 |
| animal | easy | SRC-EVAL | infonce | cosine+Platt(TGT oracle, cross-fitted) | 0.375 | 0.883 | 0.083 | 0.892 | 0.337 | 0.087 | 0.212 | 0.924 | 0.0103 |
| animal | easy | TGT-EVAL | infonce | cosine+Platt(SRC-CAL) | 0.318 | 0.730 | 0.070 | 0.888 | 0.536 | 0.172 | 0.354 | 0.757 | 0.0919 |
| animal | easy | TGT-EVAL | infonce | cosine+Platt(TGT oracle, cross-fitted) | 0.072 | 0.824 | 0.024 | 0.843 | 0.881 | 0.026 | 0.453 | 0.757 | 0.0226 |
| animal | hard | SRC-EVAL | logistic | native_sigmoid | 0.000 | nan | nan | nan | 1.000 | 0.000 | 0.500 | 0.880 | 0.4815 |
| animal | hard | SRC-EVAL | logistic | cosine+Platt(SRC-CAL) | 0.211 | 0.873 | 0.073 | 0.869 | 0.631 | 0.054 | 0.343 | 0.880 | 0.0274 |
| animal | hard | SRC-EVAL | logistic | score+Platt(SRC-CAL) | 0.211 | 0.873 | 0.073 | 0.869 | 0.632 | 0.054 | 0.343 | 0.880 | 0.0274 |
| animal | hard | SRC-EVAL | logistic | cosine+Platt(TGT oracle, cross-fitted) | 0.252 | 0.860 | 0.060 | 0.880 | 0.566 | 0.070 | 0.318 | 0.880 | 0.0181 |
| animal | hard | TGT-EVAL | logistic | native_sigmoid | 0.000 | nan | nan | nan | 1.000 | 0.000 | 0.500 | 0.837 | 0.4892 |
| animal | hard | TGT-EVAL | logistic | cosine+Platt(SRC-CAL) | 0.103 | 0.901 | 0.101 | 0.860 | 0.815 | 0.020 | 0.417 | 0.837 | 0.0706 |
| animal | hard | TGT-EVAL | logistic | score+Platt(SRC-CAL) | 0.103 | 0.901 | 0.101 | 0.860 | 0.815 | 0.020 | 0.417 | 0.837 | 0.0706 |
| animal | hard | TGT-EVAL | logistic | cosine+Platt(TGT oracle, cross-fitted) | 0.202 | 0.876 | 0.076 | 0.872 | 0.646 | 0.050 | 0.348 | 0.837 | 0.0122 |
| animal | easy | SRC-EVAL | logistic | native_sigmoid | 0.000 | nan | nan | nan | 1.000 | 0.000 | 0.500 | 0.925 | 0.4827 |
| animal | easy | SRC-EVAL | logistic | cosine+Platt(SRC-CAL) | 0.382 | 0.883 | 0.083 | 0.892 | 0.325 | 0.090 | 0.207 | 0.925 | 0.0129 |
| animal | easy | SRC-EVAL | logistic | score+Platt(SRC-CAL) | 0.382 | 0.883 | 0.083 | 0.892 | 0.325 | 0.090 | 0.207 | 0.925 | 0.0129 |
| animal | easy | SRC-EVAL | logistic | cosine+Platt(TGT oracle, cross-fitted) | 0.375 | 0.885 | 0.085 | 0.894 | 0.335 | 0.086 | 0.211 | 0.925 | 0.0113 |
| animal | easy | TGT-EVAL | logistic | native_sigmoid | 0.000 | nan | nan | nan | 1.000 | 0.000 | 0.500 | 0.761 | 0.4880 |
| animal | easy | TGT-EVAL | logistic | cosine+Platt(SRC-CAL) | 0.267 | 0.749 | 0.051 | 0.881 | 0.600 | 0.134 | 0.367 | 0.761 | 0.0800 |
| animal | easy | TGT-EVAL | logistic | score+Platt(SRC-CAL) | 0.267 | 0.749 | 0.051 | 0.881 | 0.600 | 0.134 | 0.367 | 0.761 | 0.0800 |
| animal | easy | TGT-EVAL | logistic | cosine+Platt(TGT oracle, cross-fitted) | 0.073 | 0.845 | 0.045 | 0.845 | 0.876 | 0.023 | 0.449 | 0.761 | 0.0197 |
| indoor_outdoor | hard | SRC-EVAL | raw_clip | cosine+Platt(SRC-CAL) | 0.026 | 0.791 | 0.009 | 0.825 | 0.959 | 0.011 | 0.485 | 0.680 | 0.0181 |
| indoor_outdoor | hard | SRC-EVAL | raw_clip | cosine+Platt(TGT oracle, cross-fitted) | 0.024 | 0.794 | 0.006 | 0.823 | 0.962 | 0.010 | 0.486 | 0.680 | 0.0193 |
| indoor_outdoor | hard | TGT-EVAL | raw_clip | cosine+Platt(SRC-CAL) | 0.022 | 0.670 | 0.130 | 0.830 | 0.970 | 0.015 | 0.492 | 0.574 | 0.0720 |
| indoor_outdoor | hard | TGT-EVAL | raw_clip | cosine+Platt(TGT oracle, cross-fitted) | 0.000 | nan | nan | nan | 1.000 | 0.000 | 0.500 | 0.574 | 0.0229 |
| indoor_outdoor | easy | SRC-EVAL | raw_clip | cosine+Platt(SRC-CAL) | 0.099 | 0.831 | 0.031 | 0.851 | 0.836 | 0.033 | 0.435 | 0.737 | 0.0232 |
| indoor_outdoor | easy | SRC-EVAL | raw_clip | cosine+Platt(TGT oracle, cross-fitted) | 0.105 | 0.830 | 0.030 | 0.853 | 0.826 | 0.036 | 0.431 | 0.737 | 0.0223 |
| indoor_outdoor | easy | TGT-EVAL | raw_clip | cosine+Platt(SRC-CAL) | 0.076 | 0.653 | 0.147 | 0.852 | 0.900 | 0.053 | 0.477 | 0.590 | 0.0982 |
| indoor_outdoor | easy | TGT-EVAL | raw_clip | cosine+Platt(TGT oracle, cross-fitted) | 0.000 | nan | nan | nan | 1.000 | 0.000 | 0.500 | 0.590 | 0.0220 |
| indoor_outdoor | hard | SRC-EVAL | vcs | native_(1+T)/2 | 0.312 | 0.769 | 0.031 | 0.888 | 0.520 | 0.144 | 0.332 | 0.800 | 0.0703 |
| indoor_outdoor | hard | SRC-EVAL | vcs | cosine+Platt(SRC-CAL) | 0.133 | 0.802 | 0.002 | 0.862 | 0.787 | 0.053 | 0.420 | 0.800 | 0.0231 |
| indoor_outdoor | hard | SRC-EVAL | vcs | score+Platt(SRC-CAL) | 0.115 | 0.800 | 0.001 | 0.822 | 0.816 | 0.046 | 0.431 | 0.800 | 0.0234 |
| indoor_outdoor | hard | SRC-EVAL | vcs | cosine+Platt(TGT oracle, cross-fitted) | 0.125 | 0.802 | 0.003 | 0.860 | 0.799 | 0.050 | 0.424 | 0.800 | 0.0243 |
| indoor_outdoor | hard | TGT-EVAL | vcs | native_(1+T)/2 | 0.241 | 0.534 | 0.266 | 0.880 | 0.743 | 0.225 | 0.484 | 0.531 | 0.1952 |
| indoor_outdoor | hard | TGT-EVAL | vcs | cosine+Platt(SRC-CAL) | 0.087 | 0.543 | 0.257 | 0.853 | 0.905 | 0.080 | 0.493 | 0.531 | 0.1459 |
| indoor_outdoor | hard | TGT-EVAL | vcs | score+Platt(SRC-CAL) | 0.074 | 0.543 | 0.257 | 0.819 | 0.920 | 0.067 | 0.494 | 0.531 | 0.1674 |
| indoor_outdoor | hard | TGT-EVAL | vcs | cosine+Platt(TGT oracle, cross-fitted) | 0.000 | nan | nan | nan | 1.000 | 0.000 | 0.500 | 0.531 | 0.0130 |
| indoor_outdoor | easy | SRC-EVAL | vcs | native_(1+T)/2 | 0.280 | 0.857 | 0.057 | 0.888 | 0.520 | 0.080 | 0.300 | 0.881 | 0.0248 |
| indoor_outdoor | easy | SRC-EVAL | vcs | cosine+Platt(SRC-CAL) | 0.238 | 0.866 | 0.066 | 0.882 | 0.587 | 0.064 | 0.325 | 0.881 | 0.0317 |
| indoor_outdoor | easy | SRC-EVAL | vcs | score+Platt(SRC-CAL) | 0.282 | 0.856 | 0.056 | 0.861 | 0.517 | 0.081 | 0.299 | 0.881 | 0.0344 |
| indoor_outdoor | easy | SRC-EVAL | vcs | cosine+Platt(TGT oracle, cross-fitted) | 0.268 | 0.860 | 0.060 | 0.886 | 0.540 | 0.075 | 0.307 | 0.881 | 0.0265 |
| indoor_outdoor | easy | TGT-EVAL | vcs | native_(1+T)/2 | 0.234 | 0.549 | 0.251 | 0.878 | 0.743 | 0.211 | 0.477 | 0.536 | 0.1935 |
| indoor_outdoor | easy | TGT-EVAL | vcs | cosine+Platt(SRC-CAL) | 0.192 | 0.554 | 0.246 | 0.872 | 0.788 | 0.171 | 0.479 | 0.536 | 0.1800 |
| indoor_outdoor | easy | TGT-EVAL | vcs | score+Platt(SRC-CAL) | 0.237 | 0.549 | 0.251 | 0.855 | 0.740 | 0.214 | 0.477 | 0.536 | 0.2042 |
| indoor_outdoor | easy | TGT-EVAL | vcs | cosine+Platt(TGT oracle, cross-fitted) | 0.000 | nan | nan | nan | 1.000 | 0.000 | 0.500 | 0.536 | 0.0158 |
| indoor_outdoor | hard | SRC-EVAL | infonce | cosine+Platt(SRC-CAL) | 0.111 | 0.823 | 0.023 | 0.843 | 0.817 | 0.039 | 0.428 | 0.807 | 0.0128 |
| indoor_outdoor | hard | SRC-EVAL | infonce | cosine+Platt(TGT oracle, cross-fitted) | 0.100 | 0.824 | 0.024 | 0.842 | 0.834 | 0.035 | 0.435 | 0.807 | 0.0132 |
| indoor_outdoor | hard | TGT-EVAL | infonce | cosine+Platt(SRC-CAL) | 0.135 | 0.557 | 0.243 | 0.846 | 0.850 | 0.120 | 0.485 | 0.527 | 0.1665 |
| indoor_outdoor | hard | TGT-EVAL | infonce | cosine+Platt(TGT oracle, cross-fitted) | 0.000 | nan | nan | nan | 1.000 | 0.000 | 0.500 | 0.527 | 0.0169 |
| indoor_outdoor | easy | SRC-EVAL | infonce | cosine+Platt(SRC-CAL) | 0.250 | 0.867 | 0.067 | 0.865 | 0.567 | 0.066 | 0.317 | 0.884 | 0.0168 |
| indoor_outdoor | easy | SRC-EVAL | infonce | cosine+Platt(TGT oracle, cross-fitted) | 0.281 | 0.861 | 0.061 | 0.869 | 0.515 | 0.078 | 0.297 | 0.884 | 0.0128 |
| indoor_outdoor | easy | TGT-EVAL | infonce | cosine+Platt(SRC-CAL) | 0.309 | 0.544 | 0.256 | 0.865 | 0.664 | 0.283 | 0.473 | 0.535 | 0.2132 |
| indoor_outdoor | easy | TGT-EVAL | infonce | cosine+Platt(TGT oracle, cross-fitted) | 0.000 | nan | nan | nan | 1.000 | 0.000 | 0.500 | 0.535 | 0.0193 |
| indoor_outdoor | hard | SRC-EVAL | logistic | native_sigmoid | 0.000 | nan | nan | nan | 1.000 | 0.000 | 0.500 | 0.808 | 0.4889 |
| indoor_outdoor | hard | SRC-EVAL | logistic | cosine+Platt(SRC-CAL) | 0.114 | 0.825 | 0.025 | 0.842 | 0.812 | 0.040 | 0.426 | 0.808 | 0.0140 |
| indoor_outdoor | hard | SRC-EVAL | logistic | score+Platt(SRC-CAL) | 0.114 | 0.825 | 0.025 | 0.842 | 0.812 | 0.040 | 0.426 | 0.808 | 0.0140 |
| indoor_outdoor | hard | SRC-EVAL | logistic | cosine+Platt(TGT oracle, cross-fitted) | 0.102 | 0.826 | 0.026 | 0.840 | 0.832 | 0.035 | 0.433 | 0.808 | 0.0138 |
| indoor_outdoor | hard | TGT-EVAL | logistic | native_sigmoid | 0.000 | nan | nan | nan | 1.000 | 0.000 | 0.500 | 0.529 | 0.4893 |
| indoor_outdoor | hard | TGT-EVAL | logistic | cosine+Platt(SRC-CAL) | 0.086 | 0.554 | 0.246 | 0.841 | 0.905 | 0.077 | 0.491 | 0.529 | 0.1498 |
| indoor_outdoor | hard | TGT-EVAL | logistic | score+Platt(SRC-CAL) | 0.086 | 0.554 | 0.246 | 0.841 | 0.905 | 0.077 | 0.491 | 0.529 | 0.1498 |
| indoor_outdoor | hard | TGT-EVAL | logistic | cosine+Platt(TGT oracle, cross-fitted) | 0.000 | nan | nan | nan | 1.000 | 0.000 | 0.500 | 0.529 | 0.0166 |
| indoor_outdoor | easy | SRC-EVAL | logistic | native_sigmoid | 0.000 | nan | nan | nan | 1.000 | 0.000 | 0.500 | 0.885 | 0.4901 |
| indoor_outdoor | easy | SRC-EVAL | logistic | cosine+Platt(SRC-CAL) | 0.255 | 0.867 | 0.067 | 0.864 | 0.559 | 0.068 | 0.313 | 0.885 | 0.0173 |
| indoor_outdoor | easy | SRC-EVAL | logistic | score+Platt(SRC-CAL) | 0.255 | 0.867 | 0.067 | 0.864 | 0.559 | 0.068 | 0.313 | 0.885 | 0.0173 |
| indoor_outdoor | easy | SRC-EVAL | logistic | cosine+Platt(TGT oracle, cross-fitted) | 0.283 | 0.863 | 0.063 | 0.869 | 0.511 | 0.078 | 0.295 | 0.885 | 0.0134 |
| indoor_outdoor | easy | TGT-EVAL | logistic | native_sigmoid | 0.000 | nan | nan | nan | 1.000 | 0.000 | 0.500 | 0.532 | 0.4893 |
| indoor_outdoor | easy | TGT-EVAL | logistic | cosine+Platt(SRC-CAL) | 0.234 | 0.550 | 0.250 | 0.859 | 0.743 | 0.211 | 0.477 | 0.532 | 0.1923 |
| indoor_outdoor | easy | TGT-EVAL | logistic | score+Platt(SRC-CAL) | 0.234 | 0.550 | 0.250 | 0.859 | 0.743 | 0.211 | 0.477 | 0.532 | 0.1923 |
| indoor_outdoor | easy | TGT-EVAL | logistic | cosine+Platt(TGT oracle, cross-fitted) | 0.000 | nan | nan | nan | 1.000 | 0.000 | 0.500 | 0.532 | 0.0189 |
