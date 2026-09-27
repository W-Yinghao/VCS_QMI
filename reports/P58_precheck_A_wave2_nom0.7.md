# Pre-check A wave 2 — calibration vs dependence (A-S1) and mismatch detection without a target calibration set (A-T) — 2026-09-27T14:59:54Z

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
| indoor_outdoor | random | logistic | 0.001, 40 | SRC-EVAL | nan | 0.4961 | 0.3599 | +0.1362 | 0.070 |
| indoor_outdoor | random | logistic | 0.001, 40 | TGT-EVAL | nan | 0.4960 | 0.3366 | +0.1594 | 0.123 |

## A-S1 — curve: VCS native − Platt gap vs held-out J (sorted by J; competitors' Platt-on-cosine ECE alongside)

| J | shift | pairing | split | VCS native ECE | VCS cosine+Platt | gap | infonce cosine+Platt | logistic cosine+Platt |
|---|---|---|---|---|---|---|---|---|
| -0.441 | indoor_outdoor | coarse | TGT-EVAL | 0.3080 | 0.3107 | -0.0027 | 0.1641 | 0.1641 |
| -0.188 | indoor_outdoor | topic | TGT-EVAL | 0.1922 | 0.1794 | +0.0128 | 0.2122 | 0.1930 |
| -0.091 | animal | coarse | TGT-EVAL | 0.1350 | 0.1244 | +0.0106 | 0.0896 | 0.1074 |
| -0.011 | indoor_outdoor | random | TGT-EVAL | 0.0443 | 0.3165 | -0.2722 | 0.4667 | 0.3366 |
| -0.005 | animal | random | TGT-EVAL | 0.0307 | 0.3154 | -0.2847 | 0.4667 | 0.4667 |
| -0.001 | indoor_outdoor | random | SRC-EVAL | 0.0225 | 0.3174 | -0.2949 | 0.4667 | 0.3599 |
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

## A-T — mismatch detection (adapters trained on the topic pairing; nominal precision 0.7; hard = coarse-topic caption, easy = random caption)

| shift | neg | split | method | rule | accept | precision | \|prec − nominal\| | mean p̂ acc. | FNR | FPR | bal.err | AUROC | ECE |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| animal | hard | SRC-EVAL | raw_clip | cosine+Platt(SRC-CAL) | 0.171 | 0.816 | 0.116 | 0.794 | 0.720 | 0.063 | 0.392 | 0.727 | 0.0188 |
| animal | hard | SRC-EVAL | raw_clip | cosine+Platt(TGT oracle, cross-fitted) | 0.187 | 0.811 | 0.111 | 0.799 | 0.697 | 0.071 | 0.384 | 0.727 | 0.0180 |
| animal | hard | TGT-EVAL | raw_clip | cosine+Platt(SRC-CAL) | 0.177 | 0.797 | 0.097 | 0.800 | 0.718 | 0.072 | 0.395 | 0.713 | 0.0159 |
| animal | hard | TGT-EVAL | raw_clip | cosine+Platt(TGT oracle, cross-fitted) | 0.169 | 0.802 | 0.102 | 0.795 | 0.728 | 0.067 | 0.398 | 0.713 | 0.0138 |
| animal | easy | SRC-EVAL | raw_clip | cosine+Platt(SRC-CAL) | 0.248 | 0.826 | 0.126 | 0.831 | 0.590 | 0.087 | 0.338 | 0.779 | 0.0159 |
| animal | easy | SRC-EVAL | raw_clip | cosine+Platt(TGT oracle, cross-fitted) | 0.245 | 0.830 | 0.130 | 0.829 | 0.594 | 0.083 | 0.339 | 0.779 | 0.0163 |
| animal | easy | TGT-EVAL | raw_clip | cosine+Platt(SRC-CAL) | 0.292 | 0.697 | 0.003 | 0.835 | 0.593 | 0.177 | 0.385 | 0.678 | 0.0830 |
| animal | easy | TGT-EVAL | raw_clip | cosine+Platt(TGT oracle, cross-fitted) | 0.122 | 0.763 | 0.063 | 0.771 | 0.813 | 0.058 | 0.436 | 0.678 | 0.0155 |
| animal | hard | SRC-EVAL | vcs | native_(1+T)/2 | 0.503 | 0.795 | 0.095 | 0.877 | 0.199 | 0.206 | 0.203 | 0.868 | 0.0655 |
| animal | hard | SRC-EVAL | vcs | cosine+Platt(SRC-CAL) | 0.377 | 0.826 | 0.126 | 0.820 | 0.376 | 0.131 | 0.254 | 0.868 | 0.0374 |
| animal | hard | SRC-EVAL | vcs | score+Platt(SRC-CAL) | 0.440 | 0.814 | 0.114 | 0.801 | 0.284 | 0.164 | 0.224 | 0.868 | 0.0223 |
| animal | hard | SRC-EVAL | vcs | cosine+Platt(TGT oracle, cross-fitted) | 0.395 | 0.824 | 0.124 | 0.834 | 0.349 | 0.139 | 0.244 | 0.868 | 0.0322 |
| animal | hard | TGT-EVAL | vcs | native_(1+T)/2 | 0.250 | 0.838 | 0.138 | 0.844 | 0.582 | 0.081 | 0.331 | 0.816 | 0.0708 |
| animal | hard | TGT-EVAL | vcs | cosine+Platt(SRC-CAL) | 0.142 | 0.880 | 0.180 | 0.812 | 0.751 | 0.034 | 0.393 | 0.816 | 0.1193 |
| animal | hard | TGT-EVAL | vcs | score+Platt(SRC-CAL) | 0.185 | 0.864 | 0.164 | 0.789 | 0.680 | 0.050 | 0.365 | 0.816 | 0.1314 |
| animal | hard | TGT-EVAL | vcs | cosine+Platt(TGT oracle, cross-fitted) | 0.293 | 0.823 | 0.123 | 0.832 | 0.517 | 0.104 | 0.310 | 0.816 | 0.0178 |
| animal | easy | SRC-EVAL | vcs | native_(1+T)/2 | 0.466 | 0.858 | 0.158 | 0.878 | 0.199 | 0.132 | 0.166 | 0.917 | 0.0243 |
| animal | easy | SRC-EVAL | vcs | cosine+Platt(SRC-CAL) | 0.458 | 0.861 | 0.161 | 0.873 | 0.211 | 0.128 | 0.169 | 0.917 | 0.0240 |
| animal | easy | SRC-EVAL | vcs | score+Platt(SRC-CAL) | 0.482 | 0.854 | 0.154 | 0.871 | 0.176 | 0.141 | 0.159 | 0.917 | 0.0306 |
| animal | easy | SRC-EVAL | vcs | cosine+Platt(TGT oracle, cross-fitted) | 0.450 | 0.863 | 0.163 | 0.872 | 0.223 | 0.123 | 0.173 | 0.917 | 0.0233 |
| animal | easy | TGT-EVAL | vcs | native_(1+T)/2 | 0.279 | 0.749 | 0.049 | 0.842 | 0.582 | 0.140 | 0.361 | 0.755 | 0.0856 |
| animal | easy | TGT-EVAL | vcs | cosine+Platt(SRC-CAL) | 0.268 | 0.756 | 0.056 | 0.839 | 0.596 | 0.131 | 0.363 | 0.755 | 0.0850 |
| animal | easy | TGT-EVAL | vcs | score+Platt(SRC-CAL) | 0.306 | 0.744 | 0.044 | 0.840 | 0.545 | 0.157 | 0.351 | 0.755 | 0.1152 |
| animal | easy | TGT-EVAL | vcs | cosine+Platt(TGT oracle, cross-fitted) | 0.220 | 0.772 | 0.072 | 0.799 | 0.660 | 0.101 | 0.380 | 0.755 | 0.0256 |
| animal | hard | SRC-EVAL | infonce | cosine+Platt(SRC-CAL) | 0.392 | 0.836 | 0.136 | 0.813 | 0.345 | 0.129 | 0.237 | 0.880 | 0.0275 |
| animal | hard | SRC-EVAL | infonce | cosine+Platt(TGT oracle, cross-fitted) | 0.418 | 0.832 | 0.132 | 0.829 | 0.305 | 0.140 | 0.223 | 0.880 | 0.0183 |
| animal | hard | TGT-EVAL | infonce | cosine+Platt(SRC-CAL) | 0.262 | 0.860 | 0.160 | 0.811 | 0.550 | 0.073 | 0.312 | 0.837 | 0.0409 |
| animal | hard | TGT-EVAL | infonce | cosine+Platt(TGT oracle, cross-fitted) | 0.332 | 0.831 | 0.131 | 0.825 | 0.448 | 0.112 | 0.280 | 0.837 | 0.0120 |
| animal | easy | SRC-EVAL | infonce | cosine+Platt(SRC-CAL) | 0.479 | 0.861 | 0.161 | 0.863 | 0.175 | 0.133 | 0.154 | 0.924 | 0.0119 |
| animal | easy | SRC-EVAL | infonce | cosine+Platt(TGT oracle, cross-fitted) | 0.473 | 0.863 | 0.163 | 0.865 | 0.184 | 0.130 | 0.157 | 0.924 | 0.0103 |
| animal | easy | TGT-EVAL | infonce | cosine+Platt(SRC-CAL) | 0.445 | 0.701 | 0.004 | 0.850 | 0.376 | 0.266 | 0.321 | 0.757 | 0.0919 |
| animal | easy | TGT-EVAL | infonce | cosine+Platt(TGT oracle, cross-fitted) | 0.230 | 0.754 | 0.054 | 0.777 | 0.654 | 0.113 | 0.383 | 0.757 | 0.0226 |
| animal | hard | SRC-EVAL | logistic | native_sigmoid | 0.000 | nan | nan | nan | 1.000 | 0.000 | 0.500 | 0.880 | 0.4815 |
| animal | hard | SRC-EVAL | logistic | cosine+Platt(SRC-CAL) | 0.392 | 0.835 | 0.135 | 0.814 | 0.345 | 0.129 | 0.237 | 0.880 | 0.0274 |
| animal | hard | SRC-EVAL | logistic | score+Platt(SRC-CAL) | 0.392 | 0.835 | 0.135 | 0.814 | 0.345 | 0.129 | 0.237 | 0.880 | 0.0274 |
| animal | hard | SRC-EVAL | logistic | cosine+Platt(TGT oracle, cross-fitted) | 0.417 | 0.831 | 0.131 | 0.830 | 0.306 | 0.141 | 0.223 | 0.880 | 0.0181 |
| animal | hard | TGT-EVAL | logistic | native_sigmoid | 0.000 | nan | nan | nan | 1.000 | 0.000 | 0.500 | 0.837 | 0.4892 |
| animal | hard | TGT-EVAL | logistic | cosine+Platt(SRC-CAL) | 0.219 | 0.872 | 0.172 | 0.802 | 0.618 | 0.056 | 0.337 | 0.837 | 0.0706 |
| animal | hard | TGT-EVAL | logistic | score+Platt(SRC-CAL) | 0.219 | 0.872 | 0.172 | 0.802 | 0.618 | 0.056 | 0.337 | 0.837 | 0.0706 |
| animal | hard | TGT-EVAL | logistic | cosine+Platt(TGT oracle, cross-fitted) | 0.335 | 0.829 | 0.129 | 0.825 | 0.445 | 0.115 | 0.280 | 0.837 | 0.0122 |
| animal | easy | SRC-EVAL | logistic | native_sigmoid | 0.000 | nan | nan | nan | 1.000 | 0.000 | 0.500 | 0.925 | 0.4827 |
| animal | easy | SRC-EVAL | logistic | cosine+Platt(SRC-CAL) | 0.480 | 0.862 | 0.162 | 0.865 | 0.173 | 0.132 | 0.152 | 0.925 | 0.0129 |
| animal | easy | SRC-EVAL | logistic | score+Platt(SRC-CAL) | 0.480 | 0.862 | 0.162 | 0.865 | 0.173 | 0.132 | 0.152 | 0.925 | 0.0129 |
| animal | easy | SRC-EVAL | logistic | cosine+Platt(TGT oracle, cross-fitted) | 0.474 | 0.865 | 0.165 | 0.866 | 0.181 | 0.128 | 0.155 | 0.925 | 0.0113 |
| animal | easy | TGT-EVAL | logistic | native_sigmoid | 0.000 | nan | nan | nan | 1.000 | 0.000 | 0.500 | 0.761 | 0.4880 |
| animal | easy | TGT-EVAL | logistic | cosine+Platt(SRC-CAL) | 0.398 | 0.717 | 0.017 | 0.839 | 0.429 | 0.225 | 0.327 | 0.761 | 0.0800 |
| animal | easy | TGT-EVAL | logistic | score+Platt(SRC-CAL) | 0.398 | 0.717 | 0.017 | 0.839 | 0.429 | 0.225 | 0.327 | 0.761 | 0.0800 |
| animal | easy | TGT-EVAL | logistic | cosine+Platt(TGT oracle, cross-fitted) | 0.240 | 0.758 | 0.058 | 0.777 | 0.636 | 0.116 | 0.376 | 0.761 | 0.0197 |
| indoor_outdoor | hard | SRC-EVAL | raw_clip | cosine+Platt(SRC-CAL) | 0.140 | 0.758 | 0.058 | 0.762 | 0.787 | 0.068 | 0.427 | 0.680 | 0.0181 |
| indoor_outdoor | hard | SRC-EVAL | raw_clip | cosine+Platt(TGT oracle, cross-fitted) | 0.136 | 0.759 | 0.059 | 0.759 | 0.793 | 0.066 | 0.430 | 0.680 | 0.0193 |
| indoor_outdoor | hard | TGT-EVAL | raw_clip | cosine+Platt(SRC-CAL) | 0.106 | 0.653 | 0.047 | 0.761 | 0.861 | 0.074 | 0.467 | 0.574 | 0.0720 |
| indoor_outdoor | hard | TGT-EVAL | raw_clip | cosine+Platt(TGT oracle, cross-fitted) | 0.001 | 0.500 | 0.200 | 0.709 | 0.999 | 0.001 | 0.500 | 0.574 | 0.0229 |
| indoor_outdoor | easy | SRC-EVAL | raw_clip | cosine+Platt(SRC-CAL) | 0.200 | 0.814 | 0.114 | 0.800 | 0.674 | 0.075 | 0.374 | 0.737 | 0.0232 |
| indoor_outdoor | easy | SRC-EVAL | raw_clip | cosine+Platt(TGT oracle, cross-fitted) | 0.205 | 0.811 | 0.111 | 0.803 | 0.667 | 0.077 | 0.372 | 0.737 | 0.0223 |
| indoor_outdoor | easy | TGT-EVAL | raw_clip | cosine+Platt(SRC-CAL) | 0.181 | 0.644 | 0.056 | 0.793 | 0.767 | 0.129 | 0.448 | 0.590 | 0.0982 |
| indoor_outdoor | easy | TGT-EVAL | raw_clip | cosine+Platt(TGT oracle, cross-fitted) | 0.010 | 0.636 | 0.064 | 0.721 | 0.987 | 0.007 | 0.497 | 0.590 | 0.0220 |
| indoor_outdoor | hard | SRC-EVAL | vcs | native_(1+T)/2 | 0.476 | 0.732 | 0.032 | 0.841 | 0.303 | 0.255 | 0.279 | 0.800 | 0.0703 |
| indoor_outdoor | hard | SRC-EVAL | vcs | cosine+Platt(SRC-CAL) | 0.278 | 0.779 | 0.079 | 0.802 | 0.566 | 0.123 | 0.345 | 0.800 | 0.0231 |
| indoor_outdoor | hard | SRC-EVAL | vcs | score+Platt(SRC-CAL) | 0.339 | 0.763 | 0.063 | 0.775 | 0.483 | 0.161 | 0.322 | 0.800 | 0.0234 |
| indoor_outdoor | hard | SRC-EVAL | vcs | cosine+Platt(TGT oracle, cross-fitted) | 0.270 | 0.782 | 0.082 | 0.799 | 0.577 | 0.118 | 0.348 | 0.800 | 0.0243 |
| indoor_outdoor | hard | TGT-EVAL | vcs | native_(1+T)/2 | 0.392 | 0.525 | 0.175 | 0.830 | 0.589 | 0.372 | 0.480 | 0.531 | 0.1952 |
| indoor_outdoor | hard | TGT-EVAL | vcs | cosine+Platt(SRC-CAL) | 0.212 | 0.531 | 0.169 | 0.791 | 0.774 | 0.199 | 0.487 | 0.531 | 0.1459 |
| indoor_outdoor | hard | TGT-EVAL | vcs | score+Platt(SRC-CAL) | 0.263 | 0.534 | 0.166 | 0.770 | 0.719 | 0.245 | 0.482 | 0.531 | 0.1674 |
| indoor_outdoor | hard | TGT-EVAL | vcs | cosine+Platt(TGT oracle, cross-fitted) | 0.000 | nan | nan | nan | 1.000 | 0.000 | 0.500 | 0.531 | 0.0130 |
| indoor_outdoor | easy | SRC-EVAL | vcs | native_(1+T)/2 | 0.419 | 0.830 | 0.130 | 0.843 | 0.303 | 0.142 | 0.223 | 0.881 | 0.0248 |
| indoor_outdoor | easy | SRC-EVAL | vcs | cosine+Platt(SRC-CAL) | 0.387 | 0.835 | 0.135 | 0.832 | 0.354 | 0.128 | 0.241 | 0.881 | 0.0317 |
| indoor_outdoor | easy | SRC-EVAL | vcs | score+Platt(SRC-CAL) | 0.425 | 0.829 | 0.129 | 0.826 | 0.295 | 0.145 | 0.220 | 0.881 | 0.0344 |
| indoor_outdoor | easy | SRC-EVAL | vcs | cosine+Platt(TGT oracle, cross-fitted) | 0.411 | 0.832 | 0.132 | 0.840 | 0.316 | 0.138 | 0.227 | 0.881 | 0.0265 |
| indoor_outdoor | easy | TGT-EVAL | vcs | native_(1+T)/2 | 0.389 | 0.529 | 0.171 | 0.827 | 0.589 | 0.366 | 0.477 | 0.536 | 0.1935 |
| indoor_outdoor | easy | TGT-EVAL | vcs | cosine+Platt(SRC-CAL) | 0.347 | 0.532 | 0.168 | 0.818 | 0.631 | 0.324 | 0.478 | 0.536 | 0.1800 |
| indoor_outdoor | easy | TGT-EVAL | vcs | score+Platt(SRC-CAL) | 0.396 | 0.529 | 0.171 | 0.814 | 0.581 | 0.373 | 0.477 | 0.536 | 0.2042 |
| indoor_outdoor | easy | TGT-EVAL | vcs | cosine+Platt(TGT oracle, cross-fitted) | 0.000 | nan | nan | nan | 1.000 | 0.000 | 0.500 | 0.536 | 0.0158 |
| indoor_outdoor | hard | SRC-EVAL | infonce | cosine+Platt(SRC-CAL) | 0.308 | 0.780 | 0.080 | 0.782 | 0.520 | 0.136 | 0.328 | 0.807 | 0.0128 |
| indoor_outdoor | hard | SRC-EVAL | infonce | cosine+Platt(TGT oracle, cross-fitted) | 0.295 | 0.783 | 0.083 | 0.779 | 0.537 | 0.128 | 0.333 | 0.807 | 0.0132 |
| indoor_outdoor | hard | TGT-EVAL | infonce | cosine+Platt(SRC-CAL) | 0.345 | 0.535 | 0.165 | 0.787 | 0.631 | 0.321 | 0.476 | 0.527 | 0.1665 |
| indoor_outdoor | hard | TGT-EVAL | infonce | cosine+Platt(TGT oracle, cross-fitted) | 0.000 | nan | nan | nan | 1.000 | 0.000 | 0.500 | 0.527 | 0.0169 |
| indoor_outdoor | easy | SRC-EVAL | infonce | cosine+Platt(SRC-CAL) | 0.427 | 0.831 | 0.131 | 0.819 | 0.291 | 0.144 | 0.217 | 0.884 | 0.0168 |
| indoor_outdoor | easy | SRC-EVAL | infonce | cosine+Platt(TGT oracle, cross-fitted) | 0.444 | 0.826 | 0.126 | 0.827 | 0.267 | 0.155 | 0.211 | 0.884 | 0.0128 |
| indoor_outdoor | easy | TGT-EVAL | infonce | cosine+Platt(SRC-CAL) | 0.529 | 0.518 | 0.182 | 0.818 | 0.452 | 0.510 | 0.481 | 0.535 | 0.2132 |
| indoor_outdoor | easy | TGT-EVAL | infonce | cosine+Platt(TGT oracle, cross-fitted) | 0.000 | nan | nan | nan | 1.000 | 0.000 | 0.500 | 0.535 | 0.0193 |
| indoor_outdoor | hard | SRC-EVAL | logistic | native_sigmoid | 0.000 | nan | nan | nan | 1.000 | 0.000 | 0.500 | 0.808 | 0.4889 |
| indoor_outdoor | hard | SRC-EVAL | logistic | cosine+Platt(SRC-CAL) | 0.311 | 0.781 | 0.081 | 0.782 | 0.515 | 0.136 | 0.325 | 0.808 | 0.0140 |
| indoor_outdoor | hard | SRC-EVAL | logistic | score+Platt(SRC-CAL) | 0.311 | 0.781 | 0.081 | 0.782 | 0.515 | 0.136 | 0.325 | 0.808 | 0.0140 |
| indoor_outdoor | hard | SRC-EVAL | logistic | cosine+Platt(TGT oracle, cross-fitted) | 0.298 | 0.786 | 0.086 | 0.779 | 0.531 | 0.127 | 0.329 | 0.808 | 0.0138 |
| indoor_outdoor | hard | TGT-EVAL | logistic | native_sigmoid | 0.000 | nan | nan | nan | 1.000 | 0.000 | 0.500 | 0.529 | 0.4893 |
| indoor_outdoor | hard | TGT-EVAL | logistic | cosine+Platt(SRC-CAL) | 0.264 | 0.542 | 0.158 | 0.778 | 0.713 | 0.242 | 0.478 | 0.529 | 0.1498 |
| indoor_outdoor | hard | TGT-EVAL | logistic | score+Platt(SRC-CAL) | 0.264 | 0.542 | 0.158 | 0.778 | 0.713 | 0.242 | 0.478 | 0.529 | 0.1498 |
| indoor_outdoor | hard | TGT-EVAL | logistic | cosine+Platt(TGT oracle, cross-fitted) | 0.000 | nan | nan | nan | 1.000 | 0.000 | 0.500 | 0.529 | 0.0166 |
| indoor_outdoor | easy | SRC-EVAL | logistic | native_sigmoid | 0.000 | nan | nan | nan | 1.000 | 0.000 | 0.500 | 0.885 | 0.4901 |
| indoor_outdoor | easy | SRC-EVAL | logistic | cosine+Platt(SRC-CAL) | 0.430 | 0.831 | 0.131 | 0.820 | 0.286 | 0.145 | 0.216 | 0.885 | 0.0173 |
| indoor_outdoor | easy | SRC-EVAL | logistic | score+Platt(SRC-CAL) | 0.430 | 0.831 | 0.131 | 0.820 | 0.286 | 0.145 | 0.216 | 0.885 | 0.0173 |
| indoor_outdoor | easy | SRC-EVAL | logistic | cosine+Platt(TGT oracle, cross-fitted) | 0.447 | 0.825 | 0.125 | 0.828 | 0.263 | 0.156 | 0.209 | 0.885 | 0.0134 |
| indoor_outdoor | easy | TGT-EVAL | logistic | native_sigmoid | 0.000 | nan | nan | nan | 1.000 | 0.000 | 0.500 | 0.532 | 0.4893 |
| indoor_outdoor | easy | TGT-EVAL | logistic | cosine+Platt(SRC-CAL) | 0.442 | 0.523 | 0.177 | 0.809 | 0.538 | 0.422 | 0.480 | 0.532 | 0.1923 |
| indoor_outdoor | easy | TGT-EVAL | logistic | score+Platt(SRC-CAL) | 0.442 | 0.523 | 0.177 | 0.809 | 0.538 | 0.422 | 0.480 | 0.532 | 0.1923 |
| indoor_outdoor | easy | TGT-EVAL | logistic | cosine+Platt(TGT oracle, cross-fitted) | 0.000 | nan | nan | nan | 1.000 | 0.000 | 0.500 | 0.532 | 0.0189 |
