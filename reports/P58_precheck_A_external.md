# Pre-check A, addendum 1 — external target: Localized Narratives on Open Images validation (5000 images; narrative truncation rate 0.045, mean 30 words) — 2026-09-27T20:37:54Z

Adapters = the P58 topic-pairing selection re-trained per source (lr / epochs in the JSON); pairing on the external target is exact (own narrative); mismatches are random (no topic labels).

## A-S1 quantities (mean over seeds)

| source | method | split | J (vcs) | native ECE | cosine+Platt(SRC-CAL) ECE | native − Platt | R@1 |
|---|---|---|---|---|---|---|---|
| animal | vcs | SRC-EVAL | 0.572 | 0.0231 | 0.0220 | +0.0010 | 0.080 |
| animal | vcs | EXT-EVAL | 0.559 | 0.1012 | 0.0969 | +0.0043 | 0.070 |
| animal | infonce | SRC-EVAL | — | nan | 0.0127 | +nan | 0.059 |
| animal | infonce | EXT-EVAL | — | nan | 0.1111 | +nan | 0.045 |
| animal | logistic | SRC-EVAL | — | 0.4828 | 0.0110 | +0.4718 | 0.059 |
| animal | logistic | EXT-EVAL | — | 0.4799 | 0.1011 | +0.3788 | 0.036 |
| indoor_outdoor | vcs | SRC-EVAL | 0.441 | 0.0289 | 0.0323 | -0.0034 | 0.026 |
| indoor_outdoor | vcs | EXT-EVAL | 0.322 | 0.1099 | 0.0999 | +0.0101 | 0.042 |
| indoor_outdoor | infonce | SRC-EVAL | — | nan | 0.0137 | +nan | 0.031 |
| indoor_outdoor | infonce | EXT-EVAL | — | nan | 0.1218 | +nan | 0.047 |
| indoor_outdoor | logistic | SRC-EVAL | — | 0.4900 | 0.0134 | +0.4765 | 0.031 |
| indoor_outdoor | logistic | EXT-EVAL | — | 0.4895 | 0.0865 | +0.4030 | 0.041 |

## A-T on EXT-EVAL — nominal precision 0.8 (matched = own narrative, mismatched = random other narrative; mean over seeds)

| source | method | rule | accept | precision | \|prec − nominal\| | FNR | FPR | bal.err | AUROC | ECE |
|---|---|---|---|---|---|---|---|---|---|---|
| animal | raw_clip | cosine+Platt(SRC-CAL easy) | 0.481 | 0.952 | 0.152 | 0.083 | 0.046 | 0.065 | 0.985 | 0.2290 |
| animal | raw_clip | cosine+Platt(SRC-CAL hard) | 0.377 | 0.987 | 0.187 | 0.256 | 0.010 | 0.133 | 0.985 | 0.2459 |
| animal | raw_clip | cosine+Platt(EXT oracle, cross-fitted) | 0.442 | 0.972 | 0.172 | 0.141 | 0.025 | 0.083 | 0.985 | 0.0042 |
| animal | vcs | native_(1+T)/2 | 0.436 | 0.908 | 0.108 | 0.208 | 0.080 | 0.144 | 0.946 | 0.1005 |
| animal | vcs | cosine+Platt(SRC-CAL easy) | 0.424 | 0.913 | 0.113 | 0.226 | 0.074 | 0.150 | 0.946 | 0.0961 |
| animal | vcs | cosine+Platt(SRC-CAL hard) | 0.281 | 0.946 | 0.146 | 0.469 | 0.030 | 0.249 | 0.946 | 0.1054 |
| animal | vcs | score+Platt(SRC-CAL easy) | 0.470 | 0.894 | 0.094 | 0.160 | 0.100 | 0.130 | 0.946 | 0.0980 |
| animal | vcs | cosine+Platt(EXT oracle, cross-fitted) | 0.378 | 0.925 | 0.125 | 0.301 | 0.057 | 0.179 | 0.946 | 0.0149 |
| animal | infonce | cosine+Platt(SRC-CAL easy) | 0.449 | 0.883 | 0.083 | 0.207 | 0.105 | 0.156 | 0.933 | 0.1091 |
| animal | infonce | cosine+Platt(SRC-CAL hard) | 0.292 | 0.936 | 0.136 | 0.453 | 0.037 | 0.245 | 0.933 | 0.0916 |
| animal | infonce | cosine+Platt(EXT oracle, cross-fitted) | 0.358 | 0.918 | 0.118 | 0.343 | 0.059 | 0.201 | 0.933 | 0.0072 |
| animal | logistic | native_sigmoid | 0.000 | nan | nan | 1.000 | 0.000 | 0.500 | 0.928 | 0.4801 |
| animal | logistic | cosine+Platt(SRC-CAL easy) | 0.435 | 0.880 | 0.080 | 0.234 | 0.105 | 0.169 | 0.928 | 0.0990 |
| animal | logistic | cosine+Platt(SRC-CAL hard) | 0.260 | 0.937 | 0.137 | 0.513 | 0.033 | 0.273 | 0.928 | 0.0866 |
| animal | logistic | score+Platt(SRC-CAL easy) | 0.435 | 0.880 | 0.080 | 0.234 | 0.105 | 0.169 | 0.928 | 0.0990 |
| animal | logistic | cosine+Platt(EXT oracle, cross-fitted) | 0.355 | 0.910 | 0.110 | 0.354 | 0.064 | 0.209 | 0.928 | 0.0083 |
| indoor_outdoor | raw_clip | cosine+Platt(SRC-CAL easy) | 0.354 | 0.990 | 0.190 | 0.298 | 0.007 | 0.153 | 0.985 | 0.2363 |
| indoor_outdoor | raw_clip | cosine+Platt(SRC-CAL hard) | 0.182 | 0.999 | 0.199 | 0.635 | 0.000 | 0.318 | 0.985 | 0.2671 |
| indoor_outdoor | raw_clip | cosine+Platt(EXT oracle, cross-fitted) | 0.442 | 0.972 | 0.172 | 0.141 | 0.025 | 0.083 | 0.985 | 0.0042 |
| indoor_outdoor | vcs | native_(1+T)/2 | 0.316 | 0.871 | 0.071 | 0.450 | 0.082 | 0.266 | 0.858 | 0.1089 |
| indoor_outdoor | vcs | cosine+Platt(SRC-CAL easy) | 0.274 | 0.889 | 0.089 | 0.513 | 0.061 | 0.287 | 0.858 | 0.0973 |
| indoor_outdoor | vcs | cosine+Platt(SRC-CAL hard) | 0.138 | 0.944 | 0.144 | 0.740 | 0.015 | 0.378 | 0.858 | 0.0839 |
| indoor_outdoor | vcs | score+Platt(SRC-CAL easy) | 0.318 | 0.870 | 0.070 | 0.447 | 0.083 | 0.265 | 0.858 | 0.0965 |
| indoor_outdoor | vcs | cosine+Platt(EXT oracle, cross-fitted) | 0.236 | 0.907 | 0.107 | 0.572 | 0.044 | 0.308 | 0.858 | 0.0124 |
| indoor_outdoor | infonce | cosine+Platt(SRC-CAL easy) | 0.289 | 0.884 | 0.084 | 0.489 | 0.067 | 0.278 | 0.852 | 0.1221 |
| indoor_outdoor | infonce | cosine+Platt(SRC-CAL hard) | 0.142 | 0.950 | 0.150 | 0.731 | 0.014 | 0.373 | 0.852 | 0.0915 |
| indoor_outdoor | infonce | cosine+Platt(EXT oracle, cross-fitted) | 0.220 | 0.918 | 0.118 | 0.596 | 0.036 | 0.316 | 0.852 | 0.0276 |
| indoor_outdoor | logistic | native_sigmoid | 0.000 | nan | nan | 1.000 | 0.000 | 0.500 | 0.833 | 0.4896 |
| indoor_outdoor | logistic | cosine+Platt(SRC-CAL easy) | 0.236 | 0.895 | 0.095 | 0.577 | 0.050 | 0.313 | 0.833 | 0.0870 |
| indoor_outdoor | logistic | cosine+Platt(SRC-CAL hard) | 0.108 | 0.954 | 0.154 | 0.795 | 0.010 | 0.402 | 0.833 | 0.0610 |
| indoor_outdoor | logistic | score+Platt(SRC-CAL easy) | 0.236 | 0.895 | 0.095 | 0.577 | 0.050 | 0.313 | 0.833 | 0.0870 |
| indoor_outdoor | logistic | cosine+Platt(EXT oracle, cross-fitted) | 0.195 | 0.917 | 0.117 | 0.642 | 0.032 | 0.337 | 0.833 | 0.0306 |

## A-T on EXT-EVAL — nominal precision 0.7 (matched = own narrative, mismatched = random other narrative; mean over seeds)

| source | method | rule | accept | precision | \|prec − nominal\| | FNR | FPR | bal.err | AUROC | ECE |
|---|---|---|---|---|---|---|---|---|---|---|
| animal | raw_clip | cosine+Platt(SRC-CAL easy) | 0.562 | 0.871 | 0.171 | 0.021 | 0.146 | 0.083 | 0.985 | 0.2290 |
| animal | raw_clip | cosine+Platt(SRC-CAL hard) | 0.492 | 0.944 | 0.244 | 0.071 | 0.056 | 0.063 | 0.985 | 0.2459 |
| animal | raw_clip | cosine+Platt(EXT oracle, cross-fitted) | 0.464 | 0.962 | 0.262 | 0.107 | 0.036 | 0.071 | 0.985 | 0.0042 |
| animal | vcs | native_(1+T)/2 | 0.522 | 0.864 | 0.164 | 0.098 | 0.141 | 0.120 | 0.946 | 0.1005 |
| animal | vcs | cosine+Platt(SRC-CAL easy) | 0.512 | 0.870 | 0.170 | 0.109 | 0.133 | 0.121 | 0.946 | 0.0961 |
| animal | vcs | cosine+Platt(SRC-CAL hard) | 0.409 | 0.917 | 0.217 | 0.249 | 0.068 | 0.158 | 0.946 | 0.1054 |
| animal | vcs | score+Platt(SRC-CAL easy) | 0.539 | 0.852 | 0.152 | 0.081 | 0.160 | 0.121 | 0.946 | 0.0980 |
| animal | vcs | cosine+Platt(EXT oracle, cross-fitted) | 0.436 | 0.907 | 0.207 | 0.208 | 0.081 | 0.145 | 0.946 | 0.0149 |
| animal | infonce | cosine+Platt(SRC-CAL easy) | 0.546 | 0.829 | 0.129 | 0.095 | 0.187 | 0.141 | 0.933 | 0.1091 |
| animal | infonce | cosine+Platt(SRC-CAL hard) | 0.441 | 0.887 | 0.187 | 0.217 | 0.100 | 0.159 | 0.933 | 0.0916 |
| animal | infonce | cosine+Platt(EXT oracle, cross-fitted) | 0.434 | 0.890 | 0.190 | 0.227 | 0.096 | 0.161 | 0.933 | 0.0072 |
| animal | logistic | native_sigmoid | 0.000 | nan | nan | 1.000 | 0.000 | 0.500 | 0.928 | 0.4801 |
| animal | logistic | cosine+Platt(SRC-CAL easy) | 0.534 | 0.830 | 0.130 | 0.114 | 0.182 | 0.148 | 0.928 | 0.0990 |
| animal | logistic | cosine+Platt(SRC-CAL hard) | 0.424 | 0.885 | 0.185 | 0.250 | 0.098 | 0.174 | 0.928 | 0.0866 |
| animal | logistic | score+Platt(SRC-CAL easy) | 0.534 | 0.830 | 0.130 | 0.114 | 0.182 | 0.148 | 0.928 | 0.0990 |
| animal | logistic | cosine+Platt(EXT oracle, cross-fitted) | 0.436 | 0.879 | 0.179 | 0.233 | 0.105 | 0.169 | 0.928 | 0.0083 |
| indoor_outdoor | raw_clip | cosine+Platt(SRC-CAL easy) | 0.474 | 0.957 | 0.257 | 0.093 | 0.041 | 0.067 | 0.985 | 0.2363 |
| indoor_outdoor | raw_clip | cosine+Platt(SRC-CAL hard) | 0.397 | 0.984 | 0.284 | 0.220 | 0.013 | 0.116 | 0.985 | 0.2671 |
| indoor_outdoor | raw_clip | cosine+Platt(EXT oracle, cross-fitted) | 0.464 | 0.962 | 0.262 | 0.107 | 0.036 | 0.071 | 0.985 | 0.0042 |
| indoor_outdoor | vcs | native_(1+T)/2 | 0.458 | 0.796 | 0.096 | 0.271 | 0.187 | 0.229 | 0.858 | 0.1089 |
| indoor_outdoor | vcs | cosine+Platt(SRC-CAL easy) | 0.424 | 0.816 | 0.116 | 0.309 | 0.157 | 0.233 | 0.858 | 0.0973 |
| indoor_outdoor | vcs | cosine+Platt(SRC-CAL hard) | 0.287 | 0.882 | 0.182 | 0.494 | 0.068 | 0.281 | 0.858 | 0.0839 |
| indoor_outdoor | vcs | score+Platt(SRC-CAL easy) | 0.464 | 0.793 | 0.093 | 0.264 | 0.192 | 0.228 | 0.858 | 0.0965 |
| indoor_outdoor | vcs | cosine+Platt(EXT oracle, cross-fitted) | 0.341 | 0.859 | 0.159 | 0.414 | 0.096 | 0.255 | 0.858 | 0.0124 |
| indoor_outdoor | infonce | cosine+Platt(SRC-CAL easy) | 0.459 | 0.786 | 0.086 | 0.278 | 0.196 | 0.237 | 0.852 | 0.1221 |
| indoor_outdoor | infonce | cosine+Platt(SRC-CAL hard) | 0.316 | 0.868 | 0.168 | 0.451 | 0.083 | 0.267 | 0.852 | 0.0915 |
| indoor_outdoor | infonce | cosine+Platt(EXT oracle, cross-fitted) | 0.337 | 0.857 | 0.157 | 0.422 | 0.096 | 0.259 | 0.852 | 0.0276 |
| indoor_outdoor | logistic | native_sigmoid | 0.000 | nan | nan | 1.000 | 0.000 | 0.500 | 0.833 | 0.4896 |
| indoor_outdoor | logistic | cosine+Platt(SRC-CAL easy) | 0.395 | 0.806 | 0.106 | 0.363 | 0.153 | 0.258 | 0.833 | 0.0870 |
| indoor_outdoor | logistic | cosine+Platt(SRC-CAL hard) | 0.259 | 0.881 | 0.181 | 0.544 | 0.062 | 0.303 | 0.833 | 0.0610 |
| indoor_outdoor | logistic | score+Platt(SRC-CAL easy) | 0.395 | 0.806 | 0.106 | 0.363 | 0.153 | 0.258 | 0.833 | 0.0870 |
| indoor_outdoor | logistic | cosine+Platt(EXT oracle, cross-fitted) | 0.317 | 0.848 | 0.148 | 0.462 | 0.096 | 0.279 | 0.833 | 0.0306 |
