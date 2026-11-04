Publication copy (sample names aliased; numbers and conclusions unchanged).

# CR2 results (numbers only)

protocol `CR2_ANCHOR_PRESERVATION_CONCENTRATION_V1`; frozen `PROTOCOL_FROZEN_CR2.json` sha256 `6c58a52b8a30ad8360bc36d4dae4928bac05a2d5144181d43cbeec6a4f3843ab`; DEV only, seed 1001, step 8000; 0 new estimator fits; TEST not read.

## 24 anchor-preserving repairs

| condition | unit | i2t R1 | R5 | R10 | t2i R1 | R5 | R10 | rSum | Δbase | −CR1 OBS | −CR1 parent CAP | extra | i2t resc/lost | t2i resc/lost |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Flickr00 | BASE KEEP_SOFT | 93.20 | 99.01 | 99.51 | 81.14 | 94.97 | 97.32 | 565.15 | 6.71 | -0.10 | 2.90 | minus_BCE_same_policy=-0.08; minus_NCE_same_policy=-0.16 | 36/13 | 199/84 |
| Flickr00 | BASE KEEP_TOP1 | 93.39 | 99.01 | 99.51 | 81.20 | 94.99 | 97.34 | 565.44 | 7.00 | 0.20 | 3.20 | TOP1_minus_SOFT=+0.30; minus_BCE_same_policy=+0.28; minus_NCE_same_policy=+0.04 | 38/13 | 210/92 |
| Flickr00 | J KEEP_SOFT | 92.80 | 98.92 | 99.51 | 81.22 | 95.09 | 97.36 | 564.89 | 6.45 | -0.36 | 0.47 | minus_BASE_same_policy=-0.26; minus_BCE_same_policy=-0.34; minus_NCE_same_policy=-0.41 | 34/15 | 211/92 |
| Flickr00 | J KEEP_TOP1 | 93.00 | 99.01 | 99.51 | 81.16 | 95.03 | 97.36 | 565.07 | 6.63 | -0.18 | 0.65 | TOP1_minus_SOFT=+0.18; minus_BASE_same_policy=-0.37; minus_BCE_same_policy=-0.10; minus_NCE_same_policy=-0.34 | 36/15 | 212/96 |
| Flickr00 | BCE KEEP_SOFT | 93.10 | 99.11 | 99.51 | 81.16 | 95.03 | 97.32 | 565.23 | 6.79 | -0.02 | 0.61 | minus_BASE_same_policy=+0.08; minus_NCE_same_policy=-0.08 | 36/14 | 207/91 |
| Flickr00 | BCE KEEP_TOP1 | 93.00 | 98.92 | 99.51 | 81.30 | 95.07 | 97.38 | 565.17 | 6.73 | -0.08 | 0.55 | TOP1_minus_SOFT=-0.06; minus_BASE_same_policy=-0.28; minus_NCE_same_policy=-0.24 | 36/15 | 214/91 |
| Flickr00 | NCE KEEP_SOFT | 93.10 | 99.01 | 99.61 | 81.10 | 95.07 | 97.42 | 565.31 | 6.86 | 0.06 | 1.24 | minus_BASE_same_policy=+0.16; minus_BCE_same_policy=+0.08 | 35/13 | 210/97 |
| Flickr00 | NCE KEEP_TOP1 | 93.29 | 98.92 | 99.61 | 81.30 | 94.93 | 97.36 | 565.40 | 6.96 | 0.16 | 1.34 | TOP1_minus_SOFT=+0.10; minus_BASE_same_policy=-0.04; minus_BCE_same_policy=+0.24 | 37/13 | 216/93 |
| Flickr60 | BASE KEEP_SOFT | 91.32 | 98.62 | 99.41 | 78.97 | 94.36 | 96.79 | 559.47 | 1.03 | -1.07 | -0.32 | minus_BCE_same_policy=+0.43; minus_NCE_same_policy=+0.00 | 35/31 | 155/150 |
| Flickr60 | BASE KEEP_TOP1 | 91.42 | 98.72 | 99.41 | 79.27 | 94.44 | 96.79 | 560.04 | 1.60 | -0.49 | 0.26 | TOP1_minus_SOFT=+0.57; minus_BCE_same_policy=+0.41; minus_NCE_same_policy=+1.08 | 32/27 | 186/166 |
| Flickr60 | J KEEP_SOFT | 91.52 | 98.62 | 99.41 | 78.95 | 94.20 | 96.88 | 559.59 | 1.14 | -0.95 | -0.26 | minus_BASE_same_policy=+0.12; minus_BCE_same_policy=+0.55; minus_NCE_same_policy=+0.12 | 34/28 | 158/154 |
| Flickr60 | J KEEP_TOP1 | 91.03 | 98.82 | 99.21 | 79.37 | 94.34 | 96.77 | 559.53 | 1.08 | -1.01 | -0.32 | TOP1_minus_SOFT=-0.06; minus_BASE_same_policy=-0.51; minus_BCE_same_policy=-0.10; minus_NCE_same_policy=+0.57 | 32/31 | 207/182 |
| Flickr60 | BCE KEEP_SOFT | 90.93 | 98.72 | 99.41 | 78.95 | 94.18 | 96.84 | 559.03 | 0.59 | -1.50 | -0.97 | minus_BASE_same_policy=-0.43; minus_NCE_same_policy=-0.43 | 30/30 | 153/149 |
| Flickr60 | BCE KEEP_TOP1 | 91.12 | 99.01 | 99.31 | 79.07 | 94.30 | 96.80 | 559.63 | 1.18 | -0.91 | -0.37 | TOP1_minus_SOFT=+0.59; minus_BASE_same_policy=-0.41; minus_NCE_same_policy=+0.67 | 32/30 | 202/192 |
| Flickr60 | NCE KEEP_SOFT | 91.22 | 98.72 | 99.31 | 79.27 | 94.24 | 96.71 | 559.47 | 1.03 | -1.07 | 0.26 | minus_BASE_same_policy=+0.00; minus_BCE_same_policy=+0.43 | 32/29 | 195/175 |
| Flickr60 | NCE KEEP_TOP1 | 90.73 | 98.72 | 99.21 | 79.15 | 94.36 | 96.79 | 558.95 | 0.51 | -1.58 | -0.26 | TOP1_minus_SOFT=-0.51; minus_BASE_same_policy=-1.08; minus_BCE_same_policy=-0.67 | 31/33 | 203/189 |
| CC152K_RECOVERED_20261104 | BASE KEEP_SOFT | 78.07 | 93.51 | 96.10 | 76.91 | 91.77 | 95.38 | 531.75 | 13.13 | -0.58 | 8.51 | minus_BCE_same_policy=-1.73; minus_NCE_same_policy=-1.44 | 28/15 | 39/16 |
| CC152K_RECOVERED_20261104 | BASE KEEP_TOP1 | 78.50 | 93.51 | 96.25 | 76.77 | 92.35 | 95.53 | 532.90 | 14.29 | 0.58 | 9.67 | TOP1_minus_SOFT=+1.15; minus_BCE_same_policy=-0.43; minus_NCE_same_policy=+0.00 | 30/14 | 38/16 |
| CC152K_RECOVERED_20261104 | J KEEP_SOFT | 78.50 | 93.80 | 95.96 | 76.48 | 92.21 | 95.53 | 532.47 | 13.85 | 0.14 | 4.62 | minus_BASE_same_policy=+0.72; minus_BCE_same_policy=-1.01; minus_NCE_same_policy=-0.72 | 29/13 | 36/16 |
| CC152K_RECOVERED_20261104 | J KEEP_TOP1 | 78.93 | 93.65 | 95.96 | 76.77 | 91.92 | 95.38 | 532.61 | 14.00 | 0.29 | 4.76 | TOP1_minus_SOFT=+0.14; minus_BASE_same_policy=-0.29; minus_BCE_same_policy=-0.72; minus_NCE_same_policy=-0.29 | 31/12 | 37/15 |
| CC152K_RECOVERED_20261104 | BCE KEEP_SOFT | 79.08 | 93.65 | 95.96 | 77.49 | 91.92 | 95.38 | 533.48 | 14.86 | 1.15 | 4.76 | minus_BASE_same_policy=+1.73; minus_NCE_same_policy=+0.29 | 33/13 | 42/15 |
| CC152K_RECOVERED_20261104 | BCE KEEP_TOP1 | 79.37 | 93.36 | 95.96 | 77.34 | 91.92 | 95.38 | 533.33 | 14.72 | 1.01 | 4.62 | TOP1_minus_SOFT=-0.14; minus_BASE_same_policy=+0.43; minus_NCE_same_policy=+0.43 | 34/12 | 41/15 |
| CC152K_RECOVERED_20261104 | NCE KEEP_SOFT | 78.50 | 93.65 | 96.10 | 77.20 | 92.21 | 95.53 | 533.19 | 14.57 | 0.87 | 5.77 | minus_BASE_same_policy=+1.44; minus_BCE_same_policy=-0.29 | 31/15 | 40/15 |
| CC152K_RECOVERED_20261104 | NCE KEEP_TOP1 | 78.93 | 93.36 | 95.96 | 77.06 | 92.06 | 95.53 | 532.90 | 14.29 | 0.58 | 5.48 | TOP1_minus_SOFT=-0.29; minus_BASE_same_policy=+0.00; minus_BCE_same_policy=-0.43 | 31/12 | 39/15 |

## 6 observation-selection controls

| condition | unit | i2t R1 | R5 | R10 | t2i R1 | R5 | R10 | rSum | Δbase | −CR1 OBS | −CR1 parent CAP | extra | i2t resc/lost | t2i resc/lost |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Flickr00 | OBS SUPPORT_OBS | 92.80 | 99.01 | 99.51 | 81.14 | 94.79 | 97.22 | 564.48 | 6.04 | -0.77 |  |  | 38/19 | 166/51 |
| Flickr00 | OBS RANDOM_SUPPORT | 92.41 | 99.11 | 99.41 | 81.14 | 95.03 | 97.30 | 564.40 | 5.96 | -0.85 |  | minus_SUPPORT_OBS=-0.08 | 34/19 | 217/102 |
| Flickr60 | OBS SUPPORT_OBS | 91.12 | 98.82 | 99.41 | 79.96 | 94.52 | 96.82 | 560.65 | 2.21 | 0.12 |  |  | 32/30 | 195/140 |
| Flickr60 | OBS RANDOM_SUPPORT | 91.12 | 98.52 | 99.41 | 79.23 | 94.54 | 96.82 | 559.64 | 1.20 | -0.89 |  | minus_SUPPORT_OBS=-1.01 | 34/32 | 208/190 |
| CC152K_RECOVERED_20261104 | OBS SUPPORT_OBS | 78.07 | 93.07 | 95.67 | 76.19 | 90.48 | 93.80 | 527.27 | 8.66 | -5.05 |  |  | 25/12 | 27/9 |
| CC152K_RECOVERED_20261104 | OBS RANDOM_SUPPORT | 78.21 | 93.22 | 96.10 | 76.33 | 91.92 | 95.24 | 531.02 | 12.41 | -1.30 |  | minus_SUPPORT_OBS=+3.75 | 30/16 | 36/17 |

## 2 isolated oracle diagnostics (not deployable, not an upper bound)

| condition | unit | i2t R1 | R5 | R10 | t2i R1 | R5 | R10 | rSum | Δbase | −CR1 OBS | −CR1 parent CAP | extra | i2t resc/lost | t2i resc/lost |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Flickr00 | ORACLE KEEP_ORACLE | 93.20 | 99.01 | 99.41 | 81.10 | 95.01 | 97.38 | 565.11 | 6.67 | -0.14 | 0.69 | minus_J_KEEP_TOP1=+0.04; minus_J_KEEP_SOFT=+0.22 | 37/14 | 219/106 |
| Flickr60 | ORACLE KEEP_ORACLE | 91.03 | 98.82 | 99.31 | 79.45 | 94.50 | 96.77 | 559.86 | 1.42 | -0.67 | 0.02 | minus_J_KEEP_TOP1=+0.34; minus_J_KEEP_SOFT=+0.28 | 32/31 | 205/176 |

CR1 reference rSum (reused exactly, CR1 TASK_ENDPOINTS.csv): Flickr00 OBS 565.25, BASE/J/BCE/NCE_CAP 562.25/564.42/564.62/564.06; Flickr60 OBS 560.53, BASE/J/BCE/NCE_CAP 559.78/559.84/560.00/559.21; CC152K_RECOVERED_20261104 OBS 532.32, BASE/J/BCE/NCE_CAP 523.23/527.85/528.72/527.42

## Relation graphs (CR2 CSR; truth columns Flickr evaluator-only; CC label-free only)

| condition | graph | nnz | row mass | obs mass/text | H0 obs mass | cond. entropy | max cond. mass | row ESS | max load/CR1 cap | images over cap | true-mass share | correct-obs mass lost | wrong-obs true mass | new-edge true share |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Flickr00 | CR1_BASE_ROW | 4640000 | 0.9927 | 0.3185 | 0.0000 | 2.0641 | 0.4444 | 7.6945 | 4.3544 | 1838 | 0.3208 | 0.6815 |  | 0.0000 |
| Flickr00 | CR1_BASE_CAP | 4640000 | 0.9922 | 0.3220 | 0.0000 | 2.0693 | 0.4418 | 7.8279 | 1.0000 | 8 | 0.3245 | 0.6780 |  | 0.0000 |
| Flickr00 | CR1_J_ROW | 4640000 | 0.9977 | 0.3532 | 0.0000 | 1.6845 | 0.5234 | 5.2338 | 3.7967 | 1609 | 0.3540 | 0.6468 |  | 0.0000 |
| Flickr00 | CR1_J_CAP | 4640000 | 0.9976 | 0.3561 | 0.0000 | 1.6970 | 0.5197 | 5.3556 | 1.0000 | 2 | 0.3570 | 0.6439 |  | 0.0000 |
| Flickr00 | CR1_BCE_CAP | 4640000 | 0.9981 | 0.3611 | 0.0000 | 1.6575 | 0.5287 | 5.1669 | 1.0000 | 8 | 0.3618 | 0.6389 |  | 0.0000 |
| Flickr00 | CR1_NCE_CAP | 4640000 | 0.9977 | 0.3582 | 0.0000 | 1.7369 | 0.5120 | 5.5982 | 1.0000 | 7 | 0.3591 | 0.6418 |  | 0.0000 |
| Flickr00 | CR2_Flickr00_BASE_KEEP_SOFT_s1001 | 789224 | 0.9922 | 0.9234 | 0.4957 | 0.2998 | 0.9306 | 1.3685 | 0.8095 | 0 | 0.9306 | 0.0766 |  | 0.0000 |
| Flickr00 | CR2_Flickr00_BASE_KEEP_TOP1_s1001 | 165132 | 0.9922 | 0.9234 | 0.4957 | 0.0962 | 0.9306 | 1.1388 | 1.5392 | 26 | 0.9306 | 0.0766 |  | 0.0000 |
| Flickr00 | CR2_Flickr00_BCE_KEEP_SOFT_s1001 | 789224 | 0.9981 | 0.9289 | 0.4985 | 0.2705 | 0.9306 | 1.3429 | 0.8030 | 0 | 0.9307 | 0.0711 |  | 0.0000 |
| Flickr00 | CR2_Flickr00_BCE_KEEP_TOP1_s1001 | 165132 | 0.9981 | 0.9289 | 0.4985 | 0.0962 | 0.9306 | 1.1388 | 1.3984 | 18 | 0.9307 | 0.0711 |  | 0.0000 |
| Flickr00 | CR2_Flickr00_J_KEEP_SOFT_s1001 | 789224 | 0.9976 | 0.9284 | 0.4981 | 0.2730 | 0.9306 | 1.3453 | 0.8044 | 0 | 0.9307 | 0.0716 |  | 0.0000 |
| Flickr00 | CR2_Flickr00_J_KEEP_TOP1_s1001 | 165132 | 0.9976 | 0.9284 | 0.4981 | 0.0962 | 0.9306 | 1.1388 | 1.4490 | 20 | 0.9307 | 0.0716 |  | 0.0000 |
| Flickr00 | CR2_Flickr00_NCE_KEEP_SOFT_s1001 | 789224 | 0.9977 | 0.9285 | 0.4981 | 0.2771 | 0.9306 | 1.3491 | 0.8074 | 0 | 0.9307 | 0.0715 |  | 0.0000 |
| Flickr00 | CR2_Flickr00_NCE_KEEP_TOP1_s1001 | 165132 | 0.9977 | 0.9285 | 0.4981 | 0.0962 | 0.9306 | 1.1388 | 1.4480 | 20 | 0.9307 | 0.0715 |  | 0.0000 |
| Flickr00 | CR2_Flickr00_RANDOM_SUPPORT_s1001 | 124868 | 0.8612 | 0.8612 | 0.8620 | 0.0000 | 0.8612 | 0.8612 | 0.5000 | 0 | 1.0000 | 0.1388 |  |  |
| Flickr00 | CR2_Flickr00_SUPPORT_OBS_s1001 | 124868 | 0.8612 | 0.8612 | 0.0000 | 0.0000 | 0.8612 | 0.8612 | 0.5000 | 0 | 1.0000 | 0.1388 |  |  |
| Flickr00 | CR2_Flickr00_KEEP_ORACLE_s1001 |  |  |  |  |  |  |  |  |  | 1.0000 | 0.0024 |  |  |
| Flickr60 | CR1_BASE_ROW | 4640000 | 0.8085 | 0.0137 | 0.0000 | 3.4493 | 0.0520 | 30.8353 | 3.5240 | 1083 | 0.0424 | 0.9657 | 0.0343 | 0.0259 |
| Flickr60 | CR1_BASE_CAP | 4640000 | 0.8034 | 0.0140 | 0.0000 | 3.4424 | 0.0532 | 30.4682 | 1.0000 | 0 | 0.0434 | 0.9651 | 0.0348 | 0.0265 |
| Flickr60 | CR1_J_ROW | 4640000 | 0.8570 | 0.0191 | 0.0000 | 3.4018 | 0.0712 | 28.0320 | 2.8036 | 794 | 0.0558 | 0.9522 | 0.0478 | 0.0342 |
| Flickr60 | CR1_J_CAP | 4640000 | 0.8548 | 0.0193 | 0.0000 | 3.3966 | 0.0720 | 27.7946 | 1.0000 | 0 | 0.0563 | 0.9518 | 0.0482 | 0.0346 |
| Flickr60 | CR1_BCE_CAP | 4640000 | 0.8566 | 0.0203 | 0.0000 | 3.3850 | 0.0763 | 27.1922 | 1.0000 | 0 | 0.0591 | 0.9494 | 0.0506 | 0.0363 |
| Flickr60 | CR1_NCE_CAP | 4640000 | 0.9834 | 0.1306 | 0.0000 | 1.8764 | 0.4880 | 6.1632 | 1.0000 | 5 | 0.3324 | 0.6735 | 0.3271 | 0.2301 |
| Flickr60 | CR2_Flickr60_BASE_KEEP_SOFT_s1001 | 3182408 | 0.8034 | 0.5402 | 0.4021 | 1.5807 | 0.6727 | 2.8806 | 0.7545 | 0 | 0.3850 | 0.2528 | 0.0174 | 0.0396 |
| Flickr60 | CR2_Flickr60_BASE_KEEP_TOP1_s1001 | 239919 | 0.8034 | 0.5402 | 0.4021 | 0.4537 | 0.6727 | 1.6546 | 2.4630 | 153 | 0.5031 | 0.2528 | 0.1755 | 0.4000 |
| Flickr60 | CR2_Flickr60_BCE_KEEP_SOFT_s1001 | 3182408 | 0.8566 | 0.5769 | 0.4272 | 1.5627 | 0.6727 | 2.8694 | 0.7985 | 0 | 0.3909 | 0.2009 | 0.0253 | 0.0542 |
| Flickr60 | CR2_Flickr60_BCE_KEEP_TOP1_s1001 | 239919 | 0.8566 | 0.5769 | 0.4272 | 0.4537 | 0.6727 | 1.6546 | 3.4941 | 471 | 0.4887 | 0.2009 | 0.1650 | 0.3539 |
| Flickr60 | CR2_Flickr60_J_KEEP_SOFT_s1001 | 3182408 | 0.8548 | 0.5755 | 0.4265 | 1.5664 | 0.6727 | 2.8719 | 0.7973 | 0 | 0.3899 | 0.2029 | 0.0240 | 0.0517 |
| Flickr60 | CR2_Flickr60_J_KEEP_TOP1_s1001 | 239919 | 0.8548 | 0.5755 | 0.4265 | 0.4537 | 0.6727 | 1.6546 | 2.8777 | 439 | 0.4913 | 0.2029 | 0.1685 | 0.3620 |
| Flickr60 | CR2_Flickr60_NCE_KEEP_SOFT_s1001 | 3182408 | 0.9834 | 0.6617 | 0.4913 | 1.0894 | 0.6727 | 2.3827 | 0.8456 | 0 | 0.4723 | 0.0840 | 0.1633 | 0.3047 |
| Flickr60 | CR2_Flickr60_NCE_KEEP_TOP1_s1001 | 239919 | 0.9834 | 0.6617 | 0.4913 | 0.4537 | 0.6727 | 1.6546 | 1.4327 | 109 | 0.5020 | 0.0840 | 0.2122 | 0.3958 |
| Flickr60 | CR2_Flickr60_RANDOM_SUPPORT_s1001 | 50081 | 0.3454 | 0.3454 | 0.3463 | 0.0000 | 0.3454 | 0.3454 | 0.5000 | 0 | 0.3986 | 0.6558 | 0.0000 |  |
| Flickr60 | CR2_Flickr60_SUPPORT_OBS_s1001 | 50081 | 0.3454 | 0.3454 | 0.0000 | 0.0000 | 0.3454 | 0.3454 | 0.5000 | 0 | 0.9980 | 0.1383 | 0.0000 |  |
| Flickr60 | CR2_Flickr60_KEEP_ORACLE_s1001 |  |  |  |  |  |  |  |  |  | 0.6590 | 0.1452 | 0.3689 | 1.0000 |
| CC152K_RECOVERED_20261104 | CR1_BASE_ROW | 3142784 | 0.9976 | 0.2081 | 0.0000 | 2.4261 | 0.3427 | 11.3043 | 16.0414 | 10938 |  |  |  |  |
| CC152K_RECOVERED_20261104 | CR1_BASE_CAP | 3142784 | 0.9951 | 0.2126 | 0.0000 | 2.4410 | 0.3319 | 11.4799 | 1.0000 | 28 |  |  |  |  |
| CC152K_RECOVERED_20261104 | CR1_J_ROW | 3142784 | 0.9994 | 0.2379 | 0.0000 | 1.9099 | 0.4536 | 6.2071 | 22.7903 | 10557 |  |  |  |  |
| CC152K_RECOVERED_20261104 | CR1_J_CAP | 3142784 | 0.9974 | 0.2423 | 0.0000 | 1.9794 | 0.4307 | 6.8530 | 1.0000 | 72 |  |  |  |  |
| CC152K_RECOVERED_20261104 | CR1_BCE_CAP | 3142784 | 0.9976 | 0.2464 | 0.0000 | 1.9271 | 0.4422 | 6.5214 | 1.0000 | 63 |  |  |  |  |
| CC152K_RECOVERED_20261104 | CR1_NCE_CAP | 3142784 | 0.9973 | 0.2471 | 0.0000 | 2.0202 | 0.4225 | 7.2541 | 1.0000 | 249 |  |  |  |  |
| CC152K_RECOVERED_20261104 | CR2_CC152K_RECOVERED_20261104_BASE_KEEP_SOFT_s1001 | 1102404 | 0.9951 | 0.8367 | 0.4959 | 0.7164 | 0.8402 | 1.8762 | 0.9956 | 0 |  |  |  |  |
| CC152K_RECOVERED_20261104 | CR2_CC152K_RECOVERED_20261104_BASE_KEEP_TOP1_s1001 | 129593 | 0.9951 | 0.8367 | 0.4959 | 0.2215 | 0.8402 | 1.3195 | 14.8172 | 1561 |  |  |  |  |
| CC152K_RECOVERED_20261104 | CR2_CC152K_RECOVERED_20261104_BCE_KEEP_SOFT_s1001 | 1102404 | 0.9976 | 0.8388 | 0.4969 | 0.6346 | 0.8402 | 1.8074 | 0.9992 | 0 |  |  |  |  |
| CC152K_RECOVERED_20261104 | CR2_CC152K_RECOVERED_20261104_BCE_KEEP_TOP1_s1001 | 129593 | 0.9976 | 0.8388 | 0.4969 | 0.2215 | 0.8402 | 1.3195 | 14.7506 | 1294 |  |  |  |  |
| CC152K_RECOVERED_20261104 | CR2_CC152K_RECOVERED_20261104_J_KEEP_SOFT_s1001 | 1102404 | 0.9974 | 0.8387 | 0.4968 | 0.6426 | 0.8402 | 1.8147 | 0.9991 | 0 |  |  |  |  |
| CC152K_RECOVERED_20261104 | CR2_CC152K_RECOVERED_20261104_J_KEEP_TOP1_s1001 | 129593 | 0.9974 | 0.8387 | 0.4968 | 0.2215 | 0.8402 | 1.3195 | 14.7229 | 1251 |  |  |  |  |
| CC152K_RECOVERED_20261104 | CR2_CC152K_RECOVERED_20261104_NCE_KEEP_SOFT_s1001 | 1102404 | 0.9973 | 0.8386 | 0.4966 | 0.6524 | 0.8402 | 1.8231 | 0.9990 | 0 |  |  |  |  |
| CC152K_RECOVERED_20261104 | CR2_CC152K_RECOVERED_20261104_NCE_KEEP_TOP1_s1001 | 129593 | 0.9973 | 0.8386 | 0.4966 | 0.2215 | 0.8402 | 1.3195 | 14.7434 | 1253 |  |  |  |  |
| CC152K_RECOVERED_20261104 | CR2_CC152K_RECOVERED_20261104_RANDOM_SUPPORT_s1001 | 66831 | 0.6805 | 0.6805 | 0.6834 | 0.0000 | 0.6805 | 0.6805 | 0.5000 | 0 |  |  |  |  |
| CC152K_RECOVERED_20261104 | CR2_CC152K_RECOVERED_20261104_SUPPORT_OBS_s1001 | 66831 | 0.6805 | 0.6805 | 0.0000 | 0.0000 | 0.6805 | 0.6805 | 0.5000 | 0 |  |  |  |  |

## Cost

GPU jobs 1.100 GPU·h; CPU jobs (excl. feeder) 0.144 h; per-job ledger COSTS_CR2.csv; failed/cancelled in SUBMITTED_CR2.tsv.

Files: TASK_ENDPOINTS.csv, TASK_CURVES.csv, GRAPH_STATS_CR2.csv, COSTS_CR2.csv, audit JSONs under outputs_cr2/.../audit and evaluator_only.

## Records (appended once after code/site/summarize_cr2.py)
- Package CPU tests: 1088126, 28/28 OK (scratch copy; manifest 59/59 unchanged). Real-data acceptance 1088137 ALL_PASS 7/7 (S1 sparse graphs/row mass, S2 text marginal, S3 sampler expansion, S4 soft-NCE, S5 CSR-path == CR1-path bitwise, S6 score invariance, S7 DEV bridge step-0 Flickr+CC).
- Read-only audit 1088127–1088131 (pre-build) and 1088143–1088147 (post-build; 1088138–1088142 cancelled by me before running: glob quoting bug). Builds: deployable 1088132–1088134, oracle 1088135/1088136 (isolated). Freeze `PROTOCOL_FROZEN_CR2.json` sha256 6c58a52b… before any student. Feeder 1088148; 32 students in SUBMITTED_CR2.tsv (all DONE, 0 FAILED, no rerun, no amendment).
- All 32 step-0 evaluations reproduce the base order exactly; CR1/NR1 fp32 bridge difference 0.0 at step 8000; CR2 KEEP graphs preserve CR1 row mass (max |Δ| ≤ 3.3e-16).
- KEEP_TOP1 graphs exceed the old CR1 column capacity (reported, not re-solved; no KKT certificate claimed); KEEP_SOFT graphs stay ≤ cap.
- Caveat: CR1 endpoints are reused as recorded; CSR path equals CR1 path bitwise on the same GPU, but a 250-step CR1 OBS re-run on RTX PRO 6000 vs the stored L40S checkpoint differs by max |Δparam| 0.026 (cross-GPU-type nondeterminism).
