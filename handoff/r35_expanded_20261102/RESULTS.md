Publication copy (sample names aliased; numbers and conclusions unchanged).

# R35 EXPANDED results (numbers only)
generated 2026-11-02T10:34:48Z; 42/42 units complete; fixed final step (A/B/C 8000, D 4000); full DEV 1014 img / 5070 cap; TEST not read.

## Fixed-step DEV retrieval (A/B: direct full-DEV ranking by peer-mean g; C/D: frozen-H top-100 reranked by g, rest in H order; upstream IN2R i2t/t2i)

| unit | i2t R1 | R5 | R10 | t2i R1 | R5 | R10 | rSum | Δ vs H | tie-opt | tie-pess | J DEV-clean raw | J DEV-clean affine(logistic) | GPU |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| r35_A_n00_s1001_ID_J | 81.46 | 94.38 | 97.44 | 62.62 | 85.50 | 90.49 | 511.89 | -10.18 | 511.89 | 511.89 | 0.7701 | 0.7949 | NVIDIA L40S |
| r35_A_n00_s1001_ID_BCE | 81.76 | 94.77 | 97.53 | 62.47 | 85.25 | 90.63 | 512.41 | -9.66 | 512.41 | 512.41 | 0.8075 | 0.7971 | NVIDIA L40S |
| r35_A_n00_s1001_ID_NCE | 83.04 | 95.66 | 98.03 | 63.96 | 86.31 | 91.50 | 518.50 | -3.57 | 518.50 | 518.50 | 0.8239 | 0.8148 | NVIDIA L40S |
| r35_A_n00_s1001_G_J | 76.53 | 93.98 | 97.04 | 64.75 | 86.86 | 91.81 | 510.99 | -11.08 | 510.99 | 510.99 | 0.8233 | 0.8345 | NVIDIA RTX PRO 6000 Blackwell Server Edition |
| r35_A_n00_s1001_G_BCE | 77.12 | 94.28 | 97.53 | 64.36 | 86.79 | 91.95 | 512.03 | -10.04 | 512.03 | 512.03 | 0.8544 | 0.8358 | NVIDIA RTX PRO 6000 Blackwell Server Edition |
| r35_A_n00_s1001_G_NCE | 82.94 | 95.46 | 98.03 | 64.95 | 87.32 | 92.05 | 520.75 | -1.32 | 520.75 | 520.75 | -0.9998 | 0.8336 | NVIDIA RTX PRO 6000 Blackwell Server Edition |
| r35_A_n60_s1001_ID_J | 71.99 | 90.04 | 94.87 | 49.68 | 75.98 | 83.61 | 466.17 | -39.03 | 466.17 | 466.17 | 0.5688 | 0.6242 | NVIDIA RTX PRO 6000 Blackwell Server Edition |
| r35_A_n60_s1001_ID_BCE | 73.67 | 91.62 | 95.76 | 51.58 | 77.51 | 85.11 | 475.25 | -29.96 | 475.25 | 475.25 | 0.5925 | 0.6285 | NVIDIA RTX PRO 6000 Blackwell Server Edition |
| r35_A_n60_s1001_ID_NCE | 77.12 | 93.10 | 96.25 | 54.64 | 80.24 | 87.40 | 488.74 | -16.47 | 488.74 | 488.74 | 0.5609 | 0.6178 | NVIDIA RTX PRO 6000 Blackwell Server Edition |
| r35_A_n60_s1001_G_J | 79.29 | 94.87 | 98.03 | 58.38 | 83.14 | 89.21 | 502.92 | -2.29 | 502.92 | 502.92 | 0.6143 | 0.6240 | NVIDIA RTX PRO 6000 Blackwell Server Edition |
| r35_A_n60_s1001_G_BCE | 79.68 | 94.77 | 98.32 | 58.74 | 83.91 | 89.55 | 504.97 | -0.24 | 504.97 | 504.97 | 0.6148 | 0.6413 | NVIDIA RTX PRO 6000 Blackwell Server Edition |
| r35_A_n60_s1001_G_NCE | 80.87 | 95.07 | 97.83 | 58.19 | 82.94 | 88.99 | 503.89 | -1.32 | 503.89 | 503.89 | -0.9853 | 0.6223 | NVIDIA RTX PRO 6000 Blackwell Server Edition |
| r35_B_n60_s1001_TRUST_J | 78.99 | 94.77 | 97.63 | 57.06 | 82.68 | 88.62 | 499.76 | -5.44 | 499.76 | 499.76 | 0.8434 | 0.6025 | NVIDIA RTX PRO 6000 Blackwell Server Edition |
| r35_B_n60_s1001_TRUST_BCE | 79.49 | 94.38 | 97.93 | 57.28 | 82.78 | 88.88 | 500.73 | -4.48 | 500.73 | 500.73 | 0.8408 | 0.6015 | NVIDIA RTX PRO 6000 Blackwell Server Edition |
| r35_B_n60_s1001_TRUST_NCE | 80.28 | 94.58 | 97.83 | 57.83 | 82.84 | 88.74 | 502.09 | -3.12 | 502.09 | 502.09 | -0.9987 | 0.6015 | NVIDIA RTX PRO 6000 Blackwell Server Edition |
| r35_B_n60_s1001_ORACLE_J | 75.05 | 93.59 | 96.94 | 58.82 | 83.77 | 89.90 | 498.07 | -7.14 | 498.07 | 498.07 | 0.8936 | 0.6006 | NVIDIA RTX PRO 6000 Blackwell Server Edition |
| r35_B_n60_s1001_ORACLE_BCE | 74.65 | 94.08 | 97.34 | 58.90 | 83.93 | 90.12 | 499.01 | -6.19 | 499.01 | 499.01 | 0.8969 | 0.5979 | NVIDIA RTX PRO 6000 Blackwell Server Edition |
| r35_B_n60_s1001_ORACLE_NCE | 80.28 | 95.46 | 97.83 | 60.18 | 84.20 | 90.30 | 508.24 | 3.04 | 508.24 | 508.24 | -0.2664 | 0.6007 | NVIDIA RTX PRO 6000 Blackwell Server Edition |
| r35_C_n00_s1001_G_J | 82.84 | 96.15 | 98.03 | 66.04 | 87.87 | 92.23 | 523.16 | 1.08 | 523.16 | 523.16 | -0.1941 | -0.1338 | NVIDIA RTX PRO 6000 Blackwell Server Edition |
| r35_C_n00_s1001_G_BCE | 82.84 | 95.76 | 97.83 | 65.82 | 87.67 | 92.39 | 522.31 | 0.24 | 522.31 | 522.31 | -0.1344 | -0.1351 | NVIDIA RTX PRO 6000 Blackwell Server Edition |
| r35_C_n00_s1001_G_NCE | 82.25 | 95.07 | 97.53 | 62.82 | 84.64 | 89.47 | 511.78 | -10.30 | 511.78 | 511.78 | -1.0000 | -0.1274 | NVIDIA RTX PRO 6000 Blackwell Server Edition |
| r35_C_n00_s1001_L_J | 83.04 | 96.35 | 97.93 | 66.09 | 87.97 | 92.25 | 523.63 | 1.56 | 523.63 | 523.63 | -0.1891 | -0.1368 | NVIDIA RTX PRO 6000 Blackwell Server Edition |
| r35_C_n00_s1001_L_BCE | 82.84 | 95.86 | 97.83 | 65.86 | 87.89 | 92.43 | 522.70 | 0.63 | 522.70 | 522.70 | -0.1320 | -0.1469 | NVIDIA L40S |
| r35_C_n00_s1001_L_NCE | 82.35 | 95.66 | 97.63 | 64.18 | 85.07 | 90.16 | 515.05 | -7.02 | 515.05 | 515.05 | -1.0000 | -0.1708 | NVIDIA L40S |
| r35_C_n60_s1001_G_J | 0.20 | 1.78 | 4.24 | 0.00 | 0.02 | 0.12 | 6.35 | -498.86 | 6.35 | 6.35 | -0.8645 | -0.5792 | NVIDIA RTX PRO 6000 Blackwell Server Edition |
| r35_C_n60_s1001_G_BCE | 20.91 | 30.08 | 35.50 | 1.38 | 2.17 | 2.90 | 92.94 | -412.27 | 92.94 | 92.94 | -0.4372 | -0.4596 | NVIDIA RTX PRO 6000 Blackwell Server Edition |
| r35_C_n60_s1001_G_NCE | 35.60 | 46.94 | 52.66 | 4.24 | 6.45 | 7.69 | 153.59 | -351.62 | 153.59 | 153.59 | -1.0000 | -0.2750 | NVIDIA RTX PRO 6000 Blackwell Server Edition |
| r35_C_n60_s1001_L_J | 0.00 | 1.18 | 4.24 | 0.00 | 0.02 | 0.14 | 5.58 | -499.63 | 5.58 | 5.58 | -0.8946 | -0.6245 | NVIDIA L40S |
| r35_C_n60_s1001_L_BCE | 8.58 | 18.05 | 24.46 | 0.30 | 0.53 | 0.91 | 52.82 | -452.39 | 52.82 | 52.82 | -0.5522 | -0.5924 | NVIDIA RTX PRO 6000 Blackwell Server Edition |
| r35_C_n60_s1001_L_NCE | 23.96 | 38.07 | 47.04 | 2.39 | 4.38 | 5.84 | 121.68 | -383.53 | 121.68 | 121.68 | -1.0000 | -0.3414 | NVIDIA RTX PRO 6000 Blackwell Server Edition |
| r35_D_n00_s1001_L_J | 83.63 | 96.15 | 98.22 | 66.27 | 87.91 | 92.45 | 524.64 | 2.56 | 524.64 | 524.64 | -0.1912 | -0.1418 | NVIDIA RTX PRO 6000 Blackwell Server Edition |
| r35_D_n00_s1001_L_BCE | 80.87 | 96.25 | 97.83 | 65.48 | 87.81 | 92.47 | 520.71 | -1.36 | 520.71 | 520.71 | -0.1372 | -0.1619 | NVIDIA L40S |
| r35_D_n00_s1001_L_NCE | 62.13 | 85.60 | 91.42 | 40.41 | 58.30 | 66.53 | 404.40 | -117.67 | 404.40 | 404.40 | -1.0000 | -0.1995 | NVIDIA L40S |
| r35_D_n60_s1001_L_J | 0.30 | 2.27 | 6.31 | 0.00 | 0.04 | 0.14 | 9.05 | -496.15 | 9.05 | 9.05 | -0.8541 | -0.6128 | NVIDIA L40S |
| r35_D_n60_s1001_L_BCE | 9.57 | 20.32 | 28.60 | 0.24 | 0.53 | 1.03 | 60.28 | -444.93 | 60.28 | 60.28 | -0.5436 | -0.5874 | NVIDIA RTX PRO 6000 Blackwell Server Edition |
| r35_D_n60_s1001_L_NCE | 20.51 | 34.91 | 45.27 | 1.66 | 3.57 | 5.11 | 111.03 | -394.18 | 111.03 | 111.03 | -1.0000 | -0.3951 | NVIDIA RTX PRO 6000 Blackwell Server Edition |
| r35_D_n00_s1001_L_J_HEAD_CONTROL | 83.83 | 96.15 | 98.13 | 66.19 | 87.87 | 92.45 | 524.62 | 2.54 | 524.62 | 524.62 | -0.1936 | -0.1394 | NVIDIA RTX PRO 6000 Blackwell Server Edition |
| r35_D_n00_s1001_L_BCE_HEAD_CONTROL | 81.95 | 96.25 | 97.83 | 65.74 | 87.87 | 92.52 | 522.17 | 0.10 | 522.17 | 522.17 | -0.1329 | -0.1570 | NVIDIA L40S |
| r35_D_n00_s1001_L_NCE_HEAD_CONTROL | 62.13 | 85.01 | 91.81 | 40.32 | 58.40 | 66.71 | 404.38 | -117.69 | 404.38 | 404.38 | -1.0000 | -0.1944 | NVIDIA L40S |
| r35_D_n60_s1001_L_J_HEAD_CONTROL | 0.30 | 2.27 | 6.51 | 0.00 | 0.06 | 0.14 | 9.27 | -495.94 | 9.27 | 9.27 | -0.8387 | -0.6052 | NVIDIA L40S |
| r35_D_n60_s1001_L_BCE_HEAD_CONTROL | 9.27 | 20.32 | 28.60 | 0.24 | 0.53 | 1.03 | 59.98 | -445.23 | 59.98 | 59.98 | -0.5440 | -0.5860 | NVIDIA RTX PRO 6000 Blackwell Server Edition |
| r35_D_n60_s1001_L_NCE_HEAD_CONTROL | 21.20 | 35.01 | 45.56 | 1.74 | 3.59 | 5.25 | 112.35 | -392.86 | 112.35 | 112.35 | -1.0000 | -0.3868 | NVIDIA RTX PRO 6000 Blackwell Server Edition |

Frozen H native DEV rSum: n00 522.07, n60 505.21

J readouts: A/B record U-statistic on the full DEV matrix (peer-mean score); C/D candidate-list J on DEV lists (clean annotation, 32 slots). The two objects differ and are not comparable.

## Candidate coverage (DEV, frozen-H top-100)

| unit | cov i2t | cov t2i | cond R@1 i2t | cond R@1 t2i |
|---|---|---|---|---|
| r35_C_n00_s1001_G_J | 0.9990 | 0.9878 | 82.92 | 66.85 |
| r35_C_n00_s1001_G_BCE | 0.9990 | 0.9878 | 82.92 | 66.63 |
| r35_C_n00_s1001_G_NCE | 0.9990 | 0.9878 | 82.33 | 63.60 |
| r35_C_n00_s1001_L_J | 0.9990 | 0.9878 | 83.12 | 66.91 |
| r35_C_n00_s1001_L_BCE | 0.9990 | 0.9878 | 82.92 | 66.67 |
| r35_C_n00_s1001_L_NCE | 0.9990 | 0.9878 | 82.43 | 64.98 |
| r35_C_n60_s1001_G_J | 0.9980 | 0.9840 | 0.20 | 0.00 |
| r35_C_n60_s1001_G_BCE | 0.9980 | 0.9840 | 20.95 | 1.40 |
| r35_C_n60_s1001_G_NCE | 0.9980 | 0.9840 | 35.67 | 4.31 |
| r35_C_n60_s1001_L_J | 0.9980 | 0.9840 | 0.00 | 0.00 |
| r35_C_n60_s1001_L_BCE | 0.9980 | 0.9840 | 8.60 | 0.30 |
| r35_C_n60_s1001_L_NCE | 0.9980 | 0.9840 | 24.01 | 2.43 |
| r35_D_n00_s1001_L_J | 0.9990 | 0.9878 | 83.71 | 67.09 |
| r35_D_n00_s1001_L_BCE | 0.9990 | 0.9878 | 80.95 | 66.29 |
| r35_D_n00_s1001_L_NCE | 0.9990 | 0.9878 | 62.19 | 40.91 |
| r35_D_n60_s1001_L_J | 0.9980 | 0.9840 | 0.30 | 0.00 |
| r35_D_n60_s1001_L_BCE | 0.9980 | 0.9840 | 9.58 | 0.24 |
| r35_D_n60_s1001_L_NCE | 0.9980 | 0.9840 | 20.55 | 1.68 |
| r35_D_n00_s1001_L_J_HEAD_CONTROL | 0.9990 | 0.9878 | 83.91 | 67.01 |
| r35_D_n00_s1001_L_BCE_HEAD_CONTROL | 0.9990 | 0.9878 | 82.03 | 66.55 |
| r35_D_n00_s1001_L_NCE_HEAD_CONTROL | 0.9990 | 0.9878 | 62.19 | 40.81 |
| r35_D_n60_s1001_L_J_HEAD_CONTROL | 0.9980 | 0.9840 | 0.30 | 0.00 |
| r35_D_n60_s1001_L_BCE_HEAD_CONTROL | 0.9980 | 0.9840 | 9.29 | 0.24 |
| r35_D_n60_s1001_L_NCE_HEAD_CONTROL | 0.9980 | 0.9840 | 21.25 | 1.76 |

## Cost (formal logs)

| unit | GPU | updates | train s | eval s | s/update | peak GB |
|---|---|---|---|---|---|---|
| r35_A_n00_s1001_ID_J | NVIDIA L40S | 8000 | 34 | 8 | 0.0043 | 2.0 |
| r35_A_n00_s1001_ID_BCE | NVIDIA L40S | 8000 | 30 | 7 | 0.0037 | 2.0 |
| r35_A_n00_s1001_ID_NCE | NVIDIA L40S | 8000 | 31 | 7 | 0.0039 | 2.0 |
| r35_A_n00_s1001_G_J | NVIDIA RTX PRO 6000 Blackwell Server Edition | 8000 | 51 | 16 | 0.0064 | 2.9 |
| r35_A_n00_s1001_G_BCE | NVIDIA RTX PRO 6000 Blackwell Server Edition | 8000 | 45 | 12 | 0.0057 | 2.9 |
| r35_A_n00_s1001_G_NCE | NVIDIA RTX PRO 6000 Blackwell Server Edition | 8000 | 46 | 12 | 0.0057 | 2.9 |
| r35_A_n60_s1001_ID_J | NVIDIA RTX PRO 6000 Blackwell Server Edition | 8000 | 25 | 9 | 0.0032 | 2.0 |
| r35_A_n60_s1001_ID_BCE | NVIDIA RTX PRO 6000 Blackwell Server Edition | 8000 | 20 | 4 | 0.0025 | 2.0 |
| r35_A_n60_s1001_ID_NCE | NVIDIA RTX PRO 6000 Blackwell Server Edition | 8000 | 20 | 4 | 0.0026 | 2.0 |
| r35_A_n60_s1001_G_J | NVIDIA RTX PRO 6000 Blackwell Server Edition | 8000 | 53 | 16 | 0.0066 | 2.9 |
| r35_A_n60_s1001_G_BCE | NVIDIA RTX PRO 6000 Blackwell Server Edition | 8000 | 45 | 12 | 0.0057 | 2.9 |
| r35_A_n60_s1001_G_NCE | NVIDIA RTX PRO 6000 Blackwell Server Edition | 8000 | 46 | 12 | 0.0058 | 2.9 |
| r35_B_n60_s1001_TRUST_J | NVIDIA RTX PRO 6000 Blackwell Server Edition | 8000 | 53 | 15 | 0.0066 | 2.9 |
| r35_B_n60_s1001_TRUST_BCE | NVIDIA RTX PRO 6000 Blackwell Server Edition | 8000 | 46 | 12 | 0.0058 | 2.9 |
| r35_B_n60_s1001_TRUST_NCE | NVIDIA RTX PRO 6000 Blackwell Server Edition | 8000 | 46 | 12 | 0.0058 | 2.9 |
| r35_B_n60_s1001_ORACLE_J | NVIDIA RTX PRO 6000 Blackwell Server Edition | 8000 | 49 | 13 | 0.0061 | 2.9 |
| r35_B_n60_s1001_ORACLE_BCE | NVIDIA RTX PRO 6000 Blackwell Server Edition | 8000 | 46 | 12 | 0.0057 | 2.9 |
| r35_B_n60_s1001_ORACLE_NCE | NVIDIA RTX PRO 6000 Blackwell Server Edition | 8000 | 47 | 12 | 0.0058 | 2.9 |
| r35_C_n00_s1001_G_J | NVIDIA RTX PRO 6000 Blackwell Server Edition | 8000 | 34 | 6 | 0.0042 | 1.8 |
| r35_C_n00_s1001_G_BCE | NVIDIA RTX PRO 6000 Blackwell Server Edition | 8000 | 31 | 5 | 0.0038 | 1.8 |
| r35_C_n00_s1001_G_NCE | NVIDIA RTX PRO 6000 Blackwell Server Edition | 8000 | 30 | 5 | 0.0037 | 1.8 |
| r35_C_n00_s1001_L_J | NVIDIA RTX PRO 6000 Blackwell Server Edition | 8000 | 202 | 35 | 0.0253 | 14.8 |
| r35_C_n00_s1001_L_BCE | NVIDIA L40S | 8000 | 374 | 57 | 0.0467 | 14.8 |
| r35_C_n00_s1001_L_NCE | NVIDIA L40S | 8000 | 369 | 64 | 0.0461 | 14.8 |
| r35_C_n60_s1001_G_J | NVIDIA RTX PRO 6000 Blackwell Server Edition | 8000 | 27 | 11 | 0.0034 | 1.8 |
| r35_C_n60_s1001_G_BCE | NVIDIA RTX PRO 6000 Blackwell Server Edition | 8000 | 20 | 4 | 0.0026 | 1.8 |
| r35_C_n60_s1001_G_NCE | NVIDIA RTX PRO 6000 Blackwell Server Edition | 8000 | 20 | 4 | 0.0025 | 1.8 |
| r35_C_n60_s1001_L_J | NVIDIA L40S | 8000 | 361 | 65 | 0.0451 | 14.7 |
| r35_C_n60_s1001_L_BCE | NVIDIA RTX PRO 6000 Blackwell Server Edition | 8000 | 203 | 41 | 0.0253 | 14.7 |
| r35_C_n60_s1001_L_NCE | NVIDIA RTX PRO 6000 Blackwell Server Edition | 8000 | 205 | 41 | 0.0256 | 14.7 |
| r35_D_n00_s1001_L_J | NVIDIA RTX PRO 6000 Blackwell Server Edition | 4000 | 306 | 15 | 0.0766 | 15.3 |
| r35_D_n00_s1001_L_BCE | NVIDIA L40S | 4000 | 623 | 30 | 0.1557 | 15.3 |
| r35_D_n00_s1001_L_NCE | NVIDIA L40S | 4000 | 619 | 30 | 0.1547 | 15.3 |
| r35_D_n60_s1001_L_J | NVIDIA L40S | 4000 | 628 | 30 | 0.1569 | 15.2 |
| r35_D_n60_s1001_L_BCE | NVIDIA RTX PRO 6000 Blackwell Server Edition | 4000 | 310 | 14 | 0.0775 | 15.2 |
| r35_D_n60_s1001_L_NCE | NVIDIA RTX PRO 6000 Blackwell Server Edition | 4000 | 309 | 15 | 0.0773 | 15.2 |
| r35_D_n00_s1001_L_J_HEAD_CONTROL | NVIDIA RTX PRO 6000 Blackwell Server Edition | 4000 | 102 | 15 | 0.0254 | 15.3 |
| r35_D_n00_s1001_L_BCE_HEAD_CONTROL | NVIDIA L40S | 4000 | 188 | 30 | 0.0470 | 15.3 |
| r35_D_n00_s1001_L_NCE_HEAD_CONTROL | NVIDIA L40S | 4000 | 184 | 30 | 0.0460 | 15.3 |
| r35_D_n60_s1001_L_J_HEAD_CONTROL | NVIDIA L40S | 4000 | 182 | 30 | 0.0454 | 15.2 |
| r35_D_n60_s1001_L_BCE_HEAD_CONTROL | NVIDIA RTX PRO 6000 Blackwell Server Edition | 4000 | 102 | 14 | 0.0255 | 15.2 |
| r35_D_n60_s1001_L_NCE_HEAD_CONTROL | NVIDIA RTX PRO 6000 Blackwell Server Edition | 4000 | 101 | 15 | 0.0254 | 15.2 |

Failures / retries: 0 (FAILURES.csv). Other tables: CURVES.csv, CALIBRATION_RAW_VS_AFFINE.csv (incl. R34 §6.1), RANK_TRANSITIONS.csv (incl. R34 §6.3), CANDIDATE_COVERAGE.csv, COST_LEDGER.csv.
