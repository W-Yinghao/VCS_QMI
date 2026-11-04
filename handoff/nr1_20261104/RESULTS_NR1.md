Publication copy (sample names aliased; numbers and conclusions unchanged).

# NR1_20261104 results (numbers only; DEV, top-100 rerank of base SigLIP2 candidates; TEST not read)

Model google/siglip2-base-patch16-224 @75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2 (FixRes 224), encoder frozen, protocol NR1_SAMPLEWISE_L2_V1 / NR1_IID_PRODUCT_PQ_V1 / NR1_BASE_TOP100_ALL_QUERY_V1, score dtype float64.

## Baseline (untrained base cosine, DEV 1014 images / 5070 captions)

| i2t_R1 | i2t_R5 | i2t_R10 | t2i_R1 | t2i_R5 | t2i_R10 | rSum | i2t top100 cov | t2i top100 cov |
|---|---|---|---|---|---|---|---|---|
| 90.93 | 98.52 | 99.21 | 78.88 | 94.02 | 96.88 | 558.44 | 0.9990 | 0.9980 |

## Fixed endpoint step 8000

| unit | i2t_R1 | i2t_R5 | i2t_R10 | t2i_R1 | t2i_R5 | t2i_R10 | rSum | ΔrSum | i2t resc/lost | t2i resc/lost |
|---|---|---|---|---|---|---|---|---|---|---|
| NR1_Flickr00_G_J_s1001 | 92.80 | 99.11 | 99.61 | 80.77 | 94.75 | 97.28 | 564.32 | +5.88 | 32/13 | 217/121 |
| NR1_Flickr00_G_BCE_s1001 | 92.70 | 99.21 | 99.61 | 80.83 | 94.62 | 97.34 | 564.30 | +5.86 | 34/16 | 216/117 |
| NR1_Flickr00_G_NCE_s1001 | 92.80 | 99.01 | 99.61 | 81.26 | 94.83 | 97.42 | 564.93 | +6.49 | 35/16 | 225/104 |
| NR1_Flickr00_GL_J_s1001 | 93.00 | 99.11 | 99.61 | 80.77 | 94.67 | 97.30 | 564.46 | +6.02 | 32/11 | 215/119 |
| NR1_Flickr00_GL_BCE_s1001 | 93.00 | 99.21 | 99.61 | 80.75 | 94.62 | 97.38 | 564.56 | +6.11 | 34/13 | 215/120 |
| NR1_Flickr00_GL_NCE_s1001 | 92.80 | 99.01 | 99.51 | 81.18 | 94.93 | 97.40 | 564.83 | +6.39 | 35/16 | 223/106 |
| NR1_Flickr60_G_J_s1001 | 89.64 | 97.53 | 98.92 | 77.99 | 93.14 | 96.25 | 553.47 | -4.97 | 34/47 | 194/239 |
| NR1_Flickr60_G_BCE_s1001 | 88.17 | 97.24 | 98.52 | 76.65 | 92.54 | 95.84 | 548.95 | -9.49 | 31/59 | 199/312 |
| NR1_Flickr60_G_NCE_s1001 | 91.32 | 98.52 | 99.31 | 79.49 | 94.32 | 96.82 | 559.78 | +1.34 | 34/30 | 215/184 |
| NR1_Flickr60_GL_J_s1001 | 90.04 | 97.63 | 99.01 | 77.99 | 93.16 | 96.21 | 554.04 | -4.40 | 36/45 | 189/234 |
| NR1_Flickr60_GL_BCE_s1001 | 88.66 | 97.53 | 99.01 | 76.88 | 92.70 | 95.88 | 550.67 | -7.77 | 33/56 | 198/299 |
| NR1_Flickr60_GL_NCE_s1001 | 91.52 | 98.62 | 99.41 | 79.70 | 94.28 | 96.86 | 560.39 | +1.95 | 34/28 | 220/178 |
| NR1_Flickr80_G_J_s1001 | 88.26 | 97.73 | 98.92 | 76.77 | 92.60 | 95.64 | 549.92 | -8.52 | 35/62 | 185/292 |
| NR1_Flickr80_G_BCE_s1001 | 87.08 | 97.14 | 98.72 | 75.35 | 92.21 | 95.46 | 545.96 | -12.49 | 35/74 | 196/375 |
| NR1_Flickr80_G_NCE_s1001 | 91.32 | 98.62 | 99.41 | 79.55 | 94.54 | 96.84 | 560.28 | +1.83 | 34/30 | 216/182 |
| NR1_Flickr80_GL_J_s1001 | 88.17 | 97.73 | 98.92 | 76.88 | 92.60 | 95.68 | 549.98 | -8.46 | 33/61 | 192/293 |
| NR1_Flickr80_GL_BCE_s1001 | 86.98 | 97.04 | 98.92 | 75.33 | 92.29 | 95.40 | 545.96 | -12.49 | 37/77 | 194/374 |
| NR1_Flickr80_GL_NCE_s1001 | 91.52 | 98.82 | 99.41 | 79.43 | 94.46 | 96.82 | 560.45 | +2.01 | 36/30 | 216/188 |

## rSum curves (steps 0/250/500/1000/2000/4000/8000)

| unit | 0 | 250 | 500 | 1000 | 2000 | 4000 | 8000 |
|---|---|---|---|---|---|---|---|
| NR1_Flickr00_G_J_s1001 | 558.44 | 558.48 | 561.32 | 563.41 | 563.55 | 563.85 | 564.32 |
| NR1_Flickr00_G_BCE_s1001 | 558.44 | 558.58 | 561.85 | 563.55 | 563.53 | 564.54 | 564.30 |
| NR1_Flickr00_G_NCE_s1001 | 558.44 | 560.83 | 563.41 | 564.12 | 564.34 | 564.20 | 564.93 |
| NR1_Flickr00_GL_J_s1001 | 558.44 | 558.58 | 561.36 | 563.41 | 563.51 | 563.87 | 564.46 |
| NR1_Flickr00_GL_BCE_s1001 | 558.44 | 558.56 | 561.85 | 563.51 | 563.43 | 564.52 | 564.56 |
| NR1_Flickr00_GL_NCE_s1001 | 558.44 | 560.87 | 563.63 | 564.04 | 564.36 | 564.28 | 564.83 |
| NR1_Flickr60_G_J_s1001 | 558.44 | 558.50 | 556.09 | 554.00 | 553.83 | 553.73 | 553.47 |
| NR1_Flickr60_G_BCE_s1001 | 558.44 | 556.37 | 554.87 | 551.68 | 551.03 | 550.43 | 548.95 |
| NR1_Flickr60_G_NCE_s1001 | 558.44 | 551.81 | 553.83 | 555.13 | 557.77 | 559.05 | 559.78 |
| NR1_Flickr60_GL_J_s1001 | 558.44 | 558.40 | 556.37 | 553.93 | 554.12 | 554.34 | 554.04 |
| NR1_Flickr60_GL_BCE_s1001 | 558.44 | 556.45 | 554.83 | 552.09 | 551.70 | 551.08 | 550.67 |
| NR1_Flickr60_GL_NCE_s1001 | 558.44 | 551.93 | 553.89 | 554.99 | 557.55 | 558.99 | 560.39 |
| NR1_Flickr80_G_J_s1001 | 558.44 | 556.04 | 554.04 | 551.34 | 550.39 | 551.14 | 549.92 |
| NR1_Flickr80_G_BCE_s1001 | 558.44 | 555.70 | 552.52 | 548.90 | 547.28 | 546.53 | 545.96 |
| NR1_Flickr80_G_NCE_s1001 | 558.44 | 548.66 | 550.55 | 551.52 | 555.84 | 558.70 | 560.28 |
| NR1_Flickr80_GL_J_s1001 | 558.44 | 556.02 | 554.18 | 551.56 | 550.49 | 550.45 | 549.98 |
| NR1_Flickr80_GL_BCE_s1001 | 558.44 | 555.66 | 552.54 | 548.76 | 547.18 | 546.84 | 545.96 |
| NR1_Flickr80_GL_NCE_s1001 | 558.44 | 548.82 | 550.67 | 551.48 | 555.76 | 558.50 | 560.45 |

## Unit status

| unit | status | attempts | step0 grad check | step0 order==base | train s | GPU | peak MiB | eval s/1000q |
|---|---|---|---|---|---|---|---|---|
| NR1_Flickr00_G_J_s1001 | COMPLETE | 1 | True | True | 69.0 | NVIDIA RTX PRO 6000 Blackwell Server Edition | 137 | 0.436 |
| NR1_Flickr00_G_BCE_s1001 | COMPLETE | 1 | True | True | 78.7 | NVIDIA RTX PRO 6000 Blackwell Server Edition | 137 | 0.433 |
| NR1_Flickr00_G_NCE_s1001 | COMPLETE | 1 | True | True | 184.9 | NVIDIA L40S | 137 | 0.691 |
| NR1_Flickr00_GL_J_s1001 | COMPLETE | 1 | True | True | 492.3 | NVIDIA L40S | 420 | 8.927 |
| NR1_Flickr00_GL_BCE_s1001 | COMPLETE | 1 | True | True | 188.5 | NVIDIA RTX PRO 6000 Blackwell Server Edition | 420 | 4.872 |
| NR1_Flickr00_GL_NCE_s1001 | COMPLETE | 1 | True | True | 193.0 | NVIDIA RTX PRO 6000 Blackwell Server Edition | 420 | 4.688 |
| NR1_Flickr60_G_J_s1001 | COMPLETE | 1 | True | True | 71.3 | NVIDIA RTX PRO 6000 Blackwell Server Edition | 137 | 0.43 |
| NR1_Flickr60_G_BCE_s1001 | COMPLETE | 1 | True | True | 68.4 | NVIDIA RTX PRO 6000 Blackwell Server Edition | 137 | 0.435 |
| NR1_Flickr60_G_NCE_s1001 | COMPLETE | 1 | True | True | 69.7 | NVIDIA RTX PRO 6000 Blackwell Server Edition | 137 | 0.43 |
| NR1_Flickr60_GL_J_s1001 | COMPLETE | 1 | True | True | 192.8 | NVIDIA RTX PRO 6000 Blackwell Server Edition | 420 | 4.749 |
| NR1_Flickr60_GL_BCE_s1001 | COMPLETE | 1 | True | True | 461.8 | NVIDIA L40S | 420 | 8.849 |
| NR1_Flickr60_GL_NCE_s1001 | COMPLETE | 1 | True | True | 193.3 | NVIDIA RTX PRO 6000 Blackwell Server Edition | 420 | 4.57 |
| NR1_Flickr80_G_J_s1001 | COMPLETE | 1 | True | True | 70.6 | NVIDIA RTX PRO 6000 Blackwell Server Edition | 137 | 0.427 |
| NR1_Flickr80_G_BCE_s1001 | COMPLETE | 1 | True | True | 67.6 | NVIDIA RTX PRO 6000 Blackwell Server Edition | 137 | 0.429 |
| NR1_Flickr80_G_NCE_s1001 | COMPLETE | 1 | True | True | 188.5 | NVIDIA L40S | 137 | 0.715 |
| NR1_Flickr80_GL_J_s1001 | COMPLETE | 1 | True | True | 190.3 | NVIDIA RTX PRO 6000 Blackwell Server Edition | 420 | 4.873 |
| NR1_Flickr80_GL_BCE_s1001 | COMPLETE | 1 | True | True | 342.5 | NVIDIA L40S | 420 | 6.737 |
| NR1_Flickr80_GL_NCE_s1001 | COMPLETE | 1 | True | True | 190.2 | NVIDIA RTX PRO 6000 Blackwell Server Edition | 420 | 4.556 |
| NR1_CC152K_OBS_G_J_s1001 | BLOCKED_DATA |  |  |  |  |  |  |  |
| NR1_CC152K_OBS_G_BCE_s1001 | BLOCKED_DATA |  |  |  |  |  |  |  |
| NR1_CC152K_OBS_G_NCE_s1001 | BLOCKED_DATA |  |  |  |  |  |  |  |
| NR1_CC152K_OBS_GL_J_s1001 | BLOCKED_DATA |  |  |  |  |  |  |  |
| NR1_CC152K_OBS_GL_BCE_s1001 | BLOCKED_DATA |  |  |  |  |  |  |  |
| NR1_CC152K_OBS_GL_NCE_s1001 | BLOCKED_DATA |  |  |  |  |  |  |  |
