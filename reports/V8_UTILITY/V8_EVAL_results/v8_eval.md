# V8 evaluation (V8-03 `evaluation:`) — generated from v8_eval.json (2026-10-15T15:36:13Z)

Units: 24 formal (all present).  Scorer `v7_score.py` sha256 e86af47e9c204f4e; external files = V7X-50 v3 (methods.json 548c5293b347e367).  Label file sha256 = seal (0eb955b45429f5c2).

## primary: U-CS vs B0_FGA_released_O, B1_VQAScore, B4T_TIFA, B4D_DSG (N = 9, official first-index ties; prompt-cluster bootstrap 5,000 seed 2718; Holm over 8)

| contrast | Δ selected human score [95 %] | p Holm | per seed |
|---|---|---|---|
| U-CS - B0_FGA_released_O @ DALLE_3 N9 | -0.025 [-0.055, +0.006] | 0.247 | -0.015 / -0.039 / -0.021 |
| U-CS - B0_FGA_released_O @ SDXL_Base N9 | +0.039 [+0.010, +0.069] | 0.046 | +0.045 / +0.029 / +0.044 |
| U-CS - B1_VQAScore @ DALLE_3 N9 | -0.121 [-0.168, -0.072] | 0.003 | -0.111 / -0.135 / -0.117 |
| U-CS - B1_VQAScore @ SDXL_Base N9 | -0.046 [-0.088, -0.003] | 0.146 | -0.040 / -0.057 / -0.041 |
| U-CS - B4T_TIFA @ DALLE_3 N9 | +0.043 [-0.005, +0.092] | 0.247 | +0.053 / +0.029 / +0.047 |
| U-CS - B4T_TIFA @ SDXL_Base N9 | +0.130 [+0.084, +0.177] | 0.003 | +0.136 / +0.119 / +0.135 |
| U-CS - B4D_DSG @ DALLE_3 N9 | +0.026 [-0.021, +0.073] | 0.297 | +0.035 / +0.012 / +0.030 |
| U-CS - B4D_DSG @ SDXL_Base N9 | +0.147 [+0.103, +0.190] | 0.003 | +0.153 / +0.136 / +0.152 |

| method | generator | N1 selected | N3 selected | N9 selected | N9 regret |
|---|---|---|---|---|---|
| U-CS | DALLE_3 | 4.031 | 4.103 | 4.138 | 0.476 |
| U-CS | SDXL_Base | 3.295 | 3.435 | 3.516 | 0.437 |
| B0_FGA_released_O | DALLE_3 | 4.031 | 4.119 | 4.163 | 0.451 |
| B0_FGA_released_O | SDXL_Base | 3.295 | 3.424 | 3.476 | 0.477 |
| B1_VQAScore | DALLE_3 | 4.031 | 4.154 | 4.259 | 0.355 |
| B1_VQAScore | SDXL_Base | 3.295 | 3.433 | 3.562 | 0.391 |
| B4T_TIFA | DALLE_3 | 4.031 | 4.061 | 4.095 | 0.519 |
| B4T_TIFA | SDXL_Base | 3.295 | 3.352 | 3.386 | 0.567 |
| B4D_DSG | DALLE_3 | 4.031 | 4.062 | 4.112 | 0.502 |
| B4D_DSG | SDXL_Base | 3.295 | 3.340 | 3.369 | 0.584 |

## same_conditions: U-CS vs U-RPS, U-MSE, U-RANK, U-LIST (N = 9, official first-index ties; prompt-cluster bootstrap 5,000 seed 2718; Holm over 8)

| contrast | Δ selected human score [95 %] | p Holm | per seed |
|---|---|---|---|
| U-CS - U-RPS @ DALLE_3 N9 | -0.011 [-0.030, +0.009] | 1.000 | -0.018 / -0.011 / -0.003 |
| U-CS - U-RPS @ SDXL_Base N9 | +0.003 [-0.013, +0.020] | 1.000 | +0.012 / +0.010 / -0.012 |
| U-CS - U-MSE @ DALLE_3 N9 | -0.019 [-0.037, -0.001] | 0.317 | -0.023 / -0.008 / -0.025 |
| U-CS - U-MSE @ SDXL_Base N9 | -0.002 [-0.017, +0.014] | 1.000 | -0.010 / +0.012 / -0.009 |
| U-CS - U-RANK @ DALLE_3 N9 | +0.007 [-0.017, +0.033] | 1.000 | +0.014 / -0.009 / +0.016 |
| U-CS - U-RANK @ SDXL_Base N9 | +0.010 [-0.012, +0.033] | 1.000 | +0.017 / +0.010 / +0.003 |
| U-CS - U-LIST @ DALLE_3 N9 | -0.014 [-0.037, +0.010] | 1.000 | -0.008 / -0.020 / -0.014 |
| U-CS - U-LIST @ SDXL_Base N9 | -0.018 [-0.039, +0.002] | 0.551 | -0.002 / -0.015 / -0.038 |

| method | generator | N1 selected | N3 selected | N9 selected | N9 regret |
|---|---|---|---|---|---|
| U-CS | DALLE_3 | 4.031 | 4.103 | 4.138 | 0.476 |
| U-CS | SDXL_Base | 3.295 | 3.435 | 3.516 | 0.437 |
| U-RPS | DALLE_3 | 4.031 | 4.114 | 4.148 | 0.466 |
| U-RPS | SDXL_Base | 3.295 | 3.434 | 3.512 | 0.440 |
| U-MSE | DALLE_3 | 4.031 | 4.114 | 4.156 | 0.457 |
| U-MSE | SDXL_Base | 3.295 | 3.438 | 3.518 | 0.435 |
| U-RANK | DALLE_3 | 4.031 | 4.109 | 4.131 | 0.483 |
| U-RANK | SDXL_Base | 3.295 | 3.431 | 3.505 | 0.448 |
| U-LIST | DALLE_3 | 4.031 | 4.115 | 4.151 | 0.462 |
| U-LIST | SDXL_Base | 3.295 | 3.440 | 3.534 | 0.419 |

## attribution: U-CS vs U-BCE (N = 9, official first-index ties; prompt-cluster bootstrap 5,000 seed 2718; Holm over 2)

| contrast | Δ selected human score [95 %] | p Holm | per seed |
|---|---|---|---|
| U-CS - U-BCE @ DALLE_3 N9 | -0.011 [-0.028, +0.007] | 0.480 | +0.002 / -0.031 / -0.003 |
| U-CS - U-BCE @ SDXL_Base N9 | +0.002 [-0.011, +0.016] | 0.753 | -0.010 / +0.010 / +0.007 |

| method | generator | N1 selected | N3 selected | N9 selected | N9 regret |
|---|---|---|---|---|---|
| U-CS | DALLE_3 | 4.031 | 4.103 | 4.138 | 0.476 |
| U-CS | SDXL_Base | 3.295 | 3.435 | 3.516 | 0.437 |
| U-BCE | DALLE_3 | 4.031 | 4.116 | 4.148 | 0.466 |
| U-BCE | SDXL_Base | 3.295 | 3.430 | 3.513 | 0.439 |

## full_rows: U-CS vs U-CE, U-L1 (N = 9, official first-index ties; prompt-cluster bootstrap 5,000 seed 2718; Holm over 4)

| contrast | Δ selected human score [95 %] | p Holm | per seed |
|---|---|---|---|
| U-CS - U-CE @ DALLE_3 N9 | -0.004 [-0.024, +0.016] | 1.000 | -0.009 / -0.030 / +0.026 |
| U-CS - U-CE @ SDXL_Base N9 | +0.005 [-0.013, +0.023] | 1.000 | -0.006 / +0.002 / +0.018 |
| U-CS - U-L1 @ DALLE_3 N9 | -0.007 [-0.024, +0.011] | 1.000 | -0.005 / -0.003 / -0.013 |
| U-CS - U-L1 @ SDXL_Base N9 | +0.009 [-0.007, +0.025] | 1.000 | +0.027 / +0.017 / -0.017 |

| method | generator | N1 selected | N3 selected | N9 selected | N9 regret |
|---|---|---|---|---|---|
| U-CS | DALLE_3 | 4.031 | 4.103 | 4.138 | 0.476 |
| U-CS | SDXL_Base | 3.295 | 3.435 | 3.516 | 0.437 |
| U-CE | DALLE_3 | 4.031 | 4.113 | 4.142 | 0.472 |
| U-CE | SDXL_Base | 3.295 | 3.431 | 3.511 | 0.442 |
| U-L1 | DALLE_3 | 4.031 | 4.111 | 4.144 | 0.469 |
| U-L1 | SDXL_Base | 3.295 | 3.436 | 3.507 | 0.446 |

## sensitivity subset_primary (V7X-50 v2 common-defined subset, 41 prompts removed; not pre-specified in V8-03)

| contrast | Δ [95 %] | p Holm | per seed |
|---|---|---|---|
| U-CS - B0_FGA_released_O @ DALLE_3 N9 | -0.027 [-0.059, +0.004] | 0.257 | -0.018 / -0.043 / -0.022 |
| U-CS - B0_FGA_released_O @ SDXL_Base N9 | +0.039 [+0.009, +0.070] | 0.048 | +0.043 / +0.032 / +0.042 |
| U-CS - B1_VQAScore @ DALLE_3 N9 | -0.121 [-0.169, -0.072] | 0.003 | -0.111 / -0.136 / -0.116 |
| U-CS - B1_VQAScore @ SDXL_Base N9 | -0.049 [-0.092, -0.004] | 0.133 | -0.045 / -0.055 / -0.046 |
| U-CS - B4T_TIFA @ DALLE_3 N9 | +0.039 [-0.012, +0.090] | 0.257 | +0.049 / +0.024 / +0.045 |
| U-CS - B4T_TIFA @ SDXL_Base N9 | +0.128 [+0.079, +0.175] | 0.003 | +0.132 / +0.121 / +0.131 |
| U-CS - B4D_DSG @ DALLE_3 N9 | +0.028 [-0.022, +0.078] | 0.257 | +0.038 / +0.013 / +0.033 |
| U-CS - B4D_DSG @ SDXL_Base N9 | +0.147 [+0.101, +0.194] | 0.003 | +0.151 / +0.140 / +0.150 |

## sensitivity subset_same_conditions (V7X-50 v2 common-defined subset, 41 prompts removed; not pre-specified in V8-03)

| contrast | Δ [95 %] | p Holm | per seed |
|---|---|---|---|
| U-CS - U-RPS @ DALLE_3 N9 | -0.013 [-0.033, +0.007] | 1.000 | -0.022 / -0.014 / -0.003 |
| U-CS - U-RPS @ SDXL_Base N9 | +0.003 [-0.014, +0.020] | 1.000 | +0.011 / +0.013 / -0.014 |
| U-CS - U-MSE @ DALLE_3 N9 | -0.020 [-0.039, -0.001] | 0.323 | -0.025 / -0.009 / -0.026 |
| U-CS - U-MSE @ SDXL_Base N9 | -0.003 [-0.020, +0.013] | 1.000 | -0.011 / +0.014 / -0.011 |
| U-CS - U-RANK @ DALLE_3 N9 | +0.008 [-0.018, +0.033] | 1.000 | +0.014 / -0.011 / +0.020 |
| U-CS - U-RANK @ SDXL_Base N9 | +0.010 [-0.014, +0.033] | 1.000 | +0.014 / +0.014 / -0.000 |
| U-CS - U-LIST @ DALLE_3 N9 | -0.016 [-0.041, +0.009] | 1.000 | -0.010 / -0.024 / -0.015 |
| U-CS - U-LIST @ SDXL_Base N9 | -0.018 [-0.040, +0.004] | 0.677 | -0.003 / -0.011 / -0.041 |

## lr rule and selected epochs

| arm | lr (rule) | tuning CAL_SELECT MSE | lr edge | formal selected epochs s0/s1/s2 | formal CAL_SELECT MSE s0/s1/s2 |
|---|---|---|---|---|---|
| U-CS | 1e-05 | 0.3974 | None | 2/0/2 | 0.4141/0.4059/0.3966 |
| U-BCE | 1e-05 | 0.3964 | None | 2/1/2 | 0.3978/0.4075/0.4044 |
| U-RPS | 5e-06 | 0.3932 | low | 2/0/4 | 0.3971/0.3987/0.4092 |
| U-CE | 5e-06 | 0.3867 | low | 2/1/0 | 0.3942/0.4010/0.4031 |
| U-MSE | 5e-06 | 0.3974 | low | 2/0/2 | 0.4028/0.4021/0.4164 |
| U-L1 | 1e-05 | 0.4002 | None | 3/0/3 | 0.4004/0.4011/0.4012 |
| U-RANK | 5e-06 | 0.5798 | low | 0/0/0 | 0.5178/0.5787/0.6038 |
| U-LIST | 5e-06 | 0.4242 | low | 2/1/2 | 0.4211/0.4226/0.4311 |

## distribution readouts on CAL_PROB (natural weights; mean over 3 seeds, raw outputs | after the CAL_PROB temperature)

| arm | Brier k=2/3/4/5 | RPS | hist log loss | mean MSE macro | mean MAE macro | mean MSE natural | RPS (cal) | log loss (cal) | MSE macro (cal) |
|---|---|---|---|---|---|---|---|---|---|
| U-CS | 0.0706/0.1576/0.1442/0.0834 | 0.1140 | 1.2181 | 0.4015 | 0.4975 | 0.4015 | 0.1135 | 1.2096 | 0.4014 |
| U-BCE | 0.0696/0.1575/0.1446/0.0835 | 0.1138 | 1.2149 | 0.4021 | 0.4969 | 0.4021 | 0.1130 | 1.2014 | 0.3985 |
| U-RPS | 0.0706/0.1606/0.1463/0.0837 | 0.1153 | 1.2320 | 0.4160 | 0.5068 | 0.4160 | 0.1145 | 1.2226 | 0.4126 |
| U-CE | 0.0694/0.1607/0.1453/0.0837 | 0.1148 | 1.2220 | 0.4167 | 0.5041 | 0.4167 | 0.1141 | 1.2124 | 0.4129 |
| U-MSE | 0.0809/0.1645/0.1505/0.0910 | 0.1217 | 1.3482 | 0.4171 | 0.5093 | 0.4171 | 0.1219 | 1.3477 | 0.4190 |
| U-L1 | 0.0739/0.1588/0.1504/0.0845 | 0.1169 | 1.2828 | 0.4072 | 0.4960 | 0.4072 | 0.1166 | 1.2805 | 0.4045 |
| U-RANK | 0.1287/0.1713/0.1708/0.1263 | 0.1493 | 1.6513 | 0.5434 | 0.5829 | 0.5434 | 0.1395 | 1.5444 | 0.5235 |
| U-LIST | 0.0963/0.1687/0.1568/0.0968 | 0.1296 | 1.4664 | 0.4256 | 0.5137 | 0.4256 | 0.1290 | 1.4562 | 0.4298 |

## distribution readouts on DEV (natural weights; mean over 3 seeds, raw outputs | after the CAL_PROB temperature)

| arm | Brier k=2/3/4/5 | RPS | hist log loss | mean MSE macro | mean MAE macro | mean MSE natural | RPS (cal) | log loss (cal) | MSE macro (cal) |
|---|---|---|---|---|---|---|---|---|---|
| U-CS | 0.0710/0.1489/0.1496/0.0995 | 0.1172 | 1.2348 | 0.4212 | 0.5019 | 0.4212 | 0.1169 | 1.2244 | 0.4218 |
| U-BCE | 0.0699/0.1492/0.1496/0.0988 | 0.1169 | 1.2342 | 0.4178 | 0.4974 | 0.4178 | 0.1162 | 1.2172 | 0.4161 |
| U-RPS | 0.0710/0.1512/0.1507/0.0992 | 0.1180 | 1.2438 | 0.4282 | 0.5047 | 0.4282 | 0.1173 | 1.2319 | 0.4258 |
| U-CE | 0.0702/0.1518/0.1509/0.0992 | 0.1180 | 1.2373 | 0.4322 | 0.5049 | 0.4322 | 0.1173 | 1.2252 | 0.4288 |
| U-MSE | 0.0792/0.1538/0.1554/0.1053 | 0.1234 | 1.3406 | 0.4273 | 0.5055 | 0.4273 | 0.1236 | 1.3407 | 0.4292 |
| U-L1 | 0.0738/0.1510/0.1576/0.1011 | 0.1209 | 1.3009 | 0.4313 | 0.5010 | 0.4313 | 0.1204 | 1.2950 | 0.4288 |
| U-RANK | 0.1213/0.1604/0.1761/0.1444 | 0.1505 | 1.6327 | 0.5633 | 0.5923 | 0.5633 | 0.1419 | 1.5278 | 0.5641 |
| U-LIST | 0.0930/0.1554/0.1605/0.1125 | 0.1304 | 1.4521 | 0.4252 | 0.5025 | 0.4252 | 0.1297 | 1.4409 | 0.4327 |

## Table C kernel bridges (distribution readouts; raw | PAVA tails; log loss only where the tails are monotone)

| route | role | Brier k=2/3/4/5 (raw) | RPS raw | RPS pava | log loss pava | mean MSE macro raw | mean MSE macro pava |
|---|---|---|---|---|---|---|---|
| pca0/K-KDE | CAL_PROB | 0.0803/0.2114/0.1821/0.0907 | 0.1411 | 0.1411 | 1.4047 | 0.7228 | 0.7228 |
| pca0/K-KDE | DEV | 0.0813/0.2024/0.1929/0.1115 | 0.1470 | 0.1470 | 1.4186 | 0.7889 | 0.7889 |
| pca0/K-NYSTROM | CAL_PROB | 0.0739/0.1718/0.1522/0.0847 | 0.1206 | 0.1206 | 1.2874 | 0.4859 | 0.4859 |
| pca0/K-NYSTROM | DEV | 0.0734/0.1644/0.1636/0.1040 | 0.1263 | 0.1263 | 1.3391 | 0.5345 | 0.5345 |
| pca0/K-RULSIF | CAL_PROB | 0.0753/0.1720/0.1535/0.0847 | 0.1214 | 0.1213 | 1.3963 | 0.4840 | 0.4840 |
| pca0/K-RULSIF | DEV | 0.0738/0.1655/0.1651/0.1053 | 0.1274 | 0.1274 | 1.4867 | 0.5333 | 0.5333 |
| pca256/K-KDE | CAL_PROB | 0.0868/0.2450/0.2010/0.0934 | 0.1565 | 0.1565 | 1.4931 | 0.9233 | 0.9233 |
| pca256/K-KDE | DEV | 0.0926/0.2421/0.2165/0.1168 | 0.1670 | 0.1670 | 1.5337 | 1.0502 | 1.0502 |
| pca256/K-NYSTROM | CAL_PROB | 0.0747/0.1745/0.1561/0.0856 | 0.1227 | 0.1227 | 1.4347 | 0.5036 | 0.5036 |
| pca256/K-NYSTROM | DEV | 0.0758/0.1659/0.1632/0.1041 | 0.1273 | 0.1273 | 1.4241 | 0.5380 | 0.5380 |
| pca256/K-RULSIF | CAL_PROB | 0.0744/0.1750/0.1565/0.0857 | 0.1229 | 0.1229 | 1.4578 | 0.5097 | 0.5097 |
| pca256/K-RULSIF | DEV | 0.0744/0.1675/0.1651/0.1046 | 0.1279 | 0.1279 | 1.4201 | 0.5495 | 0.5495 |
| pca64/K-KDE | CAL_PROB | 0.0805/0.2242/0.1909/0.0924 | 0.1470 | 0.1470 | 1.4408 | 0.7972 | 0.7972 |
| pca64/K-KDE | DEV | 0.0843/0.2155/0.1999/0.1128 | 0.1531 | 0.1531 | 1.4553 | 0.8697 | 0.8697 |
| pca64/K-NYSTROM | CAL_PROB | 0.0747/0.1814/0.1572/0.0865 | 0.1249 | 0.1249 | 1.3809 | 0.5406 | 0.5406 |
| pca64/K-NYSTROM | DEV | 0.0763/0.1745/0.1723/0.1067 | 0.1325 | 0.1325 | 1.3871 | 0.6114 | 0.6114 |
| pca64/K-RULSIF | CAL_PROB | 0.0766/0.1862/0.1580/0.0879 | 0.1272 | 0.1271 | 1.5335 | 0.5553 | 0.5553 |
| pca64/K-RULSIF | DEV | 0.0768/0.1768/0.1745/0.1083 | 0.1341 | 0.1341 | 1.4642 | 0.6205 | 0.6205 |
