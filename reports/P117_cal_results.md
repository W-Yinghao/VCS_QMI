# P117 — fixed-critic calibration (v5 NEXT-E-CAL): results

Mean over fit seeds (sd).  Mechanism = same frozen T0 (FIT), calibrators on CAL, readouts on EVAL; decomposition on DIAG (finest m-bins).

| condition | N | loss | seeds | candidate | posterior MSE | J bias | J RMSE | S_plug bias | S_plug RMSE | eval SE (J) | selected count |
|---|---|---|---|---|---|---|---|---|---|---|---|
| C1_gauss_mid | 4096 | js | 5 | identity | 0.0867 (0.0041) | -0.0872 | 0.0873 | -0.0547 | 0.0551 | 0.0029 | 3 |
| C1_gauss_mid | 4096 | js | 5 | latent_affine | 0.0857 (0.0038) | -0.0862 | 0.0862 | -0.0833 | 0.0861 | 0.0028 | 2 |
| C1_gauss_mid | 4096 | js | 5 | bins | 0.0906 (0.0042) | -0.0914 | 0.0914 | -0.0895 | 0.0907 | 0.0028 | 0 |
| C1_gauss_mid | 4096 | js | 5 | selected | 0.0862 (0.0036) | -0.0867 | 0.0867 | -0.0741 | 0.0796 | 0.0028 |  |
| C1_gauss_mid | 4096 | vcs | 5 | identity | 0.0964 (0.0087) | -0.0966 | 0.0970 | -0.0737 | 0.0924 | 0.0029 | 2 |
| C1_gauss_mid | 4096 | vcs | 5 | latent_affine | 0.0934 (0.0038) | -0.0935 | 0.0936 | -0.0879 | 0.0906 | 0.0028 | 3 |
| C1_gauss_mid | 4096 | vcs | 5 | bins | 0.0995 (0.0036) | -0.0994 | 0.0994 | -0.0926 | 0.0938 | 0.0028 | 0 |
| C1_gauss_mid | 4096 | vcs | 5 | selected | 0.0934 (0.0039) | -0.0935 | 0.0936 | -0.1057 | 0.1071 | 0.0027 |  |
| C1_gauss_mid | 16384 | js | 5 | identity | 0.0412 (0.0015) | -0.0405 | 0.0406 | +0.0047 | 0.0078 | 0.0028 | 0 |
| C1_gauss_mid | 16384 | js | 5 | latent_affine | 0.0389 (0.0015) | -0.0381 | 0.0383 | -0.0389 | 0.0389 | 0.0026 | 5 |
| C1_gauss_mid | 16384 | js | 5 | bins | 0.0405 (0.0016) | -0.0398 | 0.0399 | -0.0369 | 0.0371 | 0.0026 | 0 |
| C1_gauss_mid | 16384 | js | 5 | selected | 0.0389 (0.0015) | -0.0381 | 0.0383 | -0.0389 | 0.0389 | 0.0026 |  |
| C1_gauss_mid | 16384 | vcs | 5 | identity | 0.0450 (0.0013) | -0.0442 | 0.0442 | -0.0177 | 0.0205 | 0.0027 | 1 |
| C1_gauss_mid | 16384 | vcs | 5 | latent_affine | 0.0434 (0.0007) | -0.0427 | 0.0427 | -0.0412 | 0.0412 | 0.0026 | 4 |
| C1_gauss_mid | 16384 | vcs | 5 | bins | 0.0450 (0.0008) | -0.0442 | 0.0442 | -0.0418 | 0.0419 | 0.0026 | 0 |
| C1_gauss_mid | 16384 | vcs | 5 | selected | 0.0434 (0.0007) | -0.0427 | 0.0427 | -0.0373 | 0.0378 | 0.0026 |  |
| C2_gauss_pad | 4096 | js | 5 | identity | 0.5421 (0.0011) | -0.5421 | 0.5421 | -0.5382 | 0.5382 | 0.0004 | 2 |
| C2_gauss_pad | 4096 | js | 5 | latent_affine | 0.5410 (0.0015) | -0.5410 | 0.5410 | -0.5400 | 0.5400 | 0.0001 | 2 |
| C2_gauss_pad | 4096 | js | 5 | bins | 0.5425 (0.0027) | -0.5423 | 0.5423 | -0.5387 | 0.5387 | 0.0003 | 1 |
| C2_gauss_pad | 4096 | js | 5 | selected | 0.5423 (0.0029) | -0.5422 | 0.5422 | -0.5389 | 0.5389 | 0.0002 |  |
| C2_gauss_pad | 4096 | vcs | 5 | identity | 0.5457 (0.0012) | -0.5458 | 0.5458 | -0.5337 | 0.5337 | 0.0006 | 0 |
| C2_gauss_pad | 4096 | vcs | 5 | latent_affine | 0.5410 (0.0015) | -0.5409 | 0.5409 | -0.5399 | 0.5399 | 0.0001 | 5 |
| C2_gauss_pad | 4096 | vcs | 5 | bins | 0.5415 (0.0017) | -0.5414 | 0.5414 | -0.5394 | 0.5394 | 0.0002 | 0 |
| C2_gauss_pad | 4096 | vcs | 5 | selected | 0.5410 (0.0015) | -0.5409 | 0.5409 | -0.5399 | 0.5399 | 0.0001 |  |
| C2_gauss_pad | 16384 | js | 5 | identity | 0.1415 (0.0104) | -0.1422 | 0.1425 | +0.0528 | 0.0551 | 0.0040 | 0 |
| C2_gauss_pad | 16384 | js | 5 | latent_affine | 0.1037 (0.0016) | -0.1044 | 0.1044 | -0.1004 | 0.1008 | 0.0032 | 5 |
| C2_gauss_pad | 16384 | js | 5 | bins | 0.1046 (0.0016) | -0.1053 | 0.1053 | -0.0974 | 0.0975 | 0.0032 | 0 |
| C2_gauss_pad | 16384 | js | 5 | selected | 0.1037 (0.0016) | -0.1044 | 0.1044 | -0.1004 | 0.1008 | 0.0032 |  |
| C2_gauss_pad | 16384 | vcs | 5 | identity | 0.1397 (0.0085) | -0.1395 | 0.1397 | +0.0240 | 0.0316 | 0.0038 | 0 |
| C2_gauss_pad | 16384 | vcs | 5 | latent_affine | 0.1134 (0.0104) | -0.1133 | 0.1136 | -0.1113 | 0.1119 | 0.0031 | 3 |
| C2_gauss_pad | 16384 | vcs | 5 | bins | 0.1137 (0.0093) | -0.1136 | 0.1138 | -0.1074 | 0.1079 | 0.0031 | 2 |
| C2_gauss_pad | 16384 | vcs | 5 | selected | 0.1135 (0.0103) | -0.1134 | 0.1138 | -0.1083 | 0.1089 | 0.0031 |  |
| C3_xor | 4096 | js | 5 | identity | 0.5738 (0.0368) | -0.5741 | 0.5751 | -0.4083 | 0.4158 | 0.0038 | 0 |
| C3_xor | 4096 | js | 5 | latent_affine | 0.5533 (0.0434) | -0.5536 | 0.5550 | -0.5526 | 0.5550 | 0.0030 | 5 |
| C3_xor | 4096 | js | 5 | bins | 0.5599 (0.0428) | -0.5599 | 0.5613 | -0.5525 | 0.5544 | 0.0031 | 0 |
| C3_xor | 4096 | js | 5 | selected | 0.5533 (0.0434) | -0.5536 | 0.5550 | -0.5526 | 0.5550 | 0.0030 |  |
| C3_xor | 4096 | vcs | 5 | identity | 0.5621 (0.0347) | -0.5623 | 0.5632 | -0.3216 | 0.3390 | 0.0041 | 0 |
| C3_xor | 4096 | vcs | 5 | latent_affine | 0.5275 (0.0465) | -0.5276 | 0.5293 | -0.5231 | 0.5258 | 0.0031 | 5 |
| C3_xor | 4096 | vcs | 5 | bins | 0.5339 (0.0471) | -0.5343 | 0.5361 | -0.5264 | 0.5287 | 0.0032 | 0 |
| C3_xor | 4096 | vcs | 5 | selected | 0.5275 (0.0465) | -0.5276 | 0.5293 | -0.5231 | 0.5258 | 0.0031 |  |
| C3_xor | 16384 | js | 5 | identity | 0.1659 (0.0073) | -0.1693 | 0.1695 | -0.0992 | 0.0999 | 0.0032 | 0 |
| C3_xor | 16384 | js | 5 | latent_affine | 0.1621 (0.0077) | -0.1654 | 0.1655 | -0.1624 | 0.1628 | 0.0029 | 5 |
| C3_xor | 16384 | js | 5 | bins | 0.1632 (0.0077) | -0.1665 | 0.1667 | -0.1683 | 0.1685 | 0.0029 | 0 |
| C3_xor | 16384 | js | 5 | selected | 0.1621 (0.0077) | -0.1654 | 0.1655 | -0.1624 | 0.1628 | 0.0029 |  |
| C3_xor | 16384 | vcs | 5 | identity | 0.1618 (0.0033) | -0.1648 | 0.1648 | -0.1096 | 0.1105 | 0.0031 | 0 |
| C3_xor | 16384 | vcs | 5 | latent_affine | 0.1561 (0.0042) | -0.1590 | 0.1591 | -0.1547 | 0.1549 | 0.0029 | 5 |
| C3_xor | 16384 | vcs | 5 | bins | 0.1573 (0.0041) | -0.1603 | 0.1603 | -0.1614 | 0.1614 | 0.0029 | 0 |
| C3_xor | 16384 | vcs | 5 | selected | 0.1561 (0.0042) | -0.1590 | 0.1591 | -0.1547 | 0.1549 | 0.0029 |  |

## Decomposition S − J(g) = A + B (DIAG, finest m-bins) and the pre-stated readings

| condition | N | loss | A (score loss) | B identity | B selected | A share of identity gap | fixable share (B_id − B_sel) / (A + B_id) | max abs residual | reading |
|---|---|---|---|---|---|---|---|---|---|
| C1_gauss_mid | 4096 | js | 0.0838 | 0.0022 | 0.0016 | 0.97 | 0.01 | 2.3e-05 | score has lost most of the dependence (A dominates) |
| C1_gauss_mid | 4096 | vcs | 0.0912 | 0.0044 | 0.0014 | 0.96 | 0.03 | 4.4e-05 | score has lost most of the dependence (A dominates) |
| C1_gauss_mid | 16384 | js | 0.0385 | 0.0026 | 0.0004 | 0.94 | 0.05 | 5.3e-06 | score has lost most of the dependence (A dominates) |
| C1_gauss_mid | 16384 | vcs | 0.0430 | 0.0021 | 0.0005 | 0.95 | 0.04 | 1.4e-05 | score has lost most of the dependence (A dominates) |
| C2_gauss_pad | 4096 | js | 0.5362 | 0.0057 | 0.0058 | 0.99 | -0.00 | 3.8e-06 | score has lost most of the dependence (A dominates) |
| C2_gauss_pad | 4096 | vcs | 0.5361 | 0.0095 | 0.0048 | 0.98 | 0.01 | 5.2e-06 | score has lost most of the dependence (A dominates) |
| C2_gauss_pad | 16384 | js | 0.1006 | 0.0405 | 0.0030 | 0.72 | 0.26 | 7.9e-06 | score has lost most of the dependence (A dominates) |
| C2_gauss_pad | 16384 | vcs | 0.1095 | 0.0298 | 0.0037 | 0.79 | 0.19 | 1.8e-05 | score has lost most of the dependence (A dominates) |
| C3_xor | 4096 | js | 0.5498 | 0.0259 | 0.0051 | 0.95 | 0.04 | 3.4e-05 | score has lost most of the dependence (A dominates) |
| C3_xor | 4096 | vcs | 0.5240 | 0.0403 | 0.0052 | 0.93 | 0.06 | 2.2e-05 | score has lost most of the dependence (A dominates) |
| C3_xor | 16384 | js | 0.1607 | 0.0058 | 0.0019 | 0.97 | 0.02 | 2.0e-05 | score has lost most of the dependence (A dominates) |
| C3_xor | 16384 | vcs | 0.1548 | 0.0073 | 0.0018 | 0.96 | 0.03 | 1.4e-05 | score has lost most of the dependence (A dominates) |

## End-to-end at equal total sample budget (identity on FIT ∪ CAL vs T0 on FIT + selected calibrator on CAL)

| condition | N | loss | identity(FIT∪CAL) J RMSE / post MSE | calibrated J RMSE / post MSE | paired Δ post MSE (cal − id), mean ± se |
|---|---|---|---|---|---|
| C1_gauss_mid | 4096 | js | 0.0792 / 0.0785 | 0.0867 / 0.0862 | +0.0077 ± 0.0020 |
| C1_gauss_mid | 4096 | vcs | 0.0844 / 0.0845 | 0.0936 / 0.0934 | +0.0090 ± 0.0010 |
| C1_gauss_mid | 16384 | js | 0.0379 / 0.0378 | 0.0383 / 0.0389 | +0.0011 ± 0.0004 |
| C1_gauss_mid | 16384 | vcs | 0.0407 / 0.0416 | 0.0427 / 0.0434 | +0.0018 ± 0.0006 |
| C2_gauss_pad | 4096 | js | 0.5420 / 0.5420 | 0.5422 / 0.5423 | +0.0003 ± 0.0010 |
| C2_gauss_pad | 4096 | vcs | 0.5166 / 0.5168 | 0.5409 / 0.5410 | +0.0242 ± 0.0189 |
| C2_gauss_pad | 16384 | js | 0.1092 / 0.1073 | 0.1044 / 0.1037 | -0.0037 ± 0.0040 |
| C2_gauss_pad | 16384 | vcs | 0.1122 / 0.1109 | 0.1138 / 0.1135 | +0.0026 ± 0.0050 |
| C3_xor | 4096 | js | 0.4261 / 0.4260 | 0.5550 / 0.5533 | +0.1274 ± 0.0222 |
| C3_xor | 4096 | vcs | 0.4495 / 0.4494 | 0.5293 / 0.5275 | +0.0780 ± 0.0215 |
| C3_xor | 16384 | js | 0.1446 / 0.1418 | 0.1655 / 0.1621 | +0.0203 ± 0.0046 |
| C3_xor | 16384 | vcs | 0.1427 / 0.1394 | 0.1591 / 0.1561 | +0.0168 ± 0.0023 |
