# P156 (SEP-1) aggregate

Effects standardised by the clean JS error of the same cell family; Δe = e_vcs − e_js (negative = VCS more accurate).

## Primary (Holm over 3)

| test | mean [95 % t] (n seeds) | p | p Holm | cells | reading |
|---|---|---|---|---|---|
| P1_contamination_narrows_gap | -0.0583 [-0.0752, -0.0414] (n 8) | 7.996e-05 | 0.0002399 | 12 | < 0 (VCS more accurate) |
| P2_contamination_specific | -0.0137 [-0.0345, +0.0071] (n 8) | 0.1633 | 0.1872 | 12 | n.s. |
| P3_restricted_class_reversal | +0.0054 [-0.0012, +0.0120] (n 8) | 0.09361 | 0.1872 | 8 | n.s. |

Outcome candidates (mechanical rule of protocol §6): O1 saturation region (check the saturation-model table: contaminated cells on the clean line)

Secondary pooled P2-sat (contaminated vs saturation-matched clean, ε 0.3; not in the primary family): -0.0166 [-0.0419, +0.0087] (n 8) — n.s.

## Secondary: abs_delta_eps0.3_A (Holm within the list)

| cell | mean [95 % t] | p Holm | reading |
|---|---|---|---|
| ctype independent, d 20, I 2.0 | -0.0202 [-0.0308, -0.0095] (n 8) | 0.02876 | < 0 (VCS more accurate) |
| ctype independent, d 20, I 4.0 | -0.0223 [-0.0591, +0.0144] (n 8) | 0.5827 | n.s. |
| ctype independent, d 20, I 8.0 | -0.2402 [-0.4692, -0.0113] (n 8) | 0.295 | n.s. |
| ctype independent, d 64, I 2.0 | -0.0071 [-0.0120, -0.0022] (n 8) | 0.1014 | n.s. |
| ctype independent, d 64, I 4.0 | -0.0000 [-0.0049, +0.0049] (n 8) | 0.9966 | n.s. |
| ctype independent, d 64, I 8.0 | +0.0317 [-0.0027, +0.0661] (n 8) | 0.3956 | n.s. |
| ctype false_negative, d 20, I 2.0 | -0.0421 [-0.0590, -0.0251] (n 8) | 0.006757 | < 0 (VCS more accurate) |
| ctype false_negative, d 20, I 4.0 | -0.0366 [-0.1399, +0.0667] (n 8) | 0.8591 | n.s. |
| ctype false_negative, d 20, I 8.0 | -0.0581 [-0.1234, +0.0072] (n 8) | 0.3956 | n.s. |
| ctype false_negative, d 64, I 2.0 | -0.0061 [-0.0106, -0.0016] (n 8) | 0.1183 | n.s. |
| ctype false_negative, d 64, I 4.0 | +0.0043 [-0.0008, +0.0094] (n 8) | 0.3956 | n.s. |
| ctype false_negative, d 64, I 8.0 | +0.0350 [+0.0277, +0.0422] (n 8) | 0.0001088 | > 0 (JS more accurate) |

## Secondary: G_per_cell (Holm within the list)

| cell | mean [95 % t] | p Holm | reading |
|---|---|---|---|
| ctype independent, d 20, I 2.0 | -0.0129 [-0.0390, +0.0132] (n 8) | 0.2848 | n.s. |
| ctype independent, d 20, I 4.0 | -0.0791 [-0.1201, -0.0381] (n 8) | 0.02325 | < 0 (VCS more accurate) |
| ctype independent, d 20, I 8.0 | -0.3928 [-0.6101, -0.1756] (n 8) | 0.02942 | < 0 (VCS more accurate) |
| ctype independent, d 64, I 2.0 | -0.0130 [-0.0228, -0.0032] (n 8) | 0.1134 | n.s. |
| ctype independent, d 64, I 4.0 | -0.0190 [-0.0373, -0.0007] (n 8) | 0.1764 | n.s. |
| ctype independent, d 64, I 8.0 | +0.0896 [+0.0481, +0.1311] (n 8) | 0.01522 | > 0 (JS more accurate) |
| ctype false_negative, d 20, I 2.0 | -0.0348 [-0.0630, -0.0067] (n 8) | 0.1134 | n.s. |
| ctype false_negative, d 20, I 4.0 | -0.0934 [-0.2011, +0.0144] (n 8) | 0.2387 | n.s. |
| ctype false_negative, d 20, I 8.0 | -0.2107 [-0.3167, -0.1047] (n 8) | 0.02206 | < 0 (VCS more accurate) |
| ctype false_negative, d 64, I 2.0 | -0.0120 [-0.0211, -0.0029] (n 8) | 0.1134 | n.s. |
| ctype false_negative, d 64, I 4.0 | -0.0146 [-0.0356, +0.0063] (n 8) | 0.2848 | n.s. |
| ctype false_negative, d 64, I 8.0 | +0.0929 [+0.0716, +0.1142] (n 8) | 0.0002069 | > 0 (JS more accurate) |

## Secondary: H_per_cell (Holm within the list)

| cell | mean [95 % t] | p Holm | reading |
|---|---|---|---|
| ctype independent, d 20, I 2.0 | +0.0082 [-0.0157, +0.0321] (n 8) | 1 | n.s. |
| ctype independent, d 20, I 4.0 | +0.0149 [-0.0251, +0.0550] (n 8) | 1 | n.s. |
| ctype independent, d 20, I 8.0 | -0.2485 [-0.4582, -0.0388] (n 8) | 0.2116 | n.s. |
| ctype independent, d 64, I 2.0 | -0.0127 [-0.0211, -0.0043] (n 8) | 0.08154 | n.s. |
| ctype independent, d 64, I 4.0 | +0.0094 [-0.0118, +0.0306] (n 8) | 1 | n.s. |
| ctype independent, d 64, I 8.0 | +0.0317 [-0.0178, +0.0813] (n 8) | 1 | n.s. |
| ctype false_negative, d 20, I 2.0 | -0.0078 [-0.0332, +0.0177] (n 8) | 1 | n.s. |
| ctype false_negative, d 20, I 4.0 | -0.0019 [-0.1097, +0.1058] (n 8) | 1 | n.s. |
| ctype false_negative, d 20, I 8.0 | +0.0022 [-0.1034, +0.1078] (n 8) | 1 | n.s. |
| ctype false_negative, d 64, I 2.0 | -0.0110 [-0.0138, -0.0082] (n 8) | 0.0004564 | < 0 (VCS more accurate) |
| ctype false_negative, d 64, I 4.0 | +0.0094 [+0.0034, +0.0153] (n 8) | 0.07454 | n.s. |
| ctype false_negative, d 64, I 8.0 | +0.0417 [+0.0265, +0.0569] (n 8) | 0.003764 | > 0 (JS more accurate) |

## Secondary: H_sat_per_cell (Holm within the list)

| cell | mean [95 % t] | p Holm | reading |
|---|---|---|---|
| ctype independent, d 20, I 2.0 | +0.0084 [-0.0155, +0.0322] (n 8) | 1 | n.s. |
| ctype independent, d 20, I 4.0 | -0.0102 [-0.0571, +0.0367] (n 8) | 1 | n.s. |
| ctype independent, d 20, I 8.0 | -0.2192 [-0.4701, +0.0317] (n 8) | 0.7769 | n.s. |
| ctype independent, d 64, I 2.0 | +0.0043 [-0.0156, +0.0242] (n 8) | 1 | n.s. |
| ctype independent, d 64, I 4.0 | +0.0009 [-0.0084, +0.0101] (n 8) | 1 | n.s. |
| ctype independent, d 64, I 8.0 | +0.0419 [-0.0027, +0.0865] (n 8) | 0.6771 | n.s. |
| ctype false_negative, d 20, I 2.0 | -0.0181 [-0.0421, +0.0059] (n 8) | 0.9419 | n.s. |
| ctype false_negative, d 20, I 4.0 | -0.0177 [-0.1350, +0.0995] (n 8) | 1 | n.s. |
| ctype false_negative, d 20, I 8.0 | -0.0207 [-0.0754, +0.0340] (n 8) | 1 | n.s. |
| ctype false_negative, d 64, I 2.0 | -0.0014 [-0.0051, +0.0022] (n 8) | 1 | n.s. |
| ctype false_negative, d 64, I 4.0 | +0.0059 [-0.0009, +0.0126] (n 8) | 0.7769 | n.s. |
| ctype false_negative, d 64, I 8.0 | +0.0270 [+0.0087, +0.0452] (n 8) | 0.1222 | n.s. |

## Secondary: restricted_delta_eps (Holm within the list)

| cell | mean [95 % t] | p Holm | reading |
|---|---|---|---|
| cls cos2, d 20, I 4.0, eps 0.1 | -0.4247 [-0.4639, -0.3854] (n 8) | 4.621e-07 | < 0 (VCS more accurate) |
| cls cos2, d 20, I 4.0, eps 0.3 | -0.2845 [-0.3153, -0.2537] (n 8) | 1.286e-06 | < 0 (VCS more accurate) |
| cls cos2, d 20, I 8.0, eps 0.1 | -1.2522 [-1.3420, -1.1624] (n 8) | 8.581e-08 | < 0 (VCS more accurate) |
| cls cos2, d 20, I 8.0, eps 0.3 | -0.4844 [-0.5380, -0.4308] (n 8) | 1.362e-06 | < 0 (VCS more accurate) |
| cls cos2, d 64, I 4.0, eps 0.1 | -0.8575 [-0.9695, -0.7455] (n 8) | 3.501e-06 | < 0 (VCS more accurate) |
| cls cos2, d 64, I 4.0, eps 0.3 | -0.6247 [-0.6980, -0.5514] (n 8) | 1.854e-06 | < 0 (VCS more accurate) |
| cls cos2, d 64, I 8.0, eps 0.1 | -2.1037 [-2.2005, -2.0069] (n 8) | 4.435e-09 | < 0 (VCS more accurate) |
| cls cos2, d 64, I 8.0, eps 0.3 | -1.0222 [-1.0785, -0.9659] (n 8) | 1.46e-08 | < 0 (VCS more accurate) |
| cls mlp8, d 20, I 4.0, eps 0.1 | +0.0098 [-0.0034, +0.0231] (n 8) | 0.7245 | n.s. |
| cls mlp8, d 20, I 4.0, eps 0.3 | +0.0056 [-0.0018, +0.0131] (n 8) | 0.7245 | n.s. |
| cls mlp8, d 20, I 8.0, eps 0.1 | +0.0192 [-0.0051, +0.0434] (n 8) | 0.7245 | n.s. |
| cls mlp8, d 20, I 8.0, eps 0.3 | +0.0159 [+0.0031, +0.0286] (n 8) | 0.1724 | n.s. |
| cls mlp8, d 64, I 4.0, eps 0.1 | +0.0010 [-0.0024, +0.0043] (n 8) | 1 | n.s. |
| cls mlp8, d 64, I 4.0, eps 0.3 | +0.0021 [-0.0007, +0.0050] (n 8) | 0.7245 | n.s. |
| cls mlp8, d 64, I 8.0, eps 0.1 | +0.0020 [-0.0059, +0.0099] (n 8) | 1 | n.s. |
| cls mlp8, d 64, I 8.0, eps 0.3 | +0.0024 [-0.0019, +0.0068] (n 8) | 0.7245 | n.s. |

## Secondary: quad3_clean_delta (Holm within the list)

| cell | mean [95 % t] | p Holm | reading |
|---|---|---|---|
| cls quad3, d 20, I 4.0 | -0.2019 [-0.7242, +0.3205] (n 8) | 1 | n.s. |
| cls quad3, d 20, I 8.0 | -0.0921 [-0.6914, +0.5071] (n 8) | 1 | n.s. |
| cls quad3, d 64, I 4.0 | +0.2739 [-0.6282, +1.1759] (n 8) | 1 | n.s. |
| cls quad3, d 64, I 8.0 | +0.0989 [-1.4853, +1.6832] (n 8) | 1 | n.s. |

## Descriptive: lr edges and convergence (selected rows)

| block:arm | n | low edge | high edge | not converged |
|---|---|---|---|---|
| A:b05 | 432 | 125 | 10 | 27 |
| A:b2 | 432 | 41 | 4 | 22 |
| A:js | 432 | 164 | 11 | 40 |
| A:vcs | 432 | 82 | 11 | 24 |
| B:b05 | 192 | 54 | 0 | 7 |
| B:b2 | 192 | 24 | 0 | 7 |
| B:js | 192 | 63 | 0 | 12 |
| B:vcs | 192 | 41 | 0 | 10 |
| B2:b05 | 96 | 33 | 0 | 7 |
| B2:b2 | 96 | 10 | 0 | 3 |
| B2:js | 96 | 36 | 0 | 5 |
| B2:vcs | 96 | 20 | 0 | 4 |
| C:js | 112 | 32 | 1 | 4 |
| C:vcs | 112 | 26 | 18 | 14 |
| D:js | 96 | 7 | 0 | 7 |
| D:vcs | 96 | 0 | 0 | 3 |
| Dp:js | 56 | 1 | 8 | 1 |
| Dp:vcs | 56 | 7 | 8 | 6 |
| E:b05 | 288 | 0 | 111 | 26 |
| E:b2 | 288 | 0 | 141 | 33 |
| E:js | 288 | 0 | 106 | 27 |
| E:vcs | 288 | 0 | 105 | 34 |

Saturation model (rel_delta ~ 1 + sat95 + contaminated, 90 cells, descriptive): intercept -0.0061, sat95 +0.0240, contaminated -0.0327

Full descriptive tables (γ dose, saturation, blocks C / D / D′, clean-target diagnostic, selection sensitivity) are in the JSON.
