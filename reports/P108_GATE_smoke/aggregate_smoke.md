# P108 aggregate (package v4 module E)

## E1(a) mechanism (T_t = (1 − t) eta + t U on the TRUTH sample; slopes of the identity-predicted |bias| in t ≤ 0.1)

| condition | U | slope J | slope S_plug | max |D_J z| | max plug-in identity residual |
|---|---|---|---|---|---|
| C1_gauss_mid | zero | 2.000 | 0.980 | 1.35 | 1.1e-16 |
| C1_gauss_mid | fixed | 2.000 | 0.972 | 1.34 | 1.7e-16 |
| C2_gauss_pad | zero | 2.000 | 0.980 | 0.45 | 5.6e-17 |
| C2_gauss_pad | fixed | 2.000 | 0.966 | 1.39 | 1.1e-16 |
| C3_xor | zero | 2.000 | 0.980 | 0.65 | 9.7e-17 |
| C3_xor | fixed | 2.000 | 0.966 | 0.65 | 1.7e-16 |

## E1(b) identical fitted T, two readouts on the same EVAL units (lr-selected; mean over independent FIT seeds)

| condition | N | fit loss | budget | n seeds | J − S mean (sd) | J RMSE | S_plug − S mean (sd) | S_plug RMSE | posterior MSE | eval SE J / S_plug | fit s (chosen / tuning) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| C1_gauss_mid | 256 | js | 50 | 1 | -0.8023 (0.0000) | 0.8023 | -0.7975 (0.0000) | 0.7975 | 0.8143 | 0.0019 / 0.0000 | 0.2 / 0.2 |
| C1_gauss_mid | 256 | js | 100 | 1 | -0.5065 (0.0000) | 0.5065 | -0.3531 (0.0000) | 0.3531 | 0.5453 | 0.0303 / 0.0102 | 0.3 / 0.3 |
| C1_gauss_mid | 256 | vcs | 50 | 1 | -0.8080 (0.0000) | 0.8080 | -0.7946 (0.0000) | 0.7946 | 0.8205 | 0.0038 / 0.0002 | 0.2 / 0.2 |
| C1_gauss_mid | 256 | vcs | 100 | 1 | -0.4903 (0.0000) | 0.4903 | -0.3992 (0.0000) | 0.3992 | 0.5287 | 0.0280 / 0.0098 | 0.4 / 0.4 |
| C3_xor | 256 | js | 50 | 1 | -0.8355 (0.0000) | 0.8355 | -0.8303 (0.0000) | 0.8303 | 0.8349 | 0.0024 / 0.0001 | 0.2 / 0.2 |
| C3_xor | 256 | js | 100 | 1 | -0.8355 (0.0000) | 0.8355 | -0.8303 (0.0000) | 0.8303 | 0.8349 | 0.0024 / 0.0001 | 0.3 / 0.3 |
| C3_xor | 256 | vcs | 50 | 1 | -0.8439 (0.0000) | 0.8439 | -0.8205 (0.0000) | 0.8205 | 0.8440 | 0.0048 / 0.0005 | 0.2 / 0.2 |
| C3_xor | 256 | vcs | 100 | 1 | -0.8439 (0.0000) | 0.8439 | -0.8205 (0.0000) | 0.8205 | 0.8440 | 0.0048 / 0.0005 | 0.4 / 0.4 |

### Reference estimators at the largest budget (each against its own truth; no common raw number)

| condition | N | estimator | own target | relative error mean (sd) | n |
|---|---|---|---|---|---|
| C1_gauss_mid | 256 | dv | MI | -1.002 (0.000) | 1 |
| C1_gauss_mid | 256 | infonce | MI | -0.356 (0.000) | 1 |
| C1_gauss_mid | 256 | nwj | MI | -1.092 (0.000) | 1 |
| C1_gauss_mid | 256 | smile | MI | -0.854 (0.000) | 1 |
| C3_xor | 256 | dv | MI | -1.000 (0.000) | 1 |
| C3_xor | 256 | infonce | MI | -1.000 (0.000) | 1 |
| C3_xor | 256 | nwj | MI | -1.106 (0.000) | 1 |
| C3_xor | 256 | smile | MI | -1.000 (0.000) | 1 |

## E2 original vs rotated coordinates (lr / bandwidth selected; mean over seeds)

| condition | rotated | N | method | budget | n | |J − S| mean | posterior MSE | fit s chosen | tuning s total | peak CUDA MB |
|---|---|---|---|---|---|---|---|---|---|---|
| C1_gauss_mid | False | 1024 | neural:js | 50 | 1 | 0.8052 | 0.8048 | 0.2 | 0.2 | nan |
| C1_gauss_mid | False | 1024 | neural:js | 100 | 1 | 0.2815 | 0.2718 | 0.4 | 0.4 | nan |
| C1_gauss_mid | False | 1024 | neural:vcs | 50 | 1 | 0.8061 | 0.8060 | 0.2 | 0.2 | nan |
| C1_gauss_mid | False | 1024 | neural:vcs | 100 | 1 | 0.2796 | 0.2706 | 0.4 | 0.4 | nan |
| C1_gauss_mid | False | 1024 | s_kde:common_risk | None | 1 | 0.6096 | 0.6056 | 0.1 | 0.1 | nan |
| C1_gauss_mid | False | 1024 | s_kernel:rff | 50 | 1 | 0.8061 | 0.8053 | 0.1 | 0.3 | nan |
| C1_gauss_mid | False | 1024 | s_kernel:rff | 100 | 1 | 0.7938 | 0.7927 | 0.2 | 0.5 | nan |
| C2_gauss_pad | True | 1024 | neural:js | 50 | 1 | 0.5436 | 0.5522 | 0.2 | 0.2 | nan |
| C2_gauss_pad | True | 1024 | neural:js | 100 | 1 | 0.5436 | 0.5522 | 0.4 | 0.4 | nan |
| C2_gauss_pad | True | 1024 | neural:vcs | 50 | 1 | 0.5524 | 0.5596 | 0.2 | 0.2 | nan |
| C2_gauss_pad | True | 1024 | neural:vcs | 100 | 1 | 0.5524 | 0.5596 | 0.4 | 0.4 | nan |
| C2_gauss_pad | True | 1024 | s_kde:common_risk | None | 1 | 0.5346 | 0.5440 | 0.1 | 0.1 | nan |
| C2_gauss_pad | True | 1024 | s_kernel:rff | 50 | 1 | 0.5383 | 0.5484 | 0.1 | 0.3 | nan |
| C2_gauss_pad | True | 1024 | s_kernel:rff | 100 | 1 | 0.5383 | 0.5484 | 0.2 | 0.6 | nan |

## E3 cost readouts (E1 fits): smallest budget whose seed-mean |J − S| ≤ 0.02; refit spread = sd of J over independent FIT seeds

| condition | N | fit loss | budget reaching tol | refit sd of J at 4000 | sd of selected update at 4000 |
|---|---|---|---|---|---|
| C1_gauss_mid | 256 | js | — | 0.0000 | 0 |
| C1_gauss_mid | 256 | vcs | — | 0.0000 | 0 |
| C3_xor | 256 | js | — | 0.0000 | 0 |
| C3_xor | 256 | vcs | — | 0.0000 | 0 |
