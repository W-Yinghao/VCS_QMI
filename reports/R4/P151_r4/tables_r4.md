# P151 r4 — corrected common-readout aggregation (schema p151_r4.1)

Cells loaded: 81; ladder rows with provenance: 810; R = 1 − J identity failures: 0; multi-selected families: 0.

## Old (r3) vs new (r4) — same-target panel means that changed

| r3 field → r4 field | family | cells changed | max |Δ| |
|---|---|---|---|
| J_eval → common_J | neural:js:cyclic8 | 15 | 0.3589 |
| J_eval → common_J | neural:js:inbatch | 15 | 0.3594 |
| J_eval → common_J | neural:js:product | 15 | 0.3523 |
| J_eval → common_J | neural:vcs:cyclic8 | 15 | 0.0020 |
| J_eval → common_J | neural:vcs:inbatch | 15 | 0.0013 |
| J_eval → common_J | s_kde:common_risk | 15 | 0.1341 |
| J_minus_S → common_J_minus_S | neural:js:cyclic8 | 15 | 0.1860 |
| J_minus_S → common_J_minus_S | neural:js:inbatch | 15 | 0.1791 |
| J_minus_S → common_J_minus_S | neural:js:product | 15 | 0.2776 |
| J_minus_S → common_J_minus_S | neural:vcs:cyclic8 | 15 | 0.0020 |
| J_minus_S → common_J_minus_S | neural:vcs:inbatch | 15 | 0.0013 |
| J_minus_S → common_J_minus_S | s_kde:common_risk | 15 | 0.1341 |

Unchanged: 270 of 450 compared means (posterior MSE is read from posterior.* in both versions).

## cubic: common bounded J − S (mean ± sd over seeds; oracle S)

| I | S | neural:vcs:inbatch | neural:vcs:product | neural:vcs:cyclic8 | neural:js:inbatch | neural:js:product | neural:js:cyclic8 | s_kde:common_risk | s_kernel:rff | s_kernel:nystrom:c512 | rls:tanh |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 0.567 | -0.205 ± 0.005 | -0.269 ± 0.005 | -0.206 ± 0.005 | -0.198 ± 0.005 | -0.268 ± 0.003 | -0.197 ± 0.004 | -0.508 ± 0.002 | -0.433 ± 0.009 | -0.549 ± 0.001 | -0.394 ± 0.025 |
| 4 | 0.803 | -0.218 ± 0.014 | -0.278 ± 0.011 | -0.222 ± 0.008 | -0.213 ± 0.015 | -0.269 ± 0.012 | -0.205 ± 0.005 | -0.681 ± 0.006 | -0.528 ± 0.002 | -0.765 ± 0.004 | -0.508 ± 0.002 |
| 6 | 0.913 | -0.198 ± 0.000 | -0.239 ± 0.002 | -0.193 ± 0.007 | -0.186 ± 0.009 | -0.232 ± 0.002 | -0.184 ± 0.011 | -0.746 ± 0.005 | -0.551 ± 0.011 | -0.860 ± 0.005 | -0.511 ± 0.011 |
| 8 | 0.963 | -0.166 ± 0.013 | -0.204 ± 0.003 | -0.169 ± 0.005 | -0.159 ± 0.009 | -0.198 ± 0.005 | -0.158 ± 0.007 | -0.779 ± 0.008 | -0.544 ± 0.008 | -0.898 ± 0.011 | -0.493 ± 0.007 |
| 10 | 0.985 | -0.142 ± 0.009 | -0.162 ± 0.006 | -0.124 ± 0.003 | -0.132 ± 0.007 | -0.159 ± 0.008 | -0.127 ± 0.005 | -0.781 ± 0.003 | -0.521 ± 0.015 | -0.912 ± 0.003 | -0.456 ± 0.021 |

## gaussian: common bounded J − S (mean ± sd over seeds; oracle S)

| I | S | neural:vcs:inbatch | neural:vcs:product | neural:vcs:cyclic8 | neural:js:inbatch | neural:js:product | neural:js:cyclic8 | s_kde:common_risk | s_kernel:rff | s_kernel:nystrom:c512 | rls:tanh |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 0.567 | -0.066 ± 0.004 | -0.114 ± 0.005 | -0.067 ± 0.004 | -0.062 ± 0.008 | -0.115 ± 0.010 | -0.063 ± 0.006 | -0.441 ± 0.001 | -0.176 ± 0.002 | -0.405 ± 0.002 | -0.096 ± 0.003 |
| 4 | 0.803 | -0.059 ± 0.001 | -0.095 ± 0.001 | -0.059 ± 0.001 | -0.052 ± 0.003 | -0.089 ± 0.001 | -0.053 ± 0.003 | -0.601 ± 0.002 | -0.184 ± 0.001 | -0.532 ± 0.009 | -0.163 ± 0.001 |
| 6 | 0.913 | -0.050 ± 0.002 | -0.068 ± 0.002 | -0.048 ± 0.001 | -0.041 ± 0.001 | -0.062 ± 0.002 | -0.041 ± 0.000 | -0.666 ± 0.001 | -0.172 ± 0.004 | -0.578 ± 0.013 | -0.130 ± 0.003 |
| 8 | 0.963 | -0.037 ± 0.001 | -0.046 ± 0.000 | -0.034 ± 0.001 | -0.028 ± 0.000 | -0.042 ± 0.002 | -0.029 ± 0.002 | -0.693 ± 0.006 | -0.154 ± 0.005 | -0.576 ± 0.007 | -0.101 ± 0.005 |
| 10 | 0.985 | -0.026 ± 0.004 | -0.032 ± 0.001 | -0.023 ± 0.003 | -0.017 ± 0.001 | -0.026 ± 0.000 | -0.017 ± 0.001 | -0.703 ± 0.005 | -0.134 ± 0.003 | -0.558 ± 0.004 | -0.075 ± 0.002 |

## xor_mixture: common bounded J − S (mean ± sd over seeds; oracle S)

| I | S | neural:vcs:inbatch | neural:vcs:product | neural:vcs:cyclic8 | neural:js:inbatch | neural:js:product | neural:js:cyclic8 | s_kde:common_risk | s_kernel:rff | s_kernel:nystrom:c512 | rls:tanh |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 0.579 | -0.379 ± 0.035 | -0.588 ± 0.001 | -0.402 ± 0.047 | -0.365 ± 0.001 | -0.583 ± 0.001 | -0.357 ± 0.012 | -0.525 ± 0.001 | -0.576 ± 0.001 | -0.581 ± 0.003 | -0.572 ± 0.001 |
| 4 | 0.833 | -0.346 ± 0.011 | -0.563 ± 0.040 | -0.341 ± 0.033 | -0.345 ± 0.042 | -0.575 ± 0.053 | -0.333 ± 0.031 | -0.744 ± 0.002 | -0.800 ± 0.007 | -0.833 ± 0.000 | -0.795 ± 0.021 |
| 6 | 0.941 | -0.303 ± 0.060 | -0.477 ± 0.012 | -0.276 ± 0.053 | -0.232 ± 0.027 | -0.451 ± 0.020 | -0.270 ± 0.051 | -0.834 ± 0.005 | -0.887 ± 0.001 | -0.941 ± 0.001 | -0.876 ± 0.002 |
| 8 | 0.981 | -0.222 ± 0.023 | -0.385 ± 0.046 | -0.231 ± 0.022 | -0.212 ± 0.013 | -0.360 ± 0.013 | -0.203 ± 0.027 | -0.867 ± 0.000 | -0.919 ± 0.005 | -0.981 ± 0.000 | -0.907 ± 0.003 |
| 10 | 0.994 | -0.188 ± 0.027 | -0.331 ± 0.069 | -0.150 ± 0.037 | -0.156 ± 0.021 | -0.300 ± 0.034 | -0.144 ± 0.028 | -0.876 ± 0.003 | -0.926 ± 0.007 | -0.994 ± 0.000 | -0.915 ± 0.007 |
