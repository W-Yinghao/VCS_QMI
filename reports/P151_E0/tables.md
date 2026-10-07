# P151 E0 — precision / stability / resolution re-aggregated from the P85 / P86 / P108 records

Cells loaded: 81 (full-method cells; corrected SMILE rows substituted).  Ladder cells per generator: {"cubic": {"10": 3, "2": 3, "4": 3, "6": 3, "8": 3}, "gaussian": {"10": 3, "2": 3, "4": 3, "6": 3, "8": 3}, "xor_mixture": {"10": 3, "2": 3, "4": 3, "6": 3, "8": 3}}

## cubic: E-Precision (same target S; mean ± sd over seeds)

| I | S (oracle) | neural:vcs:inbatch | neural:vcs:product | neural:vcs:cyclic8 | neural:js:inbatch | neural:js:product | neural:js:cyclic8 | s_kde:common_risk | s_kernel:rff | s_kernel:nystrom:c512 | rls:tanh |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 0.567 | 0.205 ± 0.003 | 0.268 ± 0.006 | 0.206 ± 0.002 | 0.198 ± 0.003 | 0.267 ± 0.004 | 0.197 ± 0.001 | 0.507 ± 0.003 | 0.433 ± 0.009 | 0.548 ± 0.001 | 0.392 ± 0.025 |
| 4 | 0.803 | 0.218 ± 0.012 | 0.278 ± 0.010 | 0.221 ± 0.007 | 0.213 ± 0.014 | 0.268 ± 0.011 | 0.205 ± 0.004 | 0.681 ± 0.004 | 0.528 ± 0.003 | 0.764 ± 0.004 | 0.507 ± 0.003 |
| 6 | 0.913 | 0.200 ± 0.003 | 0.240 ± 0.003 | 0.195 ± 0.009 | 0.188 ± 0.011 | 0.234 ± 0.001 | 0.186 ± 0.014 | 0.746 ± 0.003 | 0.551 ± 0.010 | 0.861 ± 0.005 | 0.512 ± 0.010 |
| 8 | 0.963 | 0.166 ± 0.013 | 0.204 ± 0.002 | 0.169 ± 0.004 | 0.159 ± 0.010 | 0.198 ± 0.004 | 0.158 ± 0.008 | 0.779 ± 0.008 | 0.544 ± 0.008 | 0.899 ± 0.011 | 0.492 ± 0.008 |
| 10 | 0.985 | 0.143 ± 0.009 | 0.163 ± 0.006 | 0.124 ± 0.004 | 0.132 ± 0.007 | 0.160 ± 0.008 | 0.127 ± 0.005 | 0.781 ± 0.004 | 0.521 ± 0.015 | 0.913 ± 0.003 | 0.456 ± 0.021 |

## cubic: native-target methods (|value − own truth|; mean ± sd; MI methods on their own target)

| I | MI | neural:infonce:inbatch | neural:infonce:cyclic8 | neural:nwj:inbatch | neural:nwj:product | neural:dv:inbatch | neural:dv:product | neural:smile:inbatch | neural:smile:product |
|---|---|---|---|---|---|---|---|---|---|
| 2 | 2.0 | 1.094 (err 0.906) | 0.844 (err 1.156) | 1.131 (err 0.869) | 0.263 (err 1.737) | 1.091 (err 0.909) | 0.377 (err 1.623) | 0.996 (err 1.004) | 3.392 (err 1.392) |
| 4 | 4.0 | 2.247 (err 1.753) | 1.437 (err 2.563) | 2.309 (err 1.691) | 0.623 (err 3.377) | 2.284 (err 1.716) | -3.370 (err 7.370) | 2.283 (err 1.717) | 9.174 (err 5.174) |
| 6 | 6.0 | 3.356 (err 2.644) | 1.777 (err 4.223) | 3.354 (err 2.646) | 1.067 (err 4.933) | 3.415 (err 2.585) | 0.288 (err 5.712) | 4.206 (err 1.794) | 13.160 (err 7.160) |
| 8 | 8.0 | 4.305 (err 3.695) | 1.975 (err 6.025) | 4.208 (err 3.792) | -77.638 (err 85.638) | 4.169 (err 3.831) | -1.165 (err 9.165) | 6.308 (err 1.692) | 15.721 (err 7.721) |
| 10 | 10.0 | 5.025 (err 4.975) | 2.085 (err 7.915) | 4.623 (err 5.377) | 1.441 (err 8.559) | 4.860 (err 5.140) | 0.255 (err 9.745) | 7.875 (err 2.125) | 16.903 (err 6.903) |

## cubic: E-Stability at I = 6 (seed sd / eval boot SE / select-curve tail sd / non-finite steps / fit s)

| method | seed sd | eval boot SE | curve tail sd | nonfinite | fit s |
|---|---|---|---|---|---|
| neural:vcs:inbatch | 0.0023 | 0.0021 | 0.0317 | 0 | 8.6 |
| neural:vcs:product | 0.0019 | 0.003 | 0.0025 | 0 | 5.1 |
| neural:vcs:cyclic8 | 0.0081 | 0.0021 | 0.031 | 0 | 16.6 |
| neural:js:inbatch | 0.0284 | 0.0041 | 0.1664 | 0 | 8.4 |
| neural:js:product | 0.0035 | 0.0048 | 0.011 | 0 | 4.8 |
| neural:js:cyclic8 | 0.0158 | 0.0039 | 0.1442 | 0 | 16.4 |
| s_kde:common_risk | 0.003 | 0.0005 | None | 0 | 0.1 |
| s_kernel:rff | 0.0107 | 0.0026 | None | 0 | 3.0 |
| s_kernel:nystrom:c512 | 0.0043 | 0.0009 | None | 0 | 3.6 |
| rls:tanh | 0.0111 | 0.0035 | None | 0 | 0.4 |
| neural:infonce:inbatch | 0.0167 | 0.0118 | 0.5746 | 0 | 8.5 |
| neural:infonce:cyclic8 | 0.0065 | 0.0054 | 0.0227 | 0 | 16.3 |
| neural:nwj:inbatch | 0.0437 | 0.0145 | 0.2468 | 0 | 8.3 |
| neural:nwj:product | 0.2955 | 0.1873 | 58033325521.1915 | 0 | 4.6 |
| neural:dv:inbatch | 0.0509 | 0.0141 | 0.4149 | 0 | 8.5 |
| neural:dv:product | 0.7872 | 0.2528 | 9.854 | 270 | 4.9 |
| neural:smile:inbatch | 0.3324 | 0.0231 | 0.3178 | 0 | 3.5 |
| neural:smile:product | 0.4951 | 0.0717 | 0.3497 | 0 | 2.3 |

## cubic: E-Resolution (adjacent ladder levels; P(value_k+1 > value_k) from seed-coupled bootstrap draws; |Δ| / pooled sd)

| step | ΔS | neural:vcs:inbatch | neural:vcs:product | neural:infonce:inbatch | neural:nwj:inbatch | neural:dv:inbatch | neural:smile:inbatch |
|---|---|---|---|---|---|---|---|
| 2->4 | 0.235 | 1.00 (23.9) | 1.00 (36.7) | 1.00 (35.0) | 1.00 (27.3) | 1.00 (24.2) | 1.00 (13.8) |
| 4->6 | 0.110 | 1.00 (10.2) | 1.00 (14.0) | 1.00 (22.6) | 1.00 (19.8) | 1.00 (24.7) | 1.00 (7.5) |
| 6->8 | 0.050 | 1.00 (6.0) | 1.00 (17.3) | 1.00 (49.8) | 1.00 (15.9) | 1.00 (5.1) | 1.00 (6.0) |
| 8->10 | 0.022 | 1.00 (6.9) | 1.00 (9.2) | 1.00 (13.1) | 1.00 (5.3) | 1.00 (5.0) | 1.00 (22.8) |

## gaussian: E-Precision (same target S; mean ± sd over seeds)

| I | S (oracle) | neural:vcs:inbatch | neural:vcs:product | neural:vcs:cyclic8 | neural:js:inbatch | neural:js:product | neural:js:cyclic8 | s_kde:common_risk | s_kernel:rff | s_kernel:nystrom:c512 | rls:tanh |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 0.567 | 0.065 ± 0.003 | 0.112 ± 0.005 | 0.066 ± 0.004 | 0.061 ± 0.005 | 0.113 ± 0.010 | 0.061 ± 0.003 | 0.441 ± 0.001 | 0.174 ± 0.002 | 0.404 ± 0.001 | 0.095 ± 0.002 |
| 4 | 0.803 | 0.060 ± 0.001 | 0.095 ± 0.002 | 0.059 ± 0.000 | 0.052 ± 0.002 | 0.089 ± 0.003 | 0.052 ± 0.002 | 0.602 ± 0.003 | 0.185 ± 0.001 | 0.533 ± 0.009 | 0.163 ± 0.002 |
| 6 | 0.913 | 0.050 ± 0.000 | 0.069 ± 0.003 | 0.049 ± 0.002 | 0.041 ± 0.002 | 0.062 ± 0.003 | 0.041 ± 0.002 | 0.666 ± 0.002 | 0.173 ± 0.005 | 0.578 ± 0.012 | 0.131 ± 0.004 |
| 8 | 0.963 | 0.037 ± 0.001 | 0.046 ± 0.001 | 0.035 ± 0.000 | 0.029 ± 0.001 | 0.042 ± 0.002 | 0.029 ± 0.002 | 0.693 ± 0.006 | 0.155 ± 0.004 | 0.576 ± 0.007 | 0.101 ± 0.005 |
| 10 | 0.985 | 0.027 ± 0.005 | 0.032 ± 0.002 | 0.023 ± 0.003 | 0.018 ± 0.001 | 0.027 ± 0.001 | 0.018 ± 0.001 | 0.703 ± 0.005 | 0.134 ± 0.004 | 0.558 ± 0.004 | 0.076 ± 0.003 |

## gaussian: native-target methods (|value − own truth|; mean ± sd; MI methods on their own target)

| I | MI | neural:infonce:inbatch | neural:infonce:cyclic8 | neural:nwj:inbatch | neural:nwj:product | neural:dv:inbatch | neural:dv:product | neural:smile:inbatch | neural:smile:product |
|---|---|---|---|---|---|---|---|---|---|
| 2 | 2.0 | 1.758 (err 0.242) | 1.201 (err 0.799) | 1.760 (err 0.240) | 0.881 (err 1.119) | 1.768 (err 0.232) | 1.115 (err 0.885) | 1.670 (err 0.330) | 6.750 (err 4.750) |
| 4 | 4.0 | 3.539 (err 0.461) | 1.818 (err 2.182) | 3.657 (err 0.343) | 1.580 (err 2.420) | 3.665 (err 0.335) | 2.298 (err 1.702) | 4.272 (err 0.272) | 15.129 (err 11.129) |
| 6 | 6.0 | 4.986 (err 1.014) | 2.072 (err 3.928) | 5.368 (err 0.632) | -60.880 (err 66.880) | 5.453 (err 0.547) | -11.423 (err 17.423) | 7.797 (err 1.797) | 18.241 (err 12.241) |
| 8 | 8.0 | 5.967 (err 2.033) | 2.159 (err 5.841) | 6.712 (err 1.288) | -149697648.700 (err 149697656.700) | 7.002 (err 0.998) | -15.022 (err 23.022) | 11.352 (err 3.352) | 20.091 (err 12.091) |
| 10 | 10.0 | 6.528 (err 3.472) | 2.185 (err 7.815) | 7.718 (err 2.282) | 3.252 (err 6.748) | 8.333 (err 1.667) | -21.393 (err 31.393) | 13.386 (err 3.386) | 19.316 (err 9.316) |

## gaussian: E-Stability at I = 6 (seed sd / eval boot SE / select-curve tail sd / non-finite steps / fit s)

| method | seed sd | eval boot SE | curve tail sd | nonfinite | fit s |
|---|---|---|---|---|---|
| neural:vcs:inbatch | 0.0014 | 0.0013 | 0.005 | 0 | 9.8 |
| neural:vcs:product | 0.002 | 0.0023 | 0.0009 | 0 | 6.3 |
| neural:vcs:cyclic8 | 0.0005 | 0.0014 | 0.0044 | 0 | 20.5 |
| neural:js:inbatch | 0.0016 | 0.0023 | 0.0112 | 0 | 9.5 |
| neural:js:product | 0.0032 | 0.0039 | 0.0083 | 0 | 5.8 |
| neural:js:cyclic8 | 0.0003 | 0.0024 | 0.0105 | 0 | 20.1 |
| s_kde:common_risk | 0.0033 | 0.0003 | None | 0 | 0.1 |
| s_kernel:rff | 0.0044 | 0.002 | None | 0 | 3.4 |
| s_kernel:nystrom:c512 | 0.0125 | 0.0012 | None | 0 | 4.2 |
| rls:tanh | 0.0029 | 0.0024 | None | 0 | 0.4 |
| neural:infonce:inbatch | 0.0043 | 0.0102 | 0.0095 | 0 | 9.5 |
| neural:infonce:cyclic8 | 0.0077 | 0.0033 | 0.0031 | 0 | 20.5 |
| neural:nwj:inbatch | 0.0506 | 0.0133 | 0.0982 | 0 | 9.1 |
| neural:nwj:product | 109.3487 | 59.2121 | 2.2303793182911284e+28 | 0 | 5.7 |
| neural:dv:inbatch | 0.0078 | 0.0133 | 0.0416 | 0 | 9.4 |
| neural:dv:product | 3.303 | 2.06 | 3.5139 | 0 | 5.9 |
| neural:smile:inbatch | 0.12 | 0.0317 | 0.1337 | 0 | 3.5 |
| neural:smile:product | 0.8517 | 0.0657 | 0.4079 | 0 | 2.2 |

## gaussian: E-Resolution (adjacent ladder levels; P(value_k+1 > value_k) from seed-coupled bootstrap draws; |Δ| / pooled sd)

| step | ΔS | neural:vcs:inbatch | neural:vcs:product | neural:infonce:inbatch | neural:nwj:inbatch | neural:dv:inbatch | neural:smile:inbatch |
|---|---|---|---|---|---|---|---|
| 2->4 | 0.236 | 1.00 (84.7) | 1.00 (36.5) | 1.00 (75.7) | 1.00 (81.6) | 1.00 (81.3) | 1.00 (31.1) |
| 4->6 | 0.110 | 1.00 (32.4) | 1.00 (34.9) | 1.00 (94.1) | 1.00 (32.7) | 1.00 (90.4) | 1.00 (19.7) |
| 6->8 | 0.050 | 1.00 (23.5) | 1.00 (23.8) | 1.00 (68.6) | 1.00 (21.7) | 1.00 (41.1) | 1.00 (9.3) |
| 8->10 | 0.023 | 1.00 (8.7) | 1.00 (19.2) | 1.00 (28.7) | 1.00 (6.2) | 1.00 (23.4) | 1.00 (13.4) |

## xor_mixture: E-Precision (same target S; mean ± sd over seeds)

| I | S (oracle) | neural:vcs:inbatch | neural:vcs:product | neural:vcs:cyclic8 | neural:js:inbatch | neural:js:product | neural:js:cyclic8 | s_kde:common_risk | s_kernel:rff | s_kernel:nystrom:c512 | rls:tanh |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 0.579 | 0.378 ± 0.034 | 0.588 ± 0.002 | 0.400 ± 0.042 | 0.363 ± 0.002 | 0.582 ± 0.002 | 0.355 ± 0.009 | 0.524 ± 0.001 | 0.576 ± 0.002 | 0.580 ± 0.003 | 0.571 ± 0.002 |
| 4 | 0.833 | 0.345 ± 0.010 | 0.562 ± 0.040 | 0.340 ± 0.033 | 0.344 ± 0.041 | 0.574 ± 0.052 | 0.332 ± 0.030 | 0.744 ± 0.002 | 0.800 ± 0.007 | 0.833 ± 0.002 | 0.795 ± 0.022 |
| 6 | 0.941 | 0.303 ± 0.060 | 0.479 ± 0.012 | 0.276 ± 0.053 | 0.233 ± 0.025 | 0.453 ± 0.019 | 0.270 ± 0.050 | 0.834 ± 0.005 | 0.888 ± 0.002 | 0.941 ± 0.001 | 0.876 ± 0.002 |
| 8 | 0.981 | 0.222 ± 0.023 | 0.384 ± 0.047 | 0.230 ± 0.021 | 0.212 ± 0.012 | 0.359 ± 0.014 | 0.202 ± 0.027 | 0.866 ± 0.001 | 0.919 ± 0.005 | 0.981 ± 0.000 | 0.907 ± 0.003 |
| 10 | 0.994 | 0.188 ± 0.028 | 0.331 ± 0.069 | 0.151 ± 0.037 | 0.157 ± 0.021 | 0.300 ± 0.035 | 0.144 ± 0.028 | 0.876 ± 0.003 | 0.926 ± 0.007 | 0.995 ± 0.000 | 0.915 ± 0.007 |

## xor_mixture: native-target methods (|value − own truth|; mean ± sd; MI methods on their own target)

| I | MI | neural:infonce:inbatch | neural:infonce:cyclic8 | neural:nwj:inbatch | neural:nwj:product | neural:dv:inbatch | neural:dv:product | neural:smile:inbatch | neural:smile:product |
|---|---|---|---|---|---|---|---|---|---|
| 2 | 1.9999999999999996 | 0.699 (err 1.301) | 0.595 (err 1.405) | 0.853 (err 1.147) | -0.019 (err 2.019) | 0.736 (err 1.264) | -0.003 (err 2.003) | 0.538 (err 1.462) | -0.006 (err 2.006) |
| 4 | 4.0 | 2.286 (err 1.714) | 1.453 (err 2.547) | 2.263 (err 1.737) | -0.003 (err 4.003) | 2.071 (err 1.929) | 0.052 (err 3.948) | 1.850 (err 2.150) | 2.678 (err 1.322) |
| 6 | 5.999999999999997 | 3.730 (err 2.270) | 1.887 (err 4.113) | 2.942 (err 3.058) | 0.122 (err 5.878) | 2.860 (err 3.140) | 0.374 (err 5.626) | 3.654 (err 2.346) | 4.718 (err 1.282) |
| 8 | 8.000000000000005 | 4.927 (err 3.073) | 2.075 (err 5.925) | 3.630 (err 4.370) | 0.211 (err 7.789) | 3.682 (err 4.318) | 0.374 (err 7.626) | 5.470 (err 2.530) | 6.215 (err 1.785) |
| 10 | 9.999999999999979 | 5.861 (err 4.139) | 2.146 (err 7.854) | 4.093 (err 5.907) | 0.197 (err 9.803) | 4.023 (err 5.977) | 0.541 (err 9.459) | 7.036 (err 2.964) | 6.691 (err 3.309) |

## xor_mixture: E-Stability at I = 6 (seed sd / eval boot SE / select-curve tail sd / non-finite steps / fit s)

| method | seed sd | eval boot SE | curve tail sd | nonfinite | fit s |
|---|---|---|---|---|---|
| neural:vcs:inbatch | 0.0605 | 0.0022 | 0.0329 | 0 | 6.8 |
| neural:vcs:product | 0.0118 | 0.0038 | 0.0054 | 0 | 4.3 |
| neural:vcs:cyclic8 | 0.0521 | 0.0025 | 0.0651 | 0 | 13.6 |
| neural:js:inbatch | 0.047 | 0.0042 | 0.1205 | 0 | 6.4 |
| neural:js:product | 0.0172 | 0.0056 | 0.0257 | 0 | 4.1 |
| neural:js:cyclic8 | 0.0755 | 0.0038 | 0.1548 | 0 | 13.4 |
| s_kde:common_risk | 0.0028 | 0.0003 | None | 0 | 0.1 |
| s_kernel:rff | 0.0016 | 0.0022 | None | 0 | 2.4 |
| s_kernel:nystrom:c512 | 0.0008 | 0.0002 | None | 0 | 2.9 |
| rls:tanh | 0.0019 | 0.0018 | None | 0 | 0.4 |
| neural:infonce:inbatch | 0.0317 | 0.009 | 0.0616 | 0 | 6.7 |
| neural:infonce:cyclic8 | 0.0096 | 0.0047 | 0.0238 | 0 | 13.7 |
| neural:nwj:inbatch | 0.0949 | 0.0199 | 0.4474 | 0 | 6.5 |
| neural:nwj:product | 0.0242 | 0.0096 | 755785.6198 | 0 | 3.9 |
| neural:dv:inbatch | 0.0869 | 0.0145 | 0.9586 | 0 | 6.6 |
| neural:dv:product | 0.0951 | 0.0528 | 1.3827 | 0 | 4.2 |
| neural:smile:inbatch | 0.4768 | 0.019 | 0.1876 | 0 | 3.4 |
| neural:smile:product | 0.3712 | 0.0434 | 0.1761 | 0 | 2.2 |

## xor_mixture: E-Resolution (adjacent ladder levels; P(value_k+1 > value_k) from seed-coupled bootstrap draws; |Δ| / pooled sd)

| step | ΔS | neural:vcs:inbatch | neural:vcs:product | neural:infonce:inbatch | neural:nwj:inbatch | neural:dv:inbatch | neural:smile:inbatch |
|---|---|---|---|---|---|---|---|
| 2->4 | 0.254 | 1.00 (11.4) | 1.00 (6.8) | 1.00 (45.4) | 1.00 (18.2) | 1.00 (14.6) | 1.00 (11.9) |
| 4->6 | 0.107 | 1.00 (2.8) | 1.00 (6.3) | 1.00 (44.8) | 1.00 (3.4) | 1.00 (3.6) | 1.00 (5.2) |
| 6->8 | 0.040 | 1.00 (1.7) | 1.00 (3.9) | 1.00 (36.1) | 1.00 (3.4) | 1.00 (3.4) | 1.00 (26.7) |
| 8->10 | 0.014 | 0.89 (1.0) | 1.00 (2.8) | 1.00 (15.5) | 1.00 (2.6) | 1.00 (5.3) | 1.00 (6.2) |

## P108: J vs plug-in (signed bias / RMSE over seeds) per condition, N and budget

- C1_gauss_mid N1024: vcs@250: J -0.2183/0.2186, S_plug -0.1109/0.1238; vcs@1000: J -0.2178/0.2181, S_plug -0.1409/0.1616; vcs@4000: J -0.2178/0.2181, S_plug -0.1409/0.1616; js@250: J -0.2301/0.2314, S_plug -0.1799/0.2314; js@1000: J -0.2035/0.2036, S_plug -0.1680/0.1697; js@4000: J -0.2035/0.2036, S_plug -0.1680/0.1697; s_kde@None: J -0.6061/0.6061, S_plug -0.7351/0.7351; s_kernel_rff@250: J -0.7538/0.7538, S_plug -0.7972/0.7972; s_kernel_rff@1000: J -0.6298/0.6298, S_plug -0.7479/0.7479; s_kernel_rff@4000: J -0.5116/0.5117, S_plug -0.6309/0.6339
- C1_gauss_mid N16384: vcs@250: J -0.0623/0.0628, S_plug -0.0187/0.0255; vcs@1000: J -0.0443/0.0444, S_plug -0.0150/0.0176; vcs@4000: J -0.0414/0.0415, S_plug -0.0152/0.0200; js@250: J -0.0563/0.0567, S_plug -0.0153/0.0267; js@1000: J -0.0414/0.0416, S_plug +0.0033/0.0088; js@4000: J -0.0383/0.0385, S_plug -0.0065/0.0113; s_kde@None: J -0.5976/0.5976, S_plug -0.7349/0.7349; s_kernel_rff@250: J -0.7481/0.7481, S_plug -0.7998/0.7998; s_kernel_rff@1000: J -0.6003/0.6004, S_plug -0.7629/0.7629; s_kernel_rff@4000: J -0.4242/0.4244, S_plug -0.6099/0.6099
- C1_gauss_mid N256: vcs@250: J -0.5517/0.5528, S_plug -0.3103/0.3256; vcs@1000: J -0.5446/0.5455, S_plug -0.3892/0.4031; vcs@4000: J -0.5446/0.5455, S_plug -0.3892/0.4031; js@250: J -0.5532/0.5540, S_plug -0.3341/0.3343; js@1000: J -0.5658/0.5671, S_plug -0.3822/0.3935; js@4000: J -0.5658/0.5671, S_plug -0.3822/0.3935
- C1_gauss_mid N4096: vcs@250: J -0.1068/0.1071, S_plug -0.0801/0.1029; vcs@1000: J -0.0962/0.0965, S_plug -0.0677/0.0850; vcs@4000: J -0.0921/0.0922, S_plug -0.0747/0.0826; js@250: J -0.0963/0.0964, S_plug -0.0633/0.0636; js@1000: J -0.0862/0.0863, S_plug -0.0525/0.0531; js@4000: J -0.0861/0.0862, S_plug -0.0481/0.0500; s_kde@None: J -0.6012/0.6012, S_plug -0.7352/0.7352; s_kernel_rff@250: J -0.7498/0.7498, S_plug -0.7999/0.7999; s_kernel_rff@1000: J -0.6071/0.6071, S_plug -0.7607/0.7607; s_kernel_rff@4000: J -0.4440/0.4442, S_plug -0.6038/0.6039
- C2_gauss_pad N1024: vcs@250: J -0.5506/0.5506, S_plug -0.5300/0.5300; vcs@1000: J -0.5506/0.5506, S_plug -0.5300/0.5300; vcs@4000: J -0.5506/0.5506, S_plug -0.5300/0.5300; js@250: J -0.5433/0.5433, S_plug -0.5373/0.5373; js@1000: J -0.5433/0.5433, S_plug -0.5373/0.5373; js@4000: J -0.5433/0.5433, S_plug -0.5373/0.5373; s_kde@None: J -0.5365/0.5365, S_plug -0.5401/0.5401; s_kernel_rff@250: J -0.5406/0.5406, S_plug -0.5403/0.5403; s_kernel_rff@1000: J -0.5406/0.5406, S_plug -0.5403/0.5403; s_kernel_rff@4000: J -0.5406/0.5406, S_plug -0.5403/0.5403
- C2_gauss_pad N16384: vcs@250: J -0.1363/0.1365, S_plug +0.0233/0.0368; vcs@1000: J -0.1359/0.1364, S_plug +0.0319/0.0679; vcs@4000: J -0.1359/0.1364, S_plug +0.0319/0.0679; js@250: J -0.1427/0.1429, S_plug +0.0199/0.0565; js@1000: J -0.1406/0.1409, S_plug +0.0291/0.0497; js@4000: J -0.1406/0.1409, S_plug +0.0291/0.0497; s_kde@None: J -0.5364/0.5364, S_plug -0.5407/0.5407; s_kernel_rff@250: J -0.5402/0.5402, S_plug -0.5405/0.5405; s_kernel_rff@1000: J -0.5400/0.5400, S_plug -0.5397/0.5397; s_kernel_rff@4000: J -0.5400/0.5400, S_plug -0.5397/0.5397
- C2_gauss_pad N256: vcs@250: J -0.5623/0.5630, S_plug -0.5164/0.5173; vcs@1000: J -0.5623/0.5630, S_plug -0.5164/0.5173; vcs@4000: J -0.5623/0.5630, S_plug -0.5164/0.5173; js@250: J -0.5512/0.5515, S_plug -0.5280/0.5283; js@1000: J -0.5512/0.5515, S_plug -0.5280/0.5283; js@4000: J -0.5512/0.5515, S_plug -0.5280/0.5283
- C2_gauss_pad N4096: vcs@250: J -0.5477/0.5477, S_plug -0.4752/0.5099; vcs@1000: J -0.5477/0.5477, S_plug -0.4752/0.5099; vcs@4000: J -0.4961/0.5052, S_plug -0.3011/0.4340; js@250: J -0.5420/0.5420, S_plug -0.5382/0.5382; js@1000: J -0.5420/0.5420, S_plug -0.5382/0.5382; js@4000: J -0.5420/0.5420, S_plug -0.5382/0.5382; s_kde@None: J -0.5361/0.5361, S_plug -0.5403/0.5403; s_kernel_rff@250: J -0.5402/0.5402, S_plug -0.5401/0.5401; s_kernel_rff@1000: J -0.5404/0.5404, S_plug -0.5396/0.5396; s_kernel_rff@4000: J -0.5404/0.5404, S_plug -0.5396/0.5396
- C3_xor N1024: vcs@250: J -0.8411/0.8411, S_plug -0.8244/0.8244; vcs@1000: J -0.8411/0.8411, S_plug -0.8244/0.8244; vcs@4000: J -0.8411/0.8411, S_plug -0.8244/0.8244; js@250: J -0.8370/0.8370, S_plug -0.8295/0.8295; js@1000: J -0.8370/0.8370, S_plug -0.8295/0.8295; js@4000: J -0.8370/0.8370, S_plug -0.8295/0.8295
- C3_xor N16384: vcs@250: J -0.5785/0.5788, S_plug -0.6837/0.6840; vcs@1000: J -0.1688/0.1689, S_plug -0.1049/0.1052; vcs@4000: J -0.1628/0.1629, S_plug -0.0614/0.0631; js@250: J -0.6088/0.6092, S_plug -0.7028/0.7029; js@1000: J -0.1683/0.1688, S_plug -0.0915/0.0918; js@4000: J -0.1673/0.1680, S_plug -0.0864/0.0880
- C3_xor N256: vcs@250: J -0.8522/0.8523, S_plug -0.8147/0.8148; vcs@1000: J -0.8522/0.8523, S_plug -0.8147/0.8148; vcs@4000: J -0.8522/0.8523, S_plug -0.8147/0.8148; js@250: J -0.8390/0.8390, S_plug -0.8280/0.8280; js@1000: J -0.8390/0.8390, S_plug -0.8280/0.8280; js@4000: J -0.8390/0.8390, S_plug -0.8280/0.8280
- C3_xor N4096: vcs@250: J -0.7004/0.7007, S_plug -0.6043/0.6043; vcs@1000: J -0.5680/0.5690, S_plug -0.3544/0.3752; vcs@4000: J -0.5597/0.5606, S_plug -0.3194/0.3463; js@250: J -0.7237/0.7239, S_plug -0.6112/0.6112; js@1000: J -0.5752/0.5762, S_plug -0.4097/0.4170; js@4000: J -0.5698/0.5709, S_plug -0.3792/0.3872

## Field matrix

```
{
 "P85/P86": {
  "oracle_S": true,
  "oracle_MI": true,
  "selected_weights": false,
  "per_sample_T_eval": false,
  "bootstrap_values": true,
  "select_curve": true,
  "critic_gradient_norms": false,
  "generator_channel_derivative": true,
  "nonfinite_steps": true,
  "cost_fields": true,
  "saturation_summary": true,
  "seeds": 3
 },
 "P108": {
  "oracle_S": true,
  "selected_weights": false,
  "per_sample_T_eval": false,
  "bootstrap_values": false,
  "budgets": [
   250,
   1000,
   4000
  ],
  "critic_gradient_norms": false,
  "cost_fields": true,
  "seeds": "5 (E1) / 3 (E2)"
 },
 "missing_for_O1_s12.2": [
  "critic-parameter gradient norms per update (gradient-norm variance)",
  "critic-output histograms (only quantiles / saturation fractions)",
  "per-sample T on EVAL (needed for paired bootstrap across methods)",
  "repeated independent EVAL batches for a fixed fit (only a bootstrap of one EVAL block)"
 ]
}
```