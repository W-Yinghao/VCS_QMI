# R6-02 controlled validation — tables (descriptive)

## REG (synthetic two-modality scenes, known shift; scenes are the units)

| condition | method | scenes | success ≤ 0.5 px | median err | p90 err | capture r = 1 / 2 / 4 / 6 | median sharpness |
|---|---|---|---|---|---|---|---|
| main_nonmonotone | oracle | 60 | 1.00 | 0.16 | 0.24 | 1.00 / 1.00 / 1.00 / 0.97 | 0.73 |
| main_nonmonotone | vcs | 60 | 1.00 | 0.12 | 0.20 | 1.00 / 1.00 / 1.00 / 0.98 | 0.75 |
| main_nonmonotone | logistic | 60 | 1.00 | 0.11 | 0.18 | 1.00 / 1.00 / 1.00 / 0.97 | 0.70 |
| main_nonmonotone | mi | 60 | 1.00 | 0.17 | 0.33 | 1.00 / 1.00 / 0.98 / 0.88 | 1.64 |
| main_nonmonotone | nmi | 60 | 1.00 | 0.19 | 0.33 | 1.00 / 0.98 / 1.00 / 0.93 | 1.78 |
| main_nonmonotone | corr | 60 | 0.02 | 3.14 | 9.56 | 0.03 / 0.02 / 0.02 / 0.02 | 0.28 |
| main_monotone | oracle | 60 | 1.00 | 0.19 | 0.36 | 1.00 / 1.00 / 1.00 / 1.00 | 0.41 |
| main_monotone | vcs | 60 | 1.00 | 0.19 | 0.35 | 1.00 / 0.98 / 1.00 / 0.98 | 0.41 |
| main_monotone | logistic | 60 | 1.00 | 0.19 | 0.36 | 1.00 / 1.00 / 1.00 / 0.98 | 0.39 |
| main_monotone | mi | 60 | 0.97 | 0.22 | 0.41 | 1.00 / 1.00 / 1.00 / 0.95 | 1.13 |
| main_monotone | nmi | 60 | 0.95 | 0.22 | 0.45 | 1.00 / 1.00 / 0.98 / 0.97 | 1.32 |
| main_monotone | corr | 60 | 1.00 | 0.22 | 0.40 | 1.00 / 1.00 / 1.00 / 0.98 | 0.18 |
| periodic | vcs | 60 | 0.03 | 6.01 | 9.04 | 1.00 / 0.42 / 0.10 / 0.02 | 1.99 |
| periodic | logistic | 60 | 0.05 | 6.00 | 9.34 | 1.00 / 0.53 / 0.17 / 0.02 | 1.98 |
| periodic | mi | 60 | 0.07 | 5.99 | 9.04 | 0.15 / 0.30 / 0.07 / 0.08 | 3.18 |
| periodic | nmi | 60 | 0.03 | 6.03 | 9.04 | 0.13 / 0.27 / 0.05 / 0.05 | 3.27 |
| periodic | corr | 60 | 0.00 | 6.43 | 10.60 | 0.05 / 0.10 / 0.05 / 0.02 | 2.93 |
| weak | oracle | 60 | 0.38 | 0.65 | 1.11 | 0.48 / 0.53 / 0.52 / 0.42 | 0.58 |
| weak | vcs | 60 | 0.32 | 0.71 | 1.12 | 0.52 / 0.47 / 0.50 / 0.42 | 0.60 |
| weak | logistic | 60 | 0.38 | 0.66 | 1.12 | 0.50 / 0.63 / 0.62 / 0.48 | 0.60 |
| weak | mi | 60 | 0.13 | 0.88 | 1.66 | 0.30 / 0.45 / 0.27 / 0.20 | 1.58 |
| weak | nmi | 60 | 0.15 | 0.92 | 1.60 | 0.22 / 0.35 / 0.23 / 0.25 | 1.46 |
| weak | corr | 60 | 0.07 | 3.57 | 10.13 | 0.08 / 0.07 / 0.05 / 0.03 | 1.61 |
| scene_offset_within | vcs | 60 | 0.85 | 0.28 | 0.53 | 0.80 / 0.80 / 0.87 / 0.73 | 0.45 |
| scene_offset_within | logistic | 60 | 0.87 | 0.27 | 0.52 | 0.87 / 0.87 / 0.88 / 0.80 | 0.38 |
| scene_offset_within | mi | 60 | 1.00 | 0.15 | 0.26 | 1.00 / 1.00 / 1.00 / 0.93 | 1.69 |
| scene_offset_within | nmi | 60 | 1.00 | 0.17 | 0.30 | 1.00 / 1.00 / 1.00 / 0.93 | 1.92 |
| scene_offset_within | corr | 60 | 0.18 | 2.23 | 10.30 | 0.22 / 0.22 / 0.20 / 0.20 | 0.28 |
| scene_offset_cross | vcs | 60 | 0.78 | 0.34 | 0.80 | 0.72 / 0.73 / 0.73 / 0.72 | 0.52 |
| scene_offset_cross | logistic | 60 | 0.72 | 0.32 | 1.08 | 0.70 / 0.73 / 0.72 / 0.70 | 0.45 |
| scene_offset_cross | mi | 60 | 1.00 | 0.16 | 0.26 | 1.00 / 1.00 / 1.00 / 0.85 | 1.80 |
| scene_offset_cross | nmi | 60 | 1.00 | 0.16 | 0.31 | 1.00 / 1.00 / 1.00 / 0.97 | 1.95 |
| scene_offset_cross | corr | 60 | 0.12 | 3.00 | 10.57 | 0.13 / 0.12 / 0.13 / 0.07 | 0.30 |
| coordinate_leak | vcs | 60 | 0.17 | 0.86 | 1.47 | 0.18 / 0.18 / 0.15 / 0.17 | 0.04 |
| coordinate_leak | logistic | 60 | 0.18 | 0.90 | 1.39 | 0.18 / 0.18 / 0.17 / 0.18 | 0.02 |
| coordinate_leak | mi | 60 | 1.00 | 0.16 | 0.28 | 1.00 / 1.00 / 1.00 / 0.87 | 1.82 |
| coordinate_leak | nmi | 60 | 1.00 | 0.17 | 0.31 | 1.00 / 1.00 / 0.98 / 0.85 | 2.01 |
| coordinate_leak | corr | 60 | 0.13 | 2.26 | 9.94 | 0.20 / 0.18 / 0.15 / 0.17 | 0.31 |
| frame_overlap | vcs | 60 | 0.53 | 0.48 | 0.82 | 0.45 / 0.50 / 0.48 / 0.45 | 0.40 |
| frame_overlap | logistic | 60 | 0.17 | 0.71 | 0.99 | 0.17 / 0.23 / 0.17 / 0.23 | 0.32 |
| frame_overlap | mi | 60 | 0.55 | 0.43 | 4.70 | 0.78 / 0.77 / 0.60 / 0.50 | 1.00 |
| frame_overlap | nmi | 60 | 0.48 | 0.55 | 4.75 | 0.77 / 0.68 / 0.58 / 0.57 | 1.03 |
| frame_overlap | corr | 60 | 0.00 | 3.57 | 4.76 | 0.00 / 0.00 / 0.00 / 0.00 | 0.37 |
| frame_fixed | vcs | 60 | 1.00 | 0.12 | 0.26 | 1.00 / 1.00 / 1.00 / 0.98 | 0.76 |
| frame_fixed | logistic | 60 | 1.00 | 0.11 | 0.25 | 1.00 / 1.00 / 1.00 / 1.00 | 0.72 |
| frame_fixed | mi | 60 | 1.00 | 0.15 | 0.27 | 1.00 / 1.00 / 0.98 / 0.92 | 1.77 |
| frame_fixed | nmi | 60 | 0.98 | 0.17 | 0.35 | 1.00 / 1.00 / 1.00 / 0.92 | 1.97 |
| frame_fixed | corr | 60 | 0.10 | 1.75 | 9.37 | 0.15 / 0.12 / 0.13 / 0.08 | 0.28 |

Finite-difference vs analytic dF/dφ (VCS critic, non-integer shifts): max |diff| = 9.7e-09 over 72 checks.

## GAIN (balanced binary constructions; label noise ε = 0.1 for the fitted cells)

| construction | exact Δ (ε = 0) | exact Δ (ε = 0.1) |
|---|---|---|
| redundant | 0.000 | 0.000 |
| irrelevant | 0.000 | 0.000 |
| xor | 1.000 | 0.640 |
| context | 0.500 | 0.320 |
| context_noise | 0.500 | 0.320 |

| construction | family | class | DEV mean z | mean h | mean residual | risk cheap / full | risk at budget .5: random / uncertainty / gain-square / gain-utility / oracle d |
|---|---|---|---|---|---|---|---|
| redundant | logistic | interaction | 0.000 | -0.000 ± 0.000 | -0.000 | 0.357 / 0.357 | 0.357 / 0.357 / 0.357 / 0.357 / 0.357 |
| redundant | logistic | linear | 0.000 | -0.000 ± 0.000 | -0.000 | 0.357 / 0.357 | 0.357 / 0.357 / 0.357 / 0.357 / 0.357 |
| redundant | square | interaction | 0.000 | -0.000 ± 0.000 | -0.000 | 0.357 / 0.357 | 0.357 / 0.357 / 0.357 / 0.357 / 0.357 |
| redundant (worse full) | square | interaction | 0.641 | -0.638 ± 0.016 | -1.279 | 0.362 / 1.000 | 0.682 / 0.680 / 0.682 / 0.682 / 0.683 |
| redundant | square | linear | 0.000 | -0.000 ± 0.000 | -0.000 | 0.357 / 0.357 | 0.357 / 0.357 / 0.357 / 0.357 / 0.357 |
| irrelevant | logistic | interaction | 0.000 | -0.000 ± 0.001 | -0.001 | 0.355 / 0.355 | 0.355 / 0.355 / 0.355 / 0.355 / 0.355 |
| irrelevant | logistic | linear | 0.000 | -0.000 ± 0.000 | -0.000 | 0.355 / 0.355 | 0.355 / 0.355 / 0.355 / 0.355 / 0.355 |
| irrelevant | square | interaction | 0.000 | -0.000 ± 0.001 | -0.001 | 0.355 / 0.355 | 0.355 / 0.355 / 0.355 / 0.355 / 0.355 |
| irrelevant (worse full) | square | interaction | 0.649 | -0.645 ± 0.015 | -1.294 | 0.355 / 1.000 | 0.677 / 0.678 / 0.677 / 0.681 / 0.676 |
| irrelevant | square | linear | 0.000 | -0.000 ± 0.000 | -0.000 | 0.355 / 0.355 | 0.355 / 0.355 / 0.355 / 0.355 / 0.355 |
| xor | logistic | interaction | 0.643 | +0.638 ± 0.013 | -0.005 | 1.001 / 0.363 | 0.681 / 0.680 / 0.684 / 0.683 / 0.678 |
| xor | logistic | linear | 0.000 | -0.000 ± 0.001 | -0.001 | 1.001 / 1.001 | 1.001 / 1.002 / 1.003 / 0.988 / 1.001 |
| xor | square | interaction | 0.643 | +0.638 ± 0.013 | -0.005 | 1.001 / 0.363 | 0.681 / 0.680 / 0.684 / 0.683 / 0.678 |
| xor (worse full) | square | interaction | 0.001 | +0.001 ± 0.002 | +0.000 | 1.001 / 1.000 | 1.000 / 1.001 / 1.000 / 1.000 / 1.000 |
| xor | square | linear | 0.000 | -0.000 ± 0.001 | -0.001 | 1.001 / 1.001 | 1.001 / 1.002 / 1.000 / 0.988 / 1.001 |
| context | logistic | interaction | 0.320 | +0.320 ± 0.014 | -0.000 | 0.684 / 0.364 | 0.523 / 0.366 / 0.365 / 0.365 / 0.365 |
| context | logistic | linear | 0.000 | -0.000 ± 0.001 | -0.001 | 0.684 / 0.684 | 0.684 / 0.684 / 0.684 / 0.676 / 0.684 |
| context | square | interaction | 0.321 | +0.320 ± 0.014 | -0.001 | 0.684 / 0.364 | 0.523 / 0.366 / 0.365 / 0.365 / 0.365 |
| context (worse full) | square | interaction | 0.327 | -0.319 ± 0.011 | -0.646 | 0.681 / 1.000 | 0.837 / 0.682 / 0.998 / 0.682 / 0.682 |
| context | square | linear | 0.000 | -0.000 ± 0.000 | -0.001 | 0.684 / 0.684 | 0.684 / 0.684 / 0.682 / 0.678 / 0.684 |
| context_noise | logistic | interaction | 0.329 | +0.321 ± 0.009 | -0.009 | 1.002 / 0.682 | 0.840 / 0.818 / 0.684 / 0.684 / 0.684 |
| context_noise | logistic | linear | 0.000 | -0.001 ± 0.001 | -0.001 | 1.002 / 1.003 | 1.002 / 1.002 / 1.003 / 0.996 / 1.002 |
| context_noise | square | interaction | 0.330 | +0.321 ± 0.009 | -0.009 | 1.002 / 0.682 | 0.840 / 0.818 / 0.684 / 0.684 / 0.684 |
| context_noise (worse full) | square | interaction | 0.002 | +0.002 ± 0.002 | +0.000 | 1.002 / 1.000 | 1.001 / 1.002 / 1.000 / 1.001 / 1.001 |
| context_noise | square | linear | 0.000 | -0.001 ± 0.001 | -0.001 | 1.002 / 1.003 | 1.002 / 1.002 / 1.001 / 0.996 / 1.002 |

Reference counterexample (redundant; a = sign, b = 0): {'square': 1.0, 'residual': -2.0, 'actual_gain': -1.0}
