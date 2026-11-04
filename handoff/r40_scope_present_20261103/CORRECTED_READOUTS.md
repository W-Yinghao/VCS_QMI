Publication copy (sample names aliased; numbers and conclusions unchanged).

# Corrected (layout-free) DEV readouts (numbers only; zero training; TEST not read)

logged = trainer readout (native per-batch centring over the sequential native layout). full_global / full_decoupled = parent cache, s_H, stage-1 and (JOINT) student centring recomputed with one global DEV mean / independent random batches. Gains are relative to the parent H under the same centring. See R44_AUDIT.md.

units: 418; status: {'EVALUATED': 415, 'PENDING': 3}

## R37

| unit | enc | logged | logged-H | full_global | global-H(global) | full_decoupled | decoupled-H(decoupled) | student_global (JOINT) | status |
|---|---|---|---|---|---|---|---|---|---|
| r37_confirm_f60_h1002_N_G_BCE | frozen | 485.11 | 2.17 | 472.05 | -0.67 | 469.17 | 0.57 | NA | EVALUATED |
| r37_confirm_f60_h1002_N_G_J | frozen | 484.42 | 1.48 | 472.94 | 0.22 | 470.20 | 1.60 | NA | EVALUATED |
| r37_confirm_f60_h1002_N_G_NCE | frozen | 484.99 | 2.05 | 473.20 | 0.47 | 469.21 | 0.61 | NA | EVALUATED |
| r37_confirm_f60_h1002_T_G_BCE | frozen | 429.59 | -53.35 | 428.54 | -44.18 | 422.05 | -46.55 | NA | EVALUATED |
| r37_confirm_f60_h1002_T_G_J | frozen | 483.67 | 0.73 | 471.30 | -1.42 | 468.56 | -0.04 | NA | EVALUATED |
| r37_confirm_f60_h1002_T_G_NCE | frozen | 437.00 | -45.94 | 435.13 | -37.59 | 431.05 | -37.55 | NA | EVALUATED |
| r37_confirm_f60_h1003_N_G_BCE | frozen | 497.36 | 1.28 | 484.38 | 1.05 | 481.99 | -0.81 | NA | EVALUATED |
| r37_confirm_f60_h1003_N_G_J | frozen | 497.73 | 1.66 | 484.24 | 0.91 | 482.50 | -0.30 | NA | EVALUATED |
| r37_confirm_f60_h1003_N_G_NCE | frozen | 497.99 | 1.91 | 484.18 | 0.85 | 481.20 | -1.60 | NA | EVALUATED |
| r37_confirm_f60_h1003_T_G_BCE | frozen | 479.19 | -16.88 | 471.72 | -11.62 | 469.35 | -13.45 | NA | EVALUATED |
| r37_confirm_f60_h1003_T_G_J | frozen | 496.79 | 0.71 | 483.45 | 0.12 | 480.95 | -1.85 | NA | EVALUATED |
| r37_confirm_f60_h1003_T_G_NCE | frozen | 482.84 | -13.23 | 470.59 | -12.74 | 465.84 | -16.96 | NA | EVALUATED |
| r37_factor_f60_h1001_N_PRESENT_ONLY_G_BCE | frozen | 507.02 | 1.81 | 492.88 | -3.75 | 490.63 | -2.92 | NA | EVALUATED |
| r37_factor_f60_h1001_N_PRESENT_ONLY_G_J | frozen | 507.44 | 2.23 | 493.69 | -2.94 | 490.59 | -2.96 | NA | EVALUATED |
| r37_factor_f60_h1001_N_PRESENT_ONLY_G_NCE | frozen | 505.96 | 0.75 | 491.85 | -4.77 | 489.72 | -3.83 | NA | EVALUATED |
| r37_factor_f60_h1001_TRUST_NATIVE_NULL_G_BCE | frozen | 507.40 | 2.19 | 492.92 | -3.61 | 490.06 | -3.49 | NA | EVALUATED |
| r37_factor_f60_h1001_TRUST_NATIVE_NULL_G_J | frozen | 507.81 | 2.60 | 493.41 | -3.12 | 491.34 | -2.21 | NA | EVALUATED |
| r37_factor_f60_h1001_TRUST_NATIVE_NULL_G_NCE | frozen | 507.99 | 2.78 | 492.90 | -3.63 | 490.26 | -3.29 | NA | EVALUATED |

## R37C

| unit | enc | logged | logged-H | full_global | global-H(global) | full_decoupled | decoupled-H(decoupled) | student_global (JOINT) | status |
|---|---|---|---|---|---|---|---|---|---|
| r37c_pilot_cc_h1001_N_G_BCE | frozen | 393.10 | 3.40 | 390.20 | 5.10 | 387.90 | 3.40 | NA | EVALUATED |
| r37c_pilot_cc_h1001_N_G_J | frozen | 392.30 | 2.60 | 389.90 | 4.80 | 388.00 | 3.50 | NA | EVALUATED |
| r37c_pilot_cc_h1001_N_G_NCE | frozen | 387.50 | -2.20 | 386.30 | 1.20 | 382.60 | -1.90 | NA | EVALUATED |
| r37c_pilot_cc_h1001_T_G_BCE | frozen | 3.10 | -386.60 | 3.50 | -381.60 | 3.80 | -380.70 | NA | EVALUATED |
| r37c_pilot_cc_h1001_T_G_J | frozen | 5.20 | -384.50 | 5.30 | -379.80 | 5.50 | -379.00 | NA | EVALUATED |
| r37c_pilot_cc_h1001_T_G_NCE | frozen | 2.50 | -387.20 | 2.70 | -382.40 | 2.50 | -382.00 | NA | EVALUATED |

## R37S

| unit | enc | logged | logged-H | full_global | global-H(global) | full_decoupled | decoupled-H(decoupled) | student_global (JOINT) | status |
|---|---|---|---|---|---|---|---|---|---|
| r37s_cc_r30h_s1001_N_directional_BCE_constant | frozen | 392.50 | -0.70 | 389.20 | 2.20 | 387.90 | 0.50 | NA | EVALUATED |
| r37s_cc_r30h_s1001_N_directional_BCE_cosine | frozen | 393.20 | 0.00 | 389.10 | 2.10 | 388.80 | 1.40 | NA | EVALUATED |
| r37s_cc_r30h_s1001_N_directional_J_constant | frozen | 392.30 | -0.90 | 390.10 | 3.10 | 389.00 | 1.60 | NA | EVALUATED |
| r37s_cc_r30h_s1001_N_directional_J_cosine | frozen | 393.40 | 0.20 | 389.40 | 2.40 | 387.70 | 0.30 | NA | EVALUATED |
| r37s_cc_r30h_s1001_N_directional_NCE_constant | frozen | 390.50 | -2.70 | 386.10 | -0.90 | 384.70 | -2.70 | NA | EVALUATED |
| r37s_cc_r30h_s1001_N_directional_NCE_cosine | frozen | 390.10 | -3.10 | 386.60 | -0.40 | 385.60 | -1.80 | NA | EVALUATED |
| r37s_cc_r30h_s1001_N_shared_BCE_constant | frozen | 392.40 | -0.80 | 388.50 | 1.50 | 387.50 | 0.10 | NA | EVALUATED |
| r37s_cc_r30h_s1001_N_shared_BCE_cosine | frozen | 393.20 | 0.00 | 389.30 | 2.30 | 388.70 | 1.30 | NA | EVALUATED |
| r37s_cc_r30h_s1001_N_shared_J_constant | frozen | 392.40 | -0.80 | 390.50 | 3.50 | 389.00 | 1.60 | NA | EVALUATED |
| r37s_cc_r30h_s1001_N_shared_J_cosine | frozen | 393.40 | 0.20 | 389.20 | 2.20 | 387.40 | 0.00 | NA | EVALUATED |
| r37s_cc_r30h_s1001_N_shared_NCE_constant | frozen | 390.20 | -3.00 | 386.00 | -1.00 | 385.00 | -2.40 | NA | EVALUATED |
| r37s_cc_r30h_s1001_N_shared_NCE_cosine | frozen | 390.30 | -2.90 | 386.50 | -0.50 | 385.90 | -1.50 | NA | EVALUATED |
| r37s_cc_r30h_s1001_T_shared_BCE_constant | frozen | 18.40 | -374.80 | 19.30 | -367.70 | 18.30 | -369.20 | NA | EVALUATED |
| r37s_cc_r30h_s1001_T_shared_BCE_cosine | frozen | 360.80 | -32.40 | 356.80 | -30.20 | 356.60 | -30.90 | NA | EVALUATED |
| r37s_cc_r30h_s1001_T_shared_J_constant | frozen | 368.80 | -24.40 | 368.00 | -19.00 | 365.50 | -22.00 | NA | EVALUATED |
| r37s_cc_r30h_s1001_T_shared_J_cosine | frozen | 392.20 | -1.00 | 390.20 | 3.20 | 389.20 | 1.70 | NA | EVALUATED |
| r37s_cc_r30h_s1001_T_shared_NCE_constant | frozen | 2.10 | -391.10 | 2.80 | -384.20 | 2.00 | -385.50 | NA | EVALUATED |
| r37s_cc_r30h_s1001_T_shared_NCE_cosine | frozen | 6.30 | -386.90 | 7.20 | -379.80 | 6.50 | -381.00 | NA | EVALUATED |

## R38

| unit | enc | logged | logged-H | full_global | global-H(global) | full_decoupled | decoupled-H(decoupled) | student_global (JOINT) | status |
|---|---|---|---|---|---|---|---|---|---|
| r38_A_cc_R31_s1002_N_shared_BCE_constant | frozen | 392.40 | -2.20 | 391.40 | -1.80 | 390.50 | -0.40 | NA | EVALUATED |
| r38_A_cc_R31_s1002_N_shared_J_constant | frozen | 394.20 | -0.40 | 392.50 | -0.70 | 391.10 | 0.20 | NA | EVALUATED |
| r38_A_cc_R31_s1002_N_shared_NCE_constant | frozen | 391.10 | -3.50 | 388.90 | -4.30 | 388.10 | -2.80 | NA | EVALUATED |
| r38_A_cc_R31_s1003_N_shared_BCE_constant | frozen | 393.50 | 0.90 | 390.90 | 1.60 | 387.20 | -0.10 | NA | EVALUATED |
| r38_A_cc_R31_s1003_N_shared_J_constant | frozen | 394.60 | 2.00 | 389.60 | 0.30 | 387.40 | 0.10 | NA | EVALUATED |
| r38_A_cc_R31_s1003_N_shared_NCE_constant | frozen | 392.50 | -0.10 | 388.50 | -0.80 | 385.80 | -1.50 | NA | EVALUATED |
| r38_B_cc_R31_s1001_N_directional_BCE_constant | frozen | 393.00 | 3.30 | 390.00 | 4.90 | 388.20 | 3.70 | NA | EVALUATED |
| r38_B_cc_R31_s1001_N_directional_BCE_cosine | frozen | 392.10 | 2.40 | 390.30 | 5.20 | 388.10 | 3.60 | NA | EVALUATED |
| r38_B_cc_R31_s1001_N_directional_J_constant | frozen | 392.50 | 2.80 | 390.20 | 5.10 | 388.00 | 3.50 | NA | EVALUATED |
| r38_B_cc_R31_s1001_N_directional_J_cosine | frozen | 392.60 | 2.90 | 390.70 | 5.60 | 389.40 | 4.90 | NA | EVALUATED |
| r38_B_cc_R31_s1001_N_directional_NCE_constant | frozen | 387.50 | -2.20 | 387.20 | 2.10 | 382.50 | -2.00 | NA | EVALUATED |
| r38_B_cc_R31_s1001_N_directional_NCE_cosine | frozen | 388.20 | -1.50 | 385.40 | 0.30 | 383.00 | -1.50 | NA | EVALUATED |
| r38_B_cc_R31_s1001_N_shared_BCE_cosine | frozen | 392.80 | 3.10 | 390.20 | 5.10 | 388.50 | 4.00 | NA | EVALUATED |
| r38_B_cc_R31_s1001_N_shared_J_cosine | frozen | 392.70 | 3.00 | 390.60 | 5.50 | 389.20 | 4.70 | NA | EVALUATED |
| r38_B_cc_R31_s1001_N_shared_NCE_cosine | frozen | 388.60 | -1.10 | 385.80 | 0.70 | 383.00 | -1.50 | NA | EVALUATED |
| r38_B_cc_R31_s1002_N_directional_BCE_constant | frozen | 392.70 | -1.90 | 391.80 | -1.40 | 391.00 | 0.10 | NA | EVALUATED |
| r38_B_cc_R31_s1002_N_directional_BCE_cosine | frozen | 393.50 | -1.10 | 390.50 | -2.70 | 391.00 | 0.10 | NA | EVALUATED |
| r38_B_cc_R31_s1002_N_directional_J_constant | frozen | 393.70 | -0.90 | 392.40 | -0.80 | 391.10 | 0.20 | NA | EVALUATED |
| r38_B_cc_R31_s1002_N_directional_J_cosine | frozen | 394.50 | -0.10 | 392.00 | -1.20 | 391.40 | 0.50 | NA | EVALUATED |
| r38_B_cc_R31_s1002_N_directional_NCE_constant | frozen | 390.40 | -4.20 | 388.10 | -5.10 | 388.30 | -2.60 | NA | EVALUATED |
| r38_B_cc_R31_s1002_N_directional_NCE_cosine | frozen | 390.40 | -4.20 | 388.30 | -4.90 | 387.60 | -3.30 | NA | EVALUATED |
| r38_B_cc_R31_s1002_N_shared_BCE_cosine | frozen | 392.90 | -1.70 | 390.50 | -2.70 | 391.00 | 0.10 | NA | EVALUATED |
| r38_B_cc_R31_s1002_N_shared_J_cosine | frozen | 394.10 | -0.50 | 392.20 | -1.00 | 391.10 | 0.20 | NA | EVALUATED |
| r38_B_cc_R31_s1002_N_shared_NCE_cosine | frozen | 390.30 | -4.30 | 388.50 | -4.70 | 388.00 | -2.90 | NA | EVALUATED |
| r38_B_cc_R31_s1003_N_directional_BCE_constant | frozen | 393.30 | 0.70 | 390.80 | 1.50 | 387.40 | 0.10 | NA | EVALUATED |
| r38_B_cc_R31_s1003_N_directional_BCE_cosine | frozen | 393.60 | 1.00 | 389.80 | 0.50 | 387.70 | 0.40 | NA | EVALUATED |
| r38_B_cc_R31_s1003_N_directional_J_constant | frozen | 394.40 | 1.80 | 390.00 | 0.70 | 388.20 | 0.90 | NA | EVALUATED |
| r38_B_cc_R31_s1003_N_directional_J_cosine | frozen | 393.80 | 1.20 | 390.00 | 0.70 | 387.70 | 0.40 | NA | EVALUATED |
| r38_B_cc_R31_s1003_N_directional_NCE_constant | frozen | 392.40 | -0.20 | 387.80 | -1.50 | 385.30 | -2.00 | NA | EVALUATED |
| r38_B_cc_R31_s1003_N_directional_NCE_cosine | frozen | 391.30 | -1.30 | 389.10 | -0.20 | 386.80 | -0.50 | NA | EVALUATED |
| r38_B_cc_R31_s1003_N_shared_BCE_cosine | frozen | 393.00 | 0.40 | 390.20 | 0.90 | 387.30 | 0.00 | NA | EVALUATED |
| r38_B_cc_R31_s1003_N_shared_J_cosine | frozen | 393.50 | 0.90 | 390.30 | 1.00 | 387.40 | 0.10 | NA | EVALUATED |
| r38_B_cc_R31_s1003_N_shared_NCE_cosine | frozen | 391.30 | -1.30 | 389.30 | 0.00 | 386.90 | -0.40 | NA | EVALUATED |
| r38_B_f60_R30_s1001_N_directional_BCE_constant | frozen | 507.73 | 2.52 | 493.57 | -2.96 | 490.77 | -2.78 | NA | EVALUATED |
| r38_B_f60_R30_s1001_N_directional_BCE_cosine | frozen | 508.76 | 3.55 | 493.47 | -3.06 | 491.78 | -1.78 | NA | EVALUATED |
| r38_B_f60_R30_s1001_N_directional_J_constant | frozen | 507.51 | 2.31 | 492.92 | -3.61 | 491.30 | -2.25 | NA | EVALUATED |
| r38_B_f60_R30_s1001_N_directional_J_cosine | frozen | 508.95 | 3.75 | 493.87 | -2.66 | 491.28 | -2.27 | NA | EVALUATED |
| r38_B_f60_R30_s1001_N_directional_NCE_constant | frozen | 505.82 | 0.61 | 491.87 | -4.65 | 490.06 | -3.49 | NA | EVALUATED |
| r38_B_f60_R30_s1001_N_directional_NCE_cosine | frozen | 506.15 | 0.95 | 492.43 | -4.10 | 489.61 | -3.94 | NA | EVALUATED |
| r38_B_f60_R30_s1001_N_shared_BCE_cosine | frozen | 508.40 | 3.20 | 493.16 | -3.37 | 491.81 | -1.74 | NA | EVALUATED |
| r38_B_f60_R30_s1001_N_shared_J_cosine | frozen | 509.15 | 3.94 | 493.63 | -2.90 | 491.44 | -2.11 | NA | EVALUATED |
| r38_B_f60_R30_s1001_N_shared_NCE_cosine | frozen | 506.69 | 1.48 | 492.49 | -4.04 | 489.57 | -3.98 | NA | EVALUATED |
| r38_B_f60_R30_s1002_N_directional_BCE_constant | frozen | 485.27 | 2.33 | 472.84 | 0.12 | 469.86 | 1.26 | NA | EVALUATED |
| r38_B_f60_R30_s1002_N_directional_BCE_cosine | frozen | 487.08 | 4.14 | 473.81 | 1.08 | 470.43 | 1.83 | NA | EVALUATED |
| r38_B_f60_R30_s1002_N_directional_J_constant | frozen | 484.26 | 1.32 | 473.59 | 0.87 | 470.18 | 1.58 | NA | EVALUATED |
| r38_B_f60_R30_s1002_N_directional_J_cosine | frozen | 486.33 | 3.39 | 474.56 | 1.83 | 470.85 | 2.25 | NA | EVALUATED |
| r38_B_f60_R30_s1002_N_directional_NCE_constant | frozen | 484.40 | 1.46 | 472.80 | 0.08 | 469.39 | 0.79 | NA | EVALUATED |
| r38_B_f60_R30_s1002_N_directional_NCE_cosine | frozen | 485.29 | 2.35 | 472.31 | -0.41 | 469.57 | 0.97 | NA | EVALUATED |
| r38_B_f60_R30_s1002_N_shared_BCE_cosine | frozen | 486.51 | 3.57 | 473.91 | 1.18 | 470.12 | 1.52 | NA | EVALUATED |
| r38_B_f60_R30_s1002_N_shared_J_cosine | frozen | 485.94 | 3.00 | 474.22 | 1.50 | 470.71 | 2.11 | NA | EVALUATED |
| r38_B_f60_R30_s1002_N_shared_NCE_cosine | frozen | 485.74 | 2.80 | 473.00 | 0.28 | 470.36 | 1.76 | NA | EVALUATED |
| r38_B_f60_R30_s1003_N_directional_BCE_constant | frozen | 497.26 | 1.18 | 484.40 | 1.07 | 482.82 | 0.02 | NA | EVALUATED |
| r38_B_f60_R30_s1003_N_directional_BCE_cosine | frozen | 497.89 | 1.81 | 485.11 | 1.78 | 483.31 | 0.51 | NA | EVALUATED |
| r38_B_f60_R30_s1003_N_directional_J_constant | frozen | 497.04 | 0.97 | 483.57 | 0.24 | 482.96 | 0.16 | NA | EVALUATED |
| r38_B_f60_R30_s1003_N_directional_J_cosine | frozen | 498.36 | 2.29 | 485.07 | 1.74 | 483.25 | 0.45 | NA | EVALUATED |
| r38_B_f60_R30_s1003_N_directional_NCE_constant | frozen | 497.48 | 1.40 | 483.93 | 0.59 | 481.22 | -1.58 | NA | EVALUATED |
| r38_B_f60_R30_s1003_N_directional_NCE_cosine | frozen | 497.51 | 1.44 | 483.85 | 0.51 | 480.49 | -2.31 | NA | EVALUATED |
| r38_B_f60_R30_s1003_N_shared_BCE_cosine | frozen | 497.91 | 1.83 | 484.81 | 1.48 | 483.06 | 0.26 | NA | EVALUATED |
| r38_B_f60_R30_s1003_N_shared_J_cosine | frozen | 498.46 | 2.39 | 484.89 | 1.56 | 483.08 | 0.28 | NA | EVALUATED |
| r38_B_f60_R30_s1003_N_shared_NCE_cosine | frozen | 498.48 | 2.41 | 484.12 | 0.79 | 481.18 | -1.62 | NA | EVALUATED |
| r38_C_cc_R31_s1001_T_shared_BCE_cosine | frozen | 240.20 | -149.50 | 242.00 | -143.10 | 243.00 | -141.50 | NA | EVALUATED |
| r38_C_cc_R31_s1001_T_shared_J_cosine | frozen | 381.70 | -8.00 | 381.00 | -4.10 | 381.00 | -3.50 | NA | EVALUATED |
| r38_C_cc_R31_s1001_T_shared_NCE_cosine | frozen | 2.40 | -387.30 | 3.50 | -381.60 | 3.30 | -381.20 | NA | EVALUATED |
| r38_C_cc_R31_s1002_T_shared_BCE_constant | frozen | 3.00 | -391.60 | 3.70 | -389.50 | 3.50 | -387.40 | NA | EVALUATED |
| r38_C_cc_R31_s1002_T_shared_BCE_cosine | frozen | 219.30 | -175.30 | 221.10 | -172.10 | 219.40 | -171.50 | NA | EVALUATED |
| r38_C_cc_R31_s1002_T_shared_J_constant | frozen | 4.30 | -390.30 | 5.40 | -387.80 | 5.00 | -385.90 | NA | EVALUATED |
| r38_C_cc_R31_s1002_T_shared_J_cosine | frozen | 386.00 | -8.60 | 385.70 | -7.50 | 383.40 | -7.50 | NA | EVALUATED |
| r38_C_cc_R31_s1002_T_shared_NCE_constant | frozen | 2.20 | -392.40 | 1.80 | -391.40 | 2.20 | -388.70 | NA | EVALUATED |
| r38_C_cc_R31_s1002_T_shared_NCE_cosine | frozen | 3.00 | -391.60 | 2.80 | -390.40 | 3.50 | -387.40 | NA | EVALUATED |
| r38_C_cc_R31_s1003_T_shared_BCE_constant | frozen | 2.90 | -389.70 | 4.20 | -385.10 | 3.80 | -383.50 | NA | EVALUATED |
| r38_C_cc_R31_s1003_T_shared_BCE_cosine | frozen | 262.60 | -130.00 | 264.10 | -125.20 | 260.10 | -127.20 | NA | EVALUATED |
| r38_C_cc_R31_s1003_T_shared_J_constant | frozen | 3.80 | -388.80 | 4.90 | -384.40 | 4.50 | -382.80 | NA | EVALUATED |
| r38_C_cc_R31_s1003_T_shared_J_cosine | frozen | 384.00 | -8.60 | 383.30 | -6.00 | 381.00 | -6.30 | NA | EVALUATED |
| r38_C_cc_R31_s1003_T_shared_NCE_constant | frozen | 2.10 | -390.50 | 3.80 | -385.50 | 3.60 | -383.70 | NA | EVALUATED |
| r38_C_cc_R31_s1003_T_shared_NCE_cosine | frozen | 2.00 | -390.60 | 3.40 | -385.90 | 2.90 | -384.40 | NA | EVALUATED |
| r38_C_f60_R30_s1001_T_shared_BCE_cosine | frozen | 506.59 | 1.38 | 491.70 | -4.93 | 490.04 | -3.51 | NA | EVALUATED |
| r38_C_f60_R30_s1001_T_shared_J_cosine | frozen | 508.01 | 2.80 | 492.84 | -3.79 | 490.65 | -2.90 | NA | EVALUATED |
| r38_C_f60_R30_s1001_T_shared_NCE_cosine | frozen | 503.53 | -1.68 | 486.65 | -9.98 | 484.71 | -8.84 | NA | EVALUATED |
| r38_C_f60_R30_s1002_T_shared_BCE_cosine | frozen | 483.75 | 0.81 | 471.52 | -1.20 | 467.87 | -0.73 | NA | EVALUATED |
| r38_C_f60_R30_s1002_T_shared_J_cosine | frozen | 485.23 | 2.29 | 472.17 | -0.55 | 469.15 | 0.55 | NA | EVALUATED |
| r38_C_f60_R30_s1002_T_shared_NCE_cosine | frozen | 484.85 | 1.91 | 473.43 | 0.71 | 469.68 | 1.08 | NA | EVALUATED |
| r38_C_f60_R30_s1003_T_shared_BCE_cosine | frozen | 494.95 | -1.12 | 481.62 | -1.72 | 480.95 | -1.85 | NA | EVALUATED |
| r38_C_f60_R30_s1003_T_shared_J_cosine | frozen | 497.53 | 1.46 | 483.23 | -0.10 | 482.37 | -0.43 | NA | EVALUATED |
| r38_C_f60_R30_s1003_T_shared_NCE_cosine | frozen | 497.59 | 1.52 | 481.62 | -1.72 | 478.90 | -3.91 | NA | EVALUATED |

## R39

| unit | enc | logged | logged-H | full_global | global-H(global) | full_decoupled | decoupled-H(decoupled) | student_global (JOINT) | status |
|---|---|---|---|---|---|---|---|---|---|
| R39_CC152K_COMPLETE_frozen_BCE | frozen | 392.60 | 2.90 | 390.30 | 5.20 | 388.20 | 3.70 | NA | EVALUATED |
| R39_CC152K_COMPLETE_frozen_J | frozen | 392.60 | 2.90 | 390.60 | 5.50 | 389.20 | 4.70 | NA | EVALUATED |
| R39_CC152K_COMPLETE_frozen_NCE | frozen | 388.60 | -1.10 | 386.80 | 1.70 | 383.70 | -0.80 | NA | EVALUATED |
| R39_CC152K_COMPLETE_joint_global_BCE | joint | 390.90 | 1.20 | 388.40 | 3.20 | 388.00 | 3.60 | 389.90 | EVALUATED |
| R39_CC152K_COMPLETE_joint_global_J | joint | 391.10 | 1.40 | 388.00 | 2.80 | 387.70 | 3.30 | 390.50 | EVALUATED |
| R39_CC152K_COMPLETE_joint_global_NCE | joint | 387.70 | -2.00 | 386.00 | 0.90 | 383.50 | -1.00 | 386.50 | EVALUATED |
| R39_CC152K_N_joint_global_BCE | joint | 390.80 | 1.10 | 388.50 | 3.30 | 388.00 | 3.60 | 390.20 | EVALUATED |
| R39_CC152K_N_joint_global_J | joint | 390.80 | 1.10 | 388.20 | 3.00 | 388.20 | 3.80 | 390.30 | EVALUATED |
| R39_CC152K_N_joint_global_NCE | joint | 386.80 | -2.90 | 385.30 | 0.20 | 383.10 | -1.40 | 386.50 | EVALUATED |
| R39_Flickr60_COMPLETE_frozen_BCE | frozen | 508.50 | 3.29 | 494.02 | -2.50 | 491.97 | -1.58 | NA | EVALUATED |
| R39_Flickr60_COMPLETE_frozen_J | frozen | 509.11 | 3.91 | 494.08 | -2.45 | 491.18 | -2.37 | NA | EVALUATED |
| R39_Flickr60_COMPLETE_frozen_NCE | frozen | 506.88 | 1.68 | 493.49 | -3.04 | 489.86 | -3.69 | NA | EVALUATED |
| R39_Flickr60_COMPLETE_joint_global_BCE | joint | 516.79 | 11.58 | 490.69 | -5.84 | 487.75 | -5.80 | 497.55 | EVALUATED |
| R39_Flickr60_COMPLETE_joint_global_J | joint | 518.13 | 12.92 | 491.16 | -5.36 | 487.85 | -5.70 | 497.51 | EVALUATED |
| R39_Flickr60_COMPLETE_joint_global_NCE | joint | 517.12 | 11.91 | 489.09 | -7.44 | 486.41 | -7.14 | 496.35 | EVALUATED |
| R39_Flickr60_N_joint_global_BCE | joint | 515.68 | 10.47 | 490.24 | -6.29 | 486.77 | -6.79 | 496.23 | EVALUATED |
| R39_Flickr60_N_joint_global_J | joint | 517.87 | 12.66 | 489.96 | -6.57 | 486.80 | -6.75 | 496.53 | EVALUATED |
| R39_Flickr60_N_joint_global_NCE | joint | 515.52 | 10.32 | 488.99 | -7.53 | 485.13 | -8.42 | 495.31 | EVALUATED |
| R39_Flickr60_ORACLE_MISSING_frozen_BCE | frozen | 509.94 | 4.73 | 493.87 | -2.66 | 492.01 | -1.54 | NA | EVALUATED |
| R39_Flickr60_ORACLE_MISSING_frozen_J | frozen | 509.92 | 4.71 | 494.02 | -2.50 | 491.89 | -1.66 | NA | EVALUATED |
| R39_Flickr60_ORACLE_MISSING_frozen_NCE | frozen | 509.39 | 4.18 | 493.71 | -2.82 | 490.73 | -2.82 | NA | EVALUATED |

## R39X

| unit | enc | logged | logged-H | full_global | global-H(global) | full_decoupled | decoupled-H(decoupled) | student_global (JOINT) | status |
|---|---|---|---|---|---|---|---|---|---|
| R39X_CC152K_R31H0_s1002_N_joint_global_BCE | joint | 391.20 | -3.40 | 389.90 | -3.30 | 388.80 | -2.10 | 390.80 | EVALUATED |
| R39X_CC152K_R31H0_s1002_N_joint_global_J | joint | 392.80 | -1.80 | 389.50 | -3.70 | 389.50 | -1.40 | 392.90 | EVALUATED |
| R39X_CC152K_R31H0_s1002_N_joint_global_NCE | joint | 390.00 | -4.60 | 386.00 | -7.20 | 385.60 | -5.30 | 390.30 | EVALUATED |
| R39X_CC152K_R31H0_s1003_N_joint_global_BCE | joint | 393.40 | 0.80 | 389.90 | 0.60 | 387.00 | -0.30 | 392.50 | EVALUATED |
| R39X_CC152K_R31H0_s1003_N_joint_global_J | joint | 391.40 | -1.20 | 389.40 | 0.10 | 386.40 | -0.90 | 392.10 | EVALUATED |
| R39X_CC152K_R31H0_s1003_N_joint_global_NCE | joint | 391.10 | -1.50 | 387.10 | -2.20 | 385.70 | -1.60 | 390.00 | EVALUATED |
| R39X_CC152K_R37S_s1001_N_joint_global_BCE | joint | 391.10 | -2.10 | 389.00 | 2.00 | 387.70 | 0.30 | 391.20 | EVALUATED |
| R39X_CC152K_R37S_s1001_N_joint_global_J | joint | 390.20 | -3.00 | 386.90 | -0.10 | 386.50 | -0.90 | 389.50 | EVALUATED |
| R39X_CC152K_R37S_s1001_N_joint_global_NCE | joint | 389.70 | -3.50 | 386.60 | -0.40 | 385.00 | -2.40 | 388.90 | EVALUATED |
| R39X_Flickr00_R30H_s1001_N_frozen_shared_BCE | frozen | 521.95 | -0.12 | 510.95 | -0.91 | 509.43 | -1.40 | NA | EVALUATED |
| R39X_Flickr00_R30H_s1001_N_frozen_shared_J | frozen | 522.78 | 0.71 | 509.94 | -1.91 | 509.05 | -1.78 | NA | EVALUATED |
| R39X_Flickr00_R30H_s1001_N_frozen_shared_NCE | frozen | 524.24 | 2.17 | 510.30 | -1.56 | 510.30 | -0.53 | NA | EVALUATED |
| R39X_Flickr00_R30H_s1001_N_joint_global_BCE | joint | 525.66 | 3.59 | 510.22 | -1.64 | 510.06 | -0.77 | 519.59 | EVALUATED |
| R39X_Flickr00_R30H_s1001_N_joint_global_J | joint | 527.59 | 5.52 | 511.78 | -0.08 | 510.20 | -0.63 | 520.41 | EVALUATED |
| R39X_Flickr00_R30H_s1001_N_joint_global_NCE | joint | 551.08 | 29.01 | 497.69 | -14.16 | 495.36 | -15.46 | 504.65 | EVALUATED |
| R39X_Flickr60_R30H_s1002_N_joint_global_BCE | joint | 491.79 | 8.86 | 470.53 | -2.19 | 466.55 | -2.05 | 477.73 | EVALUATED |
| R39X_Flickr60_R30H_s1002_N_joint_global_J | joint | 493.39 | 10.45 | 470.77 | -1.95 | 466.69 | -1.91 | 478.64 | EVALUATED |
| R39X_Flickr60_R30H_s1002_N_joint_global_NCE | joint | 491.18 | 8.24 | 470.59 | -2.13 | 464.77 | -3.83 | 477.34 | EVALUATED |
| R39X_Flickr60_R30H_s1003_N_joint_global_BCE | joint | 508.66 | 12.58 | 482.78 | -0.55 | 479.41 | -3.39 | 489.19 | EVALUATED |
| R39X_Flickr60_R30H_s1003_N_joint_global_J | joint | 510.47 | 14.40 | 483.08 | -0.26 | 479.66 | -3.14 | 490.00 | EVALUATED |
| R39X_Flickr60_R30H_s1003_N_joint_global_NCE | joint | 508.40 | 12.33 | 481.32 | -2.01 | 478.56 | -4.24 | 489.33 | EVALUATED |

## R40

| unit | enc | logged | logged-H | full_global | global-H(global) | full_decoupled | decoupled-H(decoupled) | student_global (JOINT) | status |
|---|---|---|---|---|---|---|---|---|---|
| R40_A_CC31_H1001_image_only_BCE | joint | 391.50 | 1.80 | 389.30 | 4.10 | 389.10 | 4.70 | 392.00 | EVALUATED |
| R40_A_CC31_H1001_image_only_J | joint | 391.00 | 1.30 | 389.60 | 4.40 | 387.70 | 3.30 | 390.80 | EVALUATED |
| R40_A_CC31_H1001_image_only_NCE | joint | 388.50 | -1.20 | 386.40 | 1.20 | 383.30 | -1.10 | 387.40 | EVALUATED |
| R40_A_CC31_H1001_text_only_BCE | joint | 390.50 | 0.80 | 388.60 | 3.50 | 386.00 | 1.50 | 390.80 | EVALUATED |
| R40_A_CC31_H1001_text_only_J | joint | 390.50 | 0.80 | 388.80 | 3.70 | 385.90 | 1.40 | 390.40 | EVALUATED |
| R40_A_CC31_H1001_text_only_NCE | joint | 388.80 | -0.90 | 385.70 | 0.60 | 383.90 | -0.60 | 387.50 | EVALUATED |
| R40_A_F60_R30_H1001_image_only_BCE | joint | 512.21 | 7.00 | 491.78 | -4.75 | 489.66 | -3.89 | 500.30 | EVALUATED |
| R40_A_F60_R30_H1001_image_only_J | joint | 513.20 | 7.99 | 491.85 | -4.67 | 489.23 | -4.32 | 499.39 | EVALUATED |
| R40_A_F60_R30_H1001_image_only_NCE | joint | 512.37 | 7.16 | 490.55 | -5.98 | 488.56 | -4.99 | 497.63 | EVALUATED |
| R40_A_F60_R30_H1001_text_only_BCE | joint | 508.40 | 3.20 | 490.49 | -6.04 | 489.37 | -4.18 | 498.76 | EVALUATED |
| R40_A_F60_R30_H1001_text_only_J | joint | 508.90 | 3.69 | 490.85 | -5.68 | 489.35 | -4.20 | 498.15 | EVALUATED |
| R40_A_F60_R30_H1001_text_only_NCE | joint | 508.90 | 3.69 | 490.65 | -5.88 | 488.26 | -5.29 | 497.73 | EVALUATED |
| R40_B_CC31_H1001_PRESENT_both_joint_BCE | joint | 390.70 | 1.00 | 386.40 | 1.30 | 387.20 | 2.70 | 390.70 | EVALUATED |
| R40_B_CC31_H1001_PRESENT_both_joint_J | joint | 391.80 | 2.10 | 387.20 | 2.10 | 388.30 | 3.80 | 391.30 | EVALUATED |
| R40_B_CC31_H1001_PRESENT_both_joint_NCE | joint | 388.30 | -1.40 | 386.30 | 1.20 | 385.90 | 1.40 | 388.00 | EVALUATED |
| R40_B_CC31_H1001_PRESENT_frozen_BCE | frozen | 392.20 | 2.50 | 390.90 | 5.70 | 389.00 | 4.60 | NA | EVALUATED |
| R40_B_CC31_H1001_PRESENT_frozen_J | frozen | 392.30 | 2.60 | 391.30 | 6.10 | 389.50 | 5.10 | NA | EVALUATED |
| R40_B_CC31_H1001_PRESENT_frozen_NCE | frozen | 389.90 | 0.20 | 387.60 | 2.40 | 387.40 | 3.00 | NA | EVALUATED |
| R40_B_F60_R30_H1001_PRESENT_both_joint_BCE | joint | 514.08 | 8.88 | 489.63 | -6.90 | 486.65 | -6.90 | 496.61 | EVALUATED |
| R40_B_F60_R30_H1001_PRESENT_both_joint_J | joint | 516.11 | 10.91 | 489.98 | -6.55 | 486.49 | -7.06 | 496.73 | EVALUATED |
| R40_B_F60_R30_H1001_PRESENT_both_joint_NCE | joint | 517.93 | 12.72 | 486.27 | -10.26 | 483.25 | -10.30 | 490.73 | EVALUATED |
| R40_B_F60_R30_H1001_PRESENT_frozen_BCE | frozen | 508.05 | 2.84 | 492.76 | -3.87 | 490.93 | -2.62 | NA | EVALUATED |
| R40_B_F60_R30_H1001_PRESENT_frozen_J | frozen | 508.30 | 3.10 | 492.88 | -3.75 | 490.93 | -2.62 | NA | EVALUATED |
| R40_B_F60_R30_H1001_PRESENT_frozen_NCE | frozen | 506.00 | 0.79 | 491.70 | -4.93 | 489.17 | -4.38 | NA | EVALUATED |
| R40_C_F60_R30_H1001_ORACLE_joint_BCE | joint | 524.16 | 18.95 | 491.62 | -5.01 | 487.71 | -5.84 | 497.61 | EVALUATED |
| R40_C_F60_R30_H1001_ORACLE_joint_J | joint | 524.67 | 19.47 | 491.38 | -5.25 | 487.93 | -5.62 | 497.63 | EVALUATED |
| R40_C_F60_R30_H1001_ORACLE_joint_NCE | joint | 526.07 | 20.87 | 490.24 | -6.39 | 486.43 | -7.12 | 497.61 | EVALUATED |

## R41

| unit | enc | logged | logged-H | full_global | global-H(global) | full_decoupled | decoupled-H(decoupled) | student_global (JOINT) | status |
|---|---|---|---|---|---|---|---|---|---|
| R41_A_F00_R30_H1001_N_joint_16k_BCE | joint | 548.22 | 26.15 | 506.13 | -5.72 | 502.47 | -8.36 | 514.73 | EVALUATED |
| R41_A_F00_R30_H1001_N_joint_16k_J | joint | 550.65 | 28.58 | 503.57 | -8.28 | 498.74 | -12.09 | 511.66 | EVALUATED |
| R41_A_F00_R30_H1001_N_joint_16k_NCE | joint | 567.73 | 45.66 | 477.95 | -33.91 | 466.57 | -44.26 | 482.76 | EVALUATED |
| R41_A_F60_R30_H1001_N_joint_16k_BCE | joint | 520.79 | 15.58 | 486.47 | -10.16 | 480.43 | -13.12 | 491.91 | EVALUATED |
| R41_A_F60_R30_H1001_N_joint_16k_J | joint | 524.02 | 18.82 | 487.40 | -9.23 | 481.74 | -11.81 | 492.45 | EVALUATED |
| R41_A_F60_R30_H1001_N_joint_16k_NCE | joint | 518.70 | 13.49 | 483.33 | -13.29 | 477.89 | -15.66 | 489.74 | EVALUATED |
| R41_A_F60_R30_H1002_N_joint_16k_BCE | joint | 498.40 | 15.46 | 467.16 | -5.56 | 459.72 | -8.88 | 473.61 | EVALUATED |
| R41_A_F60_R30_H1002_N_joint_16k_J | joint | 500.20 | 17.26 | 468.19 | -4.54 | 461.12 | -7.48 | 474.00 | EVALUATED |
| R41_A_F60_R30_H1002_N_joint_16k_NCE | joint | 497.24 | 14.30 | 464.95 | -7.77 | 459.07 | -9.53 | 470.53 | EVALUATED |
| R41_A_F60_R30_H1003_N_joint_16k_BCE | joint | 516.31 | 20.24 | 478.70 | -4.64 | 474.22 | -8.58 | 485.11 | EVALUATED |
| R41_A_F60_R30_H1003_N_joint_16k_J | joint | 518.78 | 22.70 | 478.86 | -4.48 | 474.34 | -8.46 | 486.51 | EVALUATED |
| R41_A_F60_R30_H1003_N_joint_16k_NCE | joint | 514.46 | 18.38 | 475.40 | -7.93 | 471.97 | -10.83 | 481.32 | EVALUATED |
| R41_B_F60_R30_H1001_seed2002_N_joint_BCE | joint | 516.92 | 11.72 | 489.88 | -6.65 | 487.26 | -6.29 | 496.37 | EVALUATED |
| R41_B_F60_R30_H1001_seed2002_N_joint_J | joint | 517.02 | 11.81 | 490.97 | -5.56 | 487.50 | -6.06 | 497.91 | EVALUATED |
| R41_B_F60_R30_H1001_seed2002_N_joint_NCE | joint | 517.10 | 11.89 | 490.02 | -6.51 | 485.88 | -7.67 | 496.06 | EVALUATED |
| R41_C_F40_IN2R_H1001_N_frozen_BCE | frozen | 518.56 | 4.38 | 501.85 | -1.89 | 500.18 | -1.56 | NA | EVALUATED |
| R41_C_F40_IN2R_H1001_N_frozen_J | frozen | 518.11 | 3.93 | 501.54 | -2.21 | 500.30 | -1.44 | NA | EVALUATED |
| R41_C_F40_IN2R_H1001_N_frozen_NCE | frozen | 517.16 | 2.98 | 502.25 | -1.50 | 501.89 | 0.16 | NA | EVALUATED |
| R41_C_F40_IN2R_H1001_N_joint_BCE | joint | 535.78 | 21.60 | 499.39 | -4.36 | 496.77 | -4.97 | 506.06 | EVALUATED |
| R41_C_F40_IN2R_H1001_N_joint_J | joint | 536.90 | 22.72 | 497.26 | -6.49 | 494.16 | -7.57 | 503.33 | EVALUATED |
| R41_C_F40_IN2R_H1001_N_joint_NCE | joint | 538.05 | 23.87 | 498.48 | -5.27 | 495.35 | -6.39 | 504.73 | EVALUATED |
| R41_C_F80_IN2R_H1001_N_frozen_BCE | frozen | 456.33 | -8.60 | 446.61 | -7.24 | 445.40 | -6.75 | NA | EVALUATED |
| R41_C_F80_IN2R_H1001_N_frozen_J | frozen | 458.05 | -6.88 | 448.48 | -5.36 | 446.77 | -5.38 | NA | EVALUATED |
| R41_C_F80_IN2R_H1001_N_frozen_NCE | frozen | 456.82 | -8.11 | 445.09 | -8.76 | 443.04 | -9.11 | NA | EVALUATED |
| R41_C_F80_IN2R_H1001_N_joint_BCE | joint | 454.71 | -10.22 | 437.53 | -16.31 | 434.38 | -17.77 | 446.45 | EVALUATED |
| R41_C_F80_IN2R_H1001_N_joint_J | joint | 457.55 | -7.38 | 440.65 | -13.20 | 435.44 | -16.71 | 448.15 | EVALUATED |
| R41_C_F80_IN2R_H1001_N_joint_NCE | joint | 456.69 | -8.24 | 438.68 | -15.17 | 434.48 | -17.67 | 447.91 | EVALUATED |
| R41_D_F60_R30_H1002_ORACLE_frozen_BCE | frozen | 487.93 | 4.99 | 474.32 | 1.60 | 471.60 | 3.00 | NA | EVALUATED |
| R41_D_F60_R30_H1002_ORACLE_frozen_J | frozen | 487.18 | 4.24 | 474.83 | 2.11 | 471.66 | 3.06 | NA | EVALUATED |
| R41_D_F60_R30_H1002_ORACLE_frozen_NCE | frozen | 487.22 | 4.28 | 474.67 | 1.95 | 471.89 | 3.29 | NA | EVALUATED |
| R41_D_F60_R30_H1002_ORACLE_joint_BCE | joint | 500.65 | 17.71 | 472.96 | 0.24 | 466.43 | -2.17 | 479.57 | EVALUATED |
| R41_D_F60_R30_H1002_ORACLE_joint_J | joint | 501.64 | 18.70 | 472.47 | -0.26 | 466.73 | -1.87 | 479.45 | EVALUATED |
| R41_D_F60_R30_H1002_ORACLE_joint_NCE | joint | 502.56 | 19.63 | 472.01 | -0.71 | 464.95 | -3.65 | 478.19 | EVALUATED |
| R41_E_F60_R30_H1002_image_only_BCE | joint | 489.80 | 6.86 | 474.77 | 2.05 | 469.35 | 0.75 | 482.72 | EVALUATED |
| R41_E_F60_R30_H1002_image_only_J | joint | 489.33 | 6.39 | 475.05 | 2.33 | 469.35 | 0.75 | 483.16 | EVALUATED |
| R41_E_F60_R30_H1002_image_only_NCE | joint | 489.17 | 6.23 | 472.94 | 0.22 | 469.68 | 1.08 | 480.57 | EVALUATED |
| R41_E_F60_R30_H1003_image_only_BCE | joint | 505.70 | 9.63 | 484.87 | 1.54 | 482.07 | -0.73 | 492.78 | EVALUATED |
| R41_E_F60_R30_H1003_image_only_J | joint | 505.86 | 9.78 | 485.64 | 2.31 | 481.85 | -0.95 | 493.06 | EVALUATED |
| R41_E_F60_R30_H1003_image_only_NCE | joint | 504.46 | 8.38 | 483.79 | 0.45 | 481.38 | -1.42 | 491.50 | EVALUATED |
| R41_F_CC31_H1001_N_joint_enclr0p1_BCE | joint | 391.90 | 2.20 | 389.70 | 4.50 | 387.80 | 3.40 | 392.40 | EVALUATED |
| R41_F_CC31_H1001_N_joint_enclr0p1_J | joint | 392.30 | 2.60 | 389.40 | 4.20 | 388.80 | 4.40 | 392.60 | EVALUATED |
| R41_F_CC31_H1001_N_joint_enclr0p1_NCE | joint | 387.50 | -2.20 | 385.90 | 0.70 | 382.60 | -1.80 | 386.70 | EVALUATED |
| R41_G1_F00_R30_H1001_seed2002_N_joint_BCE | joint | 524.24 | 2.17 | 509.57 | -2.29 | 508.15 | -2.68 | 519.37 | EVALUATED |
| R41_G1_F00_R30_H1001_seed2002_N_joint_J | joint | 528.48 | 6.41 | 510.36 | -1.50 | 509.47 | -1.36 | 519.94 | EVALUATED |
| R41_G1_F00_R30_H1001_seed2002_N_joint_NCE | joint | 550.77 | 28.70 | 499.09 | -12.76 | 496.17 | -14.65 | 505.86 | EVALUATED |
| R41_G2_F60_R30_H1003_ORACLE_frozen_BCE | frozen | 499.90 | 3.83 | 485.19 | 1.85 | 483.71 | 0.91 | NA | EVALUATED |
| R41_G2_F60_R30_H1003_ORACLE_frozen_J | frozen | 499.80 | 3.73 | 485.72 | 2.39 | 483.65 | 0.85 | NA | EVALUATED |
| R41_G2_F60_R30_H1003_ORACLE_frozen_NCE | frozen | 499.98 | 3.91 | 484.83 | 1.50 | 483.35 | 0.55 | NA | EVALUATED |
| R41_G2_F60_R30_H1003_ORACLE_joint_BCE | joint | 517.10 | 21.03 | 484.65 | 1.32 | 479.47 | -3.33 | 491.14 | EVALUATED |
| R41_G2_F60_R30_H1003_ORACLE_joint_J | joint | 517.69 | 21.62 | 484.04 | 0.71 | 478.80 | -4.00 | 490.63 | EVALUATED |
| R41_G2_F60_R30_H1003_ORACLE_joint_NCE | joint | 520.67 | 24.60 | 482.17 | -1.16 | 480.36 | -2.45 | 489.55 | EVALUATED |

## R42

| unit | enc | logged | logged-H | full_global | global-H(global) | full_decoupled | decoupled-H(decoupled) | student_global (JOINT) | status |
|---|---|---|---|---|---|---|---|---|---|
| R42_N_F20_IN2R_H1001_N_frozen_BCE | frozen | 520.91 | 3.20 | 505.82 | -1.24 | 504.91 | -0.89 | NA | EVALUATED |
| R42_N_F20_IN2R_H1001_N_frozen_J | frozen | 521.40 | 3.69 | 506.13 | -0.93 | 505.23 | -0.57 | NA | EVALUATED |
| R42_N_F20_IN2R_H1001_N_frozen_NCE | frozen | 520.67 | 2.96 | 505.66 | -1.40 | 503.75 | -2.05 | NA | EVALUATED |
| R42_N_F20_IN2R_H1001_N_joint_BCE | joint | 533.12 | 15.40 | 503.61 | -3.45 | 501.79 | -4.00 | 509.82 | EVALUATED |
| R42_N_F20_IN2R_H1001_N_joint_J | joint | 535.80 | 18.09 | 502.54 | -4.52 | 498.58 | -7.22 | 506.49 | EVALUATED |
| R42_N_F20_IN2R_H1001_N_joint_NCE | joint | 539.74 | 22.03 | 502.45 | -4.62 | 501.05 | -4.75 | 508.34 | EVALUATED |
| R42_N_F20_IN2R_H1002_N_frozen_BCE | frozen | 521.81 | 5.23 | 505.84 | -0.34 | 503.91 | -0.32 | NA | EVALUATED |
| R42_N_F20_IN2R_H1002_N_frozen_J | frozen | 522.33 | 5.74 | 506.04 | -0.14 | 504.64 | 0.41 | NA | EVALUATED |
| R42_N_F20_IN2R_H1002_N_frozen_NCE | frozen | 520.59 | 4.00 | 504.89 | -1.28 | 503.35 | -0.87 | NA | EVALUATED |
| R42_N_F20_IN2R_H1002_N_joint_BCE | joint | 537.59 | 21.01 | 501.48 | -4.69 | 500.30 | -3.93 | 509.59 | EVALUATED |
| R42_N_F20_IN2R_H1002_N_joint_J | joint | 539.19 | 22.60 | 499.63 | -6.55 | 496.63 | -7.59 | 505.35 | EVALUATED |
| R42_N_F20_IN2R_H1002_N_joint_NCE | joint | 543.23 | 26.65 | 501.68 | -4.50 | 499.15 | -5.07 | 509.96 | EVALUATED |
| R42_N_F20_IN2R_H1003_N_frozen_BCE | frozen | 518.42 | 2.15 | 503.83 | -1.22 | 503.14 | -1.54 | NA | EVALUATED |
| R42_N_F20_IN2R_H1003_N_frozen_J | frozen | 518.50 | 2.23 | 504.30 | -0.75 | 503.12 | -1.56 | NA | EVALUATED |
| R42_N_F20_IN2R_H1003_N_frozen_NCE | frozen | 518.82 | 2.54 | 504.58 | -0.47 | 504.14 | -0.53 | NA | EVALUATED |
| R42_N_F20_IN2R_H1003_N_joint_BCE | joint | 529.03 | 12.76 | 499.07 | -5.98 | 497.50 | -7.18 | 506.53 | EVALUATED |
| R42_N_F20_IN2R_H1003_N_joint_J | joint | 533.53 | 17.26 | 499.96 | -5.09 | 497.24 | -7.44 | 505.64 | EVALUATED |
| R42_N_F20_IN2R_H1003_N_joint_NCE | joint | 537.46 | 21.18 | 502.19 | -2.86 | 498.28 | -6.39 | 508.74 | EVALUATED |
| R42_N_F40_IN2R_H1002_N_frozen_BCE | frozen | 512.47 | 3.02 | 498.05 | -1.54 | 496.27 | -2.62 | NA | EVALUATED |
| R42_N_F40_IN2R_H1002_N_frozen_J | frozen | 512.54 | 3.10 | 497.61 | -1.97 | 495.84 | -3.06 | NA | EVALUATED |
| R42_N_F40_IN2R_H1002_N_frozen_NCE | frozen | 512.07 | 2.62 | 497.77 | -1.81 | 495.62 | -3.27 | NA | EVALUATED |
| R42_N_F40_IN2R_H1002_N_joint_BCE | joint | 530.24 | NA | NA | NA | NA | NA | NA | PENDING |
| R42_N_F40_IN2R_H1002_N_joint_J | joint | 529.47 | NA | NA | NA | NA | NA | NA | PENDING |
| R42_N_F40_IN2R_H1002_N_joint_NCE | joint | 529.05 | NA | NA | NA | NA | NA | NA | PENDING |
| R42_N_F40_IN2R_H1003_N_frozen_BCE | frozen | 513.39 | 2.60 | 498.99 | -3.00 | 498.62 | -2.21 | NA | EVALUATED |
| R42_N_F40_IN2R_H1003_N_frozen_J | frozen | 513.33 | 2.54 | 499.76 | -2.23 | 498.52 | -2.31 | NA | EVALUATED |
| R42_N_F40_IN2R_H1003_N_frozen_NCE | frozen | 512.86 | 2.07 | 498.95 | -3.04 | 498.46 | -2.37 | NA | EVALUATED |
| R42_N_F40_IN2R_H1003_N_joint_BCE | joint | 536.61 | 25.82 | 496.15 | -5.84 | 492.13 | -8.72 | 502.88 | EVALUATED |
| R42_N_F40_IN2R_H1003_N_joint_J | joint | 536.55 | 25.76 | 496.79 | -5.21 | 492.27 | -8.58 | 500.83 | EVALUATED |
| R42_N_F40_IN2R_H1003_N_joint_NCE | joint | 537.10 | 26.31 | 497.22 | -4.77 | 494.16 | -6.69 | 503.18 | EVALUATED |
| R42_N_F80_IN2R_H1002_N_frozen_BCE | frozen | 430.47 | -7.04 | 422.66 | -3.67 | 418.82 | -6.33 | NA | EVALUATED |
| R42_N_F80_IN2R_H1002_N_frozen_J | frozen | 430.41 | -7.10 | 423.21 | -3.12 | 418.74 | -6.41 | NA | EVALUATED |
| R42_N_F80_IN2R_H1002_N_frozen_NCE | frozen | 430.53 | -6.98 | 422.19 | -4.14 | 417.97 | -7.18 | NA | EVALUATED |
| R42_N_F80_IN2R_H1002_N_joint_BCE | joint | 423.16 | -14.36 | 413.45 | -12.88 | 408.66 | -16.49 | 421.14 | EVALUATED |
| R42_N_F80_IN2R_H1002_N_joint_J | joint | 424.38 | -13.14 | 415.07 | -11.26 | 409.86 | -15.29 | 422.78 | EVALUATED |
| R42_N_F80_IN2R_H1002_N_joint_NCE | joint | 424.93 | -12.58 | 416.90 | -9.43 | 411.14 | -14.00 | 424.42 | EVALUATED |
| R42_N_F80_IN2R_H1003_N_frozen_BCE | frozen | 448.13 | -6.00 | 438.93 | -5.64 | 435.01 | -6.98 | NA | EVALUATED |
| R42_N_F80_IN2R_H1003_N_frozen_J | frozen | 449.47 | -4.65 | 439.88 | -4.69 | 435.36 | -6.63 | NA | EVALUATED |
| R42_N_F80_IN2R_H1003_N_frozen_NCE | frozen | 447.48 | -6.65 | 439.11 | -5.46 | 434.30 | -7.69 | NA | EVALUATED |
| R42_N_F80_IN2R_H1003_N_joint_BCE | joint | 448.46 | -5.66 | 432.49 | -12.09 | 427.77 | -14.22 | 441.81 | EVALUATED |
| R42_N_F80_IN2R_H1003_N_joint_J | joint | 449.25 | -4.87 | 434.67 | -9.90 | 430.00 | -11.99 | 443.12 | EVALUATED |
| R42_N_F80_IN2R_H1003_N_joint_NCE | joint | 449.86 | -4.26 | 432.50 | -12.07 | 429.43 | -12.56 | 443.06 | EVALUATED |
| R42_O_F80_IN2R_H1001_ORACLE_frozen_BCE | frozen | 466.80 | 1.87 | 454.10 | 0.26 | 452.56 | 0.41 | NA | EVALUATED |
| R42_O_F80_IN2R_H1001_ORACLE_frozen_J | frozen | 465.86 | 0.93 | 452.96 | -0.89 | 452.05 | -0.10 | NA | EVALUATED |
| R42_O_F80_IN2R_H1001_ORACLE_frozen_NCE | frozen | 465.21 | 0.28 | 452.84 | -1.01 | 451.01 | -1.14 | NA | EVALUATED |
| R42_O_F80_IN2R_H1001_ORACLE_joint_BCE | joint | 476.69 | 11.76 | 454.14 | 0.30 | 448.64 | -3.51 | 461.83 | EVALUATED |
| R42_O_F80_IN2R_H1001_ORACLE_joint_J | joint | 475.62 | 10.69 | 453.06 | -0.79 | 448.07 | -4.08 | 462.47 | EVALUATED |
| R42_O_F80_IN2R_H1001_ORACLE_joint_NCE | joint | 473.73 | 8.80 | 452.84 | -1.01 | 448.05 | -4.10 | 460.43 | EVALUATED |

## R43

| unit | enc | logged | logged-H | full_global | global-H(global) | full_decoupled | decoupled-H(decoupled) | student_global (JOINT) | status |
|---|---|---|---|---|---|---|---|---|---|
| R43_A_C20_IN2R_H1001_N_frozen_BCE | frozen | 553.78 | 7.72 | 523.68 | -8.70 | 523.72 | -9.34 | NA | EVALUATED |
| R43_A_C20_IN2R_H1001_N_frozen_J | frozen | 553.72 | 7.66 | 523.04 | -9.34 | 523.16 | -9.90 | NA | EVALUATED |
| R43_A_C20_IN2R_H1001_N_frozen_NCE | frozen | 555.18 | 9.12 | 527.14 | -5.24 | 526.66 | -6.40 | NA | EVALUATED |
| R43_A_C20_IN2R_H1001_N_joint_BCE | joint | 563.82 | 17.76 | 525.72 | -6.66 | 525.88 | -7.18 | 533.50 | EVALUATED |
| R43_A_C20_IN2R_H1001_N_joint_J | joint | 566.14 | 20.08 | 525.76 | -6.62 | 524.40 | -8.66 | 531.30 | EVALUATED |
| R43_A_C20_IN2R_H1001_N_joint_NCE | joint | 566.16 | 20.10 | 525.82 | -6.56 | 525.28 | -7.78 | 532.88 | EVALUATED |
| R43_A_C60_IN2R_H1001_N_frozen_BCE | frozen | 547.04 | 8.10 | 516.52 | -7.68 | 515.32 | -9.10 | NA | EVALUATED |
| R43_A_C60_IN2R_H1001_N_frozen_J | frozen | 547.00 | 8.06 | 516.38 | -7.82 | 515.76 | -8.66 | NA | EVALUATED |
| R43_A_C60_IN2R_H1001_N_frozen_NCE | frozen | 547.12 | 8.18 | 518.58 | -5.62 | 517.42 | -7.00 | NA | EVALUATED |
| R43_A_C60_IN2R_H1001_N_joint_BCE | joint | 556.94 | 18.00 | 517.04 | -7.16 | 515.92 | -8.50 | 524.92 | EVALUATED |
| R43_A_C60_IN2R_H1001_N_joint_J | joint | 558.16 | 19.22 | 516.16 | -8.04 | 515.88 | -8.54 | 523.90 | EVALUATED |
| R43_A_C60_IN2R_H1001_N_joint_NCE | joint | 560.32 | 21.38 | 517.24 | -6.96 | 516.16 | -8.26 | 525.46 | EVALUATED |
| R43_A_C60_IN2R_H1002_N_frozen_BCE | frozen | 546.00 | 7.56 | 516.22 | -8.20 | 515.68 | -7.52 | NA | EVALUATED |
| R43_A_C60_IN2R_H1002_N_frozen_J | frozen | 545.76 | 7.32 | 516.38 | -8.04 | 515.66 | -7.54 | NA | EVALUATED |
| R43_A_C60_IN2R_H1002_N_frozen_NCE | frozen | 546.86 | 8.42 | 517.30 | -7.12 | 516.66 | -6.54 | NA | EVALUATED |
| R43_A_C60_IN2R_H1002_N_joint_BCE | joint | 557.92 | 19.48 | 515.62 | -8.80 | 515.42 | -7.78 | 524.34 | EVALUATED |
| R43_A_C60_IN2R_H1002_N_joint_J | joint | 560.56 | 22.12 | 515.70 | -8.72 | 515.60 | -7.60 | 522.86 | EVALUATED |
| R43_A_C60_IN2R_H1002_N_joint_NCE | joint | 560.76 | 22.32 | 516.82 | -7.60 | 515.86 | -7.34 | 525.00 | EVALUATED |
| R43_A_C60_IN2R_H1003_N_frozen_BCE | frozen | 546.92 | 7.60 | 518.46 | -5.94 | 518.76 | -5.56 | NA | EVALUATED |
| R43_A_C60_IN2R_H1003_N_frozen_J | frozen | 547.02 | 7.70 | 519.08 | -5.32 | 519.04 | -5.28 | NA | EVALUATED |
| R43_A_C60_IN2R_H1003_N_frozen_NCE | frozen | 548.12 | 8.80 | 520.06 | -4.34 | 520.60 | -3.72 | NA | EVALUATED |
| R43_A_C60_IN2R_H1003_N_joint_BCE | joint | 559.90 | 20.58 | 518.36 | -6.04 | 517.72 | -6.60 | 525.72 | EVALUATED |
| R43_A_C60_IN2R_H1003_N_joint_J | joint | 560.72 | 21.40 | 519.02 | -5.38 | 517.70 | -6.62 | 524.78 | EVALUATED |
| R43_A_C60_IN2R_H1003_N_joint_NCE | joint | 561.04 | 21.72 | 518.66 | -5.74 | 518.72 | -5.60 | 526.72 | EVALUATED |
| R43_A_C80_IN2R_H1001_N_frozen_BCE | frozen | 536.46 | 9.94 | 502.50 | -9.72 | 501.18 | -9.14 | NA | EVALUATED |
| R43_A_C80_IN2R_H1001_N_frozen_J | frozen | 536.72 | 10.20 | 502.30 | -9.92 | 502.04 | -8.28 | NA | EVALUATED |
| R43_A_C80_IN2R_H1001_N_frozen_NCE | frozen | 536.82 | 10.30 | 503.60 | -8.62 | 502.32 | -8.00 | NA | EVALUATED |
| R43_A_C80_IN2R_H1001_N_joint_BCE | joint | 544.26 | 17.74 | 503.86 | -8.36 | 501.38 | -8.94 | 513.76 | EVALUATED |
| R43_A_C80_IN2R_H1001_N_joint_J | joint | 545.08 | 18.56 | 503.62 | -8.60 | 501.44 | -8.88 | 514.02 | EVALUATED |
| R43_A_C80_IN2R_H1001_N_joint_NCE | joint | 543.42 | 16.90 | 503.40 | -8.82 | 502.20 | -8.12 | 511.54 | EVALUATED |
| R43_A_C80_IN2R_H1002_N_frozen_BCE | frozen | 515.02 | 3.36 | 493.38 | -1.20 | 493.30 | -0.06 | NA | EVALUATED |
| R43_A_C80_IN2R_H1002_N_frozen_J | frozen | 514.90 | 3.24 | 494.04 | -0.54 | 493.00 | -0.36 | NA | EVALUATED |
| R43_A_C80_IN2R_H1002_N_frozen_NCE | frozen | 514.94 | 3.28 | 494.00 | -0.58 | 493.24 | -0.12 | NA | EVALUATED |
| R43_A_C80_IN2R_H1002_N_joint_BCE | joint | 522.92 | 11.26 | 495.56 | 0.98 | 493.76 | 0.40 | 503.70 | EVALUATED |
| R43_A_C80_IN2R_H1002_N_joint_J | joint | 522.44 | 10.78 | 495.16 | 0.58 | 494.26 | 0.90 | 504.04 | EVALUATED |
| R43_A_C80_IN2R_H1002_N_joint_NCE | joint | 524.36 | 12.70 | 496.40 | 1.82 | 494.96 | 1.60 | 504.66 | EVALUATED |
| R43_A_C80_IN2R_H1003_N_frozen_BCE | frozen | 515.48 | 3.30 | 494.30 | -1.44 | 494.12 | -0.78 | NA | EVALUATED |
| R43_A_C80_IN2R_H1003_N_frozen_J | frozen | 515.90 | 3.72 | 494.86 | -0.88 | 493.82 | -1.08 | NA | EVALUATED |
| R43_A_C80_IN2R_H1003_N_frozen_NCE | frozen | 515.98 | 3.80 | 494.60 | -1.14 | 492.92 | -1.98 | NA | EVALUATED |
| R43_A_C80_IN2R_H1003_N_joint_BCE | joint | 522.42 | 10.24 | 494.96 | -0.78 | 495.06 | 0.16 | 505.48 | EVALUATED |
| R43_A_C80_IN2R_H1003_N_joint_J | joint | 523.02 | 10.84 | 494.30 | -1.44 | 494.96 | 0.06 | 505.34 | EVALUATED |
| R43_A_C80_IN2R_H1003_N_joint_NCE | joint | 522.52 | 10.34 | 495.36 | -0.38 | 495.08 | 0.18 | 504.44 | EVALUATED |
| R43_B_CC31_H1001_TRUST_frozen_BCE | frozen | 391.30 | 1.60 | 390.50 | 5.30 | 387.50 | 3.10 | NA | EVALUATED |
| R43_B_CC31_H1001_TRUST_frozen_J | frozen | 391.10 | 1.40 | 390.50 | 5.30 | 388.80 | 4.40 | NA | EVALUATED |
| R43_B_CC31_H1001_TRUST_frozen_NCE | frozen | 388.30 | -1.40 | 386.60 | 1.40 | 382.90 | -1.50 | NA | EVALUATED |
| R43_B_CC31_H1001_TRUST_joint_BCE | joint | 391.00 | 1.30 | 387.10 | 1.90 | 387.10 | 2.70 | 390.30 | EVALUATED |
| R43_B_CC31_H1001_TRUST_joint_J | joint | 389.40 | -0.30 | 388.30 | 3.10 | 387.20 | 2.80 | 389.20 | EVALUATED |
| R43_B_CC31_H1001_TRUST_joint_NCE | joint | 386.50 | -3.20 | 385.60 | 0.40 | 382.60 | -1.80 | 386.70 | EVALUATED |
| R43_B_F80_IN2R_H1001_TRUST_frozen_BCE | frozen | 451.81 | -13.12 | 443.02 | -10.83 | 440.36 | -11.79 | NA | EVALUATED |
| R43_B_F80_IN2R_H1001_TRUST_frozen_J | frozen | 453.27 | -11.66 | 443.85 | -10.00 | 441.20 | -10.95 | NA | EVALUATED |
| R43_B_F80_IN2R_H1001_TRUST_frozen_NCE | frozen | 452.03 | -12.90 | 442.19 | -11.66 | 440.87 | -11.28 | NA | EVALUATED |
| R43_B_F80_IN2R_H1001_TRUST_joint_BCE | joint | 447.73 | -17.20 | 431.38 | -22.47 | 427.00 | -25.15 | 439.53 | EVALUATED |
| R43_B_F80_IN2R_H1001_TRUST_joint_J | joint | 449.72 | -15.21 | 431.83 | -22.01 | 429.19 | -22.96 | 441.58 | EVALUATED |
| R43_B_F80_IN2R_H1001_TRUST_joint_NCE | joint | 448.24 | -16.69 | 432.39 | -21.46 | 427.91 | -24.24 | 440.36 | EVALUATED |
| R43_B_F80_IN2R_H1002_TRUST_frozen_BCE | frozen | 424.79 | -12.72 | 419.92 | -6.41 | 415.74 | -9.41 | NA | EVALUATED |
| R43_B_F80_IN2R_H1002_TRUST_frozen_J | frozen | 425.82 | -11.70 | 420.34 | -6.00 | 415.74 | -9.41 | NA | EVALUATED |
| R43_B_F80_IN2R_H1002_TRUST_frozen_NCE | frozen | 427.91 | -9.61 | 420.34 | -6.00 | 414.73 | -10.41 | NA | EVALUATED |
| R43_B_F80_IN2R_H1002_TRUST_joint_BCE | joint | 416.25 | -21.26 | 406.35 | -19.98 | 402.15 | -23.00 | 414.65 | EVALUATED |
| R43_B_F80_IN2R_H1002_TRUST_joint_J | joint | 417.02 | -20.49 | 408.52 | -17.81 | 402.62 | -22.52 | 416.39 | EVALUATED |
| R43_B_F80_IN2R_H1002_TRUST_joint_NCE | joint | 418.32 | -19.19 | 409.31 | -17.02 | 406.09 | -19.05 | 417.51 | EVALUATED |
| R43_B_F80_IN2R_H1003_TRUST_frozen_BCE | frozen | 444.52 | -9.61 | 434.02 | -10.55 | 431.22 | -10.77 | NA | EVALUATED |
| R43_B_F80_IN2R_H1003_TRUST_frozen_J | frozen | 444.85 | -9.27 | 435.15 | -9.43 | 432.39 | -9.61 | NA | EVALUATED |
| R43_B_F80_IN2R_H1003_TRUST_frozen_NCE | frozen | 445.64 | -8.48 | 436.71 | -7.87 | 431.83 | -10.16 | NA | EVALUATED |
| R43_B_F80_IN2R_H1003_TRUST_joint_BCE | joint | 443.23 | -10.89 | 426.79 | -17.79 | 424.50 | -17.50 | 436.96 | EVALUATED |
| R43_B_F80_IN2R_H1003_TRUST_joint_J | joint | 445.56 | -8.56 | 428.78 | -15.80 | 425.62 | -16.37 | 439.23 | EVALUATED |
| R43_B_F80_IN2R_H1003_TRUST_joint_NCE | joint | 445.11 | -9.01 | 428.76 | -15.82 | 425.84 | -16.15 | 436.92 | EVALUATED |

## R44

| unit | enc | logged | logged-H | full_global | global-H(global) | full_decoupled | decoupled-H(decoupled) | student_global (JOINT) | status |
|---|---|---|---|---|---|---|---|---|---|
| R44_A_F60_R30_H1001_ALL_BCE_train2002 | joint | 521.42 | 16.21 | 486.92 | -9.70 | 481.58 | -11.97 | 493.08 | EVALUATED |
| R44_A_F60_R30_H1001_ALL_J_train2002 | joint | 523.69 | 18.48 | 487.16 | -9.47 | 483.47 | -10.08 | 493.79 | EVALUATED |
| R44_A_F60_R30_H1001_ALL_NCE_train2002 | joint | 519.45 | 14.24 | 483.57 | -13.06 | 479.29 | -14.26 | 488.40 | EVALUATED |
| R44_A_F60_R30_H1002_ALL_BCE_train2002 | joint | 498.80 | 15.86 | 464.62 | -8.11 | 459.88 | -8.72 | 472.11 | EVALUATED |
| R44_A_F60_R30_H1002_ALL_J_train2002 | joint | 500.04 | 17.10 | 466.90 | -5.82 | 461.70 | -6.90 | 473.20 | EVALUATED |
| R44_A_F60_R30_H1002_ALL_NCE_train2002 | joint | 497.59 | 14.65 | 464.22 | -8.50 | 459.61 | -8.99 | 469.41 | EVALUATED |
| R44_A_F60_R30_H1003_ALL_BCE_train2002 | joint | 519.57 | 23.49 | 478.26 | -5.07 | 473.16 | -9.64 | 483.83 | EVALUATED |
| R44_A_F60_R30_H1003_ALL_J_train2002 | joint | 521.28 | 25.21 | 480.43 | -2.90 | 474.52 | -8.28 | 484.46 | EVALUATED |
| R44_A_F60_R30_H1003_ALL_NCE_train2002 | joint | 518.11 | 22.03 | 476.88 | -6.45 | 472.33 | -10.47 | 481.48 | EVALUATED |
| R44_B_CC_R31_H1001_ROUTE_PRESENT_BCE_train1001 | joint | 393.30 | 3.60 | 386.60 | 1.40 | 381.90 | -2.50 | 387.50 | EVALUATED |
| R44_B_CC_R31_H1001_ROUTE_PRESENT_J_train1001 | joint | 393.30 | 3.60 | 382.80 | -2.40 | 380.90 | -3.50 | 384.90 | EVALUATED |
| R44_B_CC_R31_H1001_ROUTE_PRESENT_NCE_train1001 | joint | 394.20 | 4.50 | 376.20 | -9.00 | 372.70 | -11.70 | 378.40 | EVALUATED |
| R44_B_CC_R31_H1001_ROUTE_RANDOM_BCE_train1001 | joint | 391.20 | 1.50 | 388.10 | 3.00 | 388.60 | 4.10 | 389.70 | EVALUATED |
| R44_B_CC_R31_H1001_ROUTE_RANDOM_J_train1001 | joint | 390.00 | 0.30 | 388.80 | 3.70 | 388.00 | 3.50 | 390.50 | EVALUATED |
| R44_B_CC_R31_H1001_ROUTE_RANDOM_NCE_train1001 | joint | 387.90 | -1.80 | 385.30 | 0.20 | 384.20 | -0.30 | 387.00 | EVALUATED |
| R44_B_F00_R30_H1001_ROUTE_PRESENT_BCE_train1001 | joint | 530.02 | 7.95 | 510.22 | -1.64 | 509.13 | -1.70 | 519.25 | EVALUATED |
| R44_B_F00_R30_H1001_ROUTE_PRESENT_J_train1001 | joint | 532.52 | 10.45 | 510.04 | -1.81 | 509.39 | -1.44 | 519.96 | EVALUATED |
| R44_B_F00_R30_H1001_ROUTE_PRESENT_NCE_train1001 | joint | 558.74 | 36.67 | 489.74 | -22.11 | 483.16 | -27.67 | 495.48 | EVALUATED |
| R44_B_F00_R30_H1001_ROUTE_RANDOM_BCE_train1001 | joint | 525.66 | 3.59 | 510.10 | -1.76 | 510.14 | -0.69 | 519.55 | EVALUATED |
| R44_B_F00_R30_H1001_ROUTE_RANDOM_J_train1001 | joint | 527.87 | 5.80 | 511.79 | -0.06 | 510.32 | -0.51 | 520.26 | EVALUATED |
| R44_B_F00_R30_H1001_ROUTE_RANDOM_NCE_train1001 | joint | 550.65 | 28.58 | 497.97 | -13.89 | 495.72 | -15.11 | 504.89 | EVALUATED |
| R44_B_F60_R30_H1001_ROUTE_PRESENT_BCE_train1001 | joint | 567.48 | 62.27 | 482.62 | -14.00 | 472.58 | -20.97 | 492.23 | EVALUATED |
| R44_B_F60_R30_H1001_ROUTE_PRESENT_J_train1001 | joint | 567.50 | 62.29 | 480.85 | -15.78 | 470.18 | -23.37 | 489.39 | EVALUATED |
| R44_B_F60_R30_H1001_ROUTE_PRESENT_NCE_train1001 | joint | 569.76 | 64.56 | 487.28 | -9.35 | 479.39 | -14.16 | 496.11 | EVALUATED |
| R44_B_F60_R30_H1001_ROUTE_RANDOM_BCE_train1001 | joint | 514.60 | 9.39 | 489.98 | -6.65 | 487.79 | -5.76 | 497.65 | EVALUATED |
| R44_B_F60_R30_H1001_ROUTE_RANDOM_J_train1001 | joint | 515.35 | 10.14 | 490.65 | -5.98 | 487.57 | -5.98 | 498.17 | EVALUATED |
| R44_B_F60_R30_H1001_ROUTE_RANDOM_NCE_train1001 | joint | 514.42 | 9.21 | 490.06 | -6.57 | 487.00 | -6.55 | 496.96 | EVALUATED |
| R44_B_F80_IN2R_H1001_ROUTE_PRESENT_BCE_train1001 | joint | 536.79 | 71.85 | 447.83 | -6.02 | 441.89 | -10.26 | 457.28 | EVALUATED |
| R44_B_F80_IN2R_H1001_ROUTE_PRESENT_J_train1001 | joint | 542.50 | 77.57 | 445.36 | -8.48 | 437.73 | -14.42 | 454.62 | EVALUATED |
| R44_B_F80_IN2R_H1001_ROUTE_PRESENT_NCE_train1001 | joint | 539.29 | 74.36 | 449.49 | -4.36 | 442.76 | -9.39 | 458.38 | EVALUATED |
| R44_B_F80_IN2R_H1001_ROUTE_RANDOM_BCE_train1001 | joint | 456.06 | -8.88 | 440.39 | -13.45 | 436.57 | -15.58 | 448.99 | EVALUATED |
| R44_B_F80_IN2R_H1001_ROUTE_RANDOM_J_train1001 | joint | 457.91 | -7.02 | 442.58 | -11.26 | 438.64 | -13.51 | 450.20 | EVALUATED |
| R44_B_F80_IN2R_H1001_ROUTE_RANDOM_NCE_train1001 | joint | 456.31 | -8.62 | 441.81 | -12.03 | 437.81 | -14.34 | 451.26 | EVALUATED |
| R44_C_F00_R30_H1001_PRESENT_ALL_BCE_train1001 | joint | 525.64 | 3.57 | 510.32 | -1.54 | 509.45 | -1.38 | 519.33 | EVALUATED |
| R44_C_F00_R30_H1001_PRESENT_ALL_J_train1001 | joint | 527.28 | 5.21 | 511.70 | -0.16 | 510.00 | -0.83 | 520.45 | EVALUATED |
| R44_C_F00_R30_H1001_PRESENT_ALL_NCE_train1001 | joint | 550.37 | 28.30 | 485.38 | -26.47 | 480.00 | -30.83 | 484.99 | EVALUATED |
| R44_C_F80_IN2R_H1001_PRESENT_ALL_BCE_train1001 | joint | 454.64 | -10.30 | 437.34 | -16.51 | 434.52 | -17.63 | 447.73 | EVALUATED |
| R44_C_F80_IN2R_H1001_PRESENT_ALL_J_train1001 | joint | 455.03 | -9.90 | 438.64 | -15.21 | 435.96 | -16.19 | 448.28 | EVALUATED |
| R44_C_F80_IN2R_H1001_PRESENT_ALL_NCE_train1001 | joint | 447.14 | -17.79 | 429.17 | -24.67 | 426.49 | -25.66 | 437.65 | EVALUATED |

## R45

| unit | enc | logged | logged-H | full_global | global-H(global) | full_decoupled | decoupled-H(decoupled) | student_global (JOINT) | status |
|---|---|---|---|---|---|---|---|---|---|
| R45_C60_IN2R_H1001_N_joint_16k_BCE | joint | 568.24 | 29.30 | 511.26 | -12.94 | 510.26 | -14.16 | 518.56 | EVALUATED |
| R45_C60_IN2R_H1001_N_joint_16k_J | joint | 567.88 | 28.94 | 510.62 | -13.58 | 509.78 | -14.64 | 516.82 | EVALUATED |
| R45_C60_IN2R_H1001_N_joint_16k_NCE | joint | 567.78 | 28.84 | 513.94 | -10.26 | 513.28 | -11.14 | 521.82 | EVALUATED |
| R45_C80_IN2R_H1001_N_joint_16k_BCE | joint | 552.56 | 26.04 | 502.20 | -10.02 | 498.94 | -11.38 | 513.02 | EVALUATED |
| R45_C80_IN2R_H1001_N_joint_16k_J | joint | 553.74 | 27.22 | 502.90 | -9.32 | 501.08 | -9.24 | 512.62 | EVALUATED |
| R45_C80_IN2R_H1001_N_joint_16k_NCE | joint | 551.90 | 25.38 | 501.82 | -10.40 | 498.18 | -12.14 | 509.42 | EVALUATED |

## R46

| unit | enc | logged | logged-H | full_global | global-H(global) | full_decoupled | decoupled-H(decoupled) | student_global (JOINT) | status |
|---|---|---|---|---|---|---|---|---|---|
| R46_A_C80_IN2R_H1001_P29K_N_frozen_BCE | frozen | 537.80 | 11.28 | 505.04 | -7.18 | 504.76 | -5.56 | NA | EVALUATED |
| R46_A_C80_IN2R_H1001_P29K_N_frozen_J | frozen | 537.62 | 11.10 | 505.96 | -6.26 | 504.74 | -5.58 | NA | EVALUATED |
| R46_A_C80_IN2R_H1001_P29K_N_frozen_NCE | frozen | 537.72 | 11.20 | 506.78 | -5.44 | 506.30 | -4.02 | NA | EVALUATED |
| R46_A_C80_IN2R_H1001_P29K_N_joint_J | joint | 546.30 | 19.78 | 505.44 | -6.78 | 503.62 | -6.70 | 514.64 | EVALUATED |
| R46_A_C80_IN2R_H1001_P58K_N_frozen_BCE | frozen | 536.86 | 10.34 | 503.66 | -8.56 | 502.38 | -7.94 | NA | EVALUATED |
| R46_A_C80_IN2R_H1001_P58K_N_frozen_J | frozen | 537.32 | 10.80 | 504.40 | -7.82 | 503.32 | -7.00 | NA | EVALUATED |
| R46_A_C80_IN2R_H1001_P58K_N_frozen_NCE | frozen | 537.90 | 11.38 | 505.00 | -7.22 | 503.36 | -6.96 | NA | EVALUATED |
| R46_A_C80_IN2R_H1001_P58K_N_joint_BCE | joint | 545.18 | 18.66 | 503.62 | -8.60 | 502.08 | -8.24 | 514.22 | EVALUATED |
| R46_A_C80_IN2R_H1001_P58K_N_joint_J | joint | 546.46 | 19.94 | 504.24 | -7.98 | 503.22 | -7.10 | 514.08 | EVALUATED |
| R46_B_F60_R30_H1001_R2_CONTINUED_J | joint | 523.53 | 18.32 | 487.46 | -9.07 | 483.08 | -10.47 | 493.94 | EVALUATED |
| R46_B_F60_R30_H1001_R2_REBUILT_J | joint | 522.60 | 17.40 | 485.44 | -11.08 | 481.99 | -11.56 | 490.97 | EVALUATED |
| R46_B_F80_IN2R_H1001_R2_CONTINUED_J | joint | 450.89 | -14.04 | 431.97 | -21.87 | 426.63 | -25.52 | 439.31 | EVALUATED |
| R46_B_F80_IN2R_H1001_R2_REBUILT_J | joint | 450.57 | -14.36 | 430.10 | -23.75 | 425.03 | -27.12 | 439.09 | EVALUATED |

