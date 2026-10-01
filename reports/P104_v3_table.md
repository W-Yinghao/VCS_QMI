# P104_v3_first_batch — neutral results table (observed values only)

Generated 2026-10-01T23:11:29Z. Failed / stopped runs are listed, never dropped.

| run | method | K | seed | epochs done | linear-val (%) | kNN-val (%) | heldout-J (mean±sd) | h-rank | train time (s) | peak GPU MB (alloc/res) | status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| P104_G1_views4_800ep_seed0 | vcs_qmi | 8 | 0 | 800/800 | 87.04 | 85.32 | 0.4410±0.0016 | 62.60 | 16071 | 5269/8140 | COMPLETED |
| P104_G2F_views4_800ep_seed0 | vcs_qmi | 8 | 0 | 800/800 | 86.70 | 84.38 | 0.9670±0.0018 | 96.54 | 16134 | 5269/8140 | COMPLETED |
| P104_G2F_views4_800ep_seed1 | vcs_qmi | 8 | 1 | 800/800 | 86.08 | 84.38 | 0.9676±0.0016 | 97.25 | 27918 | 7066/10148 | COMPLETED |
| P104_G2F_views4_800ep_seed2 | vcs_qmi | 8 | 2 | 800/800 | 86.32 | 85.06 | 0.9678±0.0016 | 97.01 | 16144 | 5269/8140 | COMPLETED |
| P104_G2_views4_800ep_seed0 | vcs_qmi | 8 | 0 | 800/800 | 88.66 | 86.96 | 0.8608±0.0018 | 98.90 | 16014 | 5269/8140 | COMPLETED |
| P104_G2_views4_800ep_seed1 | vcs_qmi | 8 | 1 | 800/800 | 88.92 | 87.08 | 0.8615±0.0017 | 98.62 | 16031 | 5269/8140 | COMPLETED |
| P104_G2_views4_800ep_seed2 | vcs_qmi | 8 | 2 | 800/800 | 88.52 | 87.46 | 0.8614±0.0022 | 98.46 | 35029 | 5269/8140 | COMPLETED |
| P104_G3_views4_800ep_seed0 | vcs_qmi | 8 | 0 | 800/800 | 87.64 | 85.38 | 0.4391±0.0010 | 82.87 | 16134 | 5269/8142 | COMPLETED |
| P104_G4_views4_800ep_seed0 | vcs_qmi | 8 | 0 | 800/800 | 81.70 | 78.04 | 0.9897±0.0005 | 156.78 | 16175 | 5269/8142 | COMPLETED |
| P104_N1_views4_800ep_seed0 | vcs_qmi | 8 | 0 | 800/800 | 87.30 | 85.32 | 0.9797±0.0016 | 129.16 | 30121 | 7066/10172 | COMPLETED |
| P104_N2_views4_800ep_seed0 | vcs_qmi | 8 | 0 | 800/800 | 86.84 | 84.84 | 0.9811±0.0015 | 130.13 | 16346 | 5269/8142 | COMPLETED |
| P104_U1_views4_800ep_seed0 | vcs_qmi | 8 | 0 | 800/800 | 87.56 | 85.58 | 0.9745±0.0009 | 135.59 | 16117 | 5269/8154 | COMPLETED |
| P104_U2F_views4_800ep_seed0 | vcs_qmi | 8 | 0 | 800/800 | 87.22 | 85.44 | 0.9716±0.0007 | 118.08 | 16147 | 5269/8154 | COMPLETED |
| P104_U2F_views4_800ep_seed1 | vcs_qmi | 8 | 1 | 800/800 | 87.34 | 85.12 | 0.9714±0.0018 | 118.95 | 16150 | 5269/8154 | COMPLETED |
| P104_U2F_views4_800ep_seed2 | vcs_qmi | 8 | 2 | 800/800 | 86.86 | 85.06 | 0.9716±0.0011 | 118.65 | 16085 | 5269/8154 | COMPLETED |
| P104_U2_views4_800ep_seed0 | vcs_qmi | 8 | 0 | 800/800 | 87.96 | 86.10 | 0.4390±0.0015 | 69.38 | 16092 | 5269/8146 | COMPLETED |
| P104_U2_views4_800ep_seed1 | vcs_qmi | 8 | 1 | 800/800 | 87.40 | 85.82 | 0.4392±0.0007 | 63.98 | 16091 | 5269/8146 | COMPLETED |
| P104_U2_views4_800ep_seed2 | vcs_qmi | 8 | 2 | 800/800 | 87.44 | 85.72 | 0.4385±0.0010 | 65.16 | 28159 | 7066/10154 | COMPLETED |

## Hyper-parameters per run (from run_manifest.json)

| run | K | critic hidden | last gain | critic lr× | critic wd | proj hidden | proj out | critic input | critic impl | B | lr | epochs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| P104_G1_views4_800ep_seed0 | 8 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | FixedCosineCritic | 256 | 0.001 | 800 |
| P104_G2F_views4_800ep_seed0 | 8 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | CosineCritic | 256 | 0.001 | 800 |
| P104_G2F_views4_800ep_seed1 | 8 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | CosineCritic | 256 | 0.001 | 800 |
| P104_G2F_views4_800ep_seed2 | 8 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | CosineCritic | 256 | 0.001 | 800 |
| P104_G2_views4_800ep_seed0 | 8 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | FixedCosineCritic | 256 | 0.001 | 800 |
| P104_G2_views4_800ep_seed1 | 8 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | FixedCosineCritic | 256 | 0.001 | 800 |
| P104_G2_views4_800ep_seed2 | 8 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | FixedCosineCritic | 256 | 0.001 | 800 |
| P104_G3_views4_800ep_seed0 | 8 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | FixedCosineCritic | 256 | 0.001 | 800 |
| P104_G4_views4_800ep_seed0 | 8 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | CosineCritic | 256 | 0.001 | 800 |
| P104_N1_views4_800ep_seed0 | 8 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | NoisyCosineCritic | 256 | 0.001 | 800 |
| P104_N2_views4_800ep_seed0 | 8 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | NoisyCosineCritic | 256 | 0.001 | 800 |
| P104_U1_views4_800ep_seed0 | 8 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | CosineCritic | 256 | 0.001 | 800 |
| P104_U2F_views4_800ep_seed0 | 8 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | CosineCritic | 256 | 0.001 | 800 |
| P104_U2F_views4_800ep_seed1 | 8 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | CosineCritic | 256 | 0.001 | 800 |
| P104_U2F_views4_800ep_seed2 | 8 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | CosineCritic | 256 | 0.001 | 800 |
| P104_U2_views4_800ep_seed0 | 8 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | FixedCosineCritic | 256 | 0.001 | 800 |
| P104_U2_views4_800ep_seed1 | 8 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | FixedCosineCritic | 256 | 0.001 | 800 |
| P104_U2_views4_800ep_seed2 | 8 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | FixedCosineCritic | 256 | 0.001 | 800 |

## Epoch-0 (random init, same seed) reference and deltas

| run | linear-val ep0 (%) | linear-val final (%) | Δ linear | kNN ep0 (%) | kNN final (%) | Δ kNN | h-rank ep0 | h-rank final | heldout-J ep0 |
|---|---|---|---|---|---|---|---|---|---|
| P104_G1_views4_800ep_seed0 | 41.94 | 87.04 | 45.10 | 36.54 | 85.32 | 48.78 | 2.94 | 62.60 | -0.5686 |
| P104_G2F_views4_800ep_seed0 | 41.94 | 86.70 | 44.76 | 36.54 | 84.38 | 47.84 | 2.94 | 96.54 | -0.5569 |
| P104_G2F_views4_800ep_seed1 | 41.72 | 86.08 | 44.36 | 37.42 | 84.38 | 46.96 | 3.22 | 97.25 | -0.5511 |
| P104_G2F_views4_800ep_seed2 | 42.98 | 86.32 | 43.34 | 37.08 | 85.06 | 47.98 | 3.22 | 97.01 | -0.5471 |
| P104_G2_views4_800ep_seed0 | 41.94 | 88.66 | 46.72 | 36.54 | 86.96 | 50.42 | 2.94 | 98.90 | -0.5569 |
| P104_G2_views4_800ep_seed1 | 41.68 | 88.92 | 47.24 | 37.40 | 87.08 | 49.68 | 3.22 | 98.62 | -0.5511 |
| P104_G2_views4_800ep_seed2 | 42.98 | 88.52 | 45.54 | 37.08 | 87.46 | 50.38 | 3.22 | 98.46 | -0.5471 |
| P104_G3_views4_800ep_seed0 | 41.94 | 87.64 | 45.70 | 36.54 | 85.38 | 48.84 | 2.94 | 82.87 | -0.5686 |
| P104_G4_views4_800ep_seed0 | 41.94 | 81.70 | 39.76 | 36.54 | 78.04 | 41.50 | 2.94 | 156.78 | -0.9998 |
| P104_N1_views4_800ep_seed0 | 41.98 | 87.30 | 45.32 | 36.58 | 85.32 | 48.74 | 2.94 | 129.16 | -0.9998 |
| P104_N2_views4_800ep_seed0 | 41.94 | 86.84 | 44.90 | 36.54 | 84.84 | 48.30 | 2.94 | 130.13 | -0.9998 |
| P104_U1_views4_800ep_seed0 | 41.94 | 87.56 | 45.62 | 36.54 | 85.58 | 49.04 | 2.94 | 135.59 | -0.9998 |
| P104_U2F_views4_800ep_seed0 | 41.94 | 87.22 | 45.28 | 36.54 | 85.44 | 48.90 | 2.94 | 118.08 | -0.5686 |
| P104_U2F_views4_800ep_seed1 | 41.68 | 87.34 | 45.66 | 37.40 | 85.12 | 47.72 | 3.22 | 118.95 | -0.5658 |
| P104_U2F_views4_800ep_seed2 | 42.98 | 86.86 | 43.88 | 37.08 | 85.06 | 47.98 | 3.22 | 118.65 | -0.5639 |
| P104_U2_views4_800ep_seed0 | 41.94 | 87.96 | 46.02 | 36.54 | 86.10 | 49.56 | 2.94 | 69.38 | -0.5686 |
| P104_U2_views4_800ep_seed1 | 41.68 | 87.40 | 45.72 | 37.40 | 85.82 | 48.42 | 3.22 | 63.98 | -0.5658 |
| P104_U2_views4_800ep_seed2 | 42.98 | 87.44 | 44.46 | 37.06 | 85.72 | 48.66 | 3.22 | 65.16 | -0.5639 |

## kNN trajectory (in-training monitor, selection top-1 %)

- P104_G1_views4_800ep_seed0: ep0: 36.54, ep20: 66.32, ep50: 74.28, ep100: 80.72, ep200: 82.84, ep400: 84.76, ep600: 84.92, ep800: 85.32
- P104_G2F_views4_800ep_seed0: ep0: 36.54, ep20: 67.06, ep50: 72.88, ep100: 76.88, ep200: 80.44, ep400: 83.72, ep600: 83.98, ep800: 84.38
- P104_G2F_views4_800ep_seed1: ep0: 37.42, ep20: 67.14, ep50: 73.58, ep100: 77.68, ep200: 81.02, ep400: 83.10, ep600: 84.34, ep800: 84.38
- P104_G2F_views4_800ep_seed2: ep0: 37.08, ep20: 66.14, ep50: 73.30, ep100: 77.76, ep200: 80.54, ep400: 83.36, ep600: 84.62, ep800: 85.06
- P104_G2_views4_800ep_seed0: ep0: 36.54, ep20: 67.88, ep50: 75.00, ep100: 80.68, ep200: 83.60, ep400: 85.60, ep600: 86.66, ep800: 86.96
- P104_G2_views4_800ep_seed1: ep0: 37.40, ep20: 67.48, ep50: 75.40, ep100: 80.42, ep200: 83.80, ep400: 86.42, ep600: 86.76, ep800: 87.08
- P104_G2_views4_800ep_seed2: ep0: 37.08, ep20: 67.72, ep50: 75.40, ep100: 80.32, ep200: 84.02, ep400: 86.14, ep600: 87.34, ep800: 87.46
- P104_G3_views4_800ep_seed0: ep0: 36.54, ep20: 68.80, ep50: 75.36, ep100: 80.36, ep200: 83.74, ep400: 85.36, ep600: 85.26, ep800: 85.38
- P104_G4_views4_800ep_seed0: ep0: 36.54, ep20: 67.38, ep50: 72.66, ep100: 73.88, ep200: 76.28, ep400: 77.58, ep600: 78.00, ep800: 78.04
- P104_N1_views4_800ep_seed0: ep0: 36.58, ep20: 68.56, ep50: 74.48, ep100: 78.34, ep200: 81.88, ep400: 84.96, ep600: 85.28, ep800: 85.32
- P104_N2_views4_800ep_seed0: ep0: 36.54, ep20: 68.04, ep50: 73.44, ep100: 78.02, ep200: 81.44, ep400: 83.84, ep600: 84.74, ep800: 84.84
- P104_U1_views4_800ep_seed0: ep0: 36.54, ep20: 69.06, ep50: 74.86, ep100: 78.48, ep200: 81.98, ep400: 84.72, ep600: 85.22, ep800: 85.58
- P104_U2F_views4_800ep_seed0: ep0: 36.54, ep20: 69.08, ep50: 74.72, ep100: 78.82, ep200: 81.76, ep400: 83.96, ep600: 85.06, ep800: 85.44
- P104_U2F_views4_800ep_seed1: ep0: 37.40, ep20: 68.82, ep50: 75.26, ep100: 78.82, ep200: 82.36, ep400: 84.30, ep600: 85.04, ep800: 85.12
- P104_U2F_views4_800ep_seed2: ep0: 37.08, ep20: 68.84, ep50: 75.78, ep100: 79.08, ep200: 81.66, ep400: 83.76, ep600: 84.92, ep800: 85.06
- P104_U2_views4_800ep_seed0: ep0: 36.54, ep20: 68.40, ep50: 77.06, ep100: 81.72, ep200: 84.26, ep400: 86.02, ep600: 85.88, ep800: 86.10
- P104_U2_views4_800ep_seed1: ep0: 37.40, ep20: 68.56, ep50: 76.48, ep100: 81.24, ep200: 84.10, ep400: 85.64, ep600: 86.04, ep800: 85.82
- P104_U2_views4_800ep_seed2: ep0: 37.06, ep20: 68.66, ep50: 77.44, ep100: 81.78, ep200: 84.60, ep400: 85.38, ep600: 85.54, ep800: 85.72

## Held-out J trajectory (VCS only)

- P104_G1_views4_800ep_seed0: ep0: -0.5686, ep20: 0.4026, ep50: 0.4265, ep100: 0.4364, ep200: 0.4406, ep400: 0.4418, ep600: 0.4411, ep800: 0.4410
- P104_G2F_views4_800ep_seed0: ep0: -0.5569, ep20: 0.8471, ep50: 0.8972, ep100: 0.9295, ep200: 0.9487, ep400: 0.9603, ep600: 0.9653, ep800: 0.9670
- P104_G2F_views4_800ep_seed1: ep0: -0.5511, ep20: 0.8378, ep50: 0.9000, ep100: 0.9266, ep200: 0.9471, ep400: 0.9603, ep600: 0.9659, ep800: 0.9676
- P104_G2F_views4_800ep_seed2: ep0: -0.5471, ep20: 0.8321, ep50: 0.8976, ep100: 0.9290, ep200: 0.9477, ep400: 0.9615, ep600: 0.9662, ep800: 0.9678
- P104_G2_views4_800ep_seed0: ep0: -0.5569, ep20: 0.7812, ep50: 0.8137, ep100: 0.8415, ep200: 0.8522, ep400: 0.8587, ep600: 0.8604, ep800: 0.8608
- P104_G2_views4_800ep_seed1: ep0: -0.5511, ep20: 0.7687, ep50: 0.8189, ep100: 0.8392, ep200: 0.8523, ep400: 0.8586, ep600: 0.8608, ep800: 0.8615
- P104_G2_views4_800ep_seed2: ep0: -0.5471, ep20: 0.7693, ep50: 0.8131, ep100: 0.8385, ep200: 0.8528, ep400: 0.8590, ep600: 0.8610, ep800: 0.8614
- P104_G3_views4_800ep_seed0: ep0: -0.5686, ep20: 0.4188, ep50: 0.4323, ep100: 0.4374, ep200: 0.4426, ep400: 0.4424, ep600: 0.4399, ep800: 0.4391
- P104_G4_views4_800ep_seed0: ep0: -0.9998, ep20: 0.8958, ep50: 0.9461, ep100: 0.9687, ep200: 0.9798, ep400: 0.9859, ep600: 0.9889, ep800: 0.9897
- P104_N1_views4_800ep_seed0: ep0: -0.9998, ep20: 0.8660, ep50: 0.9178, ep100: 0.9480, ep200: 0.9666, ep400: 0.9757, ep600: 0.9784, ep800: 0.9797
- P104_N2_views4_800ep_seed0: ep0: -0.9998, ep20: 0.8690, ep50: 0.9272, ep100: 0.9575, ep200: 0.9698, ep400: 0.9773, ep600: 0.9801, ep800: 0.9811
- P104_U1_views4_800ep_seed0: ep0: -0.9998, ep20: 0.8754, ep50: 0.9231, ep100: 0.9431, ep200: 0.9595, ep400: 0.9688, ep600: 0.9730, ep800: 0.9745
- P104_U2F_views4_800ep_seed0: ep0: -0.5686, ep20: 0.8515, ep50: 0.9176, ep100: 0.9389, ep200: 0.9560, ep400: 0.9660, ep600: 0.9701, ep800: 0.9716
- P104_U2F_views4_800ep_seed1: ep0: -0.5658, ep20: 0.8442, ep50: 0.9177, ep100: 0.9393, ep200: 0.9550, ep400: 0.9653, ep600: 0.9700, ep800: 0.9714
- P104_U2F_views4_800ep_seed2: ep0: -0.5639, ep20: 0.8385, ep50: 0.9148, ep100: 0.9384, ep200: 0.9557, ep400: 0.9660, ep600: 0.9702, ep800: 0.9716
- P104_U2_views4_800ep_seed0: ep0: -0.5686, ep20: 0.4068, ep50: 0.4236, ep100: 0.4380, ep200: 0.4414, ep400: 0.4412, ep600: 0.4392, ep800: 0.4390
- P104_U2_views4_800ep_seed1: ep0: -0.5658, ep20: 0.4052, ep50: 0.4285, ep100: 0.4363, ep200: 0.4411, ep400: 0.4409, ep600: 0.4397, ep800: 0.4392
- P104_U2_views4_800ep_seed2: ep0: -0.5639, ep20: 0.4018, ep50: 0.4250, ep100: 0.4362, ep200: 0.4414, ep400: 0.4406, ep600: 0.4387, ep800: 0.4385

## Final-epoch training objective values (epoch means)

- P104_G1_views4_800ep_seed0: J_raw 0.4599, R_binary 0.5401, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P104_G2F_views4_800ep_seed0: J_raw 0.9809, R_binary 0.0191, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P104_G2F_views4_800ep_seed1: J_raw 0.9810, R_binary 0.0190, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P104_G2F_views4_800ep_seed2: J_raw 0.9812, R_binary 0.0188, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P104_G2_views4_800ep_seed0: J_raw 0.8945, R_binary 0.1055, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P104_G2_views4_800ep_seed1: J_raw 0.8946, R_binary 0.1054, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P104_G2_views4_800ep_seed2: J_raw 0.8941, R_binary 0.1059, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P104_G3_views4_800ep_seed0: J_raw 0.4628, R_binary 0.5372, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P104_G4_views4_800ep_seed0: J_raw 0.9946, R_binary 0.0054, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P104_N1_views4_800ep_seed0: J_raw 0.9799, R_binary 0.0201, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P104_N2_views4_800ep_seed0: J_raw 0.9811, R_binary 0.0189, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P104_U1_views4_800ep_seed0: J_raw 0.9866, R_binary 0.0134, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P104_U2F_views4_800ep_seed0: J_raw 0.9847, R_binary 0.0153, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P104_U2F_views4_800ep_seed1: J_raw 0.9842, R_binary 0.0158, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P104_U2F_views4_800ep_seed2: J_raw 0.9842, R_binary 0.0158, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P104_U2_views4_800ep_seed0: J_raw 0.4615, R_binary 0.5385, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P104_U2_views4_800ep_seed1: J_raw 0.4614, R_binary 0.5386, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P104_U2_views4_800ep_seed2: J_raw 0.4613, R_binary 0.5387, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}

## Cost

| run | steady step (s) | images/s | views/s | train s | in-train eval s | final eval s | seen base images | critic params | job |
|---|---|---|---|---|---|---|---|---|---|
| P104_G1_views4_800ep_seed0 | 0.1133 | 2259 | 4518 | 16071 | 50 | 11 | 35840000 | None | 1016190@node59 |
| P104_G2F_views4_800ep_seed0 | 0.1138 | 2250 | 4500 | 16134 | 52 | 11 | 35840000 | 2 | 1016863@node58 |
| P104_G2F_views4_800ep_seed1 | 0.1966 | 1302 | 2604 | 27918 | 99 | 21 | 35840000 | 2 | 1016866@nodesumo01 |
| P104_G2F_views4_800ep_seed2 | 0.1138 | 2249 | 4499 | 16144 | 54 | 11 | 35840000 | 2 | 1017041@node58 |
| P104_G2_views4_800ep_seed0 | 0.1129 | 2268 | 4535 | 16014 | 53 | 11 | 35840000 | None | 1016195@node61 |
| P104_G2_views4_800ep_seed1 | 0.1130 | 2265 | 4530 | 16031 | 51 | 13 | 35840000 | None | 1016864@node61 |
| P104_G2_views4_800ep_seed2 | 0.2486 | 1030 | 2059 | 35029 | 85 | 16 | 35840000 | None | 1016865@node61 |
| P104_G3_views4_800ep_seed0 | 0.1138 | 2250 | 4500 | 16134 | 52 | 11 | 35840000 | None | 1016191@node59 |
| P104_G4_views4_800ep_seed0 | 0.1141 | 2245 | 4489 | 16175 | 52 | 11 | 35840000 | 2 | 1016196@node59 |
| P104_N1_views4_800ep_seed0 | 0.2125 | 1205 | 2410 | 30121 | 93 | 20 | 35840000 | 2 | 1016193@node53 |
| P104_N2_views4_800ep_seed0 | 0.1153 | 2221 | 4442 | 16346 | 54 | 11 | 35840000 | 2 | 1016194@node58 |
| P104_U1_views4_800ep_seed0 | 0.1136 | 2254 | 4507 | 16117 | 56 | 12 | 35840000 | 2 | 1016192@node58 |
| P104_U2F_views4_800ep_seed0 | 0.1139 | 2248 | 4497 | 16147 | 50 | 11 | 35840000 | 2 | 1017038@node59 |
| P104_U2F_views4_800ep_seed1 | 0.1139 | 2249 | 4497 | 16150 | 56 | 12 | 35840000 | 2 | 1017039@node58 |
| P104_U2F_views4_800ep_seed2 | 0.1134 | 2257 | 4514 | 16085 | 52 | 11 | 35840000 | 2 | 1017040@node61 |
| P104_U2_views4_800ep_seed0 | 0.1135 | 2256 | 4511 | 16092 | 51 | 15 | 35840000 | None | 1016197@node59 |
| P104_U2_views4_800ep_seed1 | 0.1134 | 2257 | 4513 | 16091 | 52 | 11 | 35840000 | None | 1017036@node59 |
| P104_U2_views4_800ep_seed2 | 0.1985 | 1290 | 2579 | 28159 | 91 | 20 | 35840000 | None | 1017037@node53 |

## Provenance

- P104_G1_views4_800ep_seed0: commit `cfa03d05d116c32e9c332ccefa0849cad7434a22` dirty=True, config `a160cd498615ae64805e9a0e21c7a014bdfe7fa8c8bce13b7acd734e1d5613d1`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`
- P104_G2F_views4_800ep_seed0: commit `34f4e29ad756b86c918dded136f157c62c7c1416` dirty=True, config `b953e3fe017fd0cb033dc5df89d18db0665e040635ebfaa4251a587514863427`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`
- P104_G2F_views4_800ep_seed1: commit `34f4e29ad756b86c918dded136f157c62c7c1416` dirty=True, config `7f95ce91b8d072486c9656c6fb8fa8032c62dff391cb5fb1fe2c0bde9e828f6c`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `94c2cb882a7ebf80a65d48352d007cf5fd956df2f37cd9a10980d83b11f4a51a`
- P104_G2F_views4_800ep_seed2: commit `70f53acf0cbfec9c535edad2e26e8dd1c51706fc` dirty=True, config `26751a182ff17b700bffa489672172d122999ac2dd3538ca7954e02792dfdab3`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `4385db04f2c25cafec319bb8e1dc5725215e7eebf11ab2727a37174e73c8c9d5`
- P104_G2_views4_800ep_seed0: commit `7ef1717e9b65698bbe8d94d084c7fe8c0d101271` dirty=True, config `3f9d8cebb0992714ca9f520f860a88857e17a64e0f354f4906a1a2fc822b32df`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`
- P104_G2_views4_800ep_seed1: commit `34f4e29ad756b86c918dded136f157c62c7c1416` dirty=True, config `a4b742e4527c5efb52fc40a336650045d04a7449bd81f22009e5d1a7050f598a`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `94c2cb882a7ebf80a65d48352d007cf5fd956df2f37cd9a10980d83b11f4a51a`
- P104_G2_views4_800ep_seed2: commit `34f4e29ad756b86c918dded136f157c62c7c1416` dirty=True, config `5719d64dec9e2327cd6e0024a1926e75581bc388d8898967d64026c996dbbac8`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `4385db04f2c25cafec319bb8e1dc5725215e7eebf11ab2727a37174e73c8c9d5`
- P104_G3_views4_800ep_seed0: commit `cfa03d05d116c32e9c332ccefa0849cad7434a22` dirty=True, config `c576738df5bc563fef7153ab1ec4f00e7a2fa2803f5308d817f04a794a2995c2`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`
- P104_G4_views4_800ep_seed0: commit `7ef1717e9b65698bbe8d94d084c7fe8c0d101271` dirty=True, config `0b043bc18a7cabe250a454d6f9bfb7be5543876bd96e222c77664a3bf4a4de51`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`
- P104_N1_views4_800ep_seed0: commit `cfa03d05d116c32e9c332ccefa0849cad7434a22` dirty=True, config `baef4bd00f44fddaec5ee5189b7ee600e039f167f47a18a8e34c2f0e6131ad4e`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`
- P104_N2_views4_800ep_seed0: commit `7ef1717e9b65698bbe8d94d084c7fe8c0d101271` dirty=True, config `64225067779f01b6bae527112d7fca7851e8f525b487e9b2ba6699f69ad41c0a`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`
- P104_U1_views4_800ep_seed0: commit `cfa03d05d116c32e9c332ccefa0849cad7434a22` dirty=True, config `2026ce67f1ce1d6d5aa52c453c69c5676d5c4b21bcc41cc12b233e0f292d7b1e`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`
- P104_U2F_views4_800ep_seed0: commit `1c87c9d369c90000e6718d46fdb372cc25af9a6c` dirty=True, config `ffb00209c3814cf7f439c390640739692f10a4325c217868eb78985100d4783c`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`
- P104_U2F_views4_800ep_seed1: commit `33475fe42e117d7511891e7e19571e5521418406` dirty=True, config `c4e8a7c5f7282a3d31043ffd7bf2bd38ddebc417504eae691654627a73ea47c4`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `94c2cb882a7ebf80a65d48352d007cf5fd956df2f37cd9a10980d83b11f4a51a`
- P104_U2F_views4_800ep_seed2: commit `70f53acf0cbfec9c535edad2e26e8dd1c51706fc` dirty=True, config `02f9fe42abe24807fcf1b3ef0bbee57cd1474cc72fdd699b68dbb8d16baef4c0`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `4385db04f2c25cafec319bb8e1dc5725215e7eebf11ab2727a37174e73c8c9d5`
- P104_U2_views4_800ep_seed0: commit `7ef1717e9b65698bbe8d94d084c7fe8c0d101271` dirty=True, config `03d2a5b14551b1d2c29431fc621401554771f31961be16083fba4ca114fcd57a`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`
- P104_U2_views4_800ep_seed1: commit `4c86d83a618ddb80670a24be70b83e30286c9495` dirty=True, config `274c4ce54a341c5f5c6ec9478b431611dd5d0e23511a5056ddfca5cef40db875`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `94c2cb882a7ebf80a65d48352d007cf5fd956df2f37cd9a10980d83b11f4a51a`
- P104_U2_views4_800ep_seed2: commit `4c86d83a618ddb80670a24be70b83e30286c9495` dirty=True, config `6b035453bfbe8fd4a28f8e601614411688b43bfcf351344225439c1074f2b855`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `4385db04f2c25cafec319bb8e1dc5725215e7eebf11ab2727a37174e73c8c9d5`

- identical encoder init across runs: False
- identical split across runs: True

## Per-method aggregation across seeds (only COMPLETED runs with a final evaluation; mean ± sample SD, n seeds)

| method | n | linear-val final (%) | linear-val ep0 (%) | Δ linear | kNN final (%) | h-rank final | heldout-J final |
|---|---|---|---|---|---|---|---|
| vcs_qmi K=8 | 18 | 87.08 ± 1.54 | 42.12 ± 0.49 | 44.97 ± 1.60 | 85.17 ± 1.97 | 102.12 ± 26.98 | 0.81 ± 0.24 |

## Coverage checks

- P104_G1_views4_800ep_seed0: epoch0 eval True, final eval True (evaluation_epoch_800.json), status COMPLETED, failure None
- P104_G2F_views4_800ep_seed0: epoch0 eval True, final eval True (evaluation_epoch_800.json), status COMPLETED, failure None
- P104_G2F_views4_800ep_seed1: epoch0 eval True, final eval True (evaluation_epoch_800.json), status COMPLETED, failure None
- P104_G2F_views4_800ep_seed2: epoch0 eval True, final eval True (evaluation_epoch_800.json), status COMPLETED, failure None
- P104_G2_views4_800ep_seed0: epoch0 eval True, final eval True (evaluation_epoch_800.json), status COMPLETED, failure None
- P104_G2_views4_800ep_seed1: epoch0 eval True, final eval True (evaluation_epoch_800.json), status COMPLETED, failure None
- P104_G2_views4_800ep_seed2: epoch0 eval True, final eval True (evaluation_epoch_800.json), status COMPLETED, failure None
- P104_G3_views4_800ep_seed0: epoch0 eval True, final eval True (evaluation_epoch_800.json), status COMPLETED, failure None
- P104_G4_views4_800ep_seed0: epoch0 eval True, final eval True (evaluation_epoch_800.json), status COMPLETED, failure None
- P104_N1_views4_800ep_seed0: epoch0 eval True, final eval True (evaluation_epoch_800.json), status COMPLETED, failure None
- P104_N2_views4_800ep_seed0: epoch0 eval True, final eval True (evaluation_epoch_800.json), status COMPLETED, failure None
- P104_U1_views4_800ep_seed0: epoch0 eval True, final eval True (evaluation_epoch_800.json), status COMPLETED, failure None
- P104_U2F_views4_800ep_seed0: epoch0 eval True, final eval True (evaluation_epoch_800.json), status COMPLETED, failure None
- P104_U2F_views4_800ep_seed1: epoch0 eval True, final eval True (evaluation_epoch_800.json), status COMPLETED, failure None
- P104_U2F_views4_800ep_seed2: epoch0 eval True, final eval True (evaluation_epoch_800.json), status COMPLETED, failure None
- P104_U2_views4_800ep_seed0: epoch0 eval True, final eval True (evaluation_epoch_800.json), status COMPLETED, failure None
- P104_U2_views4_800ep_seed1: epoch0 eval True, final eval True (evaluation_epoch_800.json), status COMPLETED, failure None
- P104_U2_views4_800ep_seed2: epoch0 eval True, final eval True (evaluation_epoch_800.json), status COMPLETED, failure None
