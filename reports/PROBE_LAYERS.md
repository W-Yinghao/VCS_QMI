# Per-layer probe diagnostic — 2026-09-25T17:14:37Z

Frozen final checkpoints, clean-transform features, fit UIDs train the head / selection UIDs score it. Each cell: linear-val % / kNN % (k=200, T=0.1) / effective rank (4096 selection). l2, l3 = avg-pooled ResNet stage outputs; h = the frozen endpoint; proj_hidden = projector ReLU output; p_raw / z_l2 = projector output before / after L2.

| run | method | critic | K | negdet | l2 | l3 | h | proj_hidden | p_raw | z_l2 |
|---|---|---|---|---|---|---|---|---|---|---|
| P5_simclr_seed0 | simclr_matched | None | 1 | False | 68.46 / 57.42 / 16 (d=128) | 81.74 / 72.12 / 33 (d=256) | 86.44 / 84.02 / 91 (d=512) | 84.92 / 83.02 / 64 (d=512) | 84.52 / 83.84 / 77 (d=128) | 83.92 / 83.84 / 77 (d=128) |
| P5_vicreg_seed0 | vicreg_matched_128 | None | 1 | False | 67.02 / 56.62 / 17 (d=128) | 81.50 / 71.40 / 33 (d=256) | 85.64 / 81.46 / 75 (d=512) | 83.76 / 80.20 / 88 (d=512) | 82.38 / 80.24 / 87 (d=128) | 81.74 / 80.24 / 77 (d=128) |
| P5_vcs_seed0 | vcs_qmi | ordered_concat | 1 | False | 64.62 / 54.72 / 14 (d=128) | 74.10 / 64.32 / 17 (d=256) | 74.88 / 64.64 / 13 (d=512) | 67.84 / 63.68 / 17 (d=512) | 60.88 / 63.58 / 7 (d=128) | 60.90 / 63.58 / 7 (d=128) |
| P10_vcs_k8_seed0 | vcs_qmi | ordered_concat | 8 | False | 64.82 / 55.62 / 15 (d=128) | 75.64 / 65.50 / 20 (d=256) | 76.14 / 67.44 / 16 (d=512) | 70.42 / 66.48 / 20 (d=512) | 64.40 / 66.54 / 8 (d=128) | 63.40 / 66.54 / 8 (d=128) |
| P18_vcs_crit_cosine_seed0 | vcs_qmi | cosine | 8 | False | 64.48 / 54.72 / 16 (d=128) | 74.38 / 64.80 / 29 (d=256) | 78.32 / 72.76 / 58 (d=512) | 77.06 / 74.92 / 98 (d=512) | 74.28 / 77.16 / 42 (d=128) | 72.18 / 77.16 / 42 (d=128) |
| P24_vcs_cos_negdetach_seed0 | vcs_qmi | cosine | 8 | True | 65.28 / 57.02 / 16 (d=128) | 77.46 / 68.40 / 26 (d=256) | 80.48 / 74.62 / 31 (d=512) | 77.42 / 74.98 / 38 (d=512) | 70.96 / 74.72 / 14 (d=128) | 71.16 / 74.72 / 14 (d=128) |
| P8_vcs800_seed0 | vcs_qmi | ordered_concat | 1 | False | 66.38 / 56.90 / 16 (d=128) | 77.10 / 68.56 / 26 (d=256) | 79.24 / 71.36 / 20 (d=512) | 74.30 / 70.94 / 21 (d=512) | 67.12 / 71.02 / 9 (d=128) | 67.88 / 71.02 / 9 (d=128) |
| P16_vcs_k8_clr10_800ep_seed0 | vcs_qmi | ordered_concat | 8 | False | 65.74 / 56.04 / 16 (d=128) | 78.08 / 68.26 / 28 (d=256) | 79.56 / 74.28 / 35 (d=512) | 76.92 / 73.94 / 29 (d=512) | 70.48 / 73.14 / 12 (d=128) | 69.92 / 73.14 / 12 (d=128) |
