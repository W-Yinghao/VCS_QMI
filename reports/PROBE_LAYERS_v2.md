# Per-layer probe diagnostic — 2026-09-26T02:35:33Z

Frozen final checkpoints, clean-transform features, fit UIDs train the head / selection UIDs score it. Each cell: linear-val % / kNN % (k=200, T=0.1) / effective rank (4096 selection). l2, l3 = avg-pooled ResNet stage outputs; h = the frozen endpoint; proj_hidden = projector ReLU output; p_raw / z_l2 = projector output before / after L2.

| run | method | critic | K | negdet | l2 | l3 | h | proj_hidden | p_raw | z_l2 |
|---|---|---|---|---|---|---|---|---|---|---|
| P5_simclr_seed0 | simclr_matched | None | 1 | False | 68.46 / 57.42 / 16 (d=128) | 81.74 / 72.12 / 33 (d=256) | 86.44 / 84.02 / 91 (d=512) | 84.92 / 83.02 / 64 (d=512) | 84.52 / 83.84 / 77 (d=128) | 83.92 / 83.84 / 77 (d=128) |
| P5_vicreg_seed0 | vicreg_matched_128 | None | 1 | False | 67.02 / 56.62 / 17 (d=128) | 81.50 / 71.40 / 33 (d=256) | 85.64 / 81.46 / 75 (d=512) | 83.76 / 80.20 / 88 (d=512) | 82.38 / 80.24 / 87 (d=128) | 81.74 / 80.24 / 77 (d=128) |
| P24_vcs_cos_negdetach_seed0 | vcs_qmi | cosine | 8 | True | 65.28 / 57.02 / 16 (d=128) | 77.46 / 68.40 / 26 (d=256) | 80.48 / 74.62 / 31 (d=512) | 77.42 / 74.98 / 38 (d=512) | 70.96 / 74.72 / 14 (d=128) | 71.16 / 74.72 / 14 (d=128) |
| P26_vcs_a5_learn_seed0 | vcs_qmi | cosine | 8 | True | 64.86 / 55.66 / 16 (d=128) | 77.34 / 67.44 / 28 (d=256) | 81.90 / 77.00 / 45 (d=512) | 79.82 / 77.64 / 55 (d=512) | 75.46 / 77.42 / 21 (d=128) | 75.20 / 77.42 / 20 (d=128) |
| P28_vcs_views4_seed0 | vcs_qmi | cosine | 8 | True | 65.74 / 56.06 / 17 (d=128) | 79.56 / 69.28 / 33 (d=256) | 84.48 / 81.10 / 57 (d=512) | 83.30 / 81.68 / 60 (d=512) | 79.62 / 81.38 / 22 (d=128) | 78.48 / 81.38 / 21 (d=128) |
| P35_vcs_a5_views4_seed0 | vcs_qmi | cosine | 8 | True | 65.66 / 55.56 / 17 (d=128) | 79.62 / 68.32 / 34 (d=256) | 84.50 / 81.40 / 76 (d=512) | 83.38 / 81.90 / 79 (d=512) | 81.32 / 81.76 / 31 (d=128) | 80.44 / 81.76 / 30 (d=128) |
| P37_vcs_a5_views8_100ep_seed0 | vcs_qmi | cosine | 8 | True | 65.70 / 54.96 / 17 (d=128) | 78.40 / 67.28 / 34 (d=256) | 83.40 / 80.44 / 74 (d=512) | 83.76 / 81.20 / 83 (d=512) | 82.04 / 81.96 / 42 (d=128) | 80.76 / 81.96 / 42 (d=128) |
| P26_vcs_base_800ep_seed0 | vcs_qmi | cosine | 8 | True | 66.34 / 56.36 / 18 (d=128) | 80.86 / 71.02 / 33 (d=256) | 84.64 / 81.44 / 61 (d=512) | 82.72 / 80.56 / 42 (d=512) | 74.66 / 80.34 / 15 (d=128) | 76.10 / 80.34 / 15 (d=128) |
| P24_vcs_cos_800ep_seed0 | vcs_qmi | cosine | 8 | False | 64.14 / 55.04 / 18 (d=128) | 75.96 / 65.40 / 35 (d=256) | 80.74 / 76.48 / 89 (d=512) | 79.42 / 79.00 / 111 (d=512) | 73.28 / 80.02 / 45 (d=128) | 74.66 / 80.02 / 45 (d=128) |

## Reading (interpretation commit)
| run | l3 → h gain | gap to SimCLR at l2 / l3 / h | z_l2 − h | p_raw eff-rank | kNN proj_hidden − h |
|---|---|---|---|---|---|
| SimCLR | +4.70 | — | −2.5 | 77 | −1.0 |
| detach base (2 v, 200 ep) | +3.02 | 3.2 / 4.3 / 6.0 | −9.3 | 14 | +0.4 |
| a0 = 5 (2 v, 200 ep) | +4.56 | 3.6 / 4.4 / 4.5 | −6.7 | 21 | +0.6 |
| **4 views, a0 = 5 (200 ep)** | **+4.88** | **2.8 / 2.1 / 1.9** | −4.1 | 31 | +0.5 |
| 4 views, a0 = 1 (200 ep) | +4.92 | 2.7 / 2.2 / 2.0 | −6.0 | 22 | +0.6 |
| 8 views, a0 = 5 (100 ep) | +5.00 | 2.8 / 3.3 / 3.0 | −2.6 | 42 | +0.8 |
| detach base, 800 ep | +3.78 | 2.1 / 0.9 / 1.8 | −8.5 | 15 | −0.9 |
| plain cosine, 800 ep | +4.78 | 4.3 / 5.8 / 5.7 | −6.1 | 45 | +2.5 |

1. **The last ResNet stage is fully productive under the recipe** (+4.9 from layer3 to h, SimCLR +4.7; the original MLP recipe had +0.8).
2. **The remaining deficit is now uniform across depth** (≈ 2 points at layer2, layer3 and h for the 4-view recipe; the 800-epoch 2-view
   run is even closer at layer3, 0.9), instead of growing with depth as in every earlier VCS run (3.8 / 7.6 / 11.6 for the original).  What is
   left is a whole-network effect of the objective, not a last-block pathology.
3. **The projector still discards more than SimCLR's** (z probes 4–8 below h vs 2.5; p_raw effective rank 15–31 vs 77), most extremely in
   the 800-epoch detach run (rank 15, z −8.5) — the per-layer view of the narrow z cone in GEOMETRY_DIAG_v2 §2.  The projector keeps
   converting a low-dimensional pair-separability code into a spread h; giving it more pairs (views) or more steps widens what reaches h.

Not claimed: single seed; selection set; probe on CPU (identical h endpoints confirm parity with the GPU evaluations).
