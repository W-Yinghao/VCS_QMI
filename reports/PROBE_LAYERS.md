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

## Reading (added after the table; interpretation commit)

Consistency: the h column reproduces each run's own GPU evaluation to ≤ 0.04 (e.g. 86.44 / 74.88 vs 74.84 / 76.14 vs 76.12 / 80.48 / 79.56), so
the layer comparisons below are on the same footing as the reported endpoints.

| run | l3 → h gain (linear) | gap to SimCLR at l2 / l3 / h | z_l2 − h (linear) | p_raw eff-rank | kNN p_raw − h |
|---|---|---|---|---|---|
| SimCLR | +4.70 | — | −2.5 | 77 | −0.2 |
| VICReg | +4.14 | 1.4 / 0.2 / 0.8 | −3.9 | 87 | −1.2 |
| VCS MLP K=1 | **+0.78** | 3.8 / 7.6 / 11.6 | −14.0 | **7** | −1.1 |
| VCS MLP K=8 | **+0.50** | 3.6 / 6.1 / 10.3 | −12.7 | **8** | −0.9 |
| VCS cosine K=8 | +3.94 | 4.0 / 7.4 / 8.1 | −6.1 | 42 | **+4.4** |
| VCS cosine K=8 neg-detach | +3.02 | 3.2 / 4.3 / 6.0 | −9.3 | 14 | +0.1 |
| VCS MLP K=1, 800 ep | +2.14 | 2.1 / 4.6 / 7.2 | −11.4 | 9 | −0.3 |
| VCS MLP K=8 clr×10, 800 ep | +1.48 | 2.7 / 3.7 / 6.9 | −9.6 | 12 | −1.1 |

1. **With the concat-MLP critic the last ResNet stage learns almost nothing linearly useful.** layer3 → h adds +0.5…+0.8 at 200 epochs
   (SimCLR +4.7, VICReg +4.1); 800 epochs recover part of it (+1.5…+2.1). Similarity-type critics restore the stage (+3.0…+3.9). The
   low-rank pull of the MLP critic (synthesis §3.1) therefore acts mainly on the block that feeds the projector; the layer-3 features of the
   same networks are only 6–7.6 points behind SimCLR, not 10–12.
2. **The deficit is nevertheless present at every depth and grows with it** (gap 3–4 at layer2, 4–7.6 at layer3, 6–11.6 at h). The best VCS
   run (neg-detach) is 3.2 / 4.3 / 6.0 behind: whatever the objective under-trains, it is not confined to the last block.
3. **The VCS projector discards far more class information than the SimCLR/VICReg projector.** z_l2 probes 6–14 points below h in VCS vs
   2.5–3.9 in the controls; p_raw has effective rank 7–14 (42 for plain cosine) vs 77–87. The critic thus sees a code of ≈10 effective
   dimensions: enough to separate positive from negative pairs (heldout-J 0.93–0.97), too little to force a rich h. This is the per-layer
   view of §4.3 (degenerate optimum): the objective is satisfied by a low-dimensional projection, and the projector supplies it.
4. **For the plain cosine critic, kNN is 4.4 points higher on p_raw than on h** (77.2 vs 72.8) — the metric structure the objective shapes
   lives in z; h inherits only part of it. Neg-detach closes this (74.7 vs 74.6) by making h rather than z the spread layer (GEOMETRY_DIAG.md).
   Both facts point at the projector as the place where VCS and the controls diverge; the P28 units `proj_linear` / `proj_bnonly`
   (projector without non-linearity / no projector) test exactly this.

Not claimed: single seed; selection set; the effective-rank figures are on the raw (uncentered-by-BN) features and are not comparable across
layers of different width; CPU numerics (identical h endpoints confirm this is immaterial here).
