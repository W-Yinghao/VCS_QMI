# Pre-registration — P136: exploration of schedule, view count and batch size for VCS (A-P3 structure), CIFAR-10 and CIFAR-100, seed 0 — FROZEN 2026-10-05T03:01:12Z

Owner rules: keep ≥ 15 experiment jobs queued (2026-10-04); explore everything except the collaborator's plan and the backbone for the best VCS
performance; one full seed per new idea first, more seeds only if promising.  The (a, κ) grid (P127) and the lr check (P135) cover the scorer and
optimizer; P136 covers three remaining training-budget axes, one change per cell from that dataset's A-P3 seed-0 config (verified by diff):
| cell | change | final checkpoint | expected cost (healthy RTX / L40S) |
|---|---|---|---|
| ep1600 | train.epochs 800 → 1600 (warm-up 10, cosine over 1600; kNN / checkpoint epochs + 1200, 1600) | epoch_1600 | ≈ 9 h / 17 h |
| v8 | views.count 4 → 8 (allowed named variant; all-view tokens, 56 positive view pairs per image) | epoch_800 | ≈ 9 h / 17 h |
| b512 | train.batch_size_images 256 → 512 (lr kept 1e-3, scale_lr_with_batch false — a disclosed confound) | epoch_800 | ≈ 4.5 h / 8.5 h |
6 units (3 cells × CIFAR-10 / CIFAR-100).  `configs/make_p136_configs.py`, `configs/P136_SHA256.json`, `slurm/p136_lines.txt`.

## Pre-stated reading
- Per cell: linear (primary) and kNN vs the A-P3 seed-0 result of the same dataset (89.06 / 87.30; 60.20 / 55.92); for ep1600 also the epoch-800
  intermediate of the same run (descriptive: what the extra 800 epochs add under a schedule that is cosine over 1600).
- **Promising** = linear ≥ A-P3 + 0.30 (CIFAR-10) / + 0.50 (CIFAR-100) → seeds 1–2 of that cell (addendum, submitted as soon as the rule triggers);
  a promising cell later also gets its matched-JS counterpart (so the VCS-vs-JS comparison is kept at the new setting) — separate addendum.
- Compute is reported per cell (wall time, GPU type, peak memory); a gain bought with 2× compute is stated as such.  No early stopping.
- Not claimed: anything about SimCLR at these settings (its recipe is fixed), other backbones, or the official test set.
