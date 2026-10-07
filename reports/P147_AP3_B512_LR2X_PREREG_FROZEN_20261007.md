# Pre-registration — P147: VCS (A-P3) at batch 512 with linearly scaled learning rate (2e-3), CIFAR-10 / CIFAR-100, seed 0 — FROZEN 2026-10-07T04:42:33Z

Owner rules: keep ≥ 15 experiment jobs queued (2026-10-04; queue at 14 when registered); explore everything except the collaborator's plan and the
backbone for the best VCS performance; one full seed per new idea first.  Source: P136 report — batch 512 at the unchanged lr 1e-3 was worse
(−0.48 C10 / −0.84 C100 linear), with lr not scaled with the batch as a disclosed confound ("says nothing about b512 with a tuned lr").
**Cells (2):** `P147_AP3_c10_b512_lr2x_seed0`, `P147_AP3_c100_b512_lr2x_seed0` = the P136 b512 configs with ONLY optimizer.lr 1e-3 → 2e-3 (linear
scaling 1e-3 × 512 / 256; warm-up 10 epochs and cosine to 1 % unchanged; `configs/make_p147_configs.py`, `configs/P147_SHA256.json`,
`slurm/p147_lines.txt`; both configs pass the policy load).  800 epochs, development split.

## Pre-stated reading (P136 rule)
- Primary: final frozen-h linear vs A-P3 batch 256 seed 0 (89.06 / 60.20); kNN alongside; also vs P136 b512 at lr 1e-3 (88.58 / 59.36).
- **Promising** = linear ≥ A-P3 + 0.30 (CIFAR-10) / + 0.50 (CIFAR-100) → seeds 1–2 (addendum) and later the matched-JS counterpart; otherwise
  "batch 512 does not help at lr 1e-3 or 2e-3" and the batch axis ends.  Batch 512 halves optimizer steps per epoch; wall time reported.
- Not claimed: other lr-scaling rules (square root), other batches, multi-seed effects; official test closed.
