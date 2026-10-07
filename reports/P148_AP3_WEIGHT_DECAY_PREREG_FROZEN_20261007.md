# Pre-registration — P148: VCS (A-P3) matrix weight decay 1e-4 → {5e-4, 1e-5}, CIFAR-10 / CIFAR-100, seed 0 — FROZEN 2026-10-07T08:32:35Z

Owner rules: keep ≥ 15 experiment jobs queued (queue at 14 after the post-quota resume; no triggered follow-up or fork unit ready); explore
everything except the collaborator's plan and the backbone for the best VCS performance (backlog: optimizer lr / wd); one full seed per new idea
first.  P135 covered lr; weight decay has not been varied.
**Cells (4):** `P148_AP3_{c10,c100}_{wd5e-4,wd1e-5}_seed0` = the A-P3 seed-0 parent with ONLY `optimizer.matrix_weight_decay_encoder_projector`
1e-4 → 5e-4 or 1e-5 (bias / norm decay 0 and everything else unchanged; `configs/make_p148_configs.py`, `configs/P148_SHA256.json`,
`slurm/p148_lines.txt`; policy load checked).  800 epochs, development split.

## Pre-stated reading
- Primary: final frozen-h linear vs A-P3 seed 0 (89.06 / 60.20); kNN alongside.
- **Promising** = linear ≥ A-P3 + 0.30 (CIFAR-10) / + 0.50 (CIFAR-100) → seeds 1–2 of that cell (addendum) and later the matched-JS counterpart.
  No gain at either value → the weight-decay axis ends at 1e-4.
- Not claimed: other decays, decoupled schedules, multi-seed effects; official test closed.
