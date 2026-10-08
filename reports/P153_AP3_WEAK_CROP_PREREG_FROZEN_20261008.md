# Pre-registration — P153: VCS (A-P3) with weaker crops, RandomResizedCrop scale min 0.20 → 0.35, CIFAR-10 / CIFAR-100, seed 0 — FROZEN 2026-10-08T11:04:46Z

Owner 2026-10-08 (selected "Weak-crop cell").  Exploration of our method (memory: explore everything except the plan and the backbone).  The strong
direction is closed (P115: crop min 0.08 −0.19 linear / −1.23 kNN for A-P3, while SimCLR gained +1.58); the weaker direction is untested for A-P3
(the stage-B weak-augmentation cell — crop 0.5 + half jitter, old recipe, 200 epochs — lost ≈ 6 points, disclosed).
**Cells (2):** `P153_AP3_{c10,c100}_crop035_seed0` = the A-P3 seed-0 parent with ONLY views.random_resized_crop.scale [0.2, 1.0] → [0.35, 1.0]
(and run.stage); verified by diff; policy load passes (`configs/make_p153_configs.py`, `configs/P153_SHA256.json`, `slurm/p153_lines.txt`).
800 epochs, development split.
## Pre-stated reading
- Primary: final frozen-h linear vs A-P3 seed 0 (89.06 / 60.20); kNN alongside.
- **Promising** = linear ≥ A-P3 + 0.30 (CIFAR-10) / + 0.50 (CIFAR-100) → seeds 1–2 of that cell (addendum) and later the matched-JS counterpart.
  Otherwise the crop axis ends at 0.20.
- Not claimed: other crop scales, jitter, multi-seed effects; official test closed.
