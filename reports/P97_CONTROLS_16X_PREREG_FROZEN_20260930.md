# Pre-registration — P97: 16× compute controls (tuned SimCLR and VICReg at 4 views × 1600 epochs), 2026-09-30 — FROZEN 2026-09-30T19:40:45Z before GPU compute

Owner go 2026-09-30 ("除了imagenet的先不提交，后续都可以提交").  Purpose: complete the compute curve at 16×, where only VCS has a cell
(`P43_vcs_a5_views4_1600ep_seed0`: 87.50 linear / 85.84 kNN, one seed).  Controls at 1×/2×/4×/8× exist (P41, 3 seeds; 8× SimCLR 88.32 ± 0.30, VICReg 87.11).

**Cells.**  Tuned SimCLR and tuned VICReg, the P41 4-view 800-epoch recipes extended to 1600 epochs exactly as the VCS 16× run was extended from its recipe
(epochs, kNN epochs {0, 50, 100, 200, 400, 800, 1200, 1600}, checkpoints {200, 400, 800, 1200, 1600}; nothing else changed).  Seed 0 only, matching the VCS cell.
Configs by `configs/make_p97_configs.py --write`: `cifar10_hpW_simclr_views4_1600ep_seed0.yaml` (69de6ab5…), `cifar10_hpW_vicreg_views4_1600ep_seed0.yaml` (2216f6d9…).

**Evaluation.**  Selection split, frozen-h linear probe + kNN at epoch 1600, the recipe's probe.  Official test closed.

**Reading (pre-stated, descriptive, one seed).**  Report each method's 8× → 16× gain next to VCS's (87.01 → 87.50, +0.49; VCS seed 0 alone 86.42 → 87.50).
"The VCS–control gap narrows at 16×" if the 16× SimCLR − VCS difference is at least 1.0 smaller than the 8× 3-seed difference (1.31); "widens" if at least 1.0
larger; otherwise "unchanged".  Same for VICReg.  One seed per cell: no claim beyond the stated direction.

**Cost.**  ≈ 20 s / epoch on RTX6000PRO → ≈ 9 h per run; ≈ 15 h on H100.  Two runs, normal QOS, RTX6000PRO,H100, each as a 2-link chain (resume from last.pt) for safety.
