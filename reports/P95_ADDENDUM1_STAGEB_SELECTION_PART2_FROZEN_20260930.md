# P95 addendum 1 (part 2) — stage-A selection for the refresh family (FROZEN 2026-09-30T10:58:11Z, before its stage-B jobs)

Same rule as part 1.  Refresh cells (seed 0, selection-split linear / kNN at epoch 800): R 50: 84.96 / 82.14; R 200: 85.94 / 84.38.  Selected **R 200**
(85.94 ≥ 85.42 → seeded).  The JS seed-0 cell is not read here (the control is seeded regardless; its seeds 1–2 are already queued from part 1).

Stage-B units (`configs/make_p95_configs.py --stageB refresh=200 --write`; unit file `slurm/p95_units_stageB_part2.txt`):
- `configs/cifar10_hpV_refresh_R200_views4_800ep_seed1.yaml` 7d7c2378419a962b
- `configs/cifar10_hpV_refresh_R200_views4_800ep_seed2.yaml` b00f833ec693e2f7
