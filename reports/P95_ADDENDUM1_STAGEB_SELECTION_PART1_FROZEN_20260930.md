# P95 addendum 1 (part 1) — stage-A selection for the families whose cells are complete (FROZEN 2026-09-30T10:31:51Z, before any stage-B job)

Rule (P95 frozen prereg): per family the highest selection-split linear top-1 at epoch 800 among its stage-A cells (seed 0); ties within 0.1 → the grid value
closer to the recipe (smaller λ, τ; larger R); a family whose best seed-0 run is below 86.42 − 1.0 = 85.42 is reported and not seeded; the matched-JS control
is always seeded.  The rule is per family, so families are selected as soon as all their cells are complete; the refresh family (R 50 / R 200, both still
training) and the JS seed-0 cell are not read here.  Stage-A values are read only for the selection; the verdicts wait for stage B.

| family | cells (seed 0, linear / kNN at epoch 800) | selected | seeded? |
|---|---|---|---|
| residual | λ 0.25: 86.02 / 80.90; λ 0.5: 85.74 / 83.54 | λ 0.25 | yes (86.02 ≥ 85.42) |
| dictionary | single cell: 86.56 / 83.20 | dictionary | yes |
| noise | τ 0.1: 86.52 / 85.42; τ 0.3: 87.32 / 84.92 | τ 0.3 | yes |
| refresh | R 50, R 200: training | — (part 2) | — |
| JS control | seed 0 training | always seeded | yes |

Stage-B units (seeds 1–2; `configs/make_p95_configs.py --stageB residual=0.25 dictionary noise=0.3 js --write`, unit file `slurm/p95_units_stageB.txt`):
- `configs/cifar10_hpV_residual_lam0.25_views4_800ep_seed1.yaml` 47d08bbddb4fb25a
- `configs/cifar10_hpV_residual_lam0.25_views4_800ep_seed2.yaml` c35d60d4ae2933bd
- `configs/cifar10_hpV_dictionary_views4_800ep_seed1.yaml` 3151d903862b9c46
- `configs/cifar10_hpV_dictionary_views4_800ep_seed2.yaml` 6f9afcd9d3eba83c
- `configs/cifar10_hpV_noise_tau0.3_views4_800ep_seed1.yaml` 6c4dbbae8780b39f
- `configs/cifar10_hpV_noise_tau0.3_views4_800ep_seed2.yaml` b4fefb10b0eae52f
- `configs/cifar10_hpV_js_matched_views4_800ep_seed1.yaml` fb21808e5e789316
- `configs/cifar10_hpV_js_matched_views4_800ep_seed2.yaml` 687b236f687daa95

Launch on the normal QOS, RTX6000PRO,H100 only (owner 2026-09-30: no further runfill submissions).  Recipe reference for the stage-B reading: 87.01 ± 0.53
(P35 seeds 0–2: 86.42 / 87.16 / 87.44).
