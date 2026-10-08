# P138 addendum 2 — seeds 3–4 for the three K = 16 cells without shown non-inferiority — FROZEN 2026-10-08T11:04:46Z

Owner 2026-10-08 (selected "More P138 seeds").  Addendum 1 (`P138_ADDENDUM1_SEEDS_REPORT_20261008.md`): non-inferiority shown for VCS C10 only; not
shown for JS C10 (−0.17 [−1.02, +0.69]), VCS C100 (+0.39 [−0.92, +1.70]), JS C100 (−0.17 [−1.46, +1.13]) at 3 seeds.
Units (8): K16 seeds 3–4 for js_c10, vcs_c100, js_c100 (`configs/{cifar10,cifar100}_hpPK16_*_seed{3,4}.yaml`) and the missing K = all parent
JS-AP3 C10 seeds 3–4 (`configs/cifar10_hpJS_AP3_views4_800ep_seed{3,4}.yaml`, run ids P114_JSAP3_views4_800ep_seed{3,4}); each = the seed-0 config
with ONLY run.seed / run.stage changed (verified by diff; policy load passes; `configs/make_p138a2_configs.py`, `configs/P138A2_SHA256.json`).
Existing parents seeds 3–4: P120A1 A-P3 C100, P120 JS-AP3 C100.
Reading: per cell, K16 − parent paired by seed over seeds 0–4 (n = 5), 95 % t interval; non-inferior if the lower bound > −0.30 (CIFAR-10) / −0.50
(CIFAR-100), else "non-inferiority not shown".  The 3-seed reading of addendum 1 stays reported; no further seeds after this addendum.
