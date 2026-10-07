# P130 addendum 1 — ResNet-50 CIFAR-100 seeds 1–2 for A-P3 / JS-AP3 / SimCLR — FROZEN 2026-10-07

Trigger (seed rule fixed at the P130 freeze): |A-P3 − SimCLR| = 64.66 − 58.54 = 6.12 ≥ 0.5 linear at ResNet-50 seed 0 → all three methods get
seeds 1–2.  Units: `configs/cifar100_hpR50_{AP3,JS_AP3,simclr}_views4_800ep_seed{1,2}.yaml` = the seed-0 configs with only run.seed / run.stage
changed (`configs/make_p130a1_configs.py`, `configs/P130A1_SHA256.json`, `slurm/p130a1_lines.txt`); each a 2-link chain (afterany).
Reading: A-P3 − SimCLR and A-P3 − JS-AP3, paired by seed (0–2), 95 % t interval, P114 labels (close / clear / inconclusive); kNN alongside;
per-method ResNet-50 − ResNet-18 descriptive (ResNet-18 seeds 0–2 exist for all three).

Submission note: the normal-QOS cap (30) was reached after the A-P3 chains (jobs 1027191-94); the JS / SimCLR chains are fed by `slurm/feed_normal.sh slurm/p130a1_feed.txt` as slots free (link 2 = same job name with `--dependency=singleton`, i.e. after link 1 ends).
