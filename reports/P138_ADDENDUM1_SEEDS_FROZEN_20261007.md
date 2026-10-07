# P138 addendum 1 — seeds 1–2 of the four K = 16 cells — FROZEN 2026-10-07

Trigger (P138 frozen rule, report `P138_V7_PAIR_REPORT_20261007.md`): development retention met in all four cells (Δ linear +0.16 / +0.08 / −0.20 /
+0.38 against margins 0.30 / 0.50, and 15.3× fewer computed pair scores).  Units: `configs/{cifar10,cifar100}_hpPK16_{vcs,js}_{c10,c100}_seed{1,2}.yaml`
= the seed-0 configs with only run.seed / run.stage changed (`configs/make_p138a1_configs.py`, `configs/P138A1_SHA256.json`, `slurm/p138a1_lines.txt`).
Parents' seeds 1–2: P107 A-P3 C10 / C100, P114 JS-AP3 C10, P120 JS-AP3 C100 (all exist).
Reading: per cell, paired by seed (0–2), K = 16 − K = all, 95 % t interval; **non-inferior** if the lower bound > −0.30 (CIFAR-10) / −0.50
(CIFAR-100); otherwise "non-inferiority not shown" (no "equivalent" from a non-significant difference).  kNN alongside.
