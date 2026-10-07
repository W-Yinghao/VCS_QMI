# P145 addendum 1 — CIFAR-100 seeds 1–2 for VCS / JS / SimCLR (STRESS ε = 0.10) — FROZEN 2026-10-07

Trigger (P145 frozen rule, plan §6.3): at seed 0 on CIFAR-100, Δ_VCS = +0.10 (60.30 vs clean 60.20) and Δ_JS = +1.18 (59.94 vs 58.76), so
|Δ_VCS − Δ_JS| = 1.08 ≥ 1.00 (either direction) → seeds 1–2 for **all three methods** on CIFAR-100 (the SimCLR seed-0 cell is still running and does
not change the trigger).  CIFAR-10 did not trigger (Δ +0.02 / −0.28 / −0.08; largest gap 0.30 < 0.50).
Units: `configs/cifar100_hpST10_{vcs,js,simclr}_c100_seed{1,2}.yaml` = the seed-0 STRESS configs with only run.seed / run.stage changed
(`configs/make_p145a1_configs.py`, `configs/P145A1_SHA256.json`, `slurm/p145a1_lines.txt`).  Clean baselines seeds 1–2 exist (P107 A-P3, P120
JS-AP3, P91 SimCLR; their configs differ from seed 0 only in run.seed).
Reading: per method, Δ_m = A_m(ε) − A_m(0) paired by seed (0–2); the contamination interaction Δ_VCS − Δ_m paired by seed with a 95 % t interval
and the P114 labels (close |mean| < 0.3; clear |mean| ≥ 0.3 and the interval excludes 0; else inconclusive).  All single-seed values stay reported;
a smaller drop from a worse clean start is not called stronger.  No ε = 0.20 condition without separate budget confirmation.
