# P129 addendum 2 — matched JS, CIFAR-100, (a, κ) = (2, 0.25): seeds 1–2 — FROZEN 2026-10-05T08:42:54Z

**Trigger (P129 frozen rule, submitted early):** JS (2, 0.25) seed 0 = 59.58 / 56.42 vs JS (2, 0.5) seed 0 58.76 / 54.80 → +0.82 ≥ 0.50.
Grid status at the freeze: CIFAR-100 cells (1.5, .5) 58.86 and (2, .25) 59.58 done; **(2, .75), (3, .5), (3, .25) still running**.
Pre-stated rule for the pending cells (as addendum 1): a pending cell that ends higher than 59.58 and ≥ 58.76 + 0.50 becomes the selected cell and gets
seeds 1–2 in a further addendum; the (2, 0.25) seeds are then kept as a non-selected extra.  Otherwise (2, 0.25) is the selected CIFAR-100 JS cell.
Units: `configs/cifar100_hpG6_JS_c100_a2_k0.25_seed{1,2}.yaml` (seed-0 cell config, only run.seed / run.stage changed;
`configs/P129_ADDENDUM2_SHA256.json`, `slurm/p129a2_lines.txt`), fed by `slurm/feed_normal.sh`.
Reading (from P129): tuned-VCS (P127 selected (2, 0.5); seeds 0–4 on CIFAR-100 from P107 + P120 addendum 1) vs tuned-JS (selected cell, 3 seeds),
paired by seed, P114 labels; a clear difference still needs the P135 lr check before a general statement.  Descriptive at the freeze: JS (2, 0.25)
seed 0 kNN 56.42 is above VCS A-P3 seed 0 (55.92), linear 59.58 below it (60.20).
