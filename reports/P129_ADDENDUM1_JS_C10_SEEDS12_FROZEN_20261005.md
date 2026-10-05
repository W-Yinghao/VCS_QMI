# P129 addendum 1 — matched JS, CIFAR-10, (a, κ) = (2, 0.25): seeds 1–2 — FROZEN 2026-10-05T05:25:47Z

**Trigger (P129 frozen rule, submitted early per the owner's "submit early" instruction):** JS (2, 0.25) seed 0 = 89.26 / 87.68 vs JS (2, 0.5) seed 0
88.72 / 87.50 → +0.54 ≥ 0.30.  Grid status at the freeze: CIFAR-10 cells (1.5, .5) 88.50, (2, .75) 88.52, (3, .25) 88.28, (2, .25) 89.26 done;
**(3, .5) still running** — the dataset-selected cell is only final after it.
Pre-stated rule for the pending cell: if JS (3, 0.5) seed 0 ends **higher** than 89.26 and ≥ 88.72 + 0.30, it becomes the selected cell and gets
seeds 1–2 in a second addendum; the (2, 0.25) seeds 1–2 are then kept and reported as a non-selected extra (not cancelled, no selection from them).
Otherwise (2, 0.25) is the selected CIFAR-10 JS cell.

Units: `configs/cifar10_hpG6_JS_c10_a2_k0.25_seed{1,2}.yaml` = the seed-0 cell config with only run.seed (and run.stage) changed
(`configs/P129_ADDENDUM1_SHA256.json`, `slurm/p129a1_lines.txt`); fed by `slurm/feed_normal.sh` under the normal-QOS 30-job cap.
Reading (from P129): once the selected VCS cell (P127: (2, 0.5), seeds 0–4) and the selected JS cell have 3 seeds, tuned-VCS − tuned-JS paired by
seed with the P114 labels; a clear difference still needs the lr check (P135) before a general statement.  Descriptive note recorded at the freeze:
JS (2, 0.25) seed 0 (89.26) is above VCS A-P3 (2, 0.5) seed 0 (89.06); VCS at (2, 0.25) was 88.36 (P127).
