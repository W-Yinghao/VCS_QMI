# P129 addendum 3 — matched JS, CIFAR-100, (a, κ) = (3, 0.5): seeds 1–2 — FROZEN 2026-10-05T10:44:01Z

**Trigger (rule pre-stated in addendum 2):** JS (3, 0.5) seed 0 = 60.72 / 56.18 > 59.58 (the (2, 0.25) cell) and ≥ 58.76 + 0.50 → **(3, 0.5) becomes the
selected CIFAR-100 JS cell**; the (2, 0.25) seeds 1–2 (addendum 2, jobs 1021976 / 1021977) are kept as a non-selected extra (no selection from them).
Grid status: CIFAR-100 (1.5, .5) 58.86, (2, .25) 59.58, (2, .75) 58.24, (3, .5) 60.72 done; **(3, .25) still running** — same rule: if it ends higher
than 60.72 (and ≥ 59.26) it becomes the selected cell (further addendum) and these seeds become an extra.
Units: `configs/cifar100_hpG6_JS_c100_a3_k0.5_seed{1,2}.yaml` (seed-0 cell config, only run.seed / run.stage changed;
`configs/P129_ADDENDUM3_SHA256.json`, `slurm/p129a3_lines.txt`).  Reading unchanged (tuned-VCS (2, 0.5) seeds 0–4 vs tuned-JS selected cell, 3 seeds,
P114 labels; P135 lr check before a general statement).  Descriptive at the freeze: JS (3, 0.5) seed 0 is above VCS A-P3 (2, 0.5) seed 0 on both
readouts (60.72 / 56.18 vs 60.20 / 55.92) and above the best VCS grid cell (3, 0.5) 60.64 / 55.70.
