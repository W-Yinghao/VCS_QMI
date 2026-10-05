# P129 addendum 4 — lr check at the selected JS CIFAR-100 cell (3, 0.5) — FROZEN 2026-10-05T23:21:19Z
Trigger: P129 frozen reading — the tuned-VCS vs tuned-JS CIFAR-100 difference is clear (−0.49 [−0.77, −0.20]), so "the pre-specified lr / optimiser-
budget check for both losses" is required before a general statement.  VCS's selected cell (2, 0.5) is covered by P135; this adds JS (3, 0.5) at
lr 5e-4 and 2e-3, seed 0 (`configs/cifar100_hpG6_JS_c100_a3_k0.5_lr{0.5x,2x}_seed0.yaml`, only optimizer.lr / run.stage changed;
`configs/P129_ADDENDUM4_SHA256.json`, `slurm/p129a4_lines.txt`).  Reading: the seed-0 VCS (2, 0.5) − JS (3, 0.5) gap at each lr (VCS lrs from P135):
"holds across lr" if it keeps its sign at all three lrs, else "lr-dependent"; the 3-seed label of addendum 3 stays as reported.
