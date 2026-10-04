# Pre-registration — P135: optimizer learning-rate check for VCS (A-P3) and matched JS-AP3, CIFAR-10 and CIFAR-100, seed 0 — FROZEN 2026-10-04T20:56:15Z

Owner 2026-10-04: our method family (VCS and the matched-JS control) is new and gets tuning budget; SimCLR keeps its recipe.  The AdamW learning rate
(1e-3, linear warm-up 10 epochs + cosine) was never searched for either loss; P114 / P120 also pre-require an "lr / optimiser-budget sensitivity check"
before any general JS-vs-VCS statement.  Cells: lr ∈ {5e-4, 2e-3} × {VCS A-P3, JS-AP3} × {CIFAR-10, CIFAR-100}, seed 0 = 8 units; lr 1e-3 = the
existing seed-0 runs (A-P3 89.06 / 60.20; JS-AP3 88.72 / 58.76).  Each config differs from its parent only in optimizer.lr (verified by diff);
(a, κ) = (2, 0.5).  `configs/make_p135_configs.py`, `configs/P135_SHA256.json`, `slurm/p135_lines.txt` (node51 + node60 excluded).
Submission: when the queue drops below 15 jobs (owner rule), which also keeps it under the 30-job cap.

## Pre-stated reading
- Per loss × dataset: linear (primary) and kNN at the three learning rates; selection only on development validation labels; official test closed.
- **Selected lr** = highest linear; it replaces 1e-3 only if it beats it by ≥ 0.30 (CIFAR-10) / ≥ 0.50 (CIFAR-100); a replacing lr gets seeds 1–2.
- **JS-vs-VCS sensitivity:** the seed-0 VCS − JS gap at each lr (C10 at 1e-3: +0.34; C100: +1.44) is reported side by side; the gap "holds across
  lr" if it keeps its sign at all three learning rates, otherwise "lr-dependent" — this is the check P114 / P120 require (descriptive at one seed;
  the 3- / 5-seed labels stay those of P114 / P120 addendum 1).
- The lr check is at (a, κ) = (2, 0.5); if P127 / P129 select another (a, κ) per dataset, the lr of that cell is not re-searched in this unit.
- No early stopping on any monitor.
