# VL1-14 — does the VL critic need an SSL-style search? (critic screen) — DRAFT 2026-10-09

Owner 2026-10-09: "我现在不确定，在VL task中是不是我们的critic函数也需要像ssl上搜索，实验判定".  Order agreed: VL baselines → lessons summary
→ this experiment.  Draft only; freeze after the lessons summary.

## Why (evidence so far)
- Critic form matters: package concat scorer vs residual F2r on CLIP, N all: Top-1 65.89 vs 68.80, J 5.54 vs 7.20 (VL1-10).
- Function class matters for J: MLP vs RFF on every feature set (e.g. CLIP 7.2 vs 5.2, SigLIP 2 11.6 vs 9.3); RFF always picked its grid corner.
- **Optimisation is at the edge of the frozen grid:** every estimator-selected VCS / JS fit chose lr 2e-3, the largest value (CLIP, FG-CLIP 2, SigLIP 2,
  FG-CLIP v1, ReCLIP features).  The 3 000-update cap binds on CLIP / FG-CLIP 2 / FG-CLIP v1 (CAL-J optimum in the last 250 updates).
- The objective is not the issue: VCS ≈ matched JS on every feature set.

## Question and decision rule (to freeze)
**Q:** does a bounded search over the critic and its optimisation raise the estimator's fitted common J on CAL beyond the frozen F2r recipe?
**Decision:** "VL needs a critic search" if, on either feature set, the best screened cell beats the frozen recipe (C0) on CAL common J by ≥ 0.50
(×100) with the paired-by-seed 95 % t interval excluding 0.  Then the winner (largest CAL J; ties within 0.10 → fewer parameters) is adopted, and
Table A is re-run on all feature sets for VCS and matched JS (JS as the same-posterior control).  Otherwise: "no critic search needed"; F2r stays,
and the VL tables stand.
Selection is by CAL J only, the estimator's own objective, at every level (lr within a cell, cell across cells).  DEV Top-1 is read once, for every
cell's CAL-selected checkpoint, as a secondary column; it never selects.

## Design
Features: CLIP ViT-B/16 crops (largest learning headroom) and SigLIP 2 Base crops (strongest).  Objective: VCS.  N all (10 674 FIT images).
Init seeds 0–2.  Roles / law / evaluation as `VL1_FROZEN_PROTOCOL.md`.
| cell | critic | lr grid | update cap | other |
|---|---|---|---|---|
| C0 | frozen F2r (residual MLP 3d→256→256→1) | 1e-4, 5e-4, 2e-3 | 3 000 | existing VL1-10 / VL1-12 add. 2 fits (no re-run) |
| C1 | F2r | 2e-3, 5e-3, 1e-2 | 10 000 | optimisation only (lr past the edge + budget) |
| C2 | residual MLP 3d→1024→1024→1024→1 | 2e-3, 5e-3, 1e-2 | 10 000 | capacity |
| C3 | residual low-rank bilinear a·cos + b + ⟨uP, vQ⟩, rank 64 | 2e-3, 5e-3, 1e-2 | 10 000 | different function class |
| C4 | affine a·cos + b | 2e-3, 5e-3, 1e-2 | 10 000 | calibration-only lower bound (ranking = raw) |
Implementation: `scripts/vl1_10_fit.py` options `--scorer {residual, bilinear, affine} --hidden --depth --rank --max-updates --lrs --tag-suffix`
(tests 33 / 33, gate 1031391; defaults reproduce the frozen scorer).  Jobs: 8 CPU jobs (4 cells × 2 feature sets), outputs `outputs/VL1_14`.

## Readings (to freeze)
1. Per feature set and cell: CAL J (mean ± sd), paired difference vs C0 with 95 % t interval; the decision rule above.
2. Secondary: DEV image-macro Top-1 of each cell's CAL-selected checkpoint, paired vs C0.  Also whether J and Top-1 move together across cells.
3. Optimisation diagnostics: selected lr and the update of the selected checkpoint per cell (does the cap still bind at 10 000?).
Not claimed: anything about the task loss (softmax is not screened), or official val / test.
