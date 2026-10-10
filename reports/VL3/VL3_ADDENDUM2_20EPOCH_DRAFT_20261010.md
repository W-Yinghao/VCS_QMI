# VL3 addendum 2 — schedule sensitivity: 20 epochs for every objective (B/16, three datasets) — DRAFT 2026-10-10

Why: in VL3 (B/16) the VCS / JS runs on CLIP often selected late epochs (6–10 of 10), so the 10-epoch schedule may cap the estimator objectives.
Changing the schedule for ours only would be per-objective tuning against an untuned baseline.  So the same change is applied to all three
objectives (shared recipe, epochs 10 → 20; warm-up and cosine stretch with the step count).  Everything else as `VL3_FULL_FINETUNE_FROZEN_20261010.md`.

## Units
Seed 0 for 3 datasets × 2 B/16 backbones × 3 objectives = 18 full runs (`--epochs 20`, output tag suffix `_e20`); gate 2 as VL3.

## Readings (descriptive; one seed)
1. Per objective and cell: DEV Top-1 (CAL-selected) at 20 vs 10 epochs (seed 0 of VL3), and the selected epoch.
2. Whether VCS − softmax or VCS − JS at seed 0 changes sign or by more than 0.5 point in any cell, versus VL3 seed 0.
3. If the 20-epoch schedule raises every objective by a similar amount, the VL3 B/16 reading stands as stated.  If it raises one objective more,
   that is reported as schedule-dependence of the comparison, and any follow-up seeds need a new pre-registration.
Implementation note: the trainer needs an output-tag suffix for this unit (to be added before the freeze, with a smoke).
