# VL3 addendum 2 — schedule sensitivity (20 vs 10 epochs, all objectives, B/16, three datasets, seed 0) — report — 2026-10-10

Protocol `VL3_ADDENDUM2_20EPOCH_FROZEN_20261010.md`; results-only commit `5338ee5` (`reports/VL3/VL3_add2_results.json`,
`scripts/vl3_add2_aggregate.py`, job 1033167).  Same recipe as VL3 except epochs 10 → 20 for every objective.  **Gate 2: 18 / 18 runs pass.**
Procedural deviation: the frozen text names a waiter script for feeding after the smoke.  The 18 runs were instead submitted with
`afterok` on the smoke job (still queued at submission, so the dependency was live) and the waiter was stopped.  The gating condition is the
same.  The smoke wrote `refcocog_clip_b16_vcs_s0_e20_smoke.json` and exited 0.

## 1. DEV Top-1 at the CAL-Top-1-selected epoch (seed 0): 20 epochs vs 10 epochs (selected epoch in brackets)
| cell | VCS | matched JS | candidate softmax | VCS − softmax (20 / 10) | VCS − JS (20 / 10) |
|---|---|---|---|---|---|
| RefCOCOg × CLIP | 80.67 (15) vs 81.14 (6) | 80.39 (9) vs 80.64 (10) | 80.36 (4) vs 80.05 (6) | +0.31 / +1.09 | +0.28 / +0.49 |
| RefCOCOg × SigLIP 2 | 85.11 (13) vs 84.80 (5) | 85.31 (15) vs 84.65 (8) | 84.82 (2) vs 85.30 (2) | +0.28 / −0.50 | −0.21 / +0.15 |
| RefCOCO × CLIP | 85.21 (14) vs 84.66 (6) | 84.15 (10) vs 84.36 (5) | 84.41 (16) vs 85.26 (8) | +0.80 / −0.60 | +1.06 / +0.30 |
| RefCOCO × SigLIP 2 | 88.49 (12) vs 88.28 (5) | 88.28 (14) vs 88.39 (8) | 88.03 (8) vs 87.76 (3) | +0.47 / +0.52 | +0.21 / −0.10 |
| RefCOCO+ × CLIP | 82.03 (13) vs 82.06 (8) | 81.80 (13) vs 81.55 (7) | 81.90 (3) vs 82.01 (3) | +0.13 / +0.05 | +0.23 / +0.51 |
| RefCOCO+ × SigLIP 2 | 87.03 (14) vs 86.69 (5) | 87.14 (12) vs 86.82 (8) | 87.26 (4) vs 87.32 (2) | −0.23 / −0.63 | −0.11 / −0.14 |
| **mean over the 6 cells** | **+0.15** (20 − 10) | **+0.11** | **−0.15** | **+0.29 / −0.01** | **+0.24 / +0.20** |

## 2. Frozen readings
1. **Per objective, 20 − 10 epochs:** VCS −0.47 to +0.55 (mean +0.15), JS −0.26 to +0.66 (mean +0.11), softmax −0.85 to +0.31 (mean −0.15).
   No objective gains consistently from the longer schedule.  Every change is under 1 point.
2. **Sign / 0.5-point flags (reading 2):** VCS − softmax changes sign or moves by more than 0.5 point in 3 of 6 cells.  These are RefCOCOg ×
   CLIP (+1.09 → +0.31), RefCOCOg × SigLIP 2 (−0.50 → +0.28) and RefCOCO × CLIP (−0.60 → +0.80).  VCS − JS is flagged in 3 cells at small magnitudes.
   For scale: in VL3 the seed-to-seed sd of the paired VCS − softmax difference is about 0.5–0.8 point (from the three-seed intervals).  So single-seed
   shifts of this size are within seed noise and are not read cell by cell.
3. **Overall (reading 3):** the longer schedule moves the comparison slightly toward the estimator objectives.  Mean VCS − softmax goes from −0.01
   to +0.29, VCS is ahead in 5 of 6 cells at 20 epochs, and the shift is about 0.3 point on average.  This is reported as a **small, one-seed
   schedule-dependence in the direction of the estimator objectives**.  It is smaller than the per-cell seed noise.  The VL3 B/16 reading ("after full
   fine-tuning VCS matches the task-matched softmax") stands.  The data do not support "VCS beats softmax" at either schedule.  VCS − JS is unchanged
   (+0.20 → +0.24).
4. **Selected epochs:** at 20 epochs VCS / JS select late epochs (VCS 12–15, JS 9–15), while softmax mostly peaks early (median epoch 4; 2–16).  The
   estimator objectives keep improving on CAL after softmax has stopped.  This matches the VL3 observation that motivated the unit.  It is
   also why the 10-epoch schedule slightly favours softmax.

## 3. Caveats and what follows
- One seed per cell; descriptive by design.  Any three-seed follow-up at 20 epochs is a new pre-registered unit (as fixed in the protocol).
  The primary VL3 schedule stays 10 epochs.
- The schedule was changed for all objectives together (shared recipe); this is not per-objective tuning.
