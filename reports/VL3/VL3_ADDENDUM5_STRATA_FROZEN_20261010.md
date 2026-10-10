# VL3 addendum 5 — per-query strata of the complete 12-cell table (re-evaluation of the saved checkpoints) — FROZEN 2026-10-10

Why: the complete VL3 table (`VL3_COMPLETE_12CELL_REPORT_20261010.md`) shows a descriptive dataset pattern.  Mean VCS − softmax is +0.60 on
RefCOCOg (long, relational expressions), −0.11 on RefCOCO and −0.47 on RefCOCO+ (short, appearance-only).  The report flags it as a hypothesis.
This unit tests the most direct version of the hypothesis *within* datasets, on the existing 108 checkpoints.  It is evaluation only, with no new
training: the same models and the same DEV split, read per query.

## Procedure (`scripts/vl3_add5_strata.py`, `slurm/vl3_add5_strata.sbatch`; 12 jobs = dataset × backbone, 9 checkpoints each)
Reload each CAL-Top-1 checkpoint (fp16), recompute DEV with the VL3 evaluation code (given boxes), and store per-query hits.
**Gate S:** image-macro DEV Top-1 within 0.3 point of the run's recorded value; a failing checkpoint is excluded and listed.  The RefCOCOg ×
CLIP B/16 job runs first.  The other 11 are fed only if it exits 0 with every gate S passing.

## Strata (fixed now, before any per-query result; `scripts/vl3_add5_aggregate.py`)
- **S1 expression length** (words, whitespace split): within-dataset tertiles of the DEV expressions (short ≤ 1/3 quantile, mid, long > 2/3 quantile).
- **S2 location word present** (fixed list in the script: left, right, top, bottom, middle, center/centre, front, back, behind, near/nearest, far/
  farthest, closest, first, second, third, upper, lower, corner, side, leftmost, rightmost; word boundary, lower case).  RefCOCO and RefCOCOg only.
- **S3 number of referred objects in the image:** 2, 3, ≥ 4.

## Readings
1. **Primary:** S1 long − short of (VCS − softmax), per cell the mean over the 3 seeds (paired), pooled over the 12 cells (95 % t).  Wording fixed in
   advance.
   - Interval above 0: "within datasets, VCS's relative accuracy rises with expression length".  This supports the hypothesis.
   - Interval contains 0: "expression length does not explain the dataset pattern".
   - Interval below 0: the reverse.
2. **Secondary (descriptive with intervals):** S2 with − without location words (8 cells); S3 ≥ 4 − 2 referred objects (12 cells); the same three
   contrasts for VCS − JS; per-stratum VCS − softmax pooled over cells.
3. No new training, seeds or recipe changes follow from this unit without a new pre-registration.
