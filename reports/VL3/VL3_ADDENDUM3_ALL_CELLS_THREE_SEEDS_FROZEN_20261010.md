# VL3 addendum 3 — three seeds in every cell (removes the selection made by the seed rule) — FROZEN 2026-10-10

Why: VL3 and VL3 add. 1 added seeds 1–2 only where VCS was within 0.5 point of the best objective at seed 0.  That rule spends compute where the
comparison is close, but it also selects cells: the five cells left at one seed are exactly those where VCS trailed at seed 0 (−0.60 to −1.00).  A
table whose cells have different seed counts, chosen on the outcome, cannot support a pooled statement.  The owner asked for complete full-scale
experiments (2026-10-09 / 10).  This unit completes every cell at three seeds with the same recipe; nothing else changes.

## Units
Seeds 1 and 2 × 3 objectives (VCS, matched JS, candidate softmax) for the five one-seed cells = 30 full fine-tuning runs, recipe and selection as
`VL3_FULL_FINETUNE_FROZEN_20261010.md` (10 epochs, CAL-Top-1 selection; Large backbones with `--grad-ckpt` as add. 1):
RefCOCO × CLIP B/16; RefCOCO+ × SigLIP 2 B/16; RefCOCO × SigLIP 2 L/16; RefCOCO+ × CLIP L/14@336; RefCOCO+ × SigLIP 2 L/16.  Gate 2 as VL3.

## Readings (fixed now)
1. The full table: 3 datasets × 4 backbones, every objective at n = 3; per cell paired VCS − JS and VCS − softmax (95 % t over seeds).
2. **Pooled over the 12 cells** (cells as units; each cell's paired three-seed mean): mean VCS − softmax and VCS − JS with a 95 % t interval over
   the 12 cells, and the count of per-cell intervals that exclude 0 in each direction.  Descriptive split by scale (B/16 vs Large, 6 cells each).
3. Pooled J_recal ratio (VCS / softmax and JS / softmax), same unit.
4. The seed-rule readings already reported stay as they are; this unit adds the complete table and the pooled estimate.  No further seeds or
   recipe changes follow from it without a new pre-registration.
5. Wording fixed in advance: if the pooled VCS − softmax interval contains 0 → "VCS matches the task loss after full fine-tuning"; if it lies
   below 0 → "softmax leads by x"; if above 0 → "VCS leads by x".  VCS − JS is read as the same-posterior check.
