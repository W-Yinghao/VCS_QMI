# VL3 addendum 6 — cross-dataset transfer of the fine-tuned encoders (evaluation of the saved checkpoints) — FROZEN 2026-10-10

Why: in the complete VL3 table, VCS and softmax tie on in-domain ranking, but VCS / JS encoders expose 33 % more critic-fittable dependence
(J_recal).  If that extra dependence is general rather than dataset-specific, the estimator-trained encoders should transfer better to another
referring dataset.  This unit tests it on the existing 108 checkpoints.  It is evaluation only, with no new training.

## Procedure (`scripts/vl3_add6_transfer.py`, `slurm/vl3_add6_transfer.sbatch`; 12 jobs = source dataset × backbone, 9 checkpoints each)
- Each source checkpoint (CAL-Top-1, fp16 reloaded) is evaluated on the DEV split of each of the other two datasets, with given boxes and the VL3
  evaluation code (argmax cosine = the critic's ranking).
- **Image exclusion:** target DEV images that are in the source's FIT or CAL images are dropped.  The source model never saw a kept image in
  training or selection.  RefCOCO and RefCOCO+ share images and the same seeded partition, so their transfer keeps every DEV image.  RefCOCOg
  (UMD) ↔ UNC drops the overlap.
- Zero-shot backbone on the same kept images; in-domain models (trained on the target) on the same kept images from the add. 5 per-query hits.
- **Gate X:** zero-shot on the full target DEV within 0.3 point of the frozen raw value of that target and backbone.  The RefCOCOg → others ×
  CLIP B/16 job runs first.  The other 11 are fed only if it exits 0 with gate X passing.

## Readings (fixed now; `scripts/vl3_add6_aggregate.py`)
1. **Primary:** VCS − softmax transfer Top-1 (paired by seed), pooled over the 24 directed source → target × backbone cells (95 % t).  Wording
   fixed in advance.
   - Interval above 0: "VCS-trained encoders transfer better across referring datasets by x".
   - Interval contains 0: "transfer does not differ between the estimator objective and the task loss".
   - Interval below 0: "softmax-trained encoders transfer better by x".
2. **Secondary:** VCS − JS (same-posterior check); the transfer gap (in-domain − transfer on the same images) contrasted between objectives; the
   split UNC pair (8 cells) vs RefCOCOg ↔ UNC (16 cells); zero-shot → transfer gains.
3. Cells are not independent (shared checkpoints across targets), so the pooled interval is a descriptive summary.  No new training follows from
   this unit without a new pre-registration.
