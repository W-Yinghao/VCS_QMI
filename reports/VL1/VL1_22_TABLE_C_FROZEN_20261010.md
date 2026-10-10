# VL1-22 — Table C, detection setting: clean detector proposals ranked by the VL3 fine-tuned encoders — FROZEN 2026-10-10

Why: Table A / VL3 rank the image's *given* referred boxes.  The standard REC setting has no given boxes, so a system must pick among
query-independent proposals.  VL1-21 trains a detector without any RefCOCO / RefCOCO+ / RefCOCOg CAL / DEV / val / test image.  This unit ranks
its proposals with the VL3 models.  Following the owner's 2026-10-10 rule (full experiments, no frozen-feature probes), the scorers are the
**fine-tuned VL3 encoders** (all 108 CAL-Top-1 checkpoints: 3 datasets × 4 backbones × 3 objectives × 3 seeds, 10-epoch primary schedule).  The
frozen F2r critics named in the VL1-21 text are not used.

## Procedure (`scripts/vl1_22_tableC.py`, `slurm/vl1_22_tableC.sbatch`, 12 jobs = dataset × backbone, 9 checkpoints each)
- **Proposals:** the VL1-21 detector's DEV proposals (`outputs/VL1_21/proposals/<dataset>_DEV.pt`): top-100 detections, score ≥ 0.05, per-class
  NMS, sorted by score; clipped to the image, boxes < 1 px dropped.
- **Queries and images:** Table A's / VL3's eligible DEV images (≥ 2 referred objects) and all their referring expressions (RefCOCOg 6 442 DEV
  expressions; RefCOCO / RefCOCO+ as in VL2 / VL3).
- **Scoring:** each proposal crop goes through the checkpoint's image tower (same preprocessing as VL3 training); the expression goes through its
  text tower; the top-1 proposal is the one with the highest cosine.  The VL3 critic is a·cos + b with a = softplus(α) > 0, so this equals the
  critic's ranking.
- **Metric:** Acc@IoU 0.5 of the top-1 proposal against the referred box, DEV image-macro (primary) and per query.  Ceiling: the fraction of
  expressions for which any proposal has IoU ≥ 0.5.

## Gates
- **Gate D (VL1-21):** COCO val2017 bbox AP ≥ 33.  If it fails, no Table C (report and stop).
- **Smoke:** `--smoke` (32 DEV images, RefCOCOg × CLIP B/16, all three objectives at seed 0) runs end to end and writes `_smoke` outputs.
- **Gate T1 (per checkpoint):** the reloaded fp16 checkpoint reproduces the run's given-box DEV image-macro Top-1 within 0.3 point.  A failing
  checkpoint gets no detection number and is listed.

## Readings (fixed now; `scripts/vl1_22_aggregate.py`)
1. Per cell and objective: Acc@0.5 (mean ± sd over 3 seeds); paired VCS − JS and VCS − softmax (95 % t).
2. **Pooled over the 12 cells** (cells as units, 95 % t), with the VL3 add. 3 wording rule.  If the VCS − softmax interval contains 0: "with
   detector proposals, VCS matches the task loss".  If below 0: "softmax leads by x".  If above: "VCS leads by x".
3. Given-box → detection drop per objective (paired by seed, pooled over cells).  This shows whether one objective transfers better to proposals,
   which add unreferred objects and background boxes as distractors.  Descriptive, no direction predicted.
4. Proposal ceiling per dataset (upper bound of any ranker on these proposals).
5. **Grounding DINO-T (zero-shot; no RefCOCO data) on the same DEV expressions**, from the existing VL1-20 outputs, as the clean external row.
   MDETR trained on the RefCOCO / + / g train expressions, which contain the DEV images, so it gets no DEV row.  Its official-val numbers stay in
   the Table B / C notes as a supervised reference and are not compared.
6. Table C is never merged with the given-box tables.  Official val / test stay closed.

## Budget
12 GPU jobs, under 1 h each (B/16) to a few hours (Large); fed after gate D and the smoke.  The 20-epoch checkpoints are not scored (10 epochs is
the primary VL3 schedule).
