# Pre-registration — pre-check C: batch decoupling (P53), frozen 2026-09-27 before launch

**Property tested.** J is a sum of two independently averaged expectations, and VCS negatives are product-of-marginals samples that need not
come from the batch (fixed-critic variance 1/n + 1/m; K-insensitivity observed in SSL).  Question (brief §2, C): when the batch shrinks, does
VCS degrade less than InfoNCE and pairwise logistic, whose negatives are the in-batch pairs?

**Setting.** Pre-check A's pipeline in setting 2 (topic pairing — mid-dependence by construction; frozen CLIP ViT-B/32 features, identity-initialised
512×512 adapters, COCO source → animal target, splits and hashes of P49).  VCS uses K = 8 independent-pool negatives at every batch size; InfoNCE
and logistic use their native in-batch pairs (at B = 16 that is 15 negatives per anchor — this *is* the property under test, not an unfairness;
a cross-batch/memory-bank variant for the competitors is a follow-up, listed below).  Batch sizes B ∈ {16, 32, 64, 128, 256, 512}.
**Two axes, reported separately:** fixed updates (3 120 optimizer steps at every B = 40 epochs at B 256) and fixed exposure (40 epochs at every B).
**Equal tuning:** at every (method, B, axis) the learning rate is selected on SRC-CAL from {1e-3, 3e-4, 1e-4} by the method's own loss
(selection losses tabled); 3 seeds of the selected lr; AdamW wd 1e-4, cosine schedule.
**Metrics per point:** primary non-ranking — balanced error at the SRC-CAL quantile threshold (FNR 5 %) on SRC-EVAL and TGT-EVAL; secondary —
retrieval R@1; ECE of score + Platt; held-out J (VCS; must stay < 0.9 for the mid-dependence rule).  Retention(B) = value(B) / value(256) for
accuracy (1 − balanced error) and for R@1.

**Pre-committed reading.**
- Property holds: on both axes, VCS retention of accuracy at B = 16 exceeds each competitor's by ≥ 0.05 (and the same for R@1 on at least one
  axis), with the competitors given their selected lr.  Holds conditionally: only on one axis, or only for R@1, or only for B ≤ 32 on one axis.
  Does not hold: the three retention curves are within 0.05 of each other at B = 16, or VCS degrades more.
- If VCS's own B = 256 value is far below the competitors' (as in setting 1: R@1 0.27 vs 0.36), retention is still the pre-registered
  quantity, but absolute values are shown next to it so a "flat curve at a low level" is not read as robustness.
- Not claimed: anything about K > 8 or memory banks for the competitors (follow-up: cross-batch negatives for InfoNCE, permitted by the frozen
  list); anything about the SSL setting.

Compute: one GPU job (6 batches × 3 methods × 3 lr × 2 axes grid + 3 seeds each; cached features, minutes per point).  Probe: --smoke first.
Outputs: `reports/P54_precheck_C.md/.json`.  Code: `scripts/precheck_c_batch.py`, `slurm/precheck_c.sbatch`.
