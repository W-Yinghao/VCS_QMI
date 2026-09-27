# Pre-check C — batch decoupling (P53/P54), final

Pre-registration: `P53_PRECHECK_C_PREREG_FROZEN_20260927.md`.  Table: `P54_precheck_C.md/.json` (GPU job 1010743; smoke disclosed).  Setting 2 of
pre-check A (frozen CLIP ViT-B/32, identity-initialised 512×512 adapters, topic pairing, COCO source → animal target); lr selected per
(method, batch, axis) on SRC-CAL from {1e-3, 3e-4, 1e-4}; 3 seeds (spread ≤ 0.003 on the primary metric).  Primary metric: accuracy = 1 − balanced
error at the SRC-CAL quantile threshold (FNR 5 %).  Raw CLIP without adapters: accuracy 0.60 (source) — the adapters lift it to ≈ 0.85, so the
task has headroom.

| axis | method | accuracy at B = 16 / 32 / 64 / 128 / 256 / 512 (source) | retention at B = 16 | target accuracy at B = 16 (ret.) | held-out J (VCS, source) |
|---|---|---|---|---|---|
| fixed updates (3 120 steps) | VCS | 0.844 / 0.846 / 0.847 / 0.850 / 0.850 / 0.849 | 0.99 | 0.689 (1.01) | 0.56–0.58 |
| | InfoNCE | 0.852 / 0.855 / 0.855 / 0.856 / 0.857 / 0.856 | 0.99 | 0.672 (1.00) | — |
| | logistic | 0.820 / 0.849 / 0.852 / 0.855 / 0.856 / 0.856 | 0.96 | 0.647 (0.95) | — |
| fixed exposure (40 epochs) | VCS | 0.856 / 0.855 / 0.854 / 0.851 / 0.850 / 0.850 | 1.01 | 0.678 (0.99) | 0.57–0.59 |
| | InfoNCE | 0.858 / 0.857 / 0.856 / 0.857 / 0.857 / 0.856 | 1.00 | 0.669 (0.99) | — |
| | logistic | 0.858 / 0.859 / 0.856 / 0.857 / 0.856 / 0.857 | 1.00 | 0.670 (0.98) | — |
Exact-caption retrieval R@1 (secondary) is 0.03–0.16 for every adapter trained on topic pairs (raw CLIP 0.37) and is not read.  The InfoNCE
"ECE (Platt)" column is NaN because the Platt fit on the τ-scaled logit overflowed (same as in A2); it is not part of the reading.

## Reading (pre-committed grid)
**Does not hold.**  The three degradation curves coincide: with the batch reduced from 512 to 16 the primary metric changes by ≤ 1 % for VCS and
InfoNCE on both axes (logistic loses 4 % at B = 16 on the fixed-updates axis), far inside the ±0.05 band.  VCS's independent-pool negatives
(K = 8 at every B) did not turn into a small-batch advantage because the competitors do not degrade either: with 15 in-batch negatives per anchor
InfoNCE reaches the same accuracy as with 511.  Held-out J of the VCS model is 0.56–0.59 at every batch size (mid-dependence regime; P1 satisfied).

## Why, and what it does not say
The learning problem here is a linear adapter on frozen features — nearly convex and data-rich — where the gradient-noise / negative-count effects
that make small batches hurt contrastive *encoder training* are absent.  This pre-check therefore answers "is the decoupling an advantage when
fine-tuning adapters on frozen towers?" with no; it does not test the trainable-encoder regime (there, the SSL waves showed K-insensitivity for
VCS but no small-batch study was run; batch 128–1 024 with two views was neutral for VCS at 200 epochs, P17).  Follow-ups that would test the
property where it could matter: trainable towers or deep adapters with a real small-batch constraint; cross-batch / memory-bank negatives for
InfoNCE as the fairer competitor.  Families (brief appendix A, row C): micro-batch / large-voxel, memory-bank / streaming, k-way, asymmetric
sample sizes — **not opened** by this pre-check; evidence category: completed (frozen-tower adapters only).
