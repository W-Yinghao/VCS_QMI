# P107 — v4 A-batch screen (seed 0) — report — 2026-10-02

Pre-registration `P107_V4_A_BATCH_PREREG_FROZEN_20261001.md`; results `P107_v4_A_table.md`; selection `P107_ADDENDUM1_SELECTION_FROZEN_20261002.md`.
One seed per cell: the contrasts below are seed-0 differences, reported without verdicts (pre-stated).  Selection-split linear / kNN, 4 views, B 256, 800 ep.

| cell | scorer | loss | pairing | negative gradient | linear | kNN |
|---|---|---|---|---|---|---|
| G2 (P104, ref) | fixed (2, −1) | VCS | K8 | detach | 88.66 | 86.96 |
| G2F (P104, ref) | learned from (2, −1) | VCS | K8 | detach | 86.70 | 84.38 |
| A-L1 | fixed (2, −1) | JS | K8 | detach | 88.54 | 86.80 |
| A-L2 | learned from (2, −1) | JS | K8 | detach | 86.32 | 83.88 |
| A-P1 | fixed (2, −1) | VCS | K8 | full | 88.82 | 87.30 |
| A-P2 | fixed (2, −1) | VCS | all-view tokens | detach | 88.98 | 86.60 |
| A-P3 | fixed (2, −1) | VCS | all-view tokens | full | **89.06** | **87.30** |

**A1 (scorer × loss):** AL1 − G2 = −0.12 (loss at the fixed scorer); AL2 − G2F = −0.38 (loss at the learned scorer); G2 − G2F = +1.96 and AL1 − AL2 = +2.22
(fixed vs learned within each loss).  v4's first reading pattern: **fixing the scale improves both losses by ≈ 2 points; the loss itself matters little
at matched hyper-parameters.**  No VCS-vs-JS winner claim (the pre-stated optimiser / lr sensitivity control would be required; not run).
**A2 (pairing × gradient at fixed (2, −1)):** AP1 − G2 = +0.16 / +0.34 kNN (full gradient, K8); AP2 − G2 = +0.32 / −0.36 kNN (pairing, detach);
AP3 − AP2 = +0.08 / +0.70 kNN (full gradient, all-view); AP3 − G2 = +0.40 / +0.34 kNN (both).  With a fixed scorer the full negative gradient is safe
(cf. G4 with a learned scale: 81.70); all differences except AP3 − G2 are within G2's own seed sd (0.20 linear).
**Selection:** A-P3 and A-P2 (addendum 1), with learned controls A-P2F / A-P3F; seeds 1–2 running.  Costs equal within 1 % (GPU smoke: 0.112–0.114 s/step).
