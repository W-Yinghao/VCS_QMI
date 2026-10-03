# P107 — layer-1 confirmation of A-P3 / A-P2 with same-initialisation learned controls — report — 2026-10-03

Selection `P107_ADDENDUM1_SELECTION_FROZEN_20261002.md`; results `P107_v4_A_table.md` (309bdcc).  Linear-val / kNN, selection split, 800 epochs.

| config (all fixed scorers from (2, −1) unless "learned") | seed 0 | seed 1 | seed 2 | mean ± sd (linear) | kNN mean |
|---|---|---|---|---|---|
| **A-P3** fixed, all-view tokens, full negative gradient | 89.06 | 88.96 | 89.06 | **89.03 ± 0.06** | 87.38 |
| A-P3F learned, all-view, full negative gradient | 83.20 | 82.62 | 82.94 | 82.92 ± 0.29 | 79.39 |
| A-P2 fixed, all-view, detach | 88.98 | 88.18 | 88.04 | 88.40 ± 0.51 | 86.73 |
| A-P2F learned, all-view, detach | 86.90 | 86.74 | 86.28 | 86.64 ± 0.32 | 85.06 |
| G2 fixed, K8, detach (P104) | 88.66 | 88.92 | 88.52 | 88.70 ± 0.20 | 87.17 |
| tuned SimCLR (P41) | 88.20 | 88.10 | 88.66 | 88.32 ± 0.30 | 87.65 |

Paired (same seed): **A-P3 − SimCLR +0.86 / +0.86 / +0.40 = +0.71** (one-sided 90 % lower bound +0.42 → stably positive by the frozen rule; kNN −0.27, not
stable); A-P3 − G2 +0.33 (lower bound +0.05); **A-P3 − A-P3F +6.11** (linear) / +7.99 (kNN), every seed; A-P2 − A-P2F +1.76 / +1.67, every seed;
A-P2 − SimCLR +0.08 (not stable).

**Reading.**
- The fixed scale is necessary in both all-view variants: freeing (a, b) from the same start costs 1.8 points with detached negatives (as G2 vs G2F,
  +2.33) and 6.1 points with the full negative gradient — the learned scale + full gradient combination degrades as G4 did (81.70).  So the full negative
  gradient is only usable with a fixed scorer.
- A-P3 is the only VCS configuration so far that is stably above tuned SimCLR on linear-val at three seeds, with the smallest seed spread of any cell
  (sd 0.06); on kNN it is level with SimCLR (−0.27).  Its gain over G2 (+0.33) is small but positive on every seed.
- Layer 2 (addendum 2: seeds 3–4, strong augmentation, CIFAR-100) is running; the paper-level statement waits for it.  Interim: strong augmentation seeds
  0–1 give 88.66 / 88.26 (vs A-P3 std 89.06 / 88.96; G2 lost 1.25 under strong augmentation).

## Correction addendum (2026-10-03, v5 review)
Source: v5 review §2.1.  "So the full negative gradient is only usable with a fixed scorer" (Reading, first bullet) is narrowed to:
**in the tested optimisation settings** (recipe AdamW, lr, schedule, a/b initialisations (2, −1) and (5, 0), 800 epochs), the full negative gradient
combined with a learned (a, b) degraded (A-P3F 82.92, G4 81.70) while it worked with the fixed scorer.  This is an empirical statement about these
settings, not a general result.  Also note: A-P3's three seeds include seed 0, which was used for the P107 screen selection; seeds 3–4 are reported
separately in the layer-2 report.
