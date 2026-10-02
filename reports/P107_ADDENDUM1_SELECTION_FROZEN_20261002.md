# P107 addendum 1 — screen selection (v4 §4) and confirmation launch — FROZEN 2026-10-02T05:13:09Z

All five seed-0 screen units are complete (results-only commit: `P107_v4_A_table.md`).  Final-h linear-val / kNN, selection split:

| rank | cell | linear | kNN |
|---|---|---|---|
| 1 | **A-P3** fixed (2, −1), all-view tokens, full negative gradient | **89.06** | 87.30 |
| 2 | **A-P2** fixed (2, −1), all-view tokens, right detach | **88.98** | 86.60 |
| 3 | A-P1 fixed (2, −1), K8, full negative gradient | 88.82 | 87.30 |
| 4 | A-L1 matched JS, fixed (2, −1), K8, detach | 88.54 | 86.80 |
| 5 | A-L2 matched JS, learned from (2, −1), K8, detach | 86.32 | 83.88 |
| ref | G2 (VCS, fixed (2, −1), K8, detach), seeds 0–2 | 88.66 / 88.92 / 88.52 | 86.96 / 87.08 / 87.46 |

**Selection (frozen rule: ≤ 2 by final linear-val; < 0.1 → cheaper / simpler):** **A-P3** and **A-P2**.  They are 0.08 apart (inside the tie band), but
both are taken since two may proceed; the next cell (A-P1) is 0.16 below A-P2, outside the band.  Costs are identical within 1 % (GPU smoke), so the
tie rule does not change the pair.  No hyper-parameter changes from here; seeds 1–2 change nothing.

**Method controls (v4 §4, v3 §4.1):** both candidates are fixed-scale cells → same-initialisation learned controls **A-P2F** (a, b learned from (2, −1),
all-view, detach) and **A-P3F** (learned from (2, −1), all-view, full negative gradient), seeds 0–2.  The component control (G2, seeds 0–2) exists.
A-P3F may fail as G4 did (learned scale + full gradient → 81.70); that is a reportable outcome, not a reason to change the control.

**Units submitted now (10):** A-P2 seeds 1, 2; A-P3 seeds 1, 2; A-P2F seeds 0, 1, 2; A-P3F seeds 0, 1, 2.  `make_p107_configs.py` gained the A-P2F /
A-P3F variants (parent G2F + pairing / routing as A-P2 / A-P3); it reproduces the frozen A-P2 / A-P3 seed-0 YAMLs byte for byte (c916be8f… / 63efcb70…).

**Reading after seeds 0–2 (pre-stated, as P104 / P111):** per-seed values; paired differences vs G2 (same seed) and vs the learned control; vs SimCLR seeds
0–2 (and P111's SimCLR seeds 3–4 if a candidate proceeds); "stably positive" = positive on all three paired seeds and one-sided 90 % t lower bound > 0.
A candidate stably positive vs SimCLR proceeds to pre-fixed new seeds 3–4 with SimCLR on the same seeds (P111 already holds SimCLR seeds 3–4).
