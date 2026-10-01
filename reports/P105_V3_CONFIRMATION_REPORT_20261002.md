# P104 / P105-report — package v3: first batch, selection and layer-1 confirmation (G2, U2 with same-initialisation controls) — 2026-10-02

Pre-registrations `P104_V3_FIRST_BATCH_PREREG_FROZEN_20260930.md` and addenda 1 (early G2 seeds / G2F), 3 (selection G2 + U2, controls U2F).  Results-only
commit `cc55159` (`P104_v3_table.md`).  All values: CIFAR-10 selection split, 4 views, B 256, 800 epochs, frozen-h linear probe and kNN.
(File number P105 was taken by the T1 ablation prereg; this report is filed as `P105_V3_CONFIRMATION_REPORT` for the v3 unit.)

## 1. First batch (seed 0, spec §4) and selection
G2 88.66, U2 87.96, G3 87.64, U1 87.56, N1 87.30, G1 87.04, N2 86.84, G4 81.70 (linear).  Selected by the frozen rule: **G2** and **U2** (addendum 3).
Mechanism notes: G4 (learned scale + full negative gradient) collapses to 81.70 while G3 (fixed scale + full gradient) is fine; noise averaging (N1) and the
matched-JS noisy control (N2) are on par with the recipe.

## 2. Layer 1 — three seeds each (per-seed linear / kNN)

| config | linear seeds 0 / 1 / 2 | mean ± sd | kNN mean ± sd |
|---|---|---|---|
| **G2** fixed (a, b) = (2, −1) | 88.66 / 88.92 / 88.52 | **88.70 ± 0.20** | 87.17 ± 0.26 |
| G2F learned from (2, −1) | 86.70 / 86.08 / 86.32 | 86.37 ± 0.31 | 84.61 ± 0.39 |
| U2 fixed (1, 0) + all-view tokens | 87.96 / 87.40 / 87.44 | 87.60 ± 0.31 | 85.88 ± 0.20 |
| U2F learned from (1, 0) + all-view | 87.22 / 87.34 / 86.86 | 87.14 ± 0.25 | 85.21 ± 0.20 |
| recipe VCS (P35) | 86.42 / 87.16 / 87.44 | 87.01 ± 0.53 | 85.46 ± 0.12 |
| tuned SimCLR (P41) | 88.20 / 88.10 / 88.66 | 88.32 ± 0.30 | 87.65 ± 0.40 |

Paired differences (same seed), mean, one-sided 90 % lower bound, 95 % t interval:

| contrast | linear per seed | linear mean [95 % CI] | kNN per seed | kNN mean [95 % CI] |
|---|---|---|---|---|
| **G2 − SimCLR** | +0.46 / +0.82 / −0.14 | **+0.38** [−0.82, +1.58] (90 % lower −0.15) | −0.24 / −0.68 / −0.52 | **−0.48** [−1.03, +0.07] |
| **G2 − G2F** | +1.96 / +2.84 / +2.20 | **+2.33** [+1.20, +3.46] | +2.58 / +2.70 / +2.40 | **+2.56** [+2.18, +2.94] |
| G2 − recipe | +2.24 / +1.76 / +1.08 | +1.69 [+0.25, +3.14] | +1.36 / +1.72 / +2.04 | +1.71 [+0.86, +2.55] |
| U2 − SimCLR | −0.24 / −0.70 / −1.22 | −0.72 [−1.94, +0.50] | −1.10 / −1.94 / −2.26 | −1.77 [−3.25, −0.28] |
| U2 − U2F | +0.74 / +0.06 / +0.58 | +0.46 [−0.42, +1.34] | +0.66 / +0.70 / +0.66 | +0.67 [+0.62, +0.73] |
| U2 − recipe | +1.54 / +0.24 / 0.00 | +0.59 [−1.46, +2.65] | +0.50 / +0.46 / +0.30 | +0.42 [+0.16, +0.68] |

## 3. Reading
- **G2 is a large, seed-stable improvement of the VCS recipe** (+1.69 linear, +1.71 kNN, every seed), and **the mechanism is holding the scale fixed
  throughout training, not the starting point**: the same-initialisation learned control G2F is +2.33 / +2.56 *below* G2 on every seed and lands below the
  recipe; its free scale grows from 2 to ≈ 25 (bias −1 → −22, threshold −b/a 0.5 → 0.91; P104 log) — the critic sharpens exactly as the recipe's does.
- **G2 vs tuned SimCLR: on par, not stably ahead.**  Linear +0.38 on average (two of three seeds positive, interval includes 0); kNN −0.48 (SimCLR higher on
  all three seeds).  The 8× gap of the recipe to SimCLR (−1.31 linear / −2.19 kNN) shrinks to +0.38 / −0.48.
- **Layer-1 rule of P111** ("G2 − SimCLR > 0 on all three seeds and one-sided 90 % lower bound > 0"): **not met** (seed 2 −0.14; lower bound −0.15).  By the
  frozen rule the ten P111 runs (seeds 3–4, strong augmentation, CIFAR-100) are reported as **extra runs outside any candidate-vs-SimCLR claim**; they
  still add paired seeds and are reported descriptively.
- **U2** improves on the recipe mainly in kNN (+0.42, all seeds) and on its control by +0.46 / +0.67, but stays below SimCLR (−0.72 / −1.77).  The
  all-view pairing does not add to the fixed scale at a = 1 beyond what G2's a = 2 achieves; v4 A-P2 / A-P3 test the combination at (2, −1).

## 4. Next (already running / registered)
P107 (v4 A batch around G2) screen; P111 seeds 3–4 / strong / CIFAR-100 (extra by the rule); P109 X1 / I1 on G2 and U2; P110 V1–V3.
