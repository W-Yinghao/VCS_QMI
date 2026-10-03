# P107 — A-P3 confirmation layer 2 (fresh seeds 3–4, strong augmentation, CIFAR-100) — report — 2026-10-03

Pre-registration `P107_ADDENDUM2_AP3_CONFIRM_LAYER2_FROZEN_20261002.md` (A-P3 passed layer 1 by the frozen rule); results-only commit `d3a7c8c`
(`P107_confirm2_table.md`).  A-P3 = fixed tanh(2s − 1), all-view tokens, full gradient.  Selection split, frozen-h linear / kNN, 800 epochs.
Note (v5 review): seed 0 was the screen-selection seed; seeds 3–4 are reported separately first, then pooled.

## 1. Fresh seeds 3–4 and the five-seed pool (CIFAR-10, standard augmentation)

| | seed 0 | 1 | 2 | **3** | **4** | 5-seed mean ± sd |
|---|---|---|---|---|---|---|
| A-P3 linear | 89.06 | 88.96 | 89.06 | **89.00** | **89.04** | **89.02 ± 0.04** |
| SimCLR linear | 88.20 | 88.10 | 88.66 | 88.28 | 88.30 | 88.31 ± 0.21 |
| Δ linear | +0.86 | +0.86 | +0.40 | **+0.72** | **+0.74** | **+0.72 [+0.48, +0.95]** |
| A-P3 kNN | 87.30 | 87.48 | 87.36 | 87.46 | 87.44 | 87.41 ± 0.08 |
| Δ kNN | +0.10 | −0.28 | −0.62 | −0.90 | −0.56 | −0.45 [−0.92, +0.02] |

- Fresh seeds alone (3, 4): Δ linear +0.72 and +0.74 — the same size as the selection-stage seeds.
- **"Holds at 5 seeds" (paired 95 % interval excludes 0): met on linear** (+0.72 [+0.48, +0.95]); kNN level (interval includes 0, SimCLR ahead on 4 / 5).
- vs G2 (5 seeds, P104 / P111): +0.34 [+0.10, +0.57] linear, +0.30 [+0.02, +0.59] kNN.

## 2. Strong augmentation (P89 block), seeds 0–2

| | linear (0 / 1 / 2) | mean | Δ vs own standard | kNN mean |
|---|---|---|---|---|
| A-P3 strong | 88.66 / 88.26 / 87.82 | **88.25** | **−0.78** (all seeds negative) | 86.11 |
| G2 strong (P111) | 86.86 / 87.70 / 87.80 | 87.45 | −1.25 | 85.24 |
| SimCLR strong (P89) | 89.74 / 89.08 / 89.66 | 89.49 | +1.17 | 88.71 |

Δ_int (A-P3 vs SimCLR) = **−1.95** linear (−1.94 / −1.68 / −2.24), −2.33 kNN; A-P3 strong − SimCLR strong = −1.25 linear, −2.60 kNN.  A-P3 degrades less than G2
under strong augmentation (+0.79 over G2-strong) but, like G2, loses ground where SimCLR gains.

## 3. CIFAR-100 (P91 8× recipe), seeds 0–2

| | seed 0 | 1 | 2 | mean ± sd | paired Δ (A-P3 − ref) [95 %] |
|---|---|---|---|---|---|
| A-P3 linear | 60.20 | 60.08 | 59.72 | **60.00 ± 0.25** | — |
| recipe VCS (P91) | 60.00 | 59.68 | 60.04 | 59.91 | +0.09 [−0.83, +1.02] |
| SimCLR (P91) | 58.66 | 58.02 | 58.08 | 58.25 | +1.75 [+1.06, +2.43] |
| G2 (P111) | 59.46 | 59.88 | 60.46 | 59.93 | +0.07 [−1.79, +1.93] |
| A-P3 kNN | 55.92 | 55.96 | 55.52 | **55.80 ± 0.24** | — |
| recipe VCS kNN | 55.42 | 54.56 | 54.86 | 54.95 | +0.85 [−0.34, +2.05] |
| SimCLR kNN | 57.38 | 56.78 | 57.46 | 57.21 | −1.41 [−2.80, −0.01] |
| G2 kNN | 54.12 | 54.26 | 54.00 | 54.13 | +1.67 [+1.32, +2.03] |

On CIFAR-100, A-P3 equals the recipe and G2 on linear, improves on G2 in kNN (+1.67, every seed), and against SimCLR linear and kNN disagree (+1.75 vs
−1.41) → by the pre-stated rule **no ordering vs SimCLR on CIFAR-100**.  A-P3's CIFAR-10 linear lead does not carry over as a lead over the recipe.

## 4. Reading — the A-P3 claim and its boundaries
- **Supported:** on CIFAR-10 with standard augmentation, A-P3 is above tuned SimCLR on linear-val at five paired seeds (+0.72 [+0.48, +0.95]; fresh seeds 3–4
  +0.72 / +0.74), level on kNN, with very small seed spread (sd 0.04); the fixed scorer is necessary (A-P3F −6.11 on every seed, P107 layer 1).
- **Not supported / boundaries:** under the strong-augmentation block on both sides SimCLR leads (−1.25 linear, −2.60 kNN); on CIFAR-100 A-P3 does not beat
  the recipe and has no readout-consistent ordering vs SimCLR; on CIFAR-10→CIFAR-100 frozen transfer the recipe transfers best (P119, readout-robust).
- The matched-JS control of A-P3 (P114, running) decides whether the quadratic objective contributes beyond the scorer / pairing / gradient structure;
  the augmentation decomposition (P115) is launched by the orchestrator once the P112 strong-augmentation follow-up seeds are also complete (layer 2 is now complete).
