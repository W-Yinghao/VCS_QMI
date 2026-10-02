# P111 — G2 confirmation layer 2 (fresh seeds 3–4, strong augmentation, CIFAR-100) — report — 2026-10-02

Pre-registration `P111_G2_CONFIRM_LAYER2_PREREG_FROZEN_20261002.md`; results-only commit `2cb1541` (`P111_G2_confirm2_table.md`).  All ten units COMPLETED.
**Status by the frozen rule: extra runs.**  Layer 1 (P104 confirmation) did not make G2 stably positive vs SimCLR (seed 2 −0.14; one-sided 90 % lower
bound −0.15), so these runs enter no candidate-vs-SimCLR claim; they are reported descriptively.  Selection split, frozen-h linear and kNN, 800 epochs.

## 1. Five paired seeds, CIFAR-10 standard augmentation (seeds 0–2 from P104 / P41, seeds 3–4 new)

| | seed 0 | 1 | 2 | 3 | 4 | mean ± sd |
|---|---|---|---|---|---|---|
| G2 linear | 88.66 | 88.92 | 88.52 | 88.72 | 88.62 | **88.69 ± 0.15** |
| SimCLR linear | 88.20 | 88.10 | 88.66 | 88.28 | 88.30 | 88.31 ± 0.21 |
| Δ linear | +0.46 | +0.82 | −0.14 | +0.44 | +0.32 | **+0.38 [−0.05, +0.81]** |
| G2 kNN | 86.96 | 87.08 | 87.46 | 87.04 | 86.98 | 87.10 ± 0.20 |
| SimCLR kNN | 87.20 | 87.76 | 87.98 | 88.36 | 88.00 | 87.86 ± 0.43 |
| Δ kNN | −0.24 | −0.68 | −0.52 | −1.32 | −1.02 | **−0.76 [−1.28, −0.23]** |

"Holds at 5 seeds" (interval excludes 0) — linear **no** (G2 ahead on 4 / 5 seeds, interval touches 0); kNN: SimCLR ahead on all five, interval excludes 0.
G2 is very seed-stable (sd 0.15); the new seeds reproduce seeds 0–2.

## 2. Strong augmentation (P89 block; SimCLR and recipe strong from P89 / P43)

| | seed 0 | 1 | 2 | mean | Δ vs own standard |
|---|---|---|---|---|---|
| G2 strong, linear | 86.86 | 87.70 | 87.80 | 87.45 | **−1.25** (−1.80 / −1.22 / −0.72) |
| SimCLR strong, linear | 89.74 | 89.08 | 89.66 | 89.49 | +1.17 |
| recipe VCS strong, linear | 87.78 | 87.54 | 87.66 | 87.66 | +0.65 |
| G2 strong, kNN | 84.90 | 85.30 | 85.52 | 85.24 | −1.93 |
| SimCLR strong, kNN | 88.86 | 88.68 | 88.60 | 88.71 | +1.07 |

Interaction Δ_int = (G2 strong − G2 std) − (SimCLR strong − SimCLR std): **−2.42 linear** (−3.34 / −2.20 / −1.72; sd 0.83) and **−2.99 kNN**.
G2 strong − SimCLR strong = −2.04 linear, −3.47 kNN.  **The fixed (2, −1) scale's benefit does not survive stronger augmentation**; under the strong block
G2 is also below the recipe VCS (−0.21 linear, −0.37 vs recipe-strong kNN 85.61).  Mechanism not tested here (hypothesis: stronger augmentation lowers
positive-pair cosines, so a fixed zero-score threshold κ = 0.5 sits in the wrong place, while a learned scale adapts) — P112 tests (a, κ) under strong aug.

## 3. CIFAR-100 (P91 8× recipe; references P91 seeds 0–2)

| | seed 0 | 1 | 2 | mean ± sd | paired Δ vs G2 |
|---|---|---|---|---|---|
| G2 linear | 59.46 | 59.88 | 60.46 | 59.93 ± 0.50 | — |
| recipe VCS linear | 60.00 | 59.68 | 60.04 | 59.91 | G2 **+0.03** [−1.22, +1.28] |
| SimCLR linear | 58.66 | 58.02 | 58.08 | 58.25 | G2 +1.68 [−0.32, +3.68] |
| G2 kNN | 54.12 | 54.26 | 54.00 | 54.13 ± 0.13 | — |
| recipe VCS kNN | 55.42 | 54.56 | 54.86 | 54.95 | G2 −0.82 [−2.07, +0.43] |
| SimCLR kNN | 57.38 | 56.78 | 57.46 | 57.21 | G2 **−3.08** [−4.31, −1.85] |

**On CIFAR-100 G2 equals the recipe (linear +0.03) and is slightly below it on kNN; it does not reproduce its CIFAR-10 gain (+1.69).**  Linear and kNN
disagree on G2 vs SimCLR (+1.68 vs −3.08); under P98 (the linear probe is probe-limited for SimCLR on CIFAR-100) no ordering is stated.

## 4. Reading
- G2's improvement over the VCS recipe is specific to the condition it was found in (CIFAR-10, standard augmentation): it is large and seed-stable there
  (+1.69 linear, +1.71 kNN; 5 seeds at sd 0.15), absent on CIFAR-100, and reversed under strong augmentation.  The fixed scale is a configuration that
  must be matched to the data / augmentation regime — P112 measures whether a different fixed (a, κ) recovers the gain in the other two regimes.
- Against tuned SimCLR, G2 is on par in linear accuracy and behind in kNN at 5 seeds; under strong augmentation SimCLR leads clearly.
- For the paper: "a fixed angular scale lifts the VCS recipe by ≈ 1.7 points on CIFAR-10 (standard augmentation), and the gain comes from fixing the
  scale throughout training, not from its initial value" is supported (P104); generalisation to other datasets / augmentation is **not**.
