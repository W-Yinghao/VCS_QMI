# P99 — S2b continuation report: solo-learn protocol with the critic scalars on the recipe's optimiser — 2026-10-01

Pre-registration `P99_S2B_CONTINUATION_PREREG_FROZEN_20260930.md` (owner go after P78's gate failure); results-only commit (`P99_s2b_continuation.json`).
Protocol and test-set rule as P75/P77: solo-learn CIFAR-10, ResNet-18, 2 crops, LARS 0.4, 1000 epochs; the critic scalars a, b on AdamW 1e-3 (method vcs_s2b);
online linear Acc@1 on the official test split read once at the end of epoch 1000.

| method | seed 0 | seed 1 | seed 2 | mean ± sd |
|---|---|---|---|---|
| SimCLR (P75) | 91.65 | 90.76 | 91.32 | 91.24 ± 0.45 |
| VCS, P75 (scalars on LARS-excluded SGD) | 87.45 | 87.31 | 87.24 | 87.33 ± 0.11 |
| **VCS-S2b (scalars on AdamW)** | 88.09 | 87.96 | 87.83 | **87.96 ± 0.13** |

**Verdict (pre-stated rule): REFUTED** — gap_b = 3.28 ≥ 1.5 (pooled SE 0.27).  Seed 0 resumed from epoch 276 (its epochs 0–276 were seen in P78, disclosed).

The recipe's optimiser for the two critic scalars recovers 0.63 points (87.33 → 87.96, every seed higher) and removes the scalar runaway (P78), but 3.3 of
P76's 3.9-point gap to SimCLR remain.  Under solo-learn's 2-crop LARS protocol the gap is therefore not an artefact of the scalar optimiser; it matches the
P41/P68 pattern at a larger size (8× ssl_pilot: 1.3 selection / 1.5 test).  Continuing to tune VCS inside this protocol (LARS lr, 4 crops, projector) would be a
new unit; the v3 programme (P104) works on the ssl_pilot protocol.
