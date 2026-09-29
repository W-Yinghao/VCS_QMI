# P76 — S2 report: VCS vs SimCLR under the solo-learn CIFAR-10 protocol (P75) — 2026-09-29

Pre-registration `P75_S2_SOLO_LEARN_PREREG_FROZEN_20260928.md`; results-only commit 289b46d (`P76_s2_solo.{md,json}`).  Metric: online linear Acc@1 on the
official CIFAR-10 test split, read once per run at the end of epoch 1000 (owner exemption for this protocol only).

| method | seed 0 | seed 1 | seed 2 | mean ± sd |
|---|---|---|---|---|
| SimCLR (solo-learn recipe unchanged) | 91.65 | 90.76 | 91.32 | **91.24 ± 0.45** |
| VCS (P75 port: recipe critic, protocol projector width, LARS 0.4) | 87.45 | 87.31 | 87.24 | **87.33 ± 0.11** |

**Verdict (pre-committed rule): REFUTED** — gap 3.91 ≥ 1.5 (pooled SE 0.27, far from the threshold).  **QC passes:** SimCLR's mean is 0.50 from the
solo-learn README value 90.74, so the protocol is reproduced; all six units completed 1000 epochs and read the test split once (`fit_final_epoch_validation`).
For reference, the gap on our own protocol at 8× is 1.3 (selection) / 1.5 (official test).

**The pre-named suspect is visible in the QC sentinel.**  P75's deviation 2 put the two critic scalars under LARS-excluded plain SGD-momentum at the
protocol's lr 0.4.  In all three VCS runs a and b ran away together (a ≈ 250, b ≈ −250 at epoch 580, a + b ≈ 1; the ssl_pilot recipe ends at a 24.4,
b −22.0 under AdamW 1e-3), so the critic became a step function near cos ≈ 1 and the training J stalled near 0.69 (recipe: 0.95 by epoch 100).  P75's
prereg names this deviation as the first suspect and forbids post-hoc tuning inside P75; the one-change follow-up is P77 (S2b, frozen 2026-09-28T18:11Z:
same protocol, the two scalars on the recipe's AdamW 1e-3), read with the same 1.5 rule once its three seeds finish.

**What can be written now.**  Under the solo-learn protocol with LARS applied to the critic scalars, VCS trails SimCLR by 3.9 points (87.3 vs 91.2).
Whether the gap is a property of VCS under this protocol or of the scalar optimiser is not decided until P77 reports; the P75 number must not be quoted
without that caveat.
