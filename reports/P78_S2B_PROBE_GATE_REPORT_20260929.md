# P78 — S2b (P77) probe-gate report: the unit stops by its pre-stated rule — 2026-09-29

Pre-registration `P77_S2B_CRITIC_OPTIMISER_PREREG_FROZEN_20260928.md` (frozen 2026-09-28T18:11:12Z).  Probe gate, pre-stated: at epoch 20 of seed 0,
a ∈ [2, 40] and b ∈ [−40, 2] and train J ≥ 0.70; if it fails, the unit is stopped, the failure reported, seeds 1–2 not submitted, no re-tuning inside P77.

**Gate reading (seed 0, job 1014496, Lightning CSV, no test data): FAIL.**  a = 3.05 (pass), b = −2.23 (pass), train J = 0.519 (fails the 0.70 clause).

| epoch | a | b | train J | t_pos / t_neg | sat pos / neg | online train Acc@1 | P75 VCS seed 0 at the same epoch (a / J / Acc@1) |
|---|---|---|---|---|---|---|---|
| 0 | 5.00 | 0.00 | −0.18 | 0.65 / 0.27 | 0.38 / 0.19 | 16.0 | 4.2 / −0.04 / 16.4 |
| 20 | 3.05 | −2.23 | 0.519 | 0.42 / −0.37 | 0.00 / 0.02 | 44.2 | 35.4 / 0.27 / 34.8 |
| 100 | 10.1 | −9.3 | 0.596 | 0.46 / −0.47 | 0.00 / 0.06 | 56.3 | 109.8 / 0.55 / 52.1 |
| 200 | 15.2 | −14.3 | 0.628 | 0.48 / −0.50 | 0.00 / 0.08 | 63.4 | 160.9 / 0.59 / 59.9 |
| 270 | 17.8 | −16.9 | 0.641 | 0.49 / −0.52 | 0.00 / 0.09 | 65.4 | ≈ 205 / 0.63 / ≈ 65 |

**Disclosure of timing.**  The gate was due at epoch 20 but was read at epoch ≈ 272: the automated milestone check parsed the epoch from a fixed CSV
column that the S2b log shifts (two extra lr columns), so it never fired.  The job was moved to the preemptible `runfill` QOS on the owner's instruction and
ran unobserved for 47 min.  The later epochs above were therefore seen before the stop; they do not change the verdict, which depends on epoch 20 only.
The job was cancelled at epoch ≈ 272; its per-epoch checkpoint is kept (solo-learn auto_resume could continue it), and seeds 1–2 are not submitted.

**Reading (descriptive, not a P77 verdict — P77 has no final number).**
- The optimiser change does what it was meant to: a and b no longer run away (a 17.8 at epoch 270 vs ≈ 205 under P75), positives are not saturated.
- Train J under this protocol plateaus near 0.64 with either optimiser; the gate's 0.70 threshold was taken from the ssl_pilot recipe (4 views, milder
  augmentation) and is not reached by P75 or S2b under solo-learn's 2-view symmetric augmentation.  The J clause, not the scalar clauses, failed.
- Online train Acc@1 is 2–4 points above P75 at matched epochs up to 200 and equal by epoch 270 — the scalar optimiser is not the cause of the 3.9-point
  gap to SimCLR that P76 measured.  P76's refuted verdict therefore stands without the optimiser caveat being able to close it; the caveat is recorded as
  "tested: the runaway scalars are not the main cause" rather than "open".
- Continuing S2b (or any re-tuning of the gate) would be a new pre-registration decided by the owner.
