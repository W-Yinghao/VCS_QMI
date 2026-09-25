# P32 — cosine-critic initial scale a0 (P31), final

Pre-registration: `P31_SCALE_INIT_PREREG_FROZEN_20260925.md`.  Table: `P32_scale_init_results_table.md` (summarize job 1009130-series, see
`job_ids.json`).  Base: cosine critic, K = 8, negative detach, learnable (a, b) with b0 = 0; 200 epochs; a0 varied.

| a0 | seeds | linear (mean ± SD) | Δ vs a0 = 1 | kNN | h-rank | (a, b) at ep 200 → threshold | early path of a |
|---|---|---|---|---|---|---|---|
| 1 (P24/P26) | 3 | 80.59 ± 0.12 | — | 74.41 | 30.4 | 9.8 / −8.1 → 0.82 | grows monotonically (1 → 2.7 at ep 20 → 5.2 at ep 50) |
| 2 | 1 | 81.84 | +1.25 | 76.42 | 40.2 | 9.9 / −8.1 → 0.82 | flat to ep 10, then grows like the base |
| **5** | **3** | **81.56 ± 0.44** | **+0.97** | **76.94 ± 0.09** | 45.6 | 10.1 / −8.2 → 0.82 | dips to 3.3 (ep 20), back to 5.5 at ep 50 |
| 10 | 1 | 80.66 | +0.07 | 76.30 | 45.0 | 10.6 / −8.7 → 0.83 | dips to 6.3 (ep 50), regrows |
| 20 | 1 | 78.52 | −2.07 (HURTS) | 73.98 | 36.5 | 12.5 / −10.8 → 0.86 | decays to 11 (ep 50) and stays high |

## Reading
1. **The curve is a plateau at a0 ∈ [2, 5] for linear (+1.0…+1.3) and at [2, 10] for kNN (+1.9…+2.5); a0 = 20 hurts (−2.1).**
   a0 = 5 is confirmed on 3 seeds (≈ 3.7 pooled SD on linear; kNN +2.5 at SD 0.1–0.2).  a0 = 2 (single seed) is not distinguishable from
   a0 = 5.  The base stays at a0 = 5 (3 seeds in hand).
2. **All runs with a0 ≤ 10 converge to the same critic** (a ≈ 10, b ≈ −8, threshold cos* 0.82) — the endpoint does not explain the
   differences; the first ≈ 50 epochs do.  With a0 = 1 the score is nearly linear in the cosine for the first 20 epochs (a < 3) and the
   pair-wise regression has little curvature to exploit; with a0 = 2–5 the score is already sigmoidal but unsaturated (initial J
   −tanh²(a0·s̄) is −0.58 for a0 = 1, −1.0 for a0 ≥ 5 on the initial cone of near-identical z); with a0 = 20 essentially every pair starts
   saturated (kNN 40.9 at epoch 10 vs 53.9 for the base) and the encoder receives almost no gradient until a has decayed to ≈ 11, by which
   time the run is 50 epochs behind and never recovers.  This is the gradient-participation picture of synthesis §4.4, seen through the
   initial condition rather than through the functional form.
3. **h-rank tracks kNN, not linear** (30 → 40 → 46 → 45 → 36): the initial scale changes how many directions the encoder ends up using, and
   kNN benefits from that even at a0 = 10 where linear has returned to the base.

## Consequence
Base: cosine, K = 8, negative detach, **a0 = 5** (P35 builds on it; 4 views being added).  A fixed a = 5 (P27: 81.54) is an equivalent
choice if one prefers a parameter-free critic.  No further a0 work planned.

## Not claimed
Single seed at a0 ∈ {2, 10, 20}; selection split; the early-saturation mechanism is inferred from J, kNN and (a, b) trajectories, not from
an intervention.
