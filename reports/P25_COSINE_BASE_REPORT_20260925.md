# P25 — wave F: single factors on the cosine-critic base (13 units, all final)

Pre-registration: `P24_COSINE_BASE_PREREG_FROZEN_20260925.md`.  Table: `P25_cosine_base_results_table.md` (summarize job 1008618).
Base: `P18_vcs_crit_cosine_seed0` — cosine critic tanh(a⟨z1,z2⟩+b), K = 8, a0 = 1, b0 = 0: **78.32 / kNN 72.76 / h-rank 57.7**
(3 seeds, P30: 78.21 ± 0.13).  Rule: HELPS > +1.0 / HURTS < −1.0 vs 78.32.  Seed 0, 200 epochs unless stated.

| unit | linear | Δ | kNN | h-rank | heldout-J | learned a / b (ep 200) | verdict |
|---|---|---|---|---|---|---|---|
| **cos_negdetach** | **80.48** | **+2.16** | **74.62** | 30.9 | 0.929 | 9.81 / −8.09 | HELPS → new base (P26 ff.) |
| cos_k255 | 79.32 | +1.00 | 73.48 | 61.3 | 0.969 | 8.49 / −3.95 | borderline (3 seeds: +0.9, P30) |
| cos_k64 | 78.74 | +0.42 | 73.56 | 59.4 | 0.969 | — | neutral |
| cos_ema0.99 | 78.80 | +0.48 | 72.70 | 59.2 | 0.966 | — | neutral |
| cos_stopgrad | 78.74 | +0.42 | 73.20 | 58.1 | 0.966 | — | neutral |
| cos_proj_depth3 | 78.68 | +0.36 | 72.42 | 44.9 | 0.961 | — | neutral |
| cos_ema0.996 | 77.98 | −0.34 | 72.62 | 57.4 | 0.966 | — | neutral |
| cos_ema0.99_pred | 77.88 | −0.44 | 71.62 | 34.5 | 0.951 | — | neutral (predictor lowers rank) |
| cos_scale10 (a0 = 10) | 77.56 | −0.76 | 71.28 | 65.7 | 0.966 | 8.98 / −4.05 (a fell to 6.9 by ep 50) | neutral/negative |
| cos_sg_pred | 77.30 | −1.02 | 70.76 | 29.1 | 0.945 | — | HURTS (borderline) |
| cos_clr10 (critic lr ×10) | 77.26 | −1.06 | 71.74 | 53.4 | 0.965 | 13.74 / −8.24 | HURTS (borderline) |
| cos_on_h (critic reads L2(h)) | 72.72 | −5.60 | 76.48 | 235 | 0.968 | 9.27 / −3.94 | HURTS on linear; kNN +3.7 |
| cos_800ep (800 epochs) | 80.74 | +2.42 vs 200 ep | 76.48 | 89.1 | 0.982 | 13.03 / −6.95 | best plain-cosine result |

## Reading
1. **Gradient routing is the factor that matters: detaching the shifted partner in the negatives gives +2.2** (and +2.4 on 3 seeds, P30).
   Everything downstream (P26–P31) is built on it.
2. **More negatives help a little and only without detach** (K = 255: +0.9 on 3 seeds; K = 64: +0.4; with detach +0.2, P27).
3. **Target branches (EMA / stop-grad / predictor) are neutral or negative on the cosine base too**, as they were on the MLP base (P23).
   A predictor lowers h-rank (34 / 29 vs 58) without helping.  This closes the BYOL/SimSiam-style wiring question for this objective.
4. **The critic's scalars should not move faster or start sharper without detach**: critic lr ×10 (−1.1) drives a to 13.7; a0 = 10 (−0.8)
   first *decays* to 6.9 and then returns to 9.  Contrast P27: with detach, a0 = 5 is the best single factor (+1.3).  The initial-scale
   effect is therefore conditional on the gradient routing; P31 (a0 ∈ {2, 10, 20} with detach, seeds of a0 = 5) maps it.
5. **The projector is necessary for the cosine critic.** Reading L2(h) directly collapses ‖h‖ (rank 235 = uncentred noise) and costs 5.6 on
   linear — while kNN *rises* 3.7: the L2-normalised h has the metric structure, the raw h no longer has the linear structure.  This is the
   origin of the scale-freedom hypothesis (synthesis §4.5); the affine-free BN answers (proj_outBN, P27: −2.1; proj_bnonly, P28: running)
   have so far not converted it into a gain.
6. **800 epochs of plain cosine reach 80.74 / kNN 76.48 / rank 89** — +2.4 over 200 epochs, above both 800-epoch MLP runs (78.93 K = 1,
   79.56 K = 8 clr×10), but still below the *200-epoch* detach + a0 = 5 recipe (81.5–81.9).  The kNN curve is still rising at 800
   (74.9 → 75.7 → 76.5 at 400 / 600 / 800); a keeps growing (13.0) and b saturates (−6.95): the learned threshold cos* = 0.53 is far below
   the detach runs' 0.82.  `P26_vcs_base_800ep` (detach, 800 ep) is at kNN 79.4 at epoch 400 and will give the best-achievable estimate.

## Consequences
The cosine base of this wave is superseded: base → cosine + detach (P26) → + a0 = 5 (P31 confirming).  Dropped for good: target branches,
predictor, faster critic scalars, critic on h, 3-layer projector, K > 8 once detach is on.

## Not claimed
Single seed except where P30 gives 3; selection split; no control comparison at equal tuning budget.
