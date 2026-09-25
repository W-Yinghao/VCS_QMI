# P19 — Critic-form / pairing / update-schedule variants report (P18)

Status: **COMPLETE** (6/6 COMPLETED, 2026-09-25 08:07–09:56 UTC).  Neutral table: `reports/P19_critic_variants_results_table.md`.
Pre-registration: `reports/P18_CRITIC_VARIANTS_PREREG_FROZEN_20260925.md`.  Base: K=8, critic lr ×1, seed 0, 200 epochs;
baseline `P10_vcs_k8_seed0`: linear 76.12 %, kNN 67.44, h-rank 15.6, ≈ 8 dominant `p_raw` directions, heldout-J 0.926.
The objective J, the mixture reference, the cyclic-shift negatives and the tanh output are unchanged in every unit; only f_θ, the pairing
order, or the critic update count changed (all permitted by the collaborator's plan §9).

## Results

| unit | f_θ / change | linear | Δ | kNN | h-rank | z-rank | `p_raw` cliff | ‖h‖ | heldout-J | signal |
|---|---|---|---|---|---|---|---|---|---|---|
| **crit_cosine** | tanh(a·⟨z1,z2⟩+b), 2 params | **78.32** | **+2.20** | **72.76** | **57.7** | 42.2 | none | 16.6 | 0.968 | HELPS |
| **crit_interact** | MLP on [z1, z2, z1⊙z2, |z1−z2| (elementwise)] | 77.34 | +1.22 | 70.74 | 39.3 | 22.3 | none | 20.5 | 0.962 | HELPS |
| crit_bilinear | tanh(z1ᵀWz2 + MLP[z1;z2]) | 76.18 | +0.06 | 66.76 | 16.7 | 8.9 | 9 dirs | 25.1 | 0.931 | neutral |
| pair_sym | score both orders | 77.08 | +0.96 | 67.60 | 17.0 | — | — | — | 0.931 | neutral |
| crit_steps2 | 1 extra critic-only step | 76.76 | +0.64 | 67.54 | 18.1 | — | — | — | 0.931 | neutral |
| crit_steps5 | 4 extra critic-only steps | 75.62 | −0.50 | 68.80 | 21.1 | — | — | — | 0.937 | neutral |

kNN trajectories (ep 10/20/50/100/150/200): cosine 56.5/64.0/69.2/71.7/72.8/72.8; interact 56.1/61.4/67.0/69.7/69.8/70.7; K=8 baseline
45.6/52.4/60.3/64.8/66.8/67.4.  Probe curves for cosine/interact are smooth (77.9→78.3, 76.3→77.3); ‖h‖ drops from 24–25 to 17–21.

## Reading (per the frozen grid)
- **The critic's function form was the limiter.**  Replacing the concat-MLP f_θ by a similarity-form critic raises the encoder's effective rank
  from 16 to 58 (SimCLR-matched: 90) and removes the `p_raw` eigenvalue cliff entirely; interaction features (z1⊙z2, elementwise |z1−z2|) fed to the same
  MLP get most of the way (rank 39).  This is the pre-registered "critic form is a real limiter" outcome; the paper's wording about the critic
  class must state that f_θ needs an explicit similarity/interaction structure, not merely capacity.
- **A 2-parameter critic beats a 400k-parameter one** (78.32 vs 76.12): the MLP's capacity was never the binding constraint (consistent with
  P13), and the MLP-on-concat solution actively pulls the encoder toward a low-dimensional comparable code.
- The bilinear term added to the concat MLP did not help: with the MLP branch present the optimizer still finds the low-dimensional solution.
- Symmetric pairing and extra critic-only steps are within noise; drop them.
- heldout-J rises to 0.96–0.97 for the new critics while the representation improves — the "J near its bound = weak signal" reading of P9
  is superseded: with a similarity critic, high J and high-rank features coexist.  The saturation fractions are similar (0.84–0.94), so
  saturation per se was not the mechanism; the *shape* of f_θ was.
- Faster early learning: cosine reaches at epoch 20 (kNN 64.0) what the baseline reaches at epoch 100.

## Consequence for the "best VCS" set
New base: **cosine critic + K=8** (78.32 / 72.76 / rank 58).  Next single factors on this base: K (64, 255 — more negatives may matter more
for a similarity critic), critic lr ×10, initial scale of a, critic on h, negative detach, and the re-based target-branch variants (EMA /
stop-grad / predictor / projector depth); plus the 800-epoch version.  The concat-MLP-based P20/P22 units still pending are superseded.

**Waiting for the owner only on framing; exploration continues per the stated principle.**
