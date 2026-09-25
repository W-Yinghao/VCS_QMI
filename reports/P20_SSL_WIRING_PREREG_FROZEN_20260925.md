# Pre-registration — SSL-wiring variants permitted by the plan (P20–P21) + standardized-probe diagnostic, frozen 2026-09-25

Owner clarification: only `Variational_CS_QMI_Research_Plan (1).pdf` (sha256 `0f87809e…884da`) is fixed; the SSL wiring is the owner's
and may change (`reports/PLAN_CONSTRAINTS_20260925.md`).  The objective J, the mixture reference, product-of-marginals negatives and the
tanh critic output are untouched in every unit below.

## Units (seed 0, 200 epochs; base = current best `cifar10_hpB_k8_clr10_vcs_seed0.yaml`: K=8, critic lr ×10; baseline `P16_vcs_k8_clr10_seed0`
77.44 % / kNN 68.40 / h-rank 21.4 / J 0.940)

| unit | change | rationale |
|---|---|---|
| crit_on_h | `model.critic.feature_source=h_l2` — critic reads the L2-normalized 512-d encoder output; projector is unused (receives no gradient; disclosed) | removes the 128-d projector bottleneck from the critic's view; tests whether the low-dimensional code is imposed through the projector |
| neg_detach | `pairing.negative_detach=true` — the shifted partner z2[π(i)] is detached in negative pairs | changes gradient routing only (the "push apart" signal reaches the encoder through z1 only); the objective value is unchanged |
| crit_steps5 | `train.mode=joint_critic_steps_5` on the best base | approximates sup_T inside the joint update (critic closer to optimal for the current encoder) |

Also submitted: **standardized-probe diagnostic** (`scripts/probe_standardized.py`, GPU job): the frozen linear-probe protocol re-run on
per-dimension standardized `h` (fit statistics) for every completed run of every stage, VCS and controls alike, using the cached features.
It changes no training and is reported next to the raw protocol; its purpose is to measure how much of the VCS probe noise / VCS–control gap is
a feature-norm artefact (VCS ‖h‖ ≈ 27–34 vs 8–10 for controls).

## Reading
Primary: final linear-val vs 77.44 (HELPS > +1.0 / HURTS < −1.0 / neutral); secondary: h-rank, dominant `p_raw` directions (for crit_on_h the
projector spectrum is meaningless and is ignored), kNN, heldout-J.  Standardized probe: report raw vs standardized per run; if the VCS–control
gap shrinks by more than the seed SD under standardization, the raw protocol was partly measuring norm, and the paper's probe protocol
should standardize for all methods (decision for the owner).

## Not claimed
Single seed; selection set; controls untouched.
