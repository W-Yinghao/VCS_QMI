# Pre-registration — pre-check B, question 1: closed-form critic vs neural critic on frozen features (P47), frozen 2026-09-27 before launch

**Property tested.** For a linear class T = wᵀφ(x, y) the VCS objective is quadratic in w with closed form
w* = ½ A_M⁻¹ d, J* = ¼ dᵀ A_M⁻¹ d (d = E_P[φ] − E_Q[φ], A_M = ½(E_P[φφᵀ] + E_Q[φφᵀ])); by the gap identity S − J(T) = E_M[(T − η)²] every
J value is a valid lower bound on S.  Question 1 of the brief: on frozen features, how far is the closed-form J of a linear class from the
held-out J of the trained tanh critic; is the linear class the bottleneck?

**Data (no new data).** The 5 000 selection images of the frozen split, two train-distribution views (fixed RNG 20260927), through the frozen
encoder + projector of four checkpoints spanning the critic families and the recipe: `P5_vcs_seed0` (concat-MLP critic, K = 1, J ≈ 0.90),
`P18_vcs_crit_cosine_seed0` (cosine critic, J ≈ 0.97), `P24_vcs_cos_negdetach_seed0` (cosine + detach, J ≈ 0.93), `P35_vcs_a5_views4_800ep_seed0`
(final recipe, J ≈ 0.97); plus `P5_simclr_seed0` (no critic: closed-form S lower bounds of a SimCLR embedding for comparison).
Pairs split 2 500 fit / 2 500 eval; negatives K = 8 non-zero cyclic shifts of the second view within each split (product samples as in training).

**Feature classes** (each + intercept): P1 = z1⊙z2 (128-d, contains the cosine critic's pre-activation), P2 = [z1⊙z2, |z1−z2|] (256),
P3 = [z1⊙z2, |z1−z2|, z1, z2] (512), H1 = h1⊙h2 (512, on L2-normalised encoder outputs).  Ridge λ ∈ {1e-6, 1e-4, 1e-2} × mean diag(A_M), reported.
Quantities on the eval split: J_eval(w*) (no tanh; the |T| > 1 fraction is recorded), J_eval(tanh(c·w*ᵀφ)) with the scalar c fitted on the
fit split (frozen output form), and the checkpoint's own critic on the same eval pairs (neural J).  No training beyond the linear solve and c.

**Pre-committed reading.**
- *Within-class gap:* for each VCS checkpoint, neural J_eval − max over (class, λ) of J_eval(tanh(c·w*ᵀφ)) is the "linear-class gap".
  **Property holds** if this gap is ≤ 0.02 for every VCS checkpoint (the closed form recovers the trained critic's J up to noise) and the
  between-family spread of neural J (≈ 0.90 vs 0.97 between MLP and cosine critics, P19) exceeds it — the brief's claim "linear class is not the
  bottleneck".  **Holds conditionally** if the gap is ≤ 0.02 only for the cosine-critic checkpoints (P1 contains their pre-activation) but not for
  the MLP-critic checkpoint, or only with tanh wrapping.  **Does not hold** if the gap exceeds 0.05 everywhere.
- *Informativeness of the bound:* the closed-form J_eval (no tanh) is reported as an S lower bound; it must be ≥ 0.5 for the recipe checkpoint
  to count as informative (arbitrary but frozen: half of the sup).
- *Ridge sensitivity:* if the best λ differs by more than 0.01 in J from the worst, the class is ill-conditioned and this is stated.
- Not claimed: registration (question 2 of the brief) — separate prereg once the constructed-modality pairs are defined.

Outputs: `reports/P48_precheck_B1.md/.json`.  Code: `scripts/precheck_b1_closedform.py`, `slurm/precheck_b1.sbatch` (CPU partition).
