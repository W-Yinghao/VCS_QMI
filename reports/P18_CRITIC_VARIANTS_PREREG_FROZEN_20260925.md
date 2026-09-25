# Pre-registration — VCS-QMI critic-form / pairing / update-schedule variants (P18–P19), frozen 2026-09-25 before launch

Owner decision after the framework-level analysis: the remaining candidates for the low-dimensional code and slow convergence lie in the
critic's *function form* (an MLP on concatenated inputs represents similarity/bilinear interactions inefficiently and can be satisfied by a
few comparable coordinates), the ordered one-directional pairing, and the joint update's time scales.  These are engineering choices [E]
(spec §5.3, §6.1, §7.1); the objective's mathematics (bounded quadratic J on tanh critic scores, mixture reference) is unchanged, so every
variant is a *named* alternative critic class / sampler / schedule, disclosed as such.

## Units (seed 0, 200 epochs, K = 8, critic lr ×1; everything else = `cifar10_k8_vcs_seed0.yaml`; baseline `P10_vcs_k8_seed0`:
linear 76.12 %, kNN 67.44, h-rank 15.6, heldout-J 0.926; secondary reference: `P16_vcs_k8_clr10_seed0` 77.44 %)

| unit | change | class / rule |
|---|---|---|
| crit_interact | `model.critic.input=concat_interact` | MLP on [z1, z2, z1⊙z2, |z1−z2|] (4D → 512 → 512 → 1, tanh) |
| crit_bilinear | `bilinear_concat` | tanh(z1ᵀWz2 + MLP([z1;z2])), W Xavier gain 0.1 |
| crit_cosine | `cosine` | tanh(a·⟨z1,z2⟩ + b), two parameters (a init 1, b init 0) |
| pair_sym | `pairing.sampler=random_nonzero_cyclic_shift_symmetric` | score (z1,z2) and (z2,z1) for positives and the shifted negatives; 2B positives, 2KB negatives, same separate averaging |
| crit_steps2 / crit_steps5 | `train.mode=joint_critic_steps_N` | N−1 critic-only AdamW steps on detached features per batch, then the ordinary joint step (encoder/projector updated once per batch) |

All critics are pointwise per pair, tanh-bounded, without BN/dropout, shared between positive and negative pairs; the hold-out diagnostic
uses the same critic input / symmetry as training.  Controls remain pinned to the reference critic, sampler and joint mode (policy-checked).
Configs and sha256: `configs/HPARAM_C_CRITIC_SHA256.json`.

## Endpoints and reading
Primary: final linear-val vs 76.12 (HELPS > +1.0 / HURTS < −1.0 / neutral).  Secondary and central to the hypothesis: h effective rank and the
number of dominant `p_raw` directions (K=8 baseline: 15.6 / ≈7–8), kNN, heldout-J, saturation.  Pre-committed reading:
- If `crit_interact` or `crit_bilinear` raise rank substantially (> 25) **and** linear-val → the concat-MLP critic form is a real limiter and
  the paper's "critic class" wording must be tightened accordingly.
- If `crit_cosine` (2 parameters) matches or beats the MLP critic → the MLP capacity was never used; the objective only needs a similarity
  score, and the encoder alone solves the task.  If it collapses or fails, the flexible critic is necessary for optimization even if unused
  at the optimum.
- `pair_sym` neutral → order bias irrelevant.  `crit_steps` HELPS → critic lag is a limiter (consistent with critic lr ×10 raising rank/kNN).
- Failures (e.g. non-finite from the bilinear term) are recorded, not patched.

## Not claimed
Single seed; selection set; no control comparison; no statement about the population objective.
