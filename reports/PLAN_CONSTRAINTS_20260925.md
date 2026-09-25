# What the collaborator's plan fixes — and what is free to change (owner clarification 2026-09-25)

Source of truth: `CS_QMI/Variational_CS_QMI_Research_Plan (1).pdf`, sha256 `0f87809ebaad8fb64cfbec9e442b23104150f73294401a2177a5b9e47de884da`
(= S1 in `sources.json`).  Everything else (the server spec, the frozen configs, the first-round prohibitions) was added by the owner and may be
modified, provided the plan's content is preserved.

## Fixed by the plan (must not change)
1. Reference measure M = (P+Q)/2 with P = P_XY (joint pairs), Q = P_X P_Y (independently paired samples).  Q must be product-of-marginals
   sampling: derangement / cyclic shift / cross-batch / all i≠j pairs are all allowed (§9); *selecting* negatives by hardness or labels is not.
2. Target quantity I_VQ = S = E_M[η²] ∈ [0,1]; the CS value is the monotone transform 2·log((1+S)/(1−S)) and is **not** used for training (§3, §6, §10).
3. Exact variational objective J(T) = E_P T − E_Q T − ½E_P T² − ½E_Q T², S = sup_T J, gap S − J = E_M[(T−η)²] (Theorem 2, Eq. 5–6); mini-batch
   form Eq. (17): each distribution averaged separately; train by gradient ascent on Ĵ.
4. Critic parameterization T_θ = tanh f_θ(z) (§9): the tanh output is required (aligned with T* = tanh(PMI/2)); **f_θ is any function of the
   pair** — the plan does not prescribe concatenation, an MLP, or any width/depth.
5. No density estimation, no exponential moments, no clipping of J, report I_VQ (optionally I_VCS at evaluation only), never call it Shannon MI.

## Owner's additions that are now open to modification (with disclosure)
- SSL wiring: encoder/projector architecture, which representation the critic reads (z = normalized projector output vs h), projector dims/BN,
  L2 normalization, two-view recipe, augmentations, batch size, optimizer/schedule, joint vs alternating/two-timescale updates, gradient
  routing on the negative branch (detach or not), EMA/stop-gradient teachers, multi-view/multi-crop.
- Critic form f_θ: concat MLP (first round), interaction MLP, bilinear, cosine, critic on h, … — all satisfy §9 as long as the output is tanh.
- Pairing: K shifts, symmetric scoring, all i≠j pairs.
- Evaluation protocol details (probe standardization, kNN settings) — as long as the reported quantity is frozen-feature quality and the
  same protocol is applied to every method.

## Still outside the objective (would be a different method, to be discussed with the collaborator)
Auxiliary losses added to J (variance/covariance/orthogonality/reconstruction), changing the reference measure or its weights, replacing the
tanh output, reweighting pairs inside J, hard-negative or label-based negative selection.
