# Pre-registration — P149 (r3 R3-V0 + R3-V1): two-view dependence of frozen encoders under invertible coordinates and a common random channel — FROZEN 2026-10-08

Source: r3 package §7–9, `docs/03` §6–7, `contracts/task_manifest.json` `visual_default`; intake `reports/r3_intake/R3_DELTA_INTAKE_20261008.md`.  Owner go 2026-10-07
("准备并提交相关实验").  No encoder training; CPU only; frozen after the gate (`slurm/p149_gate.sbatch`).

## 1. Question
On the SSL object itself — two independent augmented views of the same image — how repeatable and how resolving are three measurement routes of
the same target S(P_{H1,H2}, P_{H1}P_{H2}): the direct VCS fit, the logistic fit read as common J, and a same-target RFF critic?  Repeatability across
fit initialisations and under an invertible change of coordinates; resolution along a controlled degradation channel that cannot increase S.

## 2. Fixture (R3-V0; `scripts/p149_twoview.py fixture`)
Four frozen encoders, seed 1 (not chosen by score): P107_AP3_views4_800ep_seed1, P41_simclr_views4_800ep_seed1 (CIFAR-10); P107_AP3_c100_views4_800ep_seed1,
P91_c100_simclr_views4_800ep_seed1 (CIFAR-100).  20 000 base images from each run's manifest FIT uids, roles FIT 10 000 / VAL 2 000 / EVAL 8 000 by base id
(seed 20261002, the P109 / P143 roles).  Two independent views A1, A2 per base image under the P122 *standard* measurement law of the dataset
(signature `22f3b58bfde64c92`, asserted; augmentation seed 122, per-image seeding (seed, tag, uid) as P122).  Encoder in eval mode → h (512) and z (128),
float32.  EVAL independent pools (seed 20261007): P 2 000 base images (A1_i, A2_i); Q-left 2 000 (A1); Q-right 2 000 (A2); 2 000 reserved.  Cache ≈ 100 MB per
encoder; manifest with law signature, checkpoint sha, pool seed, sizes.  All base images are pre-training images of the encoders (disclosed).  The P122
cache this replaces was deleted by cleanup R6; images that were in P122's pools receive byte-identical views.

## 3. Measurement (R3-V1; representation h)
- Standardisation per view on all FIT images: μ_j and scalar γ_j = sqrt(mean squared deviation per coordinate) (an affine bijection; ΔS = 0).
- FIT pairs: P 2 048 base images (A1, A2); Q 2 048 pairs from 2 048 + 2 048 disjoint base images (independent partner design, as P122).  TUNE from VAL:
  1 000 P pairs, 500 Q pairs.  EVAL: the fixed pools above; the same pools for every setting / estimator / seed.
- Settings (4): `identity_t0`; `orthogonal_refit_t0` (fixed Haar-orthogonal R1, R2 ∈ O(512) from seeds 149001 / 149002 applied to view 1 / 2 in every role;
  critics refitted); `identity_t0.25`; `identity_t1` (Brownian channel U_t = H̃ + W_t, W_t ~ N(0, t I) per coordinate, independent across views and base
  images, coupled across t by increments; replay by (seed 149, role, uid, view); no post-channel normalisation).  P and Q pass through the same channel law.
- Estimators (3): `vcs_mlp` f = MLP([u; v]) → 256 → 256 → 1, loss −J, T = tanh f; `js_mlp` same net, loss softplus(−2f) + softplus(2f) (half-logit
  coordinate, T = tanh f); `rff_ridge_tanh` ψ = RFF of [u; v] (D 1 024), the P116 two-stage ridge-tanh solver on [ψ, 1].  Candidates per selected cell:
  MLP lr ∈ {1e-4, 5e-4, 2e-3}; RFF bandwidth ∈ {0.5, 1, 2} × FIT median pair distance.  ≤ 1 000 full-batch AdamW steps (wd 1e-2), TUNE own risk every 10
  steps, step 0 included, **selection on TUNE only**; 3 init seeds (0, 1, 2) per (setting, estimator).  144 selected cells; ≤ 432 candidate fits.
- Readouts per cell (EVAL once): per-sample T on the P and Q pools (saved); J_common with the independent two-pool SE and the Hoeffding radius
  (δ 0.05; the interval is for J(T), the approximation bias is not covered); S_plugin; T quantiles; |T| > 0.95 fractions; TUNE and FIT J; selected
  candidate and step; seconds.  Nulls: T = 0 (reported), independent-pair null = derangements of the P pool with B = 200 shared permutations → null J
  and the permutation p of the observed J.  `pair_var_conditional_batch` = variance of J over 64 re-pairings of the Q pools with the fixed critic.
- Transport check (implementation only): the identity_t0 seed-0 critics with first-layer weights W' = W blockdiag(R1ᵀ, R2ᵀ) (MLP) / frequencies
  W' = R W per block (RFF) must reproduce their outputs on rotated inputs (max |Δf| < 1e-3).  RFF frequencies in the refit setting are redrawn in the new
  coordinates (not transported) — stated.

## 4. Pre-stated reading
Per fixture and estimator: (i) **repeatability** = refit spread over the 3 init seeds at identity_t0 and the drift identity_t0 → orthogonal_refit_t0 (paired
bootstrap over the shared EVAL pools, 1 000 reps, shared indices); (ii) **resolution** = adjacent-level ordering P(J(t_k) > J(t_{k+1})) over seeds × bootstrap
and the standardised gap; (iii) every reading next to its null J and permutation p and its saturation fraction.  Conditional labels: *numerically
resolvable* (ordering probability ≥ 0.95 and gap ≥ 2 pooled sd at both steps), *not resolvable*, *estimation-sensitive* (refit spread or re-pairing sd
exceeds the adjacent gap).  No encoder or estimator is required to win; a near-saturated t = 0 reading (P122: 0.97–0.99) with small variance is reported as
saturation, not as precision.  Real images carry no oracle: no posterior MSE; `approximation_bias_covered = false` in every record.

## 5. Not claimed / not done
Resolution of SSL checkpoints (P101 negative stands); noise as a training device; z-site and other encoder seeds (later protocol); effects of
`refit_spread_resampled_pool` (FIT sample fixed in round 1; stated); any statement about S of real images beyond the fitted J and its nulls.

## 6. Cost
Fixture: ≈ 15 s augmentation + ≈ 65 s CPU encoding per encoder (P122 / P143 rates).  Measurement: 288 MLP fits (≤ 1 000 steps on 4 096 pairs × 1 024-d) +
144 RFF solves; measured in the gate; expected ≈ 1–2 h CPU per fixture → four CPU jobs.

## 7. Decisions at the freeze (2026-10-08)
- First gate (1027900) caught an implementation error: TUNE pools of 1 000 P / 500 Q pairs, while the P122 per-unit J adds P and Q rows → TUNE is
  now **500 P / 500 Q pairs** (balanced; §3 updated).  Nothing else changed.
- Gate 1027918: `tests/test_p149.py` 6 / 6 (Brownian replay / coupling / order invariance against the reference formula, Haar orthogonality,
  standardiser, exact MLP and RFF transport, readout formulas, RFF fit on dependent Gaussian pairs), P116 regression 8 / 8; smoke fixture (600 / 200 /
  800 base images) and smoke measurement (all 4 settings × 3 estimators × 3 seeds ran; nulls negative; transport check max |Δf| 4e-7 / 3e-7 / 6e-8).
- Timing: one full-size MLP selection (3 lr × 1 000 steps on 2 048 + 2 048 pairs) 43.5 s, RFF 0.5 s → ≈ 1 h per fixture expected.  The smoke ran
  slower per step than the full-size probe (node contention on tiny matrices, not understood in detail); because `measure` has no resume, the time
  limit is 16 h per fixture (pessimistic bound ≈ 7.5 h).  Partial JSON is written after every (setting, estimator).
- Permutation floor: B = 200 derangements → minimum p = 1 / 201.
- Submission: `slurm/p149_run.sbatch <dataset> <run>`, one CPU job per fixture (4).
