# Pre-registration — P83 addendum 1: the limited supplement to the P84 frozen-checkpoint diagnostics (actual gate, dictionary disagreement bounds, converged cosine, ridge member, Jacobians) — 2026-09-28 — FROZEN 2026-09-28T18:15:38Z before GPU compute (D-line CPU gate 1013249 passed; retraining disclosure of D1 accepted by the main session: no P72 adapter was ever saved)

Status: DRAFT (D fork).  The main session freezes it before the GPU job.  Sources: package v2 Server Spec §8.3, §9.1, §4; Research Plan v2 §2.3,
§8.1; supplement §§2.3, 3.1; P83 prereg (frozen 2026-09-28) and P84 report (d3b4d5f).  Results → `outputs/P83_addendum1_supplement/<run>__<ckpt>.json`,
`reports/P84_addendum1_supplement.md`; report `P84_ADDENDUM1_REPORT_<date>.md`.

## Questions (all on the ten P84 checkpoints; nothing re-selected, EVAL read once for reporting)
1. What is the actual mean gate 1 − E_M T² of the training critic at epochs 100 / 400 / 800 (and P5 ep 200), next to 1 − J, which the Plan says
   must not stand in for it (§6.2)?
2. Could *any* convex combination of the P84 dictionary {C0, C1, C2} have gained measurably?  D(w) = diag(G)ᵀw − wᵀGw is the exact gain of the
   mix over the weighted member average, J(T_w) − max_j J(T_j) ≤ D(w); we report D at the P84 weights, max_w D(w) over the simplex (concave QP)
   and the global bound U_dict (range and pair forms), on TUNE with block-bootstrap uncertainty.  Note in advance: the P84 mix put all weight on
   C2 (w = [0, 0, 1]) at every checkpoint where it was fitted, so D(w_P84) = 0 by construction; the informative quantities are max D and U_dict.
3. Does adding the converged two-parameter cosine (L-BFGS on FIT) and the P48 tanh-wrapped ridge change that bound, and what does the extended
   simplex fit give on EVAL relative to its best single member?  (Supplement §3.1; "先补收敛成员＋现有成员的低成本比较".)
4. The converged cosine's trajectory and its (a, b) in the training critic's units versus the 2000-update C0 and the training critic itself
   (P84's caution: the 2000-step C0 and the converged C0 are different objects).
5. Scores and Jacobians |∂T/∂z| of every member on the same EVAL rows, in the training critic's units: are the higher-J refits steeper /
   more saturated critics (Plan §2.3: a steeper frozen critic does not imply a better SSL signal)?
6. Labelled diagnostic: the two-member simplex {converged C0, C2} on TUNE, read on EVAL.

## Inputs
Checkpoints and roles exactly as P84: `P35_vcs_a5_views4_800ep_seed0` (initial, 100, 400, 800), `P41_simclr_views4_800ep_seed0` (same), `P5_vcs_seed0`
(initial, 200); roles FIT / TUNE / SELECT / EVAL = 27k / 6k / 6k / 6k (seed 20260928, hash 2a2c0598… asserted against each P84 JSON); the run's own
augmentation with P84's per-role seeds; FIT pairs K = 8 cyclic (seed 20260929); TUNE / SELECT / EVAL independent blocks (seeds 20260930–32).
P84 JSONs `outputs/P83_estim_frozen_diag/*.json` (mix weights, member J_eval, converged-C0 (a, b)).

## Code
`src/vcs_estim/frozen_supplement.py` (+ driver `scripts/diag_p84_supplement.py`, `slurm/diag_p83_addendum1.sbatch`), tests in `tests/test_diag_v2.py`
(dictionary identity J(T_w) = Σ w_j J(T_j) + D(w) to 1e-12; D(w) ≤ max D ≤ U_dict; G-based bounds equal the row forms of `vcs_estim.kernel_cs`;
ridge stationarity; block-bootstrap bookkeeping).

## Protocol
- **Members.**  C0 / C1 / C2 re-fitted with P84's seeds and budget (Adam 5e-4, 2000 updates, SELECT every 100) — *disclosed*: P84 stored no
  per-member TUNE outputs, so the dictionary geometry needs the refits; the refit EVAL J and selected update are compared with the stored P84
  rows (reproduction record; GPU non-determinism may move the fourth digit).  Converged C0: full-batch L-BFGS on FIT (strong Wolfe, 25 iterations
  per round, ≤ 40 rounds, stop at |ΔJ| < 1e-12), trajectory logged per round; reported in standardised and in raw units (a_eff, b_eff).  Tanh
  ridge: P48 closed form on φ = [z1 ⊙ z2; 1] (129 dims) with FIT moments, λ = 1e-4 × mean diag A, scalar c fitted on FIT (log grid + bounded
  refinement); the raw ridge (no tanh) is listed separately with its |T| > 1 fractions and never enters a bounded dictionary.
- **Geometry on TUNE** (G = E_M T Tᵀ over the 3 000 blocks, E_M = ½ P + ½ Q): D(w_P84), refit simplex weights and D, max_w D(w) (SLSQP on the
  concave objective, uniform + vertex starts), U_dict (min of the range and pair bounds); block bootstrap (500 replicates) for D, max D, U_dict;
  the same for the extended dictionary {C0, C1, C2, C0_conv, tanh_ridge} and for {C0_conv, C2}.
- **EVAL once**: member J with block SE, mix J at the TUNE weights, Σ w_j J_j, D_EVAL(w), the identity residual (must be 0 to float precision),
  the gain over the best single member and whether it is bounded by D_TUNE(w).
- **Gate and Jacobians**: training critic on EVAL: E_M(1 − T²), per-side quantiles, 1 − J, E_M T²; on FIT as well.  For the training critic and every
  member, |∂T/∂z_left|, |∂T/∂z_right| and the gate on the first 2 048 EVAL positive and product rows (same rows for all), plus the ratio to the
  training critic.
- SimCLR checkpoints have no training critic: measurement members only (as in P84).

## Reading rules (pre-stated, descriptive)
- *No combination could have gained*: max_w D(w) on TUNE for the P84 dictionary ≤ 2 × the block SE of the training critic's EVAL J at that
  checkpoint (≈ 0.003–0.005) → the P84 null on mixes is a property of the dictionary, not of the solver.  If the extended dictionary's max D also
  stays below that level, the combination family is closed for these representations (auxiliary C line only; Spec §9.2 last paragraph).
- *A member is worth a seed*: the extended simplex's EVAL gain over its best single member exceeds 2 block SE **and** its D_TUNE ≥ that gain
  (otherwise the gain is EVAL noise by the identity).  This is the only condition under which the C line would ask for more compute.
- *Steeper, not better*: a refit member whose EVAL J exceeds the training critic's while its |∂T/∂z_left| is > 2× the training critic's and its mean
  gate < ½ of it is labelled "gain bought by steepness"; carried into the technical note as the caution against online critic refresh, not as a
  training recommendation (Spec §8.3 last paragraph).
- The actual gate table is reported as is; the 1 − J column is there to show the discrepancy, not to be read.

## Cost
P84 ran 55 s per checkpoint on a GPU (features 14 s).  The supplement adds three refits at P84's budget (≈ 30 s), L-BFGS and the 129-dim solve (< 5 s),
three dictionary blocks with 500-replicate bootstraps (≈ 10 s) and the Jacobians (< 5 s): ≈ 2 min per checkpoint, ≈ 20 min for ten (walltime 4 h; node60 excluded as in P83).

## CPU gate: job **1013249** (shared D-line gate: tests + `--smoke --cpu` on `P35…:epoch_800` with 1 024 / 256 / 256 / 256 identities, 100 updates).
Results: see the section below.

## Launch line (main session, after freezing)
```
cd /home/infres/yinwang/CS_QMI/ssl_pilot && sbatch --parsable slurm/diag_p83_addendum1.sbatch
```

## Delivery block (Spec §13.2)
```yaml
experiment_family: diagnostic
protocol_id: P83_addendum1
source_commit: <commit>
source_basis: supplement_af7172c_plus_v2
estimator: vcs_neural (training critic) + refitted C0 / C1 / C2 / converged C0 / tanh ridge (measurement critics)
estimand: S (no oracle: S_truth absent as in P84)
evaluation_readout: J_common (block EVAL), D(w), max D, U_dict, gates, Jacobians
loss_scale: minus_J
reference_measure: mixture_equal
critic_class: CosineCritic tanh(a<z1,z2>+b) (training); C0 a s + b, C1 rank-16 bilinear, C2 MLP 256-256, converged C0, tanh(c w'[z1*z2;1])
gradient_routing: measurement critics only (encoder / projector frozen, eval mode)
n_independent_units: FIT 27000, TUNE 6000, SELECT 6000, EVAL 6000 identities
n_positive_pairs / n_negative_pairs: FIT 27000 / 216000; TUNE, EVAL 3000 blocks each
split_manifest_hash: c35d7cd3… ; roles_hash 2a2c0598…
noise_target_kind: none
fit_seconds / evaluation_seconds: measured, in the JSON
status: <actual>
```

## Disclosures
- Members are re-fitted (P84 stored no TUNE outputs); reproduction against the stored P84 rows is recorded per checkpoint.
- The extended dictionary and the {C0_conv, C2} pair are new fits; their EVAL J is a single read, not a selection.  λ of the ridge is fixed in
  advance (1e-4 × mean diag A, P48's middle value); c is fitted on FIT only.
- SLSQP is a convex solver for the concave max-D problem; the KKT gap of the simplex fit and the bound checks (D_fit ≤ max D ≤ U_dict) are written to the JSON.

## CPU gate results (job 1013249, nodecpu12, 2026-09-28T18:08–18:09 UTC, 100 s wall; `slurm_logs/diag_cpu_gate_1013249.out`)
- `tests/test_diag_v2.py`: 9 passed (gate identities, posterior-logit gradients and swap ratio = gate, F_same bookkeeping, SimCLR weights / own-row
  gradient, bootstrap-by-image integrity, dictionary identity and bounds, displacement metrics, ridge stationarity, cancellation index).
- D1 smoke (animal, n = 1 024, 2 epochs, m ∈ {0, 0.4}, four methods, 19 s CPU): QC sentinel max |Δθ| instrumented vs original = 0.00e+00 in all 8 cells;
  early / final checkpoints, trajectories, JSON and markdown written (`outputs/P93_D_diag/smoke/`).  Reproduction columns differ from P72 as expected at
  1 024 images / 2 epochs (e.g. SRC R@1 0.49–0.54 vs P72's 0.08–0.19) — code path evidence only.
- D2 smoke (one 64-image batch, 6 view pairs, VCS ep 800 and SimCLR ep 800, 2 s each): VCS F_same_z 0.275 (count share 0.105, mass ratio 0.60),
  SimCLR-native F_same_z 0.224 (count share 0.108, mass ratio 0.23); cancellation index 0.99 / 0.77; gate same / diff 0.185 / 0.032 (VCS) — R = 64 images,
  not read for any verdict.
- Addendum smoke (P35 ep 800, roles 1 024 / 256 / 256 / 256, 100 updates, 31 s CPU): identity residual J(T_w) − Σ w_j J_j − D_EVAL(w) = 0 to float
  precision in all three dictionaries; bound checks D_fit ≤ max D ≤ U_dict true; D(w_P84) = 0 (w = [0, 0, 1]); the 100-update members are budget-limited
  as in P83's smoke, so the smoke's extended-dictionary numbers are not indicative.
Nothing in the design was changed after the gate.
