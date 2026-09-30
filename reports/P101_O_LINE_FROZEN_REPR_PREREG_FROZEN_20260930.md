# Pre-registration — P101: the v2 O line on frozen SSL representations (observation-noise calibration to a J_proxy scale), 2026-09-30 — FROZEN 2026-09-30T19:58:11Z before GPU compute (CPU gate 1015840: 6 tests pass, smokes end to end)

Status: FROZEN 2026-09-30T19:58:11Z (main session).  The main session freezes it before the GPU job.  Owner go 2026-09-30 ("除了imagenet的先不提交，后续都可以提交").  Sources: Server
Spec v2 §10 (O line: §10.1 noise definition and fields, §10.2 FIT-internal calibration, §10.3 subspaces and evaluation), Research Plan v2 §9.2; the
P84 checkpoints and roles (`P83_FROZEN_DIAG_PREREG_FROZEN_20260928.md`); the converged cosine of the P84 addendum.  The online SSL counterpart (noise at the
critic input during training) is P95's noise family; this unit only measures on frozen representations.

## Question
Does observation noise, calibrated on fitting data to fixed proxy-J levels, change how finely a fixed critic class resolves representations taken at
different training stages (VCS recipe epochs 100 / 400 / 800; SimCLR same), and at what cost in class information?  Noise changes the observed variable
(the target becomes S_σ, not S): nothing here is read as a more accurate estimate of the noise-free S.

## Checkpoints and roles
`P35_vcs_a5_views4_800ep_seed0` epochs 100 / 400 / 800 and `P41_simclr_views4_800ep_seed0` epochs 100 / 400 / 800 (six P84 checkpoints).  P84 roles
(seed 20260928): FIT 27k split at identity level into FIT-CRITIC 13.5k / FIT-CAL 13.5k (seed 20261001); SELECT 6k; EVAL 6k; TUNE unused; all disjoint
(asserted).  Two augmented views per identity with the P84 view seeds, recipe augmentation, eval() on deep copies.  z = L2-normalised projector output (d 128).

## Noise (Spec §10.1)
u = z + τ ε / √d on both sides of every pair, independent per row, no re-normalisation; subspace variant u = z + (τ / √r) B ε_r with B the FIT-CRITIC PCA
basis (90 % variance, r ≤ 64), computed once from noise-free z and fixed for every τ.  Fields per setting: noise_total_rms = τ, noise_coordinate_sd
(τ / √d or τ / √r), d, r, calibration_target = "J_proxy".  Noise draws are keyed by (role, τ, mode): every checkpoint sees the same realisation.

## Calibration (Spec §10.2; FIT-internal only)
Calibration critic: the converged two-parameter cosine C0 (full-batch L-BFGS on noisy FIT-CRITIC pairs, K = 8 cyclic shifts, FIT-standardised inner
product), refitted for every τ; J_proxy(τ) = its block J on noisy FIT-CAL blocks.  Targets {0.95, 0.85, 0.70, 0.50}; grid τ ∈ {0, 0.1, 0.3, 0.6, 1.0,
1.5, 2.0}; within the bracketing grid interval ≤ 10 bisection steps until |J_proxy − target| ≤ 0.03; targets outside the scanned J range are recorded
unreachable (grid not extended; no look at EVAL).  The raw curve is kept (no monotone smoothing).  The **common τ** of each target is the one calibrated
on the reference checkpoint (VCS epoch 800, the recipe endpoint) and is applied unchanged to all six checkpoints; each checkpoint's own τ*(target) is also
reported (descriptive: the noise level that brings it to the same J_proxy).

## Measurement (Spec §10.3)
At τ = 0 and at each reachable common τ (isotropic and PCA): VCS and matched-JS critics of classes C0 (standardised cosine) and C2 (MLP) trained on noisy
FIT-CRITIC pairs with the P84 budget (plus a separate row `vcs_C0conv`: the calibration class fitted to convergence, because the 2000-update C0 is
budget-limited on these features — P84) (Adam 5e-4, 2000 updates, batch 256, SELECT every 100), evaluated once on noisy EVAL blocks: block J ± SE (T = tanh f
for both losses), matched-JS native value, and kNN top-1 of noisy view-1 z (EVAL block anchors vs the FIT-CRITIC bank; k 200, T 0.1; labels used for this
diagnostic only).  S, η and posterior MSE are absent (no oracle).

## Reading (pre-stated, descriptive)
Resolution of adjacent checkpoints of one method = |ΔJ| / √(se₁² + se₂²) (EVAL block SEs), per critic, at τ = 0 and at each common τ.
**"Noise improves resolution"** only if, for some calibrated target and some critic class, the resolution of an adjacent pair increases by ≥ 1.5× relative
to τ = 0 **and** the kNN top-1 of every checkpoint in that pair drops by < 2.0 points; otherwise "no resolution gain".  Isotropic and PCA are read separately.
Reported without thresholds: τ*(target) per checkpoint, unreachable targets, VCS vs JS differences at equal τ.

## Code
`src/vcs_estim/observation_frozen.py` (new; imports `frozen`, `frozen_supplement.fit_c0_converged`, `pairing` read-only), `scripts/o_line_frozen.py`
(launcher: reference first, then the five others, then the aggregate), tests `tests/test_p98_p101.py` (dot-product variance (2τ² + τ⁴)/d by MC, calibration
stops at tolerance and never extends the grid, fixed PCA basis, noise inside the basis with total RMS τ, common random numbers, role disjointness).
Gate: CPU job 1015840 — tests 6/6 pass; smoke (768 / 256 / 256 / 256 identities, 60 updates) on VCS epochs 800 (reference) and 400 ran end to end;
calibration on the reference smoke: iso grid J_proxy 0.985 / 0.983 / 0.931 / 0.579 / 0.244 / 0.045 / 0.076 over τ 0–2, targets 0.95 / 0.85 / 0.70 / 0.50 → τ 0.30 / 0.375 /
0.544 / 0.80 (all reachable, ≤ 4 refinements); PCA r = 28 (91.2 %).  The aggregate failed once on a zero SE (constant 60-update critics) → guarded (nan),
rerun OK.  A login-node re-smoke after adding the converged-C0 row: `vcs_C0conv` EVAL J 0.997 / 0.959 / 0.856 at τ 0 / 0.30 / 0.375 (tracks the targets);
the 60-update C0 / C2 rows stay ≈ 0 at smoke budget (expected).

## Cost
Per checkpoint: features ≈ 20 s; calibration 2 modes × (7 grid + ≤ 40 refinement) L-BFGS fits on 13.5k × 9 pairs (≈ 1–3 s each on GPU) ≈ 3 min;
measurement ≤ 9 settings × (4 critics × 2000 updates ≈ 5 s each + one converged C0 ≈ 5–20 s) ≈ 5 min → ≈ 10 min per checkpoint, ≈ 1–1.5 h
for six (the CPU smoke took 8 min for one tiny checkpoint, dominated by L-BFGS on CPU).  One job, `slurm/p101_o_line.sbatch`
(normal QOS; RTX6000PRO,H100,A100,L40S; excl. node51,node60; 6 h wall).

## Delivery (Spec v2 §13.2)
```yaml
experiment_family: diagnostic (O line)
protocol_id: P101_O_line_frozen
estimator: vcs_neural (C0, C2) | js (C0, C2); calibration critic converged C0
estimand: S_sigma (noisy observation); J_proxy for calibration
evaluation_readout: J_common on noisy EVAL blocks, JS native, kNN of noisy z, tau*(target)
noise_target_kind: J_proxy
noise_tau: per target (common, calibrated on the reference checkpoint)
noise_sigma_coordinate: tau / sqrt(d) (iso) | tau / sqrt(r) (PCA)
n_independent_units: 6 checkpoints x (13.5k FIT-CRITIC / 13.5k FIT-CAL / 6k SELECT / 6k EVAL identities)
status: <status>
```
