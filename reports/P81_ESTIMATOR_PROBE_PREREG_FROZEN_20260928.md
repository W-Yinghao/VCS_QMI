# Pre-registration — P81: estimator package v1, stages 0–2 first batch (Gaussian probe, seed 0) (2026-09-28) — FROZEN before GPU compute

Source: collaborator package `CS_QMI/VCS_QMI_Estimator_Research_and_Server_v1.zip` (sha256 48a76436…; copy without PDFs in
`estimator_package_v1/`), server spec §5–7, §14 "首批范围".  Owner go 2026-09-28: "先做压缩包中的事情…优先压缩包中的".  The package supersedes
the E line of the next-round draft (spec §12); the E-fork code (`src/vcs_estim/{data,critics,estimators,oracle,staircase}.py`, `scripts/estim_*`)
is not used by this unit.  Code: 79f28e2 (`src/vcs_estim/run.py` and modules; §15 fields).  Results → `P82_*`.

## Stage 0 (done before this freeze)
CPU job 1013081: all ten §5 checks pass in float64 at floating-point tolerance (risk identity, finite gap, residual step incl. B = 0, convex
mixture, simplex weights, gradient scale at f = 0, RuLSIF α = ½ affine identity with an unpenalised constant, nested-information coarsening
identity, sample roles (synthetic streams; 27k/6k/6k/6k image identity split), noise independence) + Gaussian log-ratio vs explicit densities,
CQ contains the oracle, JS mixture log-sum-exp/KKT, the package's own validator reproduces (exit 0).  Probe smoke (I = 4, small sizes, 200 updates)
runs end to end; S − J_eval ≈ posterior MSE for every model (consistency check).

## Unit
Gaussian, d = 20, I ∈ {0.5, 4, 8} nats (ρ = √(1 − e^{−2I/d})), seed 0.  Roles per distribution: FIT 4096 / TUNE 1024 / SELECT 1024 / EVAL 32768 /
TRUTH 300000, P and Q and each role from independent seeded streams; independent-unit negatives.  Per condition:
- singles: C0 (a xᵀy + b), C1 (rank-16 bilinear + b), C2 (MLP 2×256), CQ (diagnostic class containing the oracle), each trained with VCS (−J) and
  with the matched JS loss (softplus(−2f) / softplus(2f)); Adam 5e-4, wd 0, batch 256 + 256, 2000 updates, SELECT native risk every 100 updates;
- restart control: VCS C2 with two more initialisations, best-of-3 by SELECT;
- VCS dictionary {C0, C1, C2}: exact simplex QP on TUNE (package reference core), weights applied unchanged to EVAL;
- JS dictionary control: native balanced log-loss simplex weights on TUNE (SLSQP, KKT-checked), log-sum-exp mixture;
- one VCS residual candidate: base = SELECT-best VCS single of {C0, C1, C2}; U = C2 structure trained on −J((1−λ₀)T₀ + λ₀U), λ₀ = 0.5,
  checkpoint by SELECT; λ = clip(Â/B̂, 0, 1) on TUNE;
- §6.4: dJ/dρ of every fixed VCS/JS single through the generator (TRUTH base noise), against oracle central differences (δ = 1e-2, 3e-3, 1e-3,
  common random numbers) and the envelope form.
Capacity-matched single models are deferred per spec §7.3 until a candidate shows a stable gain.

## Reported (no pre-set success thresholds — spec §12 removes them for this batch)
Per model on EVAL: J_eval, S_truth ± se, signed error, posterior MSE, excess-to-Bayes-risk, gates, saturation, selected update, fit/eval time,
parameters; FIT-side J and posterior MSE (generalisation gap); combination weights / λ / A / B.  Reading questions (spec §16): where the error
sits (CQ vs C2 = function class vs optimisation; FIT vs EVAL = generalisation; gates = saturation); whether the mix / residual reduce EVAL
posterior MSE relative to the SELECT-best single *and* the best-of-3 restart; whether the JS dictionary does the same.  A mix or residual gain
smaller than the EVAL standard error of J is written as "no measurable gain".  No seed, checkpoint or weight is changed after seeing EVAL.

## QC
Oracle J on EVAL within 3 se of S_truth; |J_oracle_truth − S_truth| within 3 se; simplex KKT gap < 1e-8; JS mixture KKT < 1e-6; oracle FD
derivatives agree across δ within 3 se and with the envelope form; every row status completed or its failure saved.

## Compute
One GPU job, three conditions sequentially (≈ 15 trainings × 2000 updates each; minutes per condition).  Unit times are recorded and used for
the §14 step-4 cost estimate of the next batch (3 seeds, non-Gaussian, d ∈ {2, 20, 50}, observation scale), which is not launched without the owner.
