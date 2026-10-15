# Pre-registration DRAFT — P158: nested increments with truth on the joint-vs-product chain (project map 2026-10-14 §11 B.2), 2026-10-15

Status: DRAFT (P158 fork).  The main session freezes it (rename to `*_FROZEN_*`, freeze time in the title) before the bundle job.  Go-ahead
field: autonomous unit under the owner's idle-GPU authorisation of 2026-10-11 ("当你看到GPU有空位时，实验不需要我批准…你觉得有用的就自主提交") and the
project map's next-step list (`reports/PROJECT_MAP_20261014.md` §11 B, item 2: "在 X11 的 A2 链上做有真值的嵌套压缩…报告 nested_exact 的 L² 可加性与
JS 的 Bregman 增量并列"); no official data, no test-set read, CPU partition (normal QOS).  Sources: `reports/P118_V5_NEST_PREREG_FROZEN_20261003.md`
§6.1 (design and frozen rule reused verbatim where applicable), `reports/P118_V5_NEST_REPORT_20261003.md` (P118's result: VCS nested_exact "error
reduced", JS failing one clause — the finding this unit tests in the joint-vs-product setting), `/home/infres/yinwang/CS_QMI/ssl_pilot_v7/scripts/
v7x/x11_anchors.py` (anchor A2 generator and analytic log ratio), package v2 §5.1 (irrelevant dimensions leave S unchanged) and §8.2 / Spec §11.3
(informative self-checks r_BA, R_orth).  Code: `scripts/p158_common.py` (generator, oracle, truths, readout helpers), `src/vcs_measure/nested_joint.py`
(joint-input nested critics and the P102 / P118 fitter; `src/vcs_measure/nested.py` is **unchanged**, asserted by a test), `scripts/p158_truth.py`
(runner), `scripts/p158_aggregate.py` (tables + rule), `tests/test_p158.py` (9 tests), `slurm/p158_gate.sbatch`, `slurm/p158_bundle.sbatch`.

## Question
P118 showed, on a *conditional-binary* Gaussian chain with an exact oracle, that nested fitting with the exact product term reduces the VCS increment
error ("error reduced": all four clauses) while matched JS failed one clause.  Does the same hold in the setting the estimator benchmark actually uses
— joint (P) vs product-of-marginals (Q) with continuous inputs, where **no exact product term exists** and every readout is sampled — and how do the
VCS L² increment (ΔJ ≈ D_T = E_M (T_B − T_A)², additive along a chain) and the matched-JS Bregman increment (ΔG ≈ D_JS = 2 E_M KL(Ber(q_B) ‖ Ber(q_A)))
compare against their own truths?  A constant-zero critic makes every consistency residual perfect; it must not count as an improvement.

## Generator and chain (`p158_common.py`; X11 anchor A2 with irrelevant coordinates)
- 10 signal pairs (x_j, y_j) ~ N(0, [[1, ρ], [ρ, 1]]) under P; 10 irrelevant coordinates per side, independent N(0, 1).  x = [x_sig (10), x_irr (10)],
  y likewise (20 per side).  Q = product of the marginals (both N(0, I₂₀)); realised in every draw as (x_i, y_{π(i)}) with one non-zero cyclic shift π
  (exact product law for i.i.d. units; negative construction named: **cyclic1**).
- Chain (deterministic coordinate deletion): **B** = all 20 + 20 ⊇ **M** = the 10 signal pairs (drops only irrelevant coordinates → true increment 0,
  S_B = S_M and G_B = G_M exactly: the irrelevant factors cancel in the density ratio) ⊇ **A** = the first 5 signal pairs (true increment > 0).
- Exact oracle for the first k signal pairs: log r_k = Σ_{j<k} [−½ log(1 − ρ²) − (u² − 2ρuv + v²)/(2(1 − ρ²)) + (u² + v²)/2] (tested against
  scipy's multivariate-normal log-densities, 1e-10); f*_k = log r_k / 2, η_k = tanh f*_k, q*_k = σ(2 f*_k).  The quadratic-feature critic class
  (features x, y, x², y², x·y per coordinate) contains f*_k exactly (tested).
- Truths by Monte-Carlo on 400 000 P units + 400 000 Q units (SEs recorded): S_k = E_M η_k² (VCS), G_k = E_P log(2q*) + E_Q log(2(1 − q*)) (JS value,
  = 2 JSD on the kept set); per step the L² increment E_M (η_B − η_A)² and the Bregman increment 2 E_M KL(Ber(q*_B) ‖ Ber(q*_A)), with their gaps to
  S_B − S_A and G_B − G_A reported (population identities; the finite-sample gaps are O(1/√n) and zero exactly for B→M).
- **Pre-named conditions (from the oracle only, before any fit; targets S_M ≈ 0.04 / 0.11 as in P118):** null ρ = 0 (all S = G = 0); weak ρ = 0.13;
  moderate ρ = 0.22.  Oracle values (gate job 1041310, `reports/P158_GATE/oracle_values.{txt,json}`, 400 000 P + 400 000 Q units):
  **weak**: S_B = S_M = 0.04096 (MC se 6.7e-5), S_A = 0.02092; G_B = G_M = 0.04281, G_A = 0.02188; true M→A increment S 0.02004 (L² form 0.02008),
  G 0.02093 (Bregman form 0.02075, se 4.0e-5).  **moderate**: S_B = S_M = 0.11144 (se 1.6e-4), S_A = 0.05848; G_B = G_M = 0.11823, G_A = 0.06108;
  true M→A increment S 0.05296 (L² 0.05298), G 0.05715 (Bregman 0.05744, se 1.0e-4).  **null**: all S = G = 0 exactly.  B→M truth 0 in every condition.

## Fitting (per repeat; 5 repeats; n = 3 000 P units for FIT and 3 000 for EVAL, fresh draws; one cyclic shift per draw)
- Objectives: VCS (−J) and matched JS (balanced logistic, softplus(−2f) on P + softplus(2f) on Q, same critic and selection).
- Critic families, VAL-picked per set by the common squared score J on the 20 % VAL split of FIT (P102 / P118 fitter: AdamW 1e-3, wd 1e-2, full batch,
  300 steps, selection every 10 steps): **quad** (linear in [x, y, x², y², x·y] + bias; contains the oracle) and **mlp** (`vcs_estim.critics.JointMLP`,
  concat → 256 → 256 → 1).
- Variants: **indep** (every set fitted independently; the P102 / P118 baseline), **nested** (A independent; M = frozen A∘map + zero-initialised residual
  of the same family; B = frozen M∘map + residual; step 0 = the coarse model is a selectable candidate), **zero** (T ≡ 0 on every set; control).
- No exact variants: with continuous inputs the Q expectation cannot be enumerated (P118's `*_exact` variants have no counterpart here — disclosed);
  fitting and every readout use the sampled Q pairs.  Nothing on EVAL is clipped, made monotone or orthogonalised; EVAL selects nothing.

## Endpoints (EVAL; per objective × variant; both readouts of the same critics)
- **Common readout** (on T = tanh f; truth = S differences): J per set, increments ΔJ (B→M, M→A, B→A) and their errors vs the oracle (mean, RMSE over
  repeats), D_T, r_BA = ΔJ − D_T, R_orth of the chain, ordering violations (J_fine < J_coarse), posterior MSE ½E_P(T − η)² + ½E_Q(T − η)² of B, M, A vs
  the exact η of each set.
- **JS native readout** (on logits; truth = G differences): G per set, ΔG and errors, Bregman D_JS, r = ΔG − D_JS, R_orth_js, violations.
- **Self-checks with uncertainty** (Spec v2 §11.3): per repeat a paired bootstrap over EVAL units (1 000 resamples, same index for every critic's P and Q
  outputs) gives 95 % CIs of r_BA (M→A, B→M) and R_orth for both readouts; the aggregate counts the repeats whose CI contains 0 (consistency evidence,
  not proof of correctness).  The telescoping sum of value differences is an algebraic identity and is not used as a check.
- Auxiliary: null-condition floor (|mean increment errors| at ρ = 0), picked family and best step per fit.

## Frozen rule (P118 §6.1, per objective **and readout**, vs the indep variant of the same objective / readout, in BOTH non-null conditions)
(1) M→A increment-error RMSE ≤ 0.75 × baseline; (2) |mean B→M error| ≤ baseline's; (3) posterior MSE at B and at A ≤ 1.10 × baseline (readout-
independent); (4) |mean M→A increment| ≥ 0.5 × the oracle increment.  All four → **"error reduced"**; (4) failing while |R_orth| falls → **"shrinkage"**;
otherwise **"no clear reduction"**.  The zero control is scored by the same rule (it fails (1) and (4) by construction; a test asserts it).  Low R_orth
or zero violations alone are never read as improvement.  Pre-committed readings: VCS nested "error reduced" on the common readout *and* JS nested
"error reduced" on its native readout → the P118 asymmetry does not replicate (nesting helps both scoring rules); VCS "error reduced" and JS not →
P118's finding replicates in the joint-vs-product setting; neither → nested fitting without an exact product term gives no clear reduction (the
sampled Q term dominates); any other pattern reported as it lands.  No claim about natural data.

## Gate and smoke (code-path evidence, not results)
- CPU gate job 1041310 (nodecpu08, gate_rc 0; `reports/P158_GATE/`): **15 / 15 tests** (9 P158 + 6 P118, nested.py unchanged); `--smoke` runner end to
  end for all three conditions (n 400, 2 repeats, 30 steps, 50 000-unit oracle; 42–50 s per condition) and the aggregate on it; oracle values above;
  full-size timing below.  Smoke observation (code path, not a result): the zero control gives R_orth 0, B→M error 0 and M→A error = −(oracle
  increment) as designed; when the nested B fit selects step 0 (residual g = 0) the B→M increment, its error and R_orth are **exactly 0 by
  construction** (T_B ≡ T_M) — clause (2) is then met trivially, so the aggregate also reports the picked family and best step of every fit, and the
  M→A clauses (1), (3), (4) carry the verdict.

## Cost
One bundle job on the CPU partition (`slurm/p158_bundle.sbatch`: null, weak, moderate sequentially with skip-if-done, then the aggregate); per full repeat
2 objectives × (3 + 2 sets) × 2 families × 300 full-batch steps on n 2 400 + 1 000-resample bootstraps: **measured 96 s wall (4 m 41 s CPU) per full
repeat on 8 CPUs including the 400 000-unit oracle** (gate `truth_timing.txt`) → 3 conditions × 5 repeats ≈ 25–30 min; bundle `--time 02:00:00`.
GPU not needed (disclosed: not measured on GPU).

## Delivery (v5 §8 fields)
```yaml
experiment_family: measurement (nested increments, synthetic truth)
protocol_id: P158 (project map 2026-10-14 §11 B.2)
estimator: quad / MLP joint critics (VCS −J and matched JS), independent vs nested, sampled product term only; zero control
estimand: S (mixture, VCS) and G = 2 JSD (JS) of nested sets; L2 and Bregman increments
evaluation_readout: common squared score and JS native score; increment error vs oracle; posterior MSE vs exact eta; r_BA / R_orth with paired bootstrap
negatives: cyclic1 (one non-zero cyclic shift per draw)
n_independent_units: 3 conditions × 5 repeats × n 3000 (FIT) + 3000 (EVAL)
status: draft
```

## Decisions for the freeze (main session)
1. CPU partition, normal QOS, one bundle job (GPU quota kept for the running VL / V8 chains).
2. Conditions, rule and controls as drafted (named before any full run); the oracle values and gate numbers are inserted verbatim from the gate output.
