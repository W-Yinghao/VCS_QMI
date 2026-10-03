# Pre-registration — P118: nested critics — truth-compression check and visual confirmation (v5 NEXT-I-NEST), 2026-10-03 — FROZEN 2026-10-03T12:09:33Z before the full runs (CPU gates 1019950 / 1019951)

Status: FROZEN 2026-10-03T12:09:33Z (main session; owner 2026-10-03: execute the v5 plan).  The main session freezes it (rename to `*_FROZEN_*`, freeze time in the title) before the GPU jobs.  Source:
`VCS_Server_Next_Round_v5_CN.md` §6, `VCS_Results_Review_and_Next_Plan_v5_CN.md` §5.3.  Code: `scripts/p118_common.py` (generator + exact oracle),
`scripts/p118_truth.py` (§6.1), `scripts/p118_visual.py` (§6.2), `scripts/p118_aggregate.py`, `slurm/p118_bundle.sbatch`, `slurm/p118_gate.sbatch`,
`tests/test_p118.py` (6 tests).  `src/vcs_measure/nested.py` is used **unchanged** (asserted by a test against HEAD), so P109 I2 reproduces.  No
encoder training; official test set closed.

## Question
P109 I2: nested_exact cut R_orth ≈ 6× while keeping the increment (colour s 0.1: ΔJ 0.0084 → 0.0083); nested_sampled also cut R_orth but shrank the
increment (0.0026).  Does nested fitting actually estimate **increments** better when the truth is known — or does it only make consistency residuals
small?  A constant-zero critic makes every residual perfect; it must not count as an improvement.

## §6.1 Truth-compression check (Gaussian conditional-binary oracle)
- **Generator** (named streams): Y uniform on 10 classes; P(N = 1 | Y) = ½ ± 0.3 by parity; z ∈ R⁶⁴ = μ_Y + (2N − 1)(δ/2)v + ε, ε ~ N(0, I),
  μ_y ~ N(0, I) fixed, v = 1/√8 on coordinates 0..7 only (coordinates 8..63 independent of N given Y).
- **Chain (coordinate deletion):** B = z (64) ⊇ M = z[:, :8] (drops only irrelevant coordinates → **true increment 0**, S_B = S_M exactly) ⊇
  A = z[:, :4] (drops half the relevant block → **true increment S_M − S_A > 0**, from two oracle S values; additivity not assumed).
- **Exact oracle:** η_K = (r − 1)/(r + 1) with r = p(z_K | n, y)/p(z_K | y) in closed form (test against a brute-force density ratio, 1e-10);
  S_K on a 400 000-unit truth sample.
- **Pre-named conditions:** null δ = 0 (all S = 0); weak δ = 1.0 (S_M ≈ 0.036, true M→A ≈ 0.017); moderate δ = 2.0 (S_M ≈ 0.109, true M→A ≈ 0.044).
  Chosen from the oracle only, before any fit.
- **Per repeat** (5 repeats; fresh FIT / EVAL / POOL, n = 3 000 each, as P102 / I2; 300 AdamW steps; VAL-picked linear / MLP Y-aware critics):
  objectives VCS (−J) and matched JS; variants indep_sampled, indep_exact, nested_sampled (A independent; M = frozen A + zero-initialised residual;
  B = frozen M + residual), nested_exact, **zero** (T ≡ 0).  Each read out with the sampled and with the exact Q term.
- **Endpoints:** primary — increment errors (B→M vs 0, M→A vs oracle, B→A) as mean and RMSE over repeats; posterior MSE of B, M and A vs the exact η
  of each set.  Auxiliary — R_orth of the chain, ordering violations (J_fine < J_coarse), null-condition floor.
- **Frozen rule** (per objective, vs indep_sampled with the sampled readout = the P102 baseline, in **both** non-null conditions):
  (1) M→A increment-error RMSE ≤ 0.75 × baseline; (2) |mean B→M error| ≤ baseline's; (3) posterior MSE at B and at A ≤ 1.10 × baseline;
  (4) |mean M→A increment| ≥ 0.5 × oracle increment.  All four → **"error reduced"**; (4) failing while R_orth falls → **"shrinkage"**; otherwise
  **"no clear reduction"**.  The zero control is scored by the same rule (it fails (1), (3), (4) by construction) — **a constant-zero critic must
  not count as improvement**, and low R_orth / zero violations alone are never read as improvement.

## §6.2 Visual confirmation (no training; P109 I1 feature caches)
- **Encoders:** A-P3 seeds 1, 2 and tuned SimCLR seeds 1, 2 (P109 / P113 features).  Recipe-VCS seed 0 (P109 I2) is a historical reference only;
  a formal encoder-family comparison would need its seeds completed symmetrically (not part of this unit).
- **Pre-named conditions** (from the seed-0 h detection power of P109 / P113, i.e. not from these seeds' EVAL): **colour s 0.1** and **blur s 0.25**
  at n = 2 000 (seed-0 h power A-P3 0.51 / 0.21, SimCLR 0.09 / 0.09 — non-saturated for both families), plus nulls (`null_label_only`,
  `null_all_planted`).
- **Presence / retention:** the P109 I1 audit runner unchanged (`scripts/p109_i1_audit.py`): power cells R = 100, level cells R = 200, 200 shared
  permutations, layers layer3 / h / z / logits, max-T across layers; new seeds 601–616.  Reported as detected / not detected (rejection rates);
  "not detected" is never written as "no information".
- **Nested increments on the P102 chain** built from cached h: h ⊇ pca64 ⊇ pca16 ⊇ logit16 and main pair h vs logit_h; maps fitted once on a fixed
  PROBE of 4 000 FIT-split base images (same ids for every encoder, never drawn again); per repeat FIT n = 3 000 from the remaining FIT split, EVAL
  3 000 from the EVAL split, POOL from the VAL split (disjoint by base image, asserted); N redrawn per repeat; 5 repeats; same variants / readouts as
  §6.1 plus the zero control.  Cells colour s 0.1, blur s 0.25, null.  No oracle on images — readouts are ΔJ, D_T, r_BA, chain steps, R_orth, violations,
  refit spread, reported per encoder seed (the two seeds of a family are not pooled as independent encoders; repeats are not encoder replicates).
- **Paired prediction changes:** read from the existing P109 effect files of the same encoders (colour s 0.1, blur s 0.25; no recomputation).
  Strong blur damaging class evidence is not called a harmful shortcut.
- Audit-assisted model selection (V3) is not extended.

## Gate and smoke (numbers are code-path evidence, not results)
- CPU gate (job 1019951, gate_rc 0): 21 tests pass (6 P118 + P102 + P109); truth and visual smokes end to end; full-size timing below.  Logs and
  outputs: `reports/P118_GATE/`.
- Local truth smoke (n 400, 30 steps, 2 repeats, CPU, 65 s): runs end to end for all 5 variants × 2 readouts × 2 objectives; zero control gives
  R_orth 0 and B→M error 0 but M→A error = −oracle increment (as designed).
- Local visual smoke (A-P3 seed 1, colour s 0.1 + null, n 400, 30 steps, 2 repeats, CPU, 145 s): end to end.

## Cost
Measured on 16 CPUs (gate): one full-size truth repeat (weak, n 3 000, 300 steps, incl. the 400 000-unit oracle) 128 s; one full-size visual
cell-repeat (A-P3 seed 2, colour s 0.1) 314 s.  → bundle A ≈ 15 × 2 min (truth) + 60 × 5 min (visual) ≈ 5.5 h; bundle B = 16 I1 audit units (500
repeats per encoder × family; P109's measured audit rate ≤ 5 s / repeat) ≈ 5 h.  **Two CPU bundle jobs** (`slurm/p118_lines.txt`), off the saturated
GPU quota; GPU timing not measured (job cancelled while queued).

## Delivery (v5 §8 fields)
```yaml
experiment_family: measurement (nested increments)
protocol_id: P118 (v5 NEXT-I-NEST)
estimator: Y-aware linear / MLP critics (VCS −J and matched JS), independent vs nested, sampled vs exact product term; zero control
evaluation_readout: increment error vs oracle, posterior MSE vs exact eta (synthetic); ΔJ / D_T / R_orth / violations (visual); I1 rejection rates
n_independent_units: synthetic — 5 repeats × n 3000; visual — 4 encoders (2 seeds × 2 families) × 5 repeats × n 3000 base images
status: draft
```

## Decisions at the freeze (main session)
1. CPU partition as drafted: the 8-GPU quota is saturated by 800-epoch SSL runs, these fits are small and CPU-bound, and CPU starts now (GPU speed
   not measured; disclosed).  Normal QOS; never runfill.
2. Conditions, rules and controls as drafted (named before any full run); disclosures above kept.
