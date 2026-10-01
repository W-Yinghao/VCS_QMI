# Pre-registration — P108: package v4 module E (E1 readouts on identical critics, E2 rotation and error–cost curves, E3 cost fields), 2026-10-01 — FROZEN 2026-10-01T17:35:47Z before GPU compute (CPU smoke 1017241; 13 new + 41 existing estimator tests green)

Status: FROZEN 2026-10-01T17:35:47Z (main session; owner 2026-10-01 shared the v4 package; standing go "除了imagenet的先不提交，后续都可以提交").  The main session freezes it (rename to `*_FROZEN_*`, freeze time in the title) before the GPU jobs.  Source:
`VCS_Next_Experiments_v4_Package.zip` → `VCS_Next_Experiment_Plan_v4_CN.md` §E (E1, E2, E3), `experiments_v4_logical.yaml` (analysis_modules E1, E2).
Code: `src/vcs_estim/p108.py` (new; imports the P85 modules unchanged), `scripts/p108_estim.py`, `scripts/p108_aggregate.py`, `slurm/p108_estim.sbatch`,
`tests/test_p108.py`.  No P85 / P86 file is modified (`git status` of `src/vcs_estim`: only `p108.py` is new; the stage-0 and v2 estimator suites pass unchanged).

## Questions
- **E1.** For one and the same fitted critic T, evaluated on the same independent EVAL units, how do the variational readout
  Ĵ(T) = mean_P(T − T²/2) + mean_Q(−T − T²/2) and the squared plug-in Ŝ_plug(T) = ½ mean_P T² + ½ mean_Q T² compare as estimators of S
  (signed bias, RMSE over independent FIT draws, evaluation SE), across FIT sizes and solver budgets, and does the answer depend on whether T was
  fitted with the VCS loss or with the JS loss (formula vs fitting-loss split)?
- **E2.** Does the learned estimator's advantage in high dimension (P86) depend on the coordinate system?  Same data under fixed per-side orthogonal
  rotations; error vs independent samples and vs updates / seconds, with tuning and chosen-model cost.
- **E3.** Cost: time and memory at fixed budget points, the budget needed for a fixed tolerance, stopping-point and refit spread.

## Theory being checked (population, M = (P + Q)/2, T = eta + e)
J(T) − S = −E_M e²  (second order in e);  S_plug(T) − S = 2 E_M[eta e] + E_M e²  (first order unless E_M[eta e] = 0).
For T_t = (1 − t) eta + t U: J bias = −t² E_M (U − eta)²; plug-in bias = 2t E_M[eta (U − eta)] + t² E_M (U − eta)²; at U = 0: −t² S and (−2t + t²) S.
On a finite sample the plug-in identity is algebraic (exact on the sample relative to Ŝ(eta)); the J identity uses E_P T − E_Q T = 2 E_M[eta T] and
holds in expectation only — its sample discrepancy D_J is reported with its SE.  Equal bias order does not imply smaller RMSE in every finite cell:
both bias and variance are reported; J may be negative on EVAL and is kept as is.

## Design (all data from the P85 generators, `benchmark.build_roles`: FIT = N, TUNE = SELECT = max(64, N/4), EVAL = 32 768, TRUTH = 200 000 per distribution)
Conditions, named before any run:
| id | setting | d_signal / d_total | generator MI (nats) | role |
|---|---|---|---|---|
| C1_gauss_mid | gaussian | 20 / 20 | 4 (S ≈ 0.80) | moderate Gaussian (P85 I_MID) |
| C2_gauss_pad | gaussian | 2 / 100 | 1.5 (S ≈ 0.54) | high-dimensional, 98 irrelevant padding coordinates (P85 pad axis, largest d) |
| C3_xor | xor_mixture | 10 pairs | 4 | nonlinear (sign-pattern) dependence |

- **E1(a) mechanism** (no fitting): on each condition's TRUTH sample, T_t = (1 − t) eta + t U, t ∈ {0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1}, U ∈ {0, U_fix},
  U_fix = 0.8·tanh(w_x·x − w_y·y + ½(w_x·x)(w_y·y) + c) with (w, c) from a named stream independent of every data stream.  Readouts, identity-predicted
  biases, log–log slopes in t ≤ 0.1, D_J z-scores, plug-in identity residual.  Exact float64 identities on a discrete toy are unit tests.
- **E1(b) fits**: 3 conditions × N ∈ {256, 1024, 4096, 16 384} × FIT seeds 0–4 (independent FIT / TUNE / SELECT / EVAL draws per seed) = 60 cells.
  Critic class: the P85 JointMLP (concat → 256 → 256 → 1), product negatives (independent Q units), the P85 trainer `train_neural` unchanged
  (Adam, batch 256 or N, SELECT native risk every 100 updates, best state kept), lr grid {1e-4, 5e-4, 2e-3} chosen by SELECT risk (all rows kept).
  Solver budgets 250 / 1 000 / 4 000 updates (same seed → each budget is the prefix of the longer run, i.e. early stopping restricted to ≤ b).
  Fit losses: VCS (T = tanh f) and matched JS (Deep-InfoMax logit, T = tanh(f/2)).  Both readouts of each T on the same EVAL units; posterior MSE
  E_M (T − eta)²; evaluation SE of each readout; fit seconds (chosen lr) and total tuning seconds (all lrs).
  References at budget 4 000 (lr by SELECT): InfoNCE (in-batch), NWJ, MINE / DV, SMILE (product), each against its OWN truth (MI) as relative error;
  no common raw number with S.
- **E2**: C1 and C2 × {original, rotated} × N ∈ {1 024, 4 096, 16 384} × seeds 0–2 = 36 cells.  Rotation: x → R_x x, y → R_y y, R_x ≠ R_y Haar-orthogonal
  from named streams, fixed before splitting; applied to FIT / TUNE / SELECT / EVAL (TRUTH and eta use the original coordinates = the generator's
  inverse map).  Methods with the same selection programme in both systems: neural VCS and JS (as E1, budgets 250 / 1 000 / 4 000), S-KDE (bandwidth
  multiple of the FIT median by max J_kernel on TUNE; both readouts), RFF S-kernel (m = 1 024, bandwidth multiple ∈ {0.5, 1, 2} by SELECT risk, lr 5e-4,
  same budgets).  Error vs N (independent samples) and vs updates / seconds; tuning total and chosen-model seconds; peak CUDA memory.
  Expected by construction (stated before the runs): isotropic Gaussian kernels are rotation-invariant (pairwise distances are preserved; RFF frequencies
  are isotropic in distribution), so a kernel-side change is sampling noise; the MLP's coordinate-wise ReLU is not invariant.
- **E3** (no extra runs): budget point reaching seed-mean |Ĵ − S| ≤ 0.02 (pre-declared tolerance), refit SD of Ĵ over FIT seeds, SD of the selected update,
  fit / eval seconds and peak memory per method.  P105's VCS closed-form vs JS L-BFGS timings are not recorded by the P105 runner (gap disclosed; a
  converged L-BFGS solution is a numerical solution, not a closed form).

## Pre-stated reading (descriptive)
1. E1(a): the identity-predicted J bias has log–log slope 2 in t for both U; the plug-in slope is ≈ 1 when E_M[eta(U − eta)] ≠ 0 (and exactly −2t + t² at
   U = 0); |D_J z| beyond ±4 at any t would flag an implementation error (not a finding).
2. E1(b): per (condition, N, budget, fit loss): signed bias and RMSE of Ĵ and Ŝ_plug over the 5 seeds, side by side.  Statements are limited to
   "Ĵ has smaller |bias| / RMSE than Ŝ_plug in k of the cells" with the cells listed; no single "best readout" across cells.  The JS-fitted T gives the
   formula-vs-loss split: if Ĵ beats Ŝ_plug for both fit losses, the advantage is a property of the readout formula.
3. E2: rotation effect = rotated − original per (method, N, budget), mean over seeds, with the seed SD; a method is called "rotation-sensitive" at a
   cell only if |effect| exceeds 2 × the pooled seed SE.  Curves (error vs N, vs seconds) are reported as such; no efficiency claim from one fit's runtime.
4. Not claimed: anything about image data, SSL, the bias of any readout outside these generators; reference estimators are not ranked against S.

## Smoke (disclosed)
CPU job 1017241 (`reports/P108_GATE_smoke/`): mechanism on all 3 conditions (TRUTH 5 000 per distribution): J slope 2.000, plug-in slope 0.966–0.980
(t ≤ 0.1), max |D_J z| ≤ 1.39, plug-in identity residual ≤ 1.7e-16; E1 cells C1 / C3 at N 256 and E2 cells C2-rot / C1-orig at N 1 024 with budgets 50 / 100
and one lr (smoke values not meaningful); aggregator ran on the smoke output.  Tests: `tests/test_p108.py` 13 passed (exact discrete-toy identities to
1e-12–1e-14, mechanism orders, rotation orthogonality / distance preservation, eta unchanged under rotation) + stage-0 and v2 estimator suites (41 total).

## Cost
Measured basis: P85 product-negative JointMLP fits take 3.4–3.7 s per 2 000 updates on a GPU (`outputs/P85_estim_benchmark`, C1-type cell), in-batch
~7 s.  Per E1 cell: VCS + JS × 3 lrs × (250 + 1 000 + 4 000) updates ≈ 60 s + references 4 kinds × 3 lrs × 4 000 updates ≈ 100–120 s + EVAL / truth ≈ 20 s
→ ≈ 3 min; per E2 cell ≈ 2 min (neural 60 s, RFF 3 × 5 250 updates ≈ 15 s, S-KDE seconds to ~1 min at N 16 384).  Total ≈ 60 × 3 + 36 × 2 ≈ 4–5 GPU-h in
5 jobs (`slurm/p108_lines.txt`, ≤ 1.5 h each).  GPU timing pilot job 1017242 (cells E1 C2 N 16 384, E2 C1-rot N 16 384, E1 C3 N 1 024 + full mechanism;
output `outputs/P108_pilot`) replaces this estimate with measured per-cell times when it runs (queued behind the quota at drafting time).

## Delivery (per unit, v4 §5 fields)
```yaml
experiment_family: estimator (synthetic)
protocol_id: P108_v4_E
source_commit: <freeze commit>
estimator: VCS-N and JS-N (JointMLP), S-KDE, RFF S-kernel; references InfoNCE / NWJ / MINE-DV / SMILE (own truths)
estimand: S (VCS readouts); own targets for references
evaluation_readout: J_hat and S_plug_hat of the same T on the same EVAL units; posterior MSE
n_independent_units: FIT N per distribution (independent per seed); EVAL 32768; TRUTH 200000
status: draft
```

## Decisions at the freeze (main session)
1. Nonlinear condition C3 = xor_mixture (as drafted). 2. E3 tolerance |J − S| ≤ 0.02 absolute (as drafted). 3. No P105 timing re-run: E3's
solver-cost comparison covers the P108 fits only; P105's closed-form / L-BFGS cost is not claimed.  4. The GPU timing pilot (1017242) was cancelled
before it started; the five full jobs record per-cell times themselves (resumable, one JSON per cell).
