# Pre-registration — P102: T line, conditional dependence increments between nested information sets on the T1 task, 2026-09-30 — FROZEN 2026-09-30T19:48:01Z before GPU compute (6 tests pass, oracle self-check exact, CPU smoke end to end; launch waits for the P74 report by the orchestrator condition)

Status: FROZEN 2026-09-30T19:48:01Z (main session).  The main session freezes it (renamed `*_FROZEN_*`, freeze time in the title) before any GPU job.  Owner go 2026-09-30
("除了imagenet的先不提交，后续都可以提交，你设置好提交程序").  Sources: Server Spec v2 §11 (T line), Research Plan v2 §8.2; T1 design and critic
settings from `P73_CONDITIONAL_TEST_T1_PREREG_FROZEN_20260928.md`.  The P74 verdict is not an input to the design; it is read next to the results (`reports/P102_t_line_results.*`).

## Question
Inside one conditional binary experiment, how much of the dependence between a frozen representation and a planted nuisance (given the class) survives
the deterministic compression to class logits, and do the finite-critic increments behave like the L² geometry predicts (orthogonal, additive steps
along a nested chain)?  Descriptive; this is the auxiliary T line, not a claim that VCS is uniquely suited (log-loss / JS has its own increment
E_M KL(Ber(q_B*) ‖ Ber(q_A*)); the squared-norm form belongs to the quadratic scoring rule, Plan §8.2).

## Design
- **Task (reused from T1).**  P45 feature files `outputs/P45_precheck_D_{vcs4v800,simclr}{,_blur}` (encoders: VCS 4-view 800-epoch seed 0; SimCLR P5 seed 0).
  N ∈ {0, 1} with P(N = 1 | Y) = ½ ± 0.3 by class parity, drawn afresh per repeat; N = 1 items carry the planted nuisance of strength s, N = 0 items are
  clean.  Strengths: colour {0, 0.05, 0.1, 0.2}, blur {0, 0.25, 0.5, 1.0}; **s = 0 is T1's `null_label_only` case** (clean features for all; no conditional
  dependence) and serves as the null / noise-floor cell.
- **Conditional P / Q.**  P = observed (h, n, y); Q = the same (h, y) with n replaced by the N of a same-class item from a disjoint POOL (equal values allowed).
- **Nested information sets (all critics receive N and Y explicitly).**  h (standardised 512-d) ⊇ pca64 ⊇ pca16 ⊇ logit16 (linear probe on pca16), and
  h ⊇ logit_h (linear probe on h).  Standardisation, PCA basis and both probes are fitted once per (encoder, family) on a fixed PROBE set of 8 000 clean
  FIT images excluded from every repeat; they are deterministic maps thereafter, so every coarser set is a function of the finer one (true nesting,
  Spec §11.2).  Main pair: B = h, A = logit_h.  Chain: h → pca64 → pca16 → logit16.
- **Roles per repeat.**  FIT / EVAL / POOL, n = 3 000 each, disjoint, fresh draw per repeat; 5 repeats per cell (refit variability).
- **Critics.**  VCS (−J) and matched JS (softplus(−2f), softplus(2f); q = σ(2f), Spec §2.2) fitted separately per information set with the T1 settings: linear
  (sign-flip with n, class entering explicitly) and MLP (hidden 128 × 2) classes, AdamW lr 1e-3, wd 1e-2, 300 full-batch steps, 80 / 20 split, selection every
  10 steps.  For both objectives the class and step are selected by the **common squared score** J(tanh f) on the 20 % (Spec §11.1: magnitude critics are
  selected by independent regression risk).  All reported quantities use T = tanh f and the common score on EVAL.
- **Readouts (Spec §11.3).**  Per set J; main pair Delta_J = J_h − J_logit_h, D_T = E_M (T_h − T_logit_h)², r_BA = Delta_J − D_T (checked against
  2 E_M[(C − T_B)(T_B − T_A)] numerically); every chain step; R_orth of the chain.  Intervals: 95 % paired bootstrap by base image on EVAL (1 000 reps, critics
  fixed — sampling error), and the spread over the 5 refit repeats (refit error), reported separately.  **No permutation p-values**; Delta = 0 is not inherited
  from T1's conditional-independence tests.
- **Oracle self-check (run first, aborts on failure).**  Exact discrete toy (3 classes, 12 states, nested coarsenings 12 → 6 → 3 → 1): Delta = S_B − S_A =
  E_M (T_B* − T_A*)² and R_orth = 0 to 1e-12; J(T*) = S.  CPU smoke: 1.0e-17 / 0.0 / 1.4e-17.

## Pre-stated reading (descriptive; per encoder × family × objective)
1. **Delta tracks the planted strength** if the repeat-mean Delta_J is non-decreasing in s and, at the two largest strengths, the lower end of its fixed-critic
   bootstrap interval exceeds the upper end of the s = 0 cell's interval (the null cell is the noise floor: finite critics give D_T > 0 and Delta_J ≠ 0
   even without dependence — the CPU smoke already shows this).  Otherwise "Delta does not separate from the null at these n".
2. **Nested increments consistent** if the R_orth interval contains 0 in a cell; if it excludes 0 the cell is reported as "fit / nesting residual", with its
   sign, not as a finding about the representation.
3. **Where the nuisance is lost**: the chain steps' Delta_J with intervals (which compression step removes the nuisance information), per encoder.
4. **VCS vs JS**: the same readouts side by side; differences are described, not ranked (both are evaluated with the same squared score).
Not claimed: that the squared increment is exclusive to VCS; that a positive D_T proves dependence (two critic errors also give D_T > 0); anything about T1's
test level or power.

## Cost
CPU smoke (n 300, 20 steps, one cell pair): 31 s.  GPU estimate: 20 critic fits per repeat (5 sets × 2 classes × 2 objectives) at n = 3 000, 300 steps,
≈ 30–60 s per repeat; 2 encoders × 2 families × 4 strengths × 5 repeats = 80 repeats → ≈ 1–1.5 h plus feature loading.  One job,
`slurm/p102_t_line.sbatch` (normal QOS, RTX6000PRO,H100,A100,L40S, excl. node51/node60, 8 h wall).  Results → `reports/P102_t_line_results.{json,md}`.

## Code and checks
`src/vcs_estim/increments.py` (quantities, paired bootstrap, exact toy, Y-aware critics and fitters), `scripts/t_line_increments.py` (runner),
`tests/test_p102.py` — 6 tests pass (oracle increment identity float64; R_orth = 0 and equal to the cross-term form at the oracle; finite residual identity on
random critics; r_BA = 0 at the oracle and ≠ 0 off it; paired bootstrap keeps images together; both fitters run and select by the common score).
CPU smoke of the runner (vcs4v800 / colour, s ∈ {0, 0.2}, `--smoke`): end-to-end, JSON + markdown written.

## Launch (after freezing)
```
sbatch --parsable --job-name=p102_t_line slurm/p102_t_line.sbatch
```

## Delivery (Spec v2 §13.2)
```yaml
experiment_family: diagnostic (T line, conditional increments)
protocol_id: P102_T_line_increments
source_commit: <freeze commit>
source_basis: supplement_af7172c_plus_v2
estimator: vcs_neural | js_matched (common squared evaluation)
estimand: S of the conditional experiment (per information set) and its increments
evaluation_readout: J_common, Delta_J, D_T, r_BA, R_orth; paired bootstrap by base image; refit spread
reference_measure: mixture_equal (conditional P / Q)
critic_class: linear (Y-aware sign flip) | MLP 128x2 on [z; onehot n; onehot y]
gradient_routing: full
n_independent_units: EVAL 3000 base images per repeat; 5 repeats per cell
split_manifest_hash: P45 feature files (uids / y / stored N shared across strengths, asserted at load)
status: frozen
```

## Disclosed choices
- F for the main pair is the linear probe's logits on h (fixed on PROBE), the Spec's "确定性 logits"; the chain's last step uses a probe on pca16 so that the
  chain stays nested (a probe on h is not a function of pca16).
- PROBE uses class labels to fit the probes (Y is part of the experiment; labels never select anything on EVAL).
- s = 0 uses T1's `null_label_only` sampling (clean features for all); T1's `null_all_planted` case is not used here.
