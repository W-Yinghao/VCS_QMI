# Pre-registration — pre-check D: leakage-detection power (P45), frozen 2026-09-27 before launch

**Property tested.** The VCS objective is bounded (|T| ≤ 1), so for a fixed critic the evaluation estimate obeys a distribution-free bound
(Var(Ĵ | T) ≤ 1/n + 1/m; with probability 1 − δ, S ≥ Ĵ_eval − τ_{n,m}(δ), τ = √(2 log(4/δ)/n) + √(2 log(4/δ)/m)).  Question: when a known
dependence between a representation Z and a nuisance N is planted, at what sample size does a VCS-based test detect it, versus HSIC
(permutation) and a classifier two-sample test (C2ST); is the conditional version S(Z; N | Y) with within-class shuffling exact when Y is discrete?

**Data (no new data).** Z = frozen h (512-d) of the 45 000 *fit* images of the frozen CIFAR-10 split (the encoder never saw labels; N is planted
after training, independent of it).  Two encoders, to show the test is representation-agnostic: (E1) the final VCS recipe
`P35_vcs_a5_views4_800ep_seed0/epoch_800.pt`; (E2) the SimCLR control `P5_simclr_seed0/epoch_200.pt`.
**Planting (known strength, by construction).** N_i ~ Bernoulli(½) i.i.d. (seed 20260927), the same draw for every strength; images with N_i = 1
receive a colour-temperature shift of strength s (R × (1+s), B × (1−s), clipped), then the clean transform and the frozen encoder.
s ∈ {0, 0.01, 0.02, 0.05, 0.1, 0.2}; s = 0 is exact independence (images untouched, N random) → null / false-alarm calibration.
**Conditional cases.** N_c | Y ~ Bernoulli(½ + 0.3·(+1 if Y even else −1)) (Y = class): (i) *label_only* — images unchanged, so Z ⊥ N_c | Y
holds exactly while Z and N_c are marginally dependent through Y; (ii) *label_colour* — the N_c = 1 images additionally shifted with s = 0.05, so
Z depends on N_c beyond Y.  Within-class shuffling gives exact samples of P_Y P_{Z|Y} P_{N|Y} (brief §2, D).

**Tests** (`scripts/precheck_d_tests.py`; per instance three disjoint samples of n items: FIT, EVAL, POOL; negatives always from POOL, never in-sample shuffles for the statistic):
| test | statistic / decision |
|---|---|
| vcs_hoeff (the guaranteed one) | critic T = tanh(MLP([z, onehot N]) 514→128→128→1) fitted on FIT with the J objective (Adam 1e-3, 300 steps); Ĵ_eval on EVAL with m = n POOL negatives; reject iff Ĵ_eval > τ_{n,n}(δ), δ = 0.05 |
| vcs_perm | same Ĵ_eval; permutation null (N permuted among EVAL items, 200 permutations) — exact at level δ under independence |
| hsic_perm | HSIC_b, Gaussian kernel on standardised z (median heuristic), delta kernel on N; 200 permutations |
| c2st | the same MLP trained with BCE on FIT to separate joint (z_i, N_i) from product (z_i, N_pool,i); EVAL accuracy vs ½, one-sided normal test |
Sample sizes n ∈ {100, 200, 500, 1000, 2000, 5000}; repeats R = 100 (R = 30 for n ≥ 2000); conditional cases at n ∈ {500, 2000}, R = 50, with
the unconditional tests also run on the same draws.  Level δ = 0.05 throughout.  Probe: one smoke run (2 strengths, tiny grid) must finish
cleanly before the fleet.

**Pre-committed reading.**
- *False alarm:* at s = 0 every test's rejection rate must be ≤ 0.09 (δ = 0.05 with binomial slack at R = 100); a test above that is invalid
  at that n and is reported as such.
- *Power (primary, brief's claim):* on the small-sample end (n ≤ 500) and for every s ≥ 0.02, vcs_perm power ≥ max(hsic_perm, c2st) − 0.05.
  vcs_hoeff is reported alongside as the *guaranteed* test; it is expected to be conservative (τ_{500,500} = 0.26), and the smallest n at which
  it reaches 80 % power per s is tabulated.
- *Monotonicity:* mean Ĵ_eval at n = 5000 strictly increases in s (Spearman = 1 over the six strengths) for both encoders.
- *Conditional exactness:* in *label_only* the conditional tests reject ≤ 0.09 while the unconditional tests reject ≥ 0.8 at n = 2000; in
  *label_colour* the conditional tests reject ≥ 0.8 at n = 2000.
- **Property holds** if all four bullets hold for both encoders.  **Holds conditionally** if power parity holds only for some s or n, or only
  for one encoder, or only for the permutation-calibrated test (then the guaranteed bound is too loose at these n and this is said explicitly).
  **Does not hold** if vcs_perm is below hsic_perm by ≥ 0.10 across the small-sample end, or false alarms exceed 0.09, or Ĵ is not monotone.
- Not claimed: anything about tasks; the planted nuisance is one family (colour temperature) — texture/subgroup nuisances are follow-ups.

**Held-out J check (brief P1).** The VCS statistic here is a dependence between Z and a binary N; Ĵ_eval values are expected far below 0.9
(medium/low dependence), which is the regime the brief asks for; values are reported in every row.

Outputs: `reports/P46_precheck_D_<encoder>.md/.json` (all instances kept), feature manifests under `outputs/P45_precheck_D_<encoder>/`.
Code: `scripts/precheck_d_features.py`, `scripts/precheck_d_tests.py`, `slurm/precheck_d.sbatch` (CPU partition).
