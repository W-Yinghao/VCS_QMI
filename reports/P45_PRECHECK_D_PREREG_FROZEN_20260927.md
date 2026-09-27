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

## Addendum 2026-09-27 (probe findings, frozen before the fleet; the two probe runs are disclosed: `P46_precheck_D_smoke.md`, `P46_precheck_D_probe.md`, `P46_precheck_D_probe2.md`)
1. **Critic over-fitting broke the guaranteed test in the first design.**  The MLP critic trained for 300 full-batch steps on FIT memorised the
   (z_i, N_i) pairs (each z_i appears with both N values) and gave saturated, uninformative outputs on EVAL: Ĵ_eval ≈ −0.3 at s = 0.1 although
   the permutation-calibrated VCS test, HSIC and C2ST all rejected.  Change (before any fleet run): three critics, all fitted on 80 % of FIT and
   selected/early-stopped on the remaining 20 % (VAL) — a low-capacity signed linear critic tanh(⟨w, z⟩(2N−1) + b), the MLP (AdamW, wd 1e-2,
   best VAL J), and the closed-form linear-class critic φ = [z(2N−1), 1] (ridge and the tanh scalar chosen on VAL, cf. B1).  The reported
   `vcs_hoeff` uses the critic with the best VAL J (chosen without touching EVAL, so the bound's premise holds); the three are also reported
   separately.  C2ST gets the same early-stopping budget.  Probe 2 after the change: at s = 0 all Ĵ_eval ≈ 0 and no test rejects; at s = 0.1 the
   closed-form Ĵ_eval grows 0.006 → 0.020 → 0.047 for n = 200 / 1000 / 5000.
2. **The planted colour shift is weak in h-space** (S ≈ 0.05 at s = 0.1), so the distribution-free bound (τ_{5000,5000} = 0.084) cannot reach
   the threshold within n ≤ 5000 at s ≤ 0.1 even with a perfect critic — a property of the bound, not of the estimator.  To map the guaranteed
   test's power curve the strength grid is extended to s ∈ {0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5} and the sample sizes to include n = 10 000
   (R = 10; HSIC is not run above n = 5 000 — O(n²) memory — and is marked "not run" there).  The reading rules of the main text are unchanged;
   the "small-sample end" comparison (n ≤ 500) is between the permutation-calibrated tests, and the guaranteed test is read on its own curve
   (smallest n reaching 80 % power per s).
3. Conditional exactness held in the probe (label_only: conditional tests 0/4 rejections at n = 2000; label_colour: vcs_perm 4/4, HSIC 2/4, C2ST 0/4).
4. **Null-calibration top-up at R = 100 (written 2026-09-27 after the fleet tables were seen, frozen before any further compute).**  The
   false-alarm rule (≤ 0.09) carries a binomial slack derived for R = 100, but the fleet's s = 0 cells at n ≥ 2000 ran with R = 30 (n = 2000,
   5000) and R = 10 (n = 10 000), where a single rejection above the expectation already prints 0.10.  Observed in the fleet: vcs_perm 3/30 at
   n = 5000 (both encoders) and 1/10 at n = 10 000 (VCS encoder); hsic_perm 5/30 at n = 5000 (SimCLR encoder); *label_only* conditional
   vcs_perm 5/50 at n = 2000 (SimCLR encoder).  Top-up: the s = 0 case at n ∈ {2000, 5000, 10 000} and the *label_only* conditional case at
   n = 2000 are re-run with R = 100 on fresh draws (seed 2, disjoint from the fleet's seed 1), both encoders; nothing else changes and no
   power cell is re-run.  The false-alarm and conditional-exactness (type-I) rules are then read on the R = 100 rates; the fleet's R = 30 / 10
   rates stay in the tables.  Cost ≈ 1.5 GPU-hours per encoder.
5. **Wave-2 supplements (frozen 2026-09-27 before launch; owner's instruction "全部都补充实验"; designs in `SECOND_APP_WAVE2_PLAN_20260927.md`, row D).**
   *D-S1 level pin:* s = 0 at n ∈ {2000, 5000, 10 000}, R = 1000 fresh draws (seed 3), both encoders, every test.  Reading: rate ≤ 0.065 →
   level 0.05 confirmed (binomial 95 % upper bound at R = 1000 is 0.061); (0.065, 0.09] → anti-conservative but not invalid; > 0.09 →
   invalid at that n.  It is read next to the letter/intent question of the report §3.1; the D verdict is not re-litigated by this unit.
   *D-S2 second encoder seeds:* VCS seed 1 (`P35_vcs_a5_views4_800ep_seed1`, epoch_800.pt) and SimCLR seed 1 (`P5_simclr_seed1`,
   epoch_200.pt); s ∈ {0, 0.05, 0.1, 0.2, 0.3}; n ∈ {100, 200, 500, 1000, 2000, 5000}; R = 100 (n ≥ 2000: 30); both conditional cases at
   n ∈ {500, 2000}, R = 50; the rules of the main text apply verbatim; "on par with HSIC" is confirmed if ≥ 90 % of the small-sample cells
   (n ≤ 500, s ≥ 0.05) pass the parity rule on seed 1 as well.
   *D-S3 second nuisance family:* Gaussian blur of σ ∈ {0, 0.25, 0.5, 0.75, 1.0} px applied to the N = 1 images at 32 × 32 (σ = 0 is exact
   independence), VCS seed 0 and SimCLR seed 0, same tests and rules (parity, monotonicity, false alarm at R = 100 cells).

6. **Null-level diagnosis (wave-2 supplement S-1)** — see `P45_ADDENDUM6_NULL_DIAGNOSIS_FROZEN_20260927.md` (frozen 2026-09-27T19:28:00Z): hypothesis that the single per-pool N draw shared by all repeats explains the 0.06–0.07 level; design = s0 with N re-drawn per repeat (R = 1000, seed 4, both encoders) + three synthetic nulls.
