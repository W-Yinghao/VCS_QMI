# Pre-check D — addendum 6 DRAFT: diagnosis of the 0.06–0.07 null rejection rate (to be frozen by the main session before GPU compute) — FROZEN 2026-09-27T19:28:00Z before launch (CPU smoke 1011783 disclosed inside; owner: "都补充上")

**What was seen.** D-S1 (R = 1000 fresh subsets, seed 3) at s = 0: `vcs_perm` 0.07 / 0.06 / 0.06 (VCS encoder, n = 2000 / 5000 / 10 000) and
0.05 / 0.07 / 0.07 (SimCLR encoder); `hsic_perm` 0.06 / 0.05 (VCS, n = 2000 / 5000) and 0.05 / 0.07 (SimCLR).  The binomial 95 % band of an
exact level-0.05 test at R = 1000 is 0.036–0.064.  HSIC has no fitted component and shows the same excess, so the cause is more likely in the
shared design than in the VCS critic.  The Hoeffding test never rejects (0 / 6000).

**Hypothesis (pre-committed).** In the P45 features every strength — including s = 0 — uses ONE nuisance draw N for the 45 000-image pool
(`precheck_d_features.py`: "the SAME assignment for every strength, so strength 0 is exact independence").  Each repeat is a random
3n-subset of that pool, so all R repeats share a single realised association between the pool's z and its N.  The permutation tests are exact
*marginally over N realisations*: for one realisation the conditional rejection rate of overlapping subsets is not 0.05 but a random quantity
around it, with a spread that grows with the overlap between repeats (3n / 45 000 = 13 % at n = 2000, 67 % at n = 10 000).  Both tests read
the same realisation, so they deviate together.  Prediction: re-drawing N for the items of every repeat restores 0.05 for both tests on the same
features; if the rate stays above 0.065 with N re-drawn, the procedure itself is anti-conservative.

**Design.**
1. *Real features, N re-drawn per repeat* (`scripts/precheck_d_tests.py --redraw-null-n`, additive flag; default byte-identical to the
   committed script — identity check in the smoke below): case s0, n ∈ {2000, 5000, 10 000}, R = 1000, seed 4 (fresh subsets), both encoders,
   every test.  Six GPU jobs via `slurm/precheck_d_cases_gpu.sbatch`; outputs `reports/P46_precheck_D_redraw1000_<enc>_n<n>_s0.{json,md}`.
2. *Synthetic null* (`scripts/precheck_d_synthetic_null.py`, the unchanged `run_instance` on Gaussian z ∈ ℝ⁵¹² with N ~ Bernoulli(½), δ = 0.05,
   200 permutations, critics as in P45): mode `fresh` (fresh z and N per repeat; n ∈ {500, 2000}, R = 1000) — the textbook level check;
   mode `fixed` (one pool of 45 000, N drawn once, repeats = subsets; n ∈ {2000, 5000, 10 000}, R = 1000) — the P45 design without any real
   data; mode `fixed_redraw` (same pool, N re-drawn per repeat).  Three GPU jobs via `slurm/precheck_d_synthetic_null.sbatch`.

**Pre-committed reading.**
- Rates ≤ 0.065 in all six real re-draw cells for both `vcs_perm` and `hsic_perm`, and `fresh` ≈ 0.05 (inside 0.036–0.064) → the excess is the
  fixed-N design of P45; the tests are exact; the D report's false-alarm cells are re-read as a design artefact of the null cases (the D
  verdict itself is re-visited by the owner, not by this addendum).
- `fixed` reproducing rates > 0.064 (in at least one n, for both tests) while `fixed_redraw` and `fresh` stay inside the band → confirms the
  mechanism independently of the real features.
- Real re-draw cells still > 0.065 for `vcs_perm` (any n) → the VCS permutation procedure is anti-conservative at the ~0.01–0.02 level as
  implemented; the false-alarm clause stands as a property of the test and the report says so; if `hsic_perm` also stays high the cause is in
  the shared sampling (`run_instance`), to be located before any exactness claim.
- Mixed outcomes are reported cell by cell; nothing is averaged across encoders.

**Not claimed.** Anything about power; the conditional cases (they draw N | Y once as well — a follow-up if the hypothesis holds); other
encoders or nuisance families.

**Smoke (CPU, disclosed; filled in after job 1011783).**
CPU job 1011783 (`slurm_logs/pdnull_diag_smoke_1011783.out`, exit 0): synthetic modes `fresh` (n = 200, R = 20), `fixed` and `fixed_redraw`
(pool 3 000, n = 200 / 500, R = 20, 50 permutations, 50 critic steps) all run end to end (rates 0–0.10 at R = 20: code check only, no
evidence).  Identity check: the committed `precheck_d_tests.py` (`git show HEAD:`) and the patched one produce IDENTICAL `cases` output on
the same smoke (`--cases s0 --smoke`, VCS features); with `--redraw-null-n` the output differs and `settings.redraw_null_n = true` is
recorded.  No design change follows from the smoke.

**Compute.** Real re-draw: six GPU jobs (n = 2000 ≈ 20 min, n = 5000 ≈ 2 h because of HSIC, n = 10 000 ≈ 1 h on PRO6000).  Synthetic: `fresh`
≈ 30 min, `fixed` and `fixed_redraw` ≈ 3 h each (one job per mode).
