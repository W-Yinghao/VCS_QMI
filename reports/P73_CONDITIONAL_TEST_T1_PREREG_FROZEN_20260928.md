# Pre-registration DRAFT — T1: the within-class conditional dependence test with fair controls, N | Y re-drawn per repeat (P73), 2026-09-28 — FROZEN 2026-09-28T14:50:01Z before GPU compute (CPU smoke 1013018 disclosed inside; uid alignment of planted and clean feature files verified; thresholds loosened per the owner)

Status: DRAFT (implementation fork); the main session freezes it (rename to `*_FROZEN_*`) before any GPU compute.  Source: `CS_QMI/VCS_QMI_Next_Round_Plan_v2.md`
T line, `reports/NEXT_ROUND_EXECUTION_PLAN_DRAFT_20260928.md` §3.  Code: `scripts/cond_test_t1.py`, `slurm/cond_test_t1.sbatch`.  Owner: thresholds looser than the plan draft.

**Question.**  Is the VCS within-class conditional test at least as powerful as *fair* controls — HSIC with a per-class bandwidth, HSIC with a learned
(deep) kernel, a C2ST with a logit statistic, and a JS statistic trained in the same critic classes — when the nuisance N is class-correlated and
the design draws N | Y afresh for every repeat (the P45 addendum-6 lesson)?  The P46 cell that motivates it (blur family, SimCLR encoder, n = 2000:
VCS 0.98 vs HSIC 0.18) was diagnosed as a bandwidth mismatch of the single-bandwidth HSIC; the fair controls remove that excuse.  If the JS statistic
matches VCS, the advantage belongs to "a learned critic in the right class", not to VCS — the report says so.

## Data and construction
- Features: the existing P45 files of both encoders and both families — colour (`P45_precheck_D_{vcs4v800,simclr}`, strengths 0.05 / 0.1 / 0.2) and
  blur (`P45_precheck_D_{vcs4v800,simclr}_blur`, σ 0.25 / 0.5).  All case files of a family share `uids`, `y` and the stored i.i.d. Bernoulli(½) draw;
  the N = 0 features of a planted file equal the clean (s = 0) features exactly — both verified at load (assertions) and in the smoke.
- Per repeat: three disjoint samples of n items (FIT / EVAL / POOL).  Classes are drawn uniformly, N | Y ~ Bernoulli(½ + 0.3·(+1 if Y even else −1)) is
  drawn *fresh*, and the item is then drawn among the class's items whose stored draw equals the fresh N (so an N = 1 item has a planted feature); the
  stored draw is i.i.d. and independent of the image, so this conditioning does not bias the image distribution.  N = 1 → planted feature of that
  strength, N = 0 → clean feature.  Product samples for the critics: N values of same-class POOL items (`within_class_pool`, as P46).
- Exact nulls (Z ⊥ N | Y by construction), same fresh N | Y: `null_label_only` — clean features for everyone; `null_all_planted` — the planted file's
  feature for everyone at the family's largest strength (nuisance-affected features present, N independent of the item given Y).
- Sizes n ∈ {200, 500, 1000, 2000}; R = 100 repeats per power cell; level checks with R = 1000 at n ∈ {500, 2000} on both nulls of the colour family
  (both encoders); the blur family's nulls at R = 100.  δ = 0.05; B = 200 within-class permutations of N on EVAL, generated once per repeat and
  **shared by every permutation test**; identical FIT / EVAL / POOL and identical 80 / 20 fit / selection split inside FIT for every learned component;
  300 optimisation steps for every learned component (as P46).

## Tests (all on the same EVAL sample)
| test | statistic | null |
|---|---|---|
| `vcs_perm` | J of the VAL-picked critic (linear / MLP / closed-form linear class) | within-class permutation |
| `vcs_hoeff` | J − τ_{n,n}(δ) > 0 (guaranteed) | none (distribution-free bound) |
| `hsic_perm` | HSIC_b, Gaussian kernel, one median bandwidth (P46 control) | permutation |
| `hsic_class` | Σ_c (n_c / n)·HSIC_b within class c, per-class median bandwidth | permutation |
| `hsic_deep` | the same per-class HSIC on φ(z), φ = 2-layer MLP (128 → 32) trained on FIT 80 % to maximise [within-class HSIC − mean of 8 permutation baselines] / their sd, selected on FIT 20 % by the same score | permutation |
| `c2st_logit` | mean logit of the observed EVAL pairs from the BCE classifier (same MLP, budget) | permutation |
| `js_perm` | Deep-InfoMax JS value E_P[−softplus(−f)] − E_Q[softplus(f)] + log 4, critic classes linear / MLP with the same fitter and early stopping, VAL-picked | permutation |
Reported alongside: the three VCS critics' guaranteed-test rates, mean J and mean JS per cell, the smallest n with 80 % power per test, the level table.

## Pre-committed reading (loosened per the owner)
- **Level**: on both null cases every test rejects ≤ 0.065 at R = 1000 (binomial 95 % upper bound 0.061); a test above 0.09 is invalid at that n and
  its power cells are not read.
- **Informative cells**: (family, strength, n) cells where at least one test has power ≥ 0.5.
- **T1 holds** if in every informative cell the VCS conditional permutation test's power ≥ max(controls) − 0.10.
- **Refuted** if the JS statistic is within 0.05 of VCS in every informative cell (then the effect is "learned critic", not VCS), or if some control
  exceeds VCS by ≥ 0.10 in at least half of the informative cells.
- **Conditional** otherwise (e.g. parity with the JS statistic in some cells only; a deficit ≥ 0.10 against one control in fewer than half the cells).
- The guaranteed test is reported on its own curve and does not enter the verdict.
- Not claimed: continuous or multi-class N, other encoders / datasets, nuisances other than the two planted families, the unconditional test.

## QC sentinels
uid / y alignment and N = 0 feature equality assert at load; `n1_frac_eval` per repeat within ± 0.05 of the class-mixture mean 0.5; the VAL-picked
critic distribution is recorded; the deep kernel's VAL score and best step are recorded (a best step of 0 in most repeats means it learned nothing and
is reported as such, not tuned).

## Compute
Per repeat at n = 2000 ≈ 5–8 s on a GPU (seven learned or kernel components, 200 shared permutations).  Power cells: 2 encoders × (3 + 2) strengths
× 4 n × 100 ≈ 6 GPU-h; level cells: 2 encoders × 2 nulls × 2 n × 1000 ≈ 10 GPU-h; blur nulls at R = 100 ≈ 1 GPU-h.  One job per (encoder, family).

## Smoke (disclosed) — filled in after the CPU job
**CPU smoke, job 1013018 (`slurm_logs/cond_test_t1_t1_smoke_1013018.out`, exit 0; VCS encoder, colour family, s = 0.1, n = 200, R = 5, 20 permutations,
60 steps; ≈ 28 s per cell on 8 CPU threads).**  Load-time assertions passed for all seven colour case files (uids and y identical; N = 0 features equal to
the clean file).  All seven tests ran on the planted case and both nulls: `colour_s0.1` — vcs_perm 0.20, hsic_perm 0.40, hsic_class 0.00, hsic_deep 0.40,
c2st_logit 0.00, js_perm 0.20 (J ≈ 0.001, τ_{200} = 0.42: no guaranteed rejections, as expected at this n); `null_label_only` — 0 / 0 / 0.20 / 0 / 0.20 / 0;
`null_all_planted0.1` — 0 / 0.20 / 0 / 0 / 0 / 0 (R = 5: each rejection is one repeat; code-path evidence only).  Nothing in the design was changed after the
smoke.  Cost projection for the GPU jobs from the smoke's per-repeat time scaled to 300 steps / 200 permutations / n ≤ 2000: ≈ 8–12 s per repeat →
colour jobs (1 200 power + 2 200 null repeats) ≈ 7–10 h each, blur jobs (800 + 800) ≈ 3–5 h each; ≈ 25 GPU-h in total, four jobs.
