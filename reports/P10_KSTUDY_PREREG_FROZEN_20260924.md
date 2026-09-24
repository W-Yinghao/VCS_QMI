# Pre-registration — VCS-QMI single-factor K study (P10–P11), frozen 2026-09-24 before launch

Owner decision after `reports/P7_confirm200_report.md`: run K = 8 and K = 64, VCS only.

## Single factor
`pairing.k` ∈ {8, 64} (number of *distinct nonzero cyclic shifts* per step; negatives per step = K·B = 2,048 / 16,384; all K·B negative
scores are averaged as one distribution, spec §3.2/§6.1; reference sampler `cyclic_negative_indices` unchanged).  **Nothing else
changes**: configs `configs/cifar10_k{8,64}_vcs_seed{0,1,2}.yaml` are the frozen 200-epoch VCS configs with `run.stage=P10_kstudy200`
and `pairing.k` overridden (`configs/KSTUDY_SHA256.json`).  Same initial weights per seed (seed 0 `bdc3a4e3…`, seed 1 `94c2cb88…`,
seed 2 `4385db04…`), same split, 200 epochs, warm-up 10, B = 256, K = 1 comparator = the P5 VCS runs.
Code change: the first-round policy check `K == 1` was relaxed to `1 ≤ K ≤ B − 1` (spec §6.1 interface); tests updated
(`test_8_config_strictness`, new `test_k_greater_than_one_objective_and_config`).  The critic hold-out diagnostic uses the run's own K.

## Question
Does increasing the per-step negative pool (K) change the VCS representation's dimensional concentration (h effective rank ≈ 13, exactly
7 dominant `p_raw` directions at K = 1) and its frozen-feature quality (74.34 ± 0.46 % linear, 63.9 % kNN at K = 1), and how does the
critic objective (`heldout_J` ≈ 0.90 at K = 1) respond?

## Endpoints
Primary: `linear_val_top1_pct` of frozen `h` at `epoch_200.pt` (pilot probe protocol unchanged) per seed and mean ± SD, versus the P5
K = 1 values for the same seeds.  Secondary: kNN, h- and z-rank, number of dominant `p_raw` directions (first > 5× eigenvalue drop),
`heldout_J` (mean of 4 repeats, with the run's K), saturation fractions, per-step cost of the critic (K·B pairs) and memory.

## Pre-committed reading
- Monotone effect if linear-val(K=64) > linear-val(K=8) > linear-val(K=1) in all 3 seeds with differences > the K=1 seed SD (0.46);
  otherwise "no monotone K effect" or "non-monotone", reported as observed.
- Concentration hypothesis supported if h-rank and the number of dominant `p_raw` directions increase with K in all seeds; refuted for
  this range if they stay at ≈ 13 / 7 at K = 64.
- `heldout_J` is expected to change with K even if the population objective does not (finite-sample pairing); a lower J at higher K is not
  a failure.  Report J together with `t_pos`/`t_neg` means and saturation.
- Cost: report step time vs K on the same GPU type; the critic's K·B forward is the only added computation.
- Failures / resumes reported; no method change, no regularizer, no tuning.

## Not claimed
No test accuracy; no comparison to SimCLR/VICReg beyond the P5 200-epoch matched runs (which remain K-independent references);
no independence claim for K·B pairs; no Shannon-MI / convergence-to-S language.

## After P11
Neutral table → report → wait for the owner.
