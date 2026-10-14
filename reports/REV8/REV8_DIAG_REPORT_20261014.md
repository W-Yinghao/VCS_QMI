# REV8 — critic-only diagnostics of the frozen D1 selectors (review r8 items R8-01 / R8-02 / R8-04): report — 2026-10-14

- **Go-ahead:** owner, 2026-10-13 ~21:19 UTC ("跑"); protocol reports/REV8/REV8_DIAG_PROTOCOL_FROZEN_20261013.yaml (a977b21), frozen before any
  diagnostic number.  Read-only: no host call, no DEV or official-test read, no frozen unit re-selected or re-trained.
- **Units:** 36 frozen selector units: LLaVA 10k, LLaVA FULL and Qwen 10k × {VCS, logistic, direct risk} × seeds 0–2 at the frozen learning rate,
  plus the seed-0 unit at the other learning rate; every CAL question (7,989 × 12 D1-10 masks).
- **Guard:** each unit's eval-mode critic reproduces its logged pooled CAL Spearman; largest |recomputed − logged| 6.73e-07 over the 36 units
  (tolerance 1e-4).  Jobs 1038583–1038585 (GPU), summary job 1038964; results-only commit 5dd8f37 (reports/REV8/REV8_DIAG_RESULTS/, unedited).
  Tables below: scripts/rev8_tables.py (formatting only; job 1038981) on that file.  Image-grouped bootstrap (2,000 replicates) for the
  intervals behind the flags; mean ± sd over seeds in the main rows; no hypothesis test.
- **Limits (frozen):** critic-only quantities.  None measures host accuracy, and none establishes a cause of the D1 shortfall; that needs a paired
  retraining intervention, which is an owner decision.

## A (R8-04): what the CAL score measures, and the within-question mask choice

| scale | objective | pooled ρ | within-budget ρ | within (budget, mech.) ρ | within-question ρ | pairwise acc. (q, k) | top-1 hit | regret: critic | random choice | always grid | teacher d | seeds with acc. CI ∋ 0.5 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| LLaVA 10k | vcs | 0.088 ± 0.004 | 0.059 ± 0.009 | 0.060 ± 0.009 | 0.115 ± 0.005 | 0.499 ± 0.001 | 0.328 ± 0.002 | 0.145 ± 0.003 | 0.147 ± 0.000 | 0.149 ± 0.000 | 0.093 ± 0.000 | 3 / 3 |
| LLaVA 10k | logistic | 0.093 ± 0.004 | 0.072 ± 0.005 | 0.074 ± 0.005 | 0.136 ± 0.009 | 0.501 ± 0.003 | 0.330 ± 0.004 | 0.145 ± 0.001 | 0.147 ± 0.000 | 0.149 ± 0.000 | 0.093 ± 0.000 | 2 / 3 |
| LLaVA 10k | direct | 0.171 ± 0.003 | 0.121 ± 0.003 | 0.121 ± 0.003 | 0.270 ± 0.010 | 0.525 ± 0.004 | 0.375 ± 0.009 | 0.137 ± 0.003 | 0.147 ± 0.000 | 0.149 ± 0.000 | 0.093 ± 0.000 | 0 / 3 |
| LLaVA FULL | vcs | 0.095 ± 0.000 | 0.053 ± 0.003 | 0.058 ± 0.003 | 0.153 ± 0.005 | 0.500 ± 0.003 | 0.328 ± 0.004 | 0.143 ± 0.001 | 0.147 ± 0.000 | 0.149 ± 0.000 | 0.093 ± 0.000 | 3 / 3 |
| LLaVA FULL | logistic | 0.098 ± 0.005 | 0.058 ± 0.006 | 0.062 ± 0.006 | 0.168 ± 0.008 | 0.504 ± 0.005 | 0.330 ± 0.006 | 0.143 ± 0.002 | 0.147 ± 0.000 | 0.149 ± 0.000 | 0.093 ± 0.000 | 2 / 3 |
| LLaVA FULL | direct | 0.210 ± 0.004 | 0.169 ± 0.003 | 0.168 ± 0.003 | 0.253 ± 0.007 | 0.544 ± 0.005 | 0.394 ± 0.004 | 0.129 ± 0.001 | 0.147 ± 0.000 | 0.149 ± 0.000 | 0.093 ± 0.000 | 0 / 3 |
| Qwen 10k | vcs | 0.127 ± 0.009 | 0.085 ± 0.008 | 0.087 ± 0.009 | 0.158 ± 0.007 | 0.498 ± 0.002 | 0.333 ± 0.003 | 0.172 ± 0.002 | 0.173 ± 0.000 | 0.159 ± 0.000 | 0.087 ± 0.000 | 3 / 3 |
| Qwen 10k | logistic | 0.140 ± 0.005 | 0.097 ± 0.005 | 0.099 ± 0.005 | 0.186 ± 0.004 | 0.498 ± 0.003 | 0.330 ± 0.004 | 0.170 ± 0.002 | 0.173 ± 0.000 | 0.159 ± 0.000 | 0.087 ± 0.000 | 2 / 3 |
| Qwen 10k | direct | 0.214 ± 0.013 | 0.128 ± 0.014 | 0.128 ± 0.013 | 0.341 ± 0.027 | 0.527 ± 0.014 | 0.364 ± 0.014 | 0.160 ± 0.002 | 0.173 ± 0.000 | 0.159 ± 0.000 | 0.087 ± 0.000 | 0 / 3 |

## B (R8-02): the soft–hard gap decomposed (critic in eval mode, τ_end 0.1, smallest and 25 % budgets)

| scale | objective | units | budget | HD | SD − HD (relaxation) | HN − HD (noise, hard) | SN − HN (relaxation, noisy) | SN − HD (legacy) | logged | soft mass outside hard set (SN) | Σm − k (SN) | dominant |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| LLaVA 10k | vcs | main (3 seeds), lr 0.0003 | 72 | 0.184 ± 0.049 | -0.057 ± 0.006 | 0.000 ± 0.000 | -0.057 ± 0.006 | -0.057 ± 0.006 | -0.057 ± 0.006 | 0.001 ± 0.000 | -0.00 ± 0.00 | relaxation |
| LLaVA 10k | vcs | main (3 seeds), lr 0.0003 | 144 | 0.130 ± 0.029 | -0.035 ± 0.002 | -0.000 ± 0.000 | -0.035 ± 0.002 | -0.035 ± 0.002 | -0.036 ± 0.002 | 0.000 ± 0.000 | 0.00 ± 0.00 | relaxation |
| LLaVA 10k | vcs | other lr, seed 0, lr 0.0001 | 72 | 0.223 | -0.001 | -0.001 | -0.001 | -0.001 | -0.001 | 0.001 | -0.00 | relaxation |
| LLaVA 10k | vcs | other lr, seed 0, lr 0.0001 | 144 | 0.185 | -0.000 | -0.000 | -0.000 | -0.000 | -0.000 | 0.000 | -0.00 | relaxation |
| LLaVA 10k | logistic | main (3 seeds), lr 0.0001 | 72 | 0.254 ± 0.098 | -0.004 ± 0.001 | 0.000 ± 0.000 | -0.004 ± 0.001 | -0.004 ± 0.001 | -0.004 ± 0.001 | 0.000 ± 0.000 | -0.00 ± 0.00 | relaxation |
| LLaVA 10k | logistic | main (3 seeds), lr 0.0001 | 144 | 0.198 ± 0.081 | -0.004 ± 0.003 | -0.000 ± 0.000 | -0.004 ± 0.003 | -0.004 ± 0.003 | -0.004 ± 0.003 | 0.000 ± 0.000 | 0.00 ± 0.00 | relaxation |
| LLaVA 10k | logistic | other lr, seed 0, lr 0.0003 | 72 | 0.381 | -0.047 | -0.000 | -0.047 | -0.047 | -0.047 | 0.000 | 0.00 | relaxation |
| LLaVA 10k | logistic | other lr, seed 0, lr 0.0003 | 144 | 0.254 | -0.019 | -0.000 | -0.019 | -0.020 | -0.020 | 0.000 | -0.00 | relaxation |
| LLaVA 10k | direct | main (3 seeds), lr 0.0001 | 72 | 0.026 ± 0.068 | -0.000 ± 0.000 | -0.000 ± 0.000 | -0.000 ± 0.000 | -0.000 ± 0.000 | -0.000 ± 0.000 | 0.000 ± 0.000 | 0.00 ± 0.00 | mixed, noise, relaxation |
| LLaVA 10k | direct | main (3 seeds), lr 0.0001 | 144 | 0.003 ± 0.068 | -0.000 ± 0.000 | 0.000 ± 0.000 | -0.000 ± 0.000 | 0.000 ± 0.000 | -0.000 ± 0.000 | 0.000 ± 0.000 | -0.00 ± 0.00 | mixed, noise, relaxation |
| LLaVA 10k | direct | other lr, seed 0, lr 0.0003 | 72 | 0.049 | -0.007 | -0.000 | -0.007 | -0.007 | -0.007 | 0.000 | 0.00 | relaxation |
| LLaVA 10k | direct | other lr, seed 0, lr 0.0003 | 144 | 0.037 | -0.004 | -0.000 | -0.004 | -0.004 | -0.004 | 0.000 | -0.00 | relaxation |
| LLaVA FULL | vcs | main (3 seeds), lr 0.0001 | 72 | 0.078 ± 0.013 | -0.002 ± 0.001 | -0.000 ± 0.000 | -0.002 ± 0.001 | -0.003 ± 0.001 | -0.002 ± 0.001 | 0.001 ± 0.000 | 0.00 ± 0.00 | relaxation |
| LLaVA FULL | vcs | main (3 seeds), lr 0.0001 | 144 | 0.063 ± 0.009 | -0.001 ± 0.000 | 0.000 ± 0.000 | -0.001 ± 0.000 | -0.001 ± 0.000 | -0.001 ± 0.000 | 0.001 ± 0.000 | 0.00 ± 0.00 | relaxation |
| LLaVA FULL | vcs | other lr, seed 0, lr 0.0003 | 72 | 0.073 | -0.031 | -0.000 | -0.031 | -0.031 | -0.031 | 0.001 | 0.00 | relaxation |
| LLaVA FULL | vcs | other lr, seed 0, lr 0.0003 | 144 | 0.049 | -0.022 | 0.000 | -0.022 | -0.022 | -0.022 | 0.000 | 0.00 | relaxation |
| LLaVA FULL | logistic | main (3 seeds), lr 0.0001 | 72 | 0.083 ± 0.028 | -0.001 ± 0.001 | 0.000 ± 0.000 | -0.001 ± 0.001 | -0.001 ± 0.001 | -0.001 ± 0.001 | 0.001 ± 0.000 | -0.00 ± 0.00 | relaxation |
| LLaVA FULL | logistic | main (3 seeds), lr 0.0001 | 144 | 0.055 ± 0.021 | -0.001 ± 0.000 | 0.000 ± 0.000 | -0.001 ± 0.000 | -0.001 ± 0.000 | -0.001 ± 0.001 | 0.000 ± 0.000 | 0.00 ± 0.00 | relaxation |
| LLaVA FULL | logistic | other lr, seed 0, lr 0.0003 | 72 | 0.084 | -0.026 | -0.000 | -0.026 | -0.026 | -0.026 | 0.001 | -0.00 | relaxation |
| LLaVA FULL | logistic | other lr, seed 0, lr 0.0003 | 144 | 0.054 | -0.016 | 0.000 | -0.016 | -0.016 | -0.016 | 0.000 | -0.00 | relaxation |
| LLaVA FULL | direct | main (3 seeds), lr 0.0001 | 72 | 0.075 ± 0.113 | -0.002 ± 0.004 | -0.000 ± 0.000 | -0.002 ± 0.004 | -0.002 ± 0.004 | -0.002 ± 0.004 | 0.000 ± 0.000 | 0.00 ± 0.00 | relaxation |
| LLaVA FULL | direct | main (3 seeds), lr 0.0001 | 144 | 0.037 ± 0.078 | -0.002 ± 0.003 | 0.000 ± 0.000 | -0.002 ± 0.003 | -0.002 ± 0.003 | -0.002 ± 0.003 | 0.000 ± 0.000 | 0.00 ± 0.00 | relaxation |
| LLaVA FULL | direct | other lr, seed 0, lr 0.0003 | 72 | 0.050 | 0.011 | -0.000 | 0.011 | 0.011 | 0.011 | 0.000 | 0.00 | relaxation |
| LLaVA FULL | direct | other lr, seed 0, lr 0.0003 | 144 | 0.029 | 0.001 | 0.000 | 0.001 | 0.001 | 0.001 | 0.000 | 0.00 | relaxation |
| Qwen 10k | vcs | main (3 seeds), lr 0.0003 | 0.125 | 0.139 ± 0.007 | -0.034 ± 0.005 | 0.000 ± 0.000 | -0.034 ± 0.005 | -0.034 ± 0.005 | -0.034 ± 0.005 | 0.000 ± 0.000 | -0.00 ± 0.00 | relaxation |
| Qwen 10k | vcs | main (3 seeds), lr 0.0003 | 0.25 | 0.110 ± 0.005 | -0.026 ± 0.003 | -0.000 ± 0.000 | -0.026 ± 0.003 | -0.027 ± 0.003 | -0.026 ± 0.003 | 0.000 ± 0.000 | -0.00 ± 0.00 | relaxation |
| Qwen 10k | vcs | other lr, seed 0, lr 0.0001 | 0.125 | 0.063 | -0.001 | 0.000 | -0.001 | -0.001 | -0.001 | 0.000 | -0.00 | relaxation |
| Qwen 10k | vcs | other lr, seed 0, lr 0.0001 | 0.25 | 0.040 | -0.001 | -0.000 | -0.001 | -0.001 | -0.001 | 0.000 | -0.00 | relaxation |
| Qwen 10k | logistic | main (3 seeds), lr 0.0001 | 0.125 | 0.104 ± 0.009 | -0.003 ± 0.003 | 0.000 ± 0.000 | -0.003 ± 0.003 | -0.003 ± 0.003 | -0.003 ± 0.004 | 0.000 ± 0.000 | -0.00 ± 0.00 | relaxation |
| Qwen 10k | logistic | main (3 seeds), lr 0.0001 | 0.25 | 0.073 ± 0.011 | -0.002 ± 0.002 | -0.000 ± 0.000 | -0.002 ± 0.002 | -0.002 ± 0.002 | -0.002 ± 0.002 | 0.000 ± 0.000 | 0.00 ± 0.00 | relaxation |
| Qwen 10k | logistic | other lr, seed 0, lr 0.0003 | 0.125 | 0.159 | -0.045 | -0.000 | -0.045 | -0.045 | -0.045 | 0.000 | 0.00 | relaxation |
| Qwen 10k | logistic | other lr, seed 0, lr 0.0003 | 0.25 | 0.129 | -0.037 | 0.000 | -0.037 | -0.037 | -0.037 | 0.000 | 0.00 | relaxation |
| Qwen 10k | direct | main (3 seeds), lr 0.0003 | 0.125 | 0.103 ± 0.053 | -0.038 ± 0.033 | -0.000 ± 0.000 | -0.038 ± 0.033 | -0.038 ± 0.033 | -0.038 ± 0.033 | 0.000 ± 0.000 | -0.00 ± 0.00 | relaxation |
| Qwen 10k | direct | main (3 seeds), lr 0.0003 | 0.25 | 0.034 ± 0.029 | -0.017 ± 0.017 | -0.000 ± 0.000 | -0.017 ± 0.017 | -0.017 ± 0.017 | -0.017 ± 0.017 | 0.000 ± 0.000 | 0.00 ± 0.00 | relaxation |
| Qwen 10k | direct | other lr, seed 0, lr 0.0001 | 0.125 | 0.048 | 0.000 | -0.000 | 0.000 | -0.000 | 0.000 | 0.000 | -0.00 | mixed |
| Qwen 10k | direct | other lr, seed 0, lr 0.0001 | 0.25 | -0.003 | 0.000 | 0.000 | -0.000 | 0.000 | 0.000 | 0.000 | 0.00 | mixed |

## C (R8-01): dropout in phase B (R = 8 train-mode draws; 25 % budget)

| scale | objective | eval readout | train readout | distortion ratio | rank agreement R 8 / R 4 | gate top-k Jaccard | phase-B loss: train mode | eval mode (SN) | var T_A | self-gap 2 var T_B |
|---|---|---|---|---|---|---|---|---|---|---|
| LLaVA 10k | vcs | 0.052 ± 0.021 | 0.053 ± 0.020 | 0.042 ± 0.005 | 0.863 ± 0.027 / 0.850 ± 0.026 | 0.643 ± 0.064 | 0.081 ± 0.029 | 0.094 ± 0.029 | 0.0022 ± 0.0007 | 0.0021 ± 0.0007 |
| LLaVA 10k | logistic | 0.081 ± 0.043 | 0.085 ± 0.043 | 0.055 ± 0.025 | 0.873 ± 0.002 / 0.856 ± 0.004 | 0.684 ± 0.037 | 0.181 ± 0.078 | 0.195 ± 0.081 | 0.0075 ± 0.0037 | 0.0097 ± 0.0051 |
| LLaVA 10k | direct | 0.063 ± 0.032 | 0.062 ± 0.033 | 0.001 ± 0.000 | 0.980 ± 0.008 / 0.973 ± 0.012 | 0.849 ± 0.044 | 0.002 ± 0.069 | 0.003 ± 0.068 | — | — |
| LLaVA FULL | vcs | 0.012 ± 0.003 | 0.013 ± 0.003 | 0.056 ± 0.013 | 0.816 ± 0.019 / 0.792 ± 0.017 | 0.615 ± 0.044 | 0.046 ± 0.016 | 0.062 ± 0.009 | 0.0019 ± 0.0006 | 0.0021 ± 0.0005 |
| LLaVA FULL | logistic | 0.014 ± 0.003 | 0.015 ± 0.003 | 0.078 ± 0.023 | 0.819 ± 0.008 / 0.796 ± 0.006 | 0.619 ± 0.040 | 0.041 ± 0.015 | 0.055 ± 0.021 | 0.0021 ± 0.0006 | 0.0020 ± 0.0004 |
| LLaVA FULL | direct | 0.093 ± 0.003 | 0.093 ± 0.003 | 0.003 ± 0.001 | 0.966 ± 0.003 / 0.955 ± 0.005 | 0.713 ± 0.029 | 0.027 ± 0.063 | 0.035 ± 0.075 | — | — |
| Qwen 10k | vcs | 0.050 ± 0.004 | 0.052 ± 0.003 | 0.043 ± 0.003 | 0.852 ± 0.005 / 0.830 ± 0.003 | 0.671 ± 0.003 | 0.081 ± 0.007 | 0.084 ± 0.007 | 0.0045 ± 0.0002 | 0.0055 ± 0.0006 |
| Qwen 10k | logistic | 0.052 ± 0.007 | 0.054 ± 0.004 | 0.032 ± 0.009 | 0.880 ± 0.017 / 0.863 ± 0.018 | 0.814 ± 0.021 | 0.072 ± 0.008 | 0.071 ± 0.009 | 0.0044 ± 0.0008 | 0.0048 ± 0.0018 |
| Qwen 10k | direct | 0.113 ± 0.018 | 0.111 ± 0.018 | 0.001 ± 0.001 | 0.974 ± 0.010 / 0.966 ± 0.011 | 0.829 ± 0.029 | 0.017 ± 0.017 | 0.017 ± 0.017 | — | — |

## Flags (pre-registered, descriptive)

- R8_04_within_qk_not_better_than_chance: False × 15; True × 21
- R8_04_pooled_mostly_between_budgets: False × 35; True × 1
- R8_02_dominant: mixed × 2; noise × 1; relaxation × 33
- R8_02_legacy_reproduced: True × 36
- R8_01_dropout_material_for_ranking: False × 34; True × 2
- R8_01_gate_inputs_perturbed: False × 8; True × 28

Guard: 36 units; max |recomputed − logged| pooled CAL Spearman = 6.73e-07.

## Reading
- **R8-04 (what the selection score measures).**  The frozen pooled CAL score is not mostly between-budget variation (that flag is set for 1 of
  36 units, an other-learning-rate unit: LLaVA FULL vcs lr 3e-4 seed 0).  But inside a question at a fixed budget, the VCS and logistic critics
  choose among the three mask mechanisms at chance: pairwise accuracy 0.498–0.500 (VCS) and 0.498–0.504 (logistic), top-1 hit 0.328–0.333
  (chance 1/3), and a regret equal to random choice (e.g. LLaVA 10k 0.145 vs 0.147; Qwen 10k 0.172 / 0.170 vs 0.173).  The 95 % interval
  contains 0.5 for 12 of 12 VCS units and 9 of 12 logistic units.  The direct-risk critic is modestly above chance (0.525–0.544, 0 of 12
  intervals contain 0.5) with lower regret (0.129–0.160).  The teacher's own d choice, a reference that uses the host's outputs, has regret
  0.087–0.093.
- **R8-02 (the soft–hard gap).**  The logged gap is reproduced in 36 of 36 units and is relaxation, not Gumbel noise: the noise part HN − HD is
  0.000 in every row, so SN − HD equals SD − HD.  The gap is material only where phase A used learning rate 3e-4: e.g. LLaVA 10k vcs (frozen
  3e-4) −0.057 at 72 tokens, Qwen 10k vcs (frozen 3e-4) −0.034 at 12.5 %, Qwen 10k logistic at 3e-4 −0.045; at 1e-4 every |SD − HD| ≤ 0.004.
  The soft masks are almost hard (soft mass outside the hard set ≤ 0.001; Σm − k ≈ 0), so at 3e-4 the critic reads a lower risk from
  near-identical fractional masks than from the hard top-k it is deployed with.  The 'noise' / 'mixed' labels (3 units) are direct-risk units
  whose gaps are ~0, where the dominance label carries no information.
- **R8-01 (dropout in phase B).**  Dropout barely moves the critic's ranking (distortion ratio 0.001–0.078 at 25 %, above the 0.25 marker at some
  budget in 2 of 36 units: LLaVA FULL logistic 1e-4 seed 1 and vcs 3e-4 seed 0; rank agreement 0.816–0.980 at R = 8).  It does change the
  gate's inputs: the train-vs-eval top-k Jaccard at the 25 % budget is 0.615–0.671 for VCS and 0.619–0.814 for logistic (below 0.8 in 11 / 12
  and 10 / 12 units; 7 / 12 for direct risk), i.e. phase B trains the gate on token features that differ from the eval-mode features it sees at
  deployment.  The phase-B loss under dropout is below its eval-mode value for VCS on all three scales (0.081 vs 0.094, 0.046 vs 0.062, 0.081 vs
  0.084).
- **For D1.**  Three candidate contributors, each measured on the critic only: (i) near-chance within-question mechanism choice by the VCS and
  logistic critics (the direct-risk critic is better there), (ii) a soft–hard relaxation gap that appears with phase-A learning rate 3e-4, and
  (iii) dropout-perturbed gate inputs during phase B.  Whether any of them causes the D1 shortfall is not identified here; a paired retraining
  (e.g. phase B in eval mode, or hard-mask evaluation during gate training) would test it and is the owner's decision.
