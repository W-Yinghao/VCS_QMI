# Pre-registration — P109: package v4 modules X1 (training score vs independent measurement), I1 (paired conditional nuisance audit), I2 (nested-critic / exact-product pilot), 2026-10-01 — FROZEN 2026-10-01T18:35:27Z before the measurement jobs (CPU gate 1017246, CPU timing 1017271)

Status: FROZEN (main session, see the title time); owner 2026-10-01 shared the v4 package; standing go "除了imagenet的先不提交，后续都可以提交".
Source: `VCS_Next_Experiments_v4_Package.zip` → `VCS_Next_Experiment_Plan_v4_CN.md` §X, §I, §4–5; owner's standing go "除了imagenet的先不提交，后续都可以提交".
No SSL pretraining. Only the official TRAIN file is used: the selection split and FIT images of the frozen 45k/5k split. The official test set stays closed.

Code: `src/vcs_measure/{common,xmeasure,audit,nested}.py`, `scripts/p109_{x1,i1_features,i1_audit,i2_pilot}.py`, `slurm/p109_{unit,gate,timing}.sbatch`,
`tests/test_p109.py` (9 tests). The code reuses `precheck_d_features.PlantedDataset` (the P45 colour and blur constructors), the T1 / P105 statistics
(`closed_form_critic`, `exact_js_critic`, `within_class_pool`) and the P102 task (`t_line_increments`: Family, maps, information sets) unchanged, by import.

## Encoders (frozen, read-only; only COMPLETED runs are measured — the loader refuses anything else)
| family | runs now | runs added when P104 confirmation completes |
|---|---|---|
| B0 recipe VCS | `P35_vcs_a5_views4_800ep_seed{0,1,2}` | — |
| G2 (fixed a, b = 2, −1) | `P104_G2_views4_800ep_seed0` | seeds 1, 2 |
| U2 (fixed 1, 0, all-view tokens) | `P104_U2_views4_800ep_seed0` | seeds 1, 2 |
| tuned SimCLR, same protocol | `P41_simclr_views4_800ep_seed{0,1,2}` | — |

"Confirmed G2 / U2" in the plan means these runs are included after the P104 confirmation seeds finish. Inclusion does not depend on how those seeds score.

## X1 — training score vs an independent measurement-critic family vs downstream quality
- **Data.**
  - The 5 000 selection images, which SSL fitting never used, split by base-image ID into FIT 3 000 / TUNE 1 000 / EVAL 1 000. The split seed is fixed and identical for every run.
  - P pairs are two train-distribution augmentations of the same base image. Q pairs are K = 8 random nonzero cyclic-shift partners, taken inside the same split and the same augmentation draw, so a Q pair never joins two augmentations of one base image.
  - FIT uses 2 augmentation draws; TUNE and EVAL use 1 draw each. All seeds are fixed.
  - Representations: z, the L2 projector output that the VCS critic reads, and L2(h), the evaluated representation.
- **Measurement family.** Every candidate is fitted on FIT only.
  - `zero`: T ≡ 0.
  - `cosine`: calibrated angular critic, (a, b) fitted by L-BFGS on −J.
  - `mlp`: symmetric pair MLP on [u1 ⊙ u2, |u1 − u2|], hidden 256, AdamW; the step is chosen on TUNE.
  - `prod_ridge`: closed-form ridge of the squared score in φ = [u1 ⊙ u2, 1], with tanh(c φw) and both λ and c chosen on TUNE.
  - `rff_ridge`: the same closed form on 1 024 random Fourier features, bandwidth = median × {0.5, 1, 2} chosen on TUNE.
  - The pick is the candidate with the highest TUNE J, with `zero` included. **EVAL J is recorded unclipped for every candidate and for the pick.**
- **Reported alongside, per encoder:**
  - the run's own training critic scored on the identical TUNE / EVAL pairs (VCS runs only; SimCLR has no critic);
  - the training evaluation's held-out J;
  - the final frozen linear and kNN accuracy, as stored by training.
- **Labels are not used in X1.**
- **Pre-stated reading (descriptive; no ranking of methods by J).**
  - Measurement J is a lower readout over a finite function class. It is never called S, and the highest J does not stand in for representation quality.
  - Per encoder: the gap between measurement J and training-critic J on the same pairs. Per family: mean ± sd over seeds.
  - The spread across measurement candidates on one encoder is reported as measurement uncertainty.
  - The question "does G2 reach better downstream quality at a lower training score, while an independent refit still recovers a strong pairing" is answered by the table only.
  - Cross-model correlations are described, not tested: at most 12 encoders, and seeds of one family are not independent methods.

## I1 — paired conditional nuisance audit (presence → retention → prediction use)
- **Base images.**
  - 20 000 of the 45 000 FIT images, split by base-image ID into FIT 10 000 / VAL 2 000 / EVAL 8 000, with a fixed seed shared by every run.
  - Every version of a base image stays in its split.
  - Sampling unit = base image: per repeat, base images are drawn without replacement inside a split, and no image is reused across FIT / EVAL / POOL.
- **Versions** (the P45 constructors; one clean and one planted version of every base image):
  - clean;
  - colour shift s ∈ {0.05, 0.1, 0.2};
  - Gaussian blur radius s ∈ {0.25, 0.5, 1.0} px.
- **Predefined layer set:** layer3 (global-average-pooled, 256), h (512), z (128), logits (10).
  - The logits come from a multinomial logistic classifier on standardised clean h, fitted on the **FIT base images only**.
  - **Label use, disclosed:** class labels train this analysis classifier and define Y in the conditional design. They never enter SSL pretraining.
- **Presence and retention tests.**
  - Per repeat: N | Y ~ Bernoulli(½ ± 0.3 by class parity), drawn afresh.
  - The observed feature is the planted version iff N = 1. Because each base image has both versions stored, N can be redrawn freely without conditioning on a stored draw.
  - Exact nulls: `null_label_only` (clean for all) and `null_all_planted` (planted version at the family's largest strength for all; N independent of the image given Y).
  - Product negatives: the N of a same-class POOL item (VAL split).
  - Statistics per layer: the exact linear critics of P105, i.e. the VCS closed form and the exact JS solve. Both are fitted on the FIT sample, with standardisation from the FIT sample.
    These two are chosen because P106 showed they are the most powerful of the T1 statistics and that the two objectives are on par with them.
  - B = 200 within-class permutations of N on EVAL, shared by every layer and both statistics.
  - **Multiplicity:** per-layer raw p-values, plus family-wise adjusted p-values across the four predefined layers by the **max-statistic** over layers of the permutation-standardised statistic, using the same permutation index for every layer.
  - Primary presence readout = h, raw p. Retention = the per-layer max-T-adjusted rejections. "Any layer" = the max-T family rejection.
- **Cells.**
  - Power: one seed-0 encoder of each family (P35 s0, G2 s0, U2 s0, SimCLR s0) × {colour, blur} × 3 strengths × n ∈ {500, 2 000}, R = 100.
  Seeds are 401 onwards, one per job.
  - Level: `null_label_only` at n ∈ {500, 2 000} and `null_all_planted` at n = 2 000, R = 200.
- **Prediction effect (no test).**
  - On all 8 000 EVAL base images, compare the clean and planted version of the same image, for every planted version.
  - Measures: change in true-class probability, change in margin (true logit − best other), accuracy clean → planted, prediction flip rate.
  - Reported per class and overall, with a 95 % bootstrap CI over base images (1 000 reps).
  - Effects are computed for every encoder, all seeds.
- **Pre-stated reading.**
  - Level: a statistic above 0.09 at R = 200 is flagged, and its power cells at that layer are not read. Nominal level is 0.05; the binomial 95 % half-width is about 0.03.
  - Presence, retention and prediction use are reported **separately**. In particular, high dependence with near-zero prediction change is reported as "retained but not used".
  - Strength need not be monotone in the dependence, and blur at σ 1.0 can damage real class evidence. Accuracy drops are not labelled "shortcut".
  - Descriptive comparison across encoder families. No method ranking is made from one seed per family.

## I2 — pilot: nested critics and the exact product term (P102 task, measurement-algorithm validation only)
- **Task.** Identical to P102: P45 features, information sets h ⊇ pca64 ⊇ pca16 ⊇ logit16 and h ⊇ logit_h, maps fitted on PROBE, n = 3 000, 5 repeats, 300 steps, VCS and matched JS.
- **Cells.**
  - Zero dependence: VCS-encoder colour s = 0.
  - Two non-saturated P102 cells: VCS-encoder colour s = 0.1 and VCS-encoder blur σ = 0.5.
- **Four fitting variants on the same draws:**
  - `indep_sampled`: the P102 baseline.
  - `indep_exact`: the Q term enumerated over N ∈ {0, 1} with the known P(N | Y), in the fit loss.
  - `nested_sampled`: the fine critic = frozen coarse critic + residual with a zero-initialised output, fitted bottom-up along the chain (and logit_h → h for the main pair). Step 0 equals the coarse model and is a selectable candidate.
  - `nested_exact`: nesting and exact enumeration together.
- **Readouts.** Every variant is read out with both the sampled and the exact Q readout, so the fitting change and the readout change are separable.
- **Recorded:** J per set, main-pair ΔJ / D_T / r_BA, chain steps, R_orth, nesting violations (J_fine < J_coarse), and refit spread (sd over repeats).
- **Pre-stated reading.**
  - "Error reduced" for a variant needs **all three** of the following in the two non-saturated cells, against `indep_sampled` with the sampled readout:
    1. |R_orth| is at least 50 % smaller;
    2. the mean nesting violations are lower;
    3. the null cell's |ΔJ| floor is not larger.
  - Otherwise: "no clear reduction".
  - Only if a variant reduces the error does the plan's broader measurement use it (a separate prereg).
  - **Prohibited:** clipping EVAL increments, enforcing monotonicity or orthogonality on EVAL, or calling a nested fit's R_orth ≈ 0 a validation of the theory. A nested fine critic satisfies nesting by construction, so its R_orth measures **fit quality, not representation geometry**.

## Smoke (disclosed)
- **CPU gate, job 1017246 (gate_rc 0), outputs in `reports/P109_GATE_1017246/`.**
  - Tests: `tests/test_p109.py` 9/9 and the P102 suite 6/6 pass.
  - **X1** smoke (300 / 200 / 200 images, 1 draw, 100 MLP steps) on G2 s0 and SimCLR s0, all candidates ran:

    | encoder | rep | picked | EVAL J (pick) | candidate EVAL J range | training critic EVAL J |
    |---|---|---|---|---|---|
    | G2 s0 | z | rff | 0.894 | 0.894–0.904 | 0.843 |
    | SimCLR s0 | z | cosine | 0.920 | — | — |

    This took 22 s on CPU.
  - **I1 features** smoke (600 / 300 / 600 base images, G2 s0): 24 s on CPU. Classifier accuracy is 0.863 clean (EVAL), 0.855 at colour s0.2 and 0.667 at blur s1.
  - **I1 audit** smoke (R = 3, n = 200, 20 permutations):
    - planted colour s0.2: the any-layer max-T rejection fires in 3 of 3 repeats for both statistics;
    - null: 0 rejections;
    - effects: Δ P(true) −0.003 [−0.009, 0.002] at colour s0.2, flip rate 0.03.
  - **I2** smoke (n = 300, 2 repeats, 30 steps), colour s = 0:

    | variant | ΔJ | R_orth |
    |---|---|---|
    | independent, sampled | +0.0070 | +0.021 |
    | nested | +0.0012 | +0.0002 |

  - All smoke values are code-path evidence only. Nothing in the design changed after the smoke.
- GPU timing job: see Cost.

## Cost
Measured on the CPU partition with 16 CPUs, at full settings with one unit per module.
Job 1017271, `slurm/p109_timing.sbatch` run with `-p CPU`; outputs in `outputs/P109_timing_1017271/` (not results).
A GPU timing job, 1017247, was queued behind the P104 runs and cancelled. The CPU numbers show every unit fits the CPU partition.

| unit | measured | count | total |
|---|---|---|---|
| X1, one encoder (z and h, all candidates) | 196 s | 8 now + 4 later | ≈ 40 min |
| I1 features, one encoder (20 000 base images × 7 versions × 4 layers + classifier) | 550 s on CPU (a GPU is expected to take ~1–2 min) | 8 + 4 | ≈ 1.8 CPU-h, or ≈ 0.4 GPU-h |
| I1 audit, n = 2 000 | 3.3 s per repeat | | |
| I1 audit, n = 500 | ≤ 5 s per repeat (L-BFGS converges more slowly at small n) | | |
| I1 audit jobs | ≤ 5 s per repeat | 8 power jobs × 600 repeats + 8 level jobs × 600 repeats | ≤ 13 CPU-h; ≤ 50 min wall each |
| I1 effects | seconds (plus ≈ 10 s feature load) | 8 + 8 effects-only jobs | negligible |
| I2, one cell × one repeat (two objectives, four variants, two readouts) | 273 s | 3 cells × 5 repeats | ≈ 70 min |

**Total ≈ 16 CPU-node-hours plus ≤ 0.5 GPU-h** (feature extraction). Stages 1 and 2 take about 2–3 h of wall clock if the CPU partition is free.
The CPU units do not compete with the P104 / P107 / P108 GPU work.
The launch lines are `slurm/p109_lines.txt` (37 lines, DRAFT header): 35 on the CPU partition and the 2 feature extractions on one GPU, normal QOS.

**Timing-run observations.** These are code-path checks at full settings, 10 repeats or 1 repeat — not results and not read.
- I1 on P35 s0, colour s0.1:
  - n = 2 000: h presence power 1.00 for both statistics; logits 0.50.
  - n = 500: h 0.70 (VCS closed form) and 1.00 (exact JS).
  - Effects at colour s0.2: Δ P(true) −0.025 [−0.029, −0.021], accuracy 0.830 → 0.803, flips 0.099.
  - Classifier EVAL accuracy at blur s1 is 0.427, against 0.830 clean.
- I2, colour s0.1, one repeat, VCS objective:

  | variant | R_orth | nesting violations |
  |---|---|---|
  | `indep_sampled` | +0.022 | 2 |
  | `indep_exact` | +0.025 | — (r_BA −0.037 → −0.015) |
  | `nested_sampled` | −0.004 | 0 |
  | `nested_exact` | +0.008 | 0 |


## Delivery (v4 §5 fields, per unit)
Research question, primary endpoint, controls and completed scope.
Splits and base-image counts, numbers of P / Q pairs, repeats.
Raw objective, critic / score form, training path.
Config / checkpoint hash, run status.
Per-seed or per-repeat values, mean, sample sd, paired differences.
GPU / CPU time and peak memory.
Training score and independent measurement kept separate; variational J kept separate from any S.
Main conclusion, mechanism inference and open interpretations kept separate.

## Decisions at the freeze (main session)
1. Scope as drafted: X1 on the four families of v4 (recipe VCS P35 s0–2, tuned SimCLR P41 s0–2, G2, U2); G2F / U2F are not added (their role is
   the §4.1 SSL control, read in P104).  I1 power cells on each family's seed-0 encoder; effects-only units on the other seeds.  X2 and I3 not built.
2. Launch: stage 1 now; stage 2 with a SLURM dependency on the stage-1 feature job (afterok); stage 3 through the orchestrator once
   `P104_{G2,U2}_views4_800ep_seed{1,2}` have their epoch-800 evaluation.  Normal QOS only.
