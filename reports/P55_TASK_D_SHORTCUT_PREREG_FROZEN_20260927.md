# Pre-registration — task-level pre-check D-T: shortcut-reliance detection in trained classifiers (P55), FROZEN 2026-09-27T14:48:57Z before any GPU compute

Status: FROZEN (this commit); the CPU smoke probe (jobs 1011080/1011081) ran before freezing and is disclosed below; the D-S3 blur family is covered by P45 addendum 5.  Design source:
`SECOND_APP_WAVE2_PLAN_20260927.md`, row D, unit D-T.  Code: `scripts/task_d_shortcut.py`, `slurm/task_d_shortcut.sbatch`; tests unchanged
(`scripts/precheck_d_tests.py`, one case per job via `slurm/precheck_d_cases_gpu.sbatch`).

**Question.** Given a classifier trained with a known colour shortcut of controllable strength, does the conditional VCS statistic
S(h; N ∣ Y) — permutation-calibrated and, separately, with the distribution-free bound — detect the representation's dependence on the
nuisance beyond the label, with power comparable to HSIC and above C2ST at small n, and does its magnitude track the model's measured
reliance?  (Family: leakage / shortcut detection; brief appendix A row D.)

**Data.** CIFAR-10 official training partition only, frozen split `dev45k_val5k` (manifest sha256 c35d7cd3…).  Fit pool 45 000 → a fixed
permutation with seed 20260927 gives HELD (first 5 000 uids; detection only) and TRAIN (40 000); the 5 000 selection images measure reliance.
Sha256 of each uid list is written to every model's `manifest.json`.  No official test set.

**Models (8).** Supervised ResNet-18 with the CIFAR stem (the SSL trunk; no pretrained weights) + linear head; SGD momentum 0.9 (Nesterov),
lr 0.1 with per-step cosine to 0, no warm-up, weight decay 5e-4, batch 256, 15 epochs, augmentation RandomCrop(32, pad 4) + horizontal flip;
ρ ∈ {0.5, 0.8, 0.95, 1.0} × seeds {0, 1}.  Shortcut: for each training image N_i = parity(y_i) with probability ρ, else 1 − parity(y_i)
(rng seeded by [seed, 20260927]); images with N_i = 1 get the colour-temperature shift R × 1.2, B × 0.8 on uint8 (s = 0.2), planted once.
ρ = 0.5 is the no-shortcut reference (colour present but uninformative).

**Reliance (ground truth).** On the 5 000 selection images: acc_clean, acc_matched (N = parity(y), shift planted), acc_flipped
(N = 1 − parity(y)); reliance = acc_matched − acc_flipped.

**Detection.** HELD images with N ~ Bernoulli(½) drawn independently of Y (rng [20260927, seed]); case `cond_colour0.2` = shift planted on
N = 1; case `cond_null` = same N, nothing planted (exact conditional independence).  h = penultimate 512-d features.  Tests as P45/P46 with
within-class shuffling: vcs_perm, vcs_hoeff (VAL-picked critic; linear / MLP / closed-form reported), hsic_perm, c2st; n ∈ {200, 500, 1000},
R = 50, δ = 0.05, 200 permutations (`--cond-sizes 200,500,1000 --cond-repeats 50`).  The unconditional tests on the same draws are reported
alongside (they should also reject in `cond_colour0.2`, since N ⊥ Y here).

**Pre-committed reading (verbatim from the plan).**
(i) mean conditional Ĵ increases with ρ and Spearman(Ĵ, accuracy drop) ≥ 0.9 over the 8 (ρ, seed) models; (ii) at n = 500 the VCS
permutation test detects every ρ ≥ 0.8 model with power ≥ 0.8 and HSIC is not more than 0.05 above it; (iii) s = 0 rejection ≤ 0.09.
Holds / conditionally (one of i–iii) / does not hold (Ĵ not monotone in ρ, or power < 0.5 at ρ = 0.95).
Clarifications fixed now: "mean conditional Ĵ" = Ĵ_eval of the VAL-picked critic at n = 1000 in `cond_colour0.2`, averaged over repeats;
the Spearman is computed over the 8 models; (iii) is read on `cond_null` at every n for every test (R = 50 → ≤ 0.09 means ≤ 4/50).
The guaranteed test is reported on its own curve (smallest n reaching 80 % per ρ) and does not enter the verdict.

**QC sentinels.** Training accuracy ≥ 0.9 at ρ = 0.5 after 15 epochs (else the model is under-trained and the unit is re-run, not read);
`corr_N_Y_even` in HELD within ± 0.05; `train_agree_frac` within ± 0.02 of ρ; reliance at ρ = 1.0 > reliance at ρ = 0.5.

**Not claimed.** Anything beyond one shortcut family (global colour temperature) on CIFAR-10; nothing about mitigation; no statement on
real-world shortcuts; C2ST is the equal-budget MLP of P46 (a stronger classifier would move its curve).

**Probe.** CPU smoke (`SMOKE=1`: 2 epochs, 2 000 training images, ρ = 1.0, HELD 1 000, tests `--smoke`) must finish end-to-end with
reliance > 0 and the tests accepting the feature dir, before the 8 GPU training jobs are submitted.

**Compute.** 8 training jobs (one GPU each, ≈ 5–10 min) + 16 test jobs (2 cases × 8 models, ≈ 5 min each).
