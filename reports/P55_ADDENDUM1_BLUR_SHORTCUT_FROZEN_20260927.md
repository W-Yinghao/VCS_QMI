# P55 addendum 1 — DRAFT: D-T second shortcut family (Gaussian-blur shortcut), 2026-09-27 — FROZEN 2026-09-27T19:39:44Z before GPU compute (CPU probes 1011781, 1011797 disclosed inside; owner: "都补充上"; the radius-1.5 contingency is pre-declared here and may be used once)

Status: DRAFT written by the implementation fork; the main session freezes it (rename `*_FROZEN_*`) before the six GPU training jobs.
Parent prereg: `P55_TASK_D_SHORTCUT_PREREG_FROZEN_20260927.md` (colour-temperature shortcut).  Owner 2026-09-27: "都补充上".
Code: `scripts/task_d_shortcut.py --shortcut blur` (additive; `--shortcut colour`, the default, is byte-identical to the frozen P55 path — checked
on a synthetic batch: identical tensors; the manifest gains the key `shortcut`), `slurm/task_d_shortcut.sbatch` (`SHORTCUT=blur`); tests unchanged
(`scripts/precheck_d_tests.py`, one case per job via `slurm/precheck_d_cases_gpu.sbatch`).

**Question.** Does the D-T result hold for a *texture* shortcut, i.e. one that lives in the high-frequency content rather than in colour?

**Design (only the shortcut changes).** Same splits (TRAIN 40 000 / HELD 5 000 by the fixed permutation, seed 20260927; the 5 000 selection
images for reliance; sha256 of every uid list in each model's `manifest.json`), same models and training (supervised CIFAR ResNet-18 + linear
head, SGD 0.9 Nesterov, lr 0.1 per-step cosine, wd 5e-4, batch 256, 15 epochs, RandomCrop(32, pad 4) + flip), same shortcut rule N_i = parity(y_i)
with probability ρ.  Shortcut: images with N_i = 1 are blurred with PIL `GaussianBlur(radius = 1.0 px)` at 32 × 32 (the same operator as
`precheck_d_features.py --nuisance blur`), planted once, deterministic per image.  Models: ρ ∈ {0.5, 0.95, 1.0} × seeds {0, 1} = 6 (ρ = 0.8 dropped
against the colour family, which already showed the ρ-monotonicity there; ρ = 0.5 is the no-shortcut reference with blur present but uninformative).

**Reliance (ground truth).** On the 5 000 selection images: acc_clean, acc_matched (blur on N = parity(y)), acc_flipped (blur on N = 1 − parity(y));
reliance = acc_matched − acc_flipped.

**Detection.** HELD images with N ~ Bernoulli(½) independent of Y (rng [20260927, seed]); case `cond_blur1` = blur planted on N = 1; case `cond_null`
= same N, nothing planted (exact conditional independence).  h = penultimate 512-d features.  Tests as P45/P46 with within-class shuffling:
vcs_perm, vcs_hoeff (VAL-picked critic; linear / MLP / closed-form reported), hsic_perm, c2st; n ∈ {200, 500, 1000}, R = 50, δ = 0.05,
200 permutations (`--cond-sizes 200,500,1000 --cond-repeats 50`); the unconditional tests on the same draws reported alongside.

**Pre-committed reading (verbatim from the frozen P55 prereg, with ρ = 0.8 dropped and "colour" read as "blur").**
(i) mean conditional Ĵ increases with ρ and Spearman(Ĵ, accuracy drop) ≥ 0.9 over the 6 (ρ, seed) models; (ii) at n = 500 the VCS
permutation test detects every ρ ≥ 0.95 model with power ≥ 0.8 and HSIC is not more than 0.05 above it; (iii) s = 0 rejection ≤ 0.09.
Holds / conditionally (one of i–iii) / does not hold (Ĵ not monotone in ρ, or power < 0.5 at ρ = 0.95).
Clarifications as in P55: "mean conditional Ĵ" = Ĵ_eval of the VAL-picked critic at n = 1000 in `cond_blur1`, averaged over repeats; the Spearman
is computed over the 6 models; (iii) is read on `cond_null` at every n for every test (R = 50 → ≤ 0.09 means ≤ 4/50).  The guaranteed test is
reported on its own curve (smallest n reaching 80 % per ρ) and does not enter the verdict.  With ρ = 0.8 absent, clause (ii) applies to the four
ρ ≥ 0.95 models.

**QC sentinels.** Training accuracy ≥ 0.9 at ρ = 0.5 after 15 epochs (else under-trained: re-run, not read); `corr_N_Y_even` in HELD within
± 0.05; `train_agree_frac` within ± 0.02 of ρ; reliance at ρ = 1.0 > reliance at ρ = 0.5.

**Not claimed.** Anything beyond two planted shortcut families (global colour temperature, global Gaussian blur) on CIFAR-10; nothing about
mitigation, about real-world shortcuts, or about localised / object-level shortcuts; C2ST is the equal-budget MLP of P46.

**Probe.** CPU smoke (`SMOKE=1 SHORTCUT=blur`: ρ = 1.0, 2 epochs, 2 000 training images, HELD 1 000, tests `--smoke`) must finish end to end with
reliance > 0 and the tests accepting the feature dir before the six GPU training jobs are submitted.  Result appended below by the fork.

**Compute.** 6 training jobs (one GPU each, ≈ 5–10 min) + 12 test jobs (2 cases × 6 models, ≈ 5 min each).

**Probe results (CPU, disclosed).**
- Smoke 1 (job 1011781, `slurm_logs/task_d_shortcut_tds_blur_smoke_1011781.out`; 2 epochs, 2 000 images, ρ = 1.0): pipeline end to end, cases
  `cond_blur1` / `cond_null` written (h 1000 × 512, N1 0.51), tests accepted the dir; but the 2-epoch net had not learned anything (train acc 0.12,
  selection acc 0.11) and reliance was −0.004 — the gate "reliance > 0" was not met, so a longer probe was run.
- Smoke 2 (job 1011797, `slurm_logs/task_d_shortcut_tds_blur_smoke2_1011797.out`; 12 epochs, 4 000 images via the new `--smoke-epochs /
  --smoke-train` knobs, ρ = 1.0): train acc 0.29, selection acc clean 0.267 / matched 0.268 / flipped 0.262, **reliance +0.006 > 0**; tests
  `--smoke` (R = 3) ran on both cases.  Gate met, weakly: at this training scale the blur shortcut is far less readily exploited than the colour
  one (colour smoke at 2 epochs: reliance +0.087).  The full 15-epoch / 40 000-image models decide whether it is exploited at all.
- Pre-committed contingency (proposed to the main session for the freeze): if the full ρ = 1.0 models show reliance < 0.05 the unit is reported as
  "shortcut not learned at radius 1.0" (a property of the classifier, not of the test) and re-run once at radius 1.5 with the same rules, both
  runs disclosed.
