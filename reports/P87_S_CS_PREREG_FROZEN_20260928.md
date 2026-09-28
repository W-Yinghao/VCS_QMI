# Pre-registration — P87: S-CS, the direct CS-estimator comparison inside the full visual SSL protocol (VCS-N vs S-Kernel vs CS-K-native), 2026-09-28 — FROZEN 2026-09-28T20:14:12Z before GPU compute (S-line CPU gate 1013329 passed: ssl_core 29, integration 40, v2 14; frozen recipe first-step gradients unchanged to 1e-4; CIFAR-100 manifest sha256 96dbd6d4…5430)

Status: FROZEN 2026-09-28T20:14:12Z (main session).  The main session freezes it (renamed `*_FROZEN_*`, freeze time in the title) before any GPU job.  GPU priority follows the
owner's rule (package v2 units first).  Sources: Server Spec v2 §3.1–3.4 (control definitions, kernel conventions, numerical checks), §6.1–6.3 (tables A / B,
implementation alignment, full training), §12–13 (tests, delivery); Plan v2 §4.1, §6.1; `reports/V2_RECONCILIATION_20260928.md` (row P87); owner go 2026-09-28
("整理一下现在的所有文档、成果 … 然后开始工作 … 完成 CVPR 的所有准备").

## Question
Two tables, two questions (Spec §6.1; they are not the same comparison):

- **Table A — same target, different estimation.**  The VCS objective J, the K = 8 cyclic-shift negatives, the negative-partner detach, the 4-view
  averaging, the encoder input z_l2 and the whole training recipe are held fixed; only the critic's function class changes: the neural cosine–tanh critic
  (**VCS-N**, 2 trainable parameters a, b) versus a fixed random-Fourier-feature critic with a trainable linear read-out (**S-Kernel**).  Question: does the
  neural critic give a better frozen representation than a same-target kernel-feature critic?
- **Table B — the classical CS learning method.**  **CS-K-native** trains the same encoder with the classical kernel (plug-in) Cauchy–Schwarz QMI on the paired
  representations: fixed reference measure, native autodiff through its three kernel statistics, no critic, no negative branch.  Question: does the classical
  CS route reach the VCS-N representation quality under the same encoder, augmentation, data, evaluation and budget?  This comparison includes the change of
  objective definition and of computation and is not attributed to "estimator replacement" alone.

## Definitions (code: `src/vcs_ssl/models/critic.py::RFFTanhCritic`, `src/vcs_ssl/kernel_cs.py`, `src/vcs_ssl/objectives.py`, `src/vcs_ssl/train.py::calibrate_bandwidths`)
- **S-Kernel** (Spec §3.4): w = [z1; z2] ∈ R^{256}; φ(w) = √(2/m) cos(Ω w + b), Ω = Ω0/σ with Ω0 ~ N(0, I) and b ~ U(0, 2π) drawn once from the critic's own
  seed stream (sha256 of Ω0 and b recorded); T = tanh(θᵀ[φ(w); 1]); θ (m + 1 parameters) is the only trainable part, initialised Xavier-uniform with the recipe's
  gain 0.1 (an exactly zero θ would block the encoder gradient at step 1).  σ = `rff_bandwidth_multiple` × the median pairwise distance of the pair vectors w
  over the positive pairs plus one cyclic-shift partner per image (an M-like pool), measured once at the start of the run on the first 1 024 fit images.
  Everything else (J, K = 8, detach of the shifted partner, 6 view pairs, AdamW 1e-3, warm-up 10, 800 epochs) is the VCS-N recipe.
- **CS-K-native** (Spec §3.2): for every view pair, K_ij = exp(−‖z1_i − z1_j‖²/2σ²), L_ij = exp(−‖z2_i − z2_j‖²/2σ²) on the L2-normalised projector outputs,
  A = B⁻² Σ K_ij L_ij, B_q = (B⁻² Σ K_ij)(B⁻² Σ L_ij), C = B⁻³ Σ_i (Σ_j K_ij)(Σ_j L_ij), D_CS = log A + log B_q − 2 log C; loss = −D_CS averaged over the 6 view
  pairs.  All three terms are computed in the log domain (no product can underflow); the fraction of entries whose naive float32 product would underflow is
  logged as `kcs_underflow_frac` (the guard's activation frequency, Spec §3.2).  σ = `kernel_cs_bandwidth_multiple` × the median pairwise distance of z_l2
  (both views pooled) on the same calibration batch.  Kernel convention (disclosed): the Gaussian kernel above is used *directly* as the kernel of the plug-in
  statistic; it equals the KDE-integrated "effective" kernel of a KDE with bandwidth σ/√2.  The E-line module's `kernel_cs_native(effective=True)` uses the
  √2 convention; the two implementations agree exactly when called with the same Gram kernel (test `test_kernel_cs_extremes_gradcheck_chunk_and_estim_equivalence`).
  Extremes verified: all-ones kernels → 0; identity kernels → log B (the plug-in diagonal effect, not dependence); gradcheck in float64; chunked = dense.
- **Calibration batch** (both): the first 1 024 fit UIDs, two training-distribution views (RNG seed = run seed + 515151), forward in training mode (batch
  statistics = the distribution the objective sees at step 0) with BN buffers restored afterwards; recorded in `bandwidth_calibration.json`, the run manifest
  and every checkpoint; resume refuses a differing kernel σ.  Disclosed choice: training-mode BN (the existing `calibrate_cosine_bias` precedent), not eval mode.
- Multi-view: CS-K-native uses the same "objective averaged over the 6 view pairs" as the VCS recipe (policy check extended for it; `run.control_tuning` is not
  needed).  It keeps its own kernel statistics; no "same negative branch" claim is made for it (Spec §6.1).

## Cells
**Dev grid (stage `P87_s_cs_dev`, 1× budget = 4 views, B 128, 100 epochs, seed 0; base config `cifar10_hpM_a5_views4_b128_100ep_vcs_seed0.yaml`).**
Reference VCS-N at 1×: 83.19 ± 0.40 linear / 78.43 kNN (P39 seeds 0–2: 83.52 / 82.74 / 83.30).

| family | grid | configs (`configs/S_LINE_SHA256.json` → stage P87_s_cs_dev) |
|---|---|---|
| S-Kernel | m ∈ {256, 1024, 4096} × bandwidth multiple ∈ {0.5, 1, 2} (9) | `cifar10_hpS_skernel_m{m}_bw{bw}_1x_seed0.yaml`, run `P87_skernel_m{m}_bw{bw}_1x_seed0` |
| CS-K-native | bandwidth multiple ∈ {0.5, 1, 2} (3) | `cifar10_hpS_kcs_bw{bw}_1x_seed0.yaml`, run `P87_kcs_bw{bw}_1x_seed0` |

**Selection rule (pre-stated).**  Per family, the configuration with the highest selection-split linear top-1 at epoch 100.  Ties within 0.1 points → the
smaller m, then the multiple 1.0.  A cell that fails numerically (non-finite loss / gradient, collapse flag at two consecutive kNN epochs, or
`kcs_underflow_frac` > 0.5 sustained over an epoch) is recorded and excluded; if a whole family fails, the failure is the result and the 8× cells of that family
are not run.  Nothing is re-selected after the 8× runs.

**8× confirmation (stage `P87_s_cs`, 4 views, B 256, 800 epochs, seeds 0–2; base configs `cifar10_hpK_a5_views4_800ep_vcs_seed{0,1,2}.yaml`).**
The selected S-Kernel (m*, bw*) × 3 seeds and the selected CS-K-native (bw*) × 3 seeds — six long runs generated by
`python configs/make_s_line_configs.py --p87-8x m* bw* kbw* --write` (a frozen addendum records the selection and the six shas before submission).
VCS-N at 8× is reused, never re-run: 87.01 ± 0.53 linear / 85.46 ± 0.12 kNN (P35 seeds 0–2: 86.42 / 87.16 / 87.44; test 86.65 ± 0.26, cited only).

## Evaluation (unchanged protocol)
Selection split only (the official test set is closed since P68; every new SSL unit reads the selection split, Spec §1.2).  Frozen h before the projector,
linear probe (SGD 0.1, 100 epochs, cosine, seed 20260925, head 512 → 10), kNN k = 200 T = 0.1 on the 45k fit bank; spectrum / effective rank; the critic
hold-out J for the VCS-N and S-Kernel runs (same code path, the RFF critic is a critic).  Cost record per run (Spec §6.2): steady-state step seconds, GPU
model, peak memory, wall clock, `n_pos` / `n_neg` per step from the step log (VCS-N and S-Kernel: B × 8 × 6 negative pairs; CS-K-native: 6 kernel pairs of
B × B entries, no negative pairs), trainable critic parameters (cosine 2; RFF m + 1; CS-K 0), encoder forwards per step (4B for all).

## Reading rules (descriptive; 3 seeds give a spread, not a significance statement)
- Table A: mean ± sd of S-Kernel(m*, bw*) vs VCS-N at 8× (linear, kNN), per-seed values, and the J trajectories of both (the same objective, so J is
  comparable).  Reading: |Δ linear| ≥ 1.0 → "the critic's function class matters at this budget" (direction stated); |Δ| < 1.0 → "no measurable difference
  between the neural and the kernel-feature critic under the same J".  Also reported: the dev-grid curve over m (does the kernel critic improve with m?).
- Table B: mean ± sd of CS-K-native(bw*) vs VCS-N at 8×; |Δ linear| ≥ 1.0 → direction stated; otherwise parity.  D_CS, its three log terms and the underflow
  fraction over training are reported with it; D_CS is never put in the same error column as S or J.
- Mechanism records (post hoc, read-only, `scripts/s4_mechanism_records.py` at epochs 100 / 400 / 800): actual gate 1 − E_M T², residuals, saturation,
  input-gradient norms, cosine medians, alignment / uniformity and effective rank on the same selection images for VCS-N, S-Kernel and CS-K-native.
- No claim about "matched trainable budget": the cosine critic has 2 parameters, the smallest RFF read-out 257; m is reported as an axis (disclosed).

## Cost (measured per-epoch speeds, 4 views B 256: RTX6000PRO ≈ 20 s, H100 ≈ 33 s, A100 ≈ 66 s; node60 ≈ 2× slower, excluded by preference)
| stage | runs | per run | total |
|---|---|---|---|
| dev grid 1× | 12 | 0.7 h PRO / 1 h H100 / 2 h A100 | 8–24 GPU-h |
| 8× S-Kernel + CS-K | 6 | 4.5 h PRO / 7.5 h H100 / 15 h A100 (all < 23 h; no chain) | 27–90 GPU-h |
RFF m = 4096 adds ≈ 2 GFLOP per view pair per step; the dense 256 × 256 kernels are negligible.  Memory: unchanged (≈ 5–8 GB).

## Launch (after the freeze; the main session submits)
```
# dev grid (feeder respects the 30-job cap; package units first)
sbatch --job-name=p87_dev_babysit --export=ALL,UNITS_FILE=slurm/s_units_P87_dev.txt,STAGE=P87_s_cs_dev,OUT=reports/P87_s_cs_dev_table.md,FINAL_CKPT=epoch_100.pt slurm/babysit_submit.sbatch
# after the selection addendum:
python configs/make_s_line_configs.py --p87-8x <m*> <bw*> <kbw*> --write
sbatch --job-name=p87_8x_babysit --export=ALL,UNITS_FILE=slurm/s_units_P87_8x.txt,STAGE=P87_s_cs,OUT=reports/P87_s_cs_table.md,FINAL_CKPT=epoch_800.pt slurm/babysit_submit.sbatch
#   (or per unit with --partition=H100,RTX6000PRO through slurm/submit_chain.sh <run_id> <cfg> epoch_800.pt 1 for the fast GPUs)
```

## Gate (CPU job 1013253, `slurm/s_line_cpu_gate.sbatch`; results copied to `reports/S_LINE_GATE_1013253/`)
Gate 1 = CPU job **1013253** (58 min, `reports/S_LINE_GATE_1013253/`, run before two script fixes); gate 2 = CPU job **1013329**, the same script with
(i) the `val_per_class` policy pin removed (it broke the frozen integration test 12 on synthetic manifests; the value is cross-checked against the manifest at
load time instead), (ii) `pipefail`, (iii) the mechanism-record CPU smoke added.  Gate-1 numbers (identical code on the training / evaluation paths):
- Test suites: `test_ssl_core` 29 passed; `test_integration` 39 passed + test 12 failed on the policy pin (fixed); `test_vcs_ssl_v2` **14 passed** (config
  policy, generated-config shas, RFF critic forward / grad / state round-trip, kernel-CS extremes 0 and log B, float64 gradcheck, chunked = dense, equality with
  the E-line `kernel_cs_native`, synthetic CIFAR-100 loader incl. sentinel / md5 / class-count refusals, 100-class manifest + split text, kNN / probe class
  plumbing, trainer smokes rff x {2, 4 views} and cs_kernel_native x {2, 4 views} with epoch evaluation and critic hold-out, kernel-sigma stop / resume / tamper
  refusal, a 100-class trainer + `evaluate --protocol pilot` run and the `final_official_test` refusal).
- Frozen recipe unchanged: `cifar10_hpK_a5_views4_800ep_vcs_seed0.yaml`, first optimizer step on CPU vs the P35 GPU run: gradient norms encoder
  1.08068 vs 1.08078 (rel. 9.5e-5), projector 1.485646 vs 1.485638 (5.3e-6), critic 0.2958448 vs 0.2958448 (5.3e-9).
- S-Kernel 4 views (m 1024, bw 1, the 1x dev config), 6 CPU steps: calibration median pair distance 1.5634 -> sigma 1.5634, 1 025 trainable critic parameters,
  Omega0 / b shas recorded; `evaluate --protocol pilot` ran the probe, kNN and the critic hold-out with the RFF critic (39.74 / 31.96 after 6 steps:
  code-path evidence only).  2-view stop at step 2 -> resume -> step 4, sigma buffer identical in `initial.pt` and `last.pt`.
- CS-K-native 4 views (bw 1): calibration median pair distance of z_l2 1.1158 -> sigma 1.1158, `kernel_cs` / log-term / underflow stats logged per step,
  pilot evaluation 37.66 / 29.56 after 6 steps; 2-view stop / resume with `kernel_sigma` equal in both checkpoints (a tampered value is refused, tested).
- CIFAR-100: link `data/cifar100/cifar-100-python -> /projects/EEG-foundation-model/yinghao/FMCA-AV/cifar100`; md5 of `train` / `meta` match the published
  values, sentinel and 100 x 500 counts pass, `test` hashed only; manifest **created** `manifests/cifar100_dev45k_val5k.json` sha256 `96dbd6d427807a0ef94fbcbbff9277bd4c8ebca3f5f7bb44e3246c0a45ad5430`
  (fit 45 000 = 100 x 450, selection 5 000 = 100 x 50).  VCS 2x config, 6 steps with epoch evaluation: 100-class kNN at epochs 0 and 2, pilot probe with a
  512 -> 100 head (17.92 / 10.84 after 6 steps), `final_official_test --standin-selection` refused with the CIFAR-10-only PermissionError.
- Gate 2 verdict: **PASS** (GATE_RC=0, 59 min, `reports/S_LINE_GATE_1013329/`): `test_ssl_core` 29, `test_integration` 40 (incl. test 12), `test_vcs_ssl_v2` 14 passed; every smoke, resume, evaluation and refusal step identical to gate 1 (same numbers); manifest EXISTS_VERIFIED (same sha); mechanism-record smoke on P35 / P41 / P43 epoch 800 (128 images, CPU) produced the JSONs and `mech_smoke.md` (e.g. actual gate_M 0.126 for the standard VCS model vs 0.340 for the strong-aug model under their own augmentation and 0.272 for the strong model under the standard block; SimCLR negative mass 0.67, effective negatives 104 of 510 — code-path evidence on 128 images, not a result).


## Delivery block (Spec §13.2), one per run in the results JSON
```yaml
experiment_family: full_ssl
protocol_id: P87_s_cs            # or P87_s_cs_dev
source_commit: <commit of the run>
source_basis: supplement_af7172c_plus_v2
estimator: vcs_neural | kernel_S_rff_tanh | kernel_cs_native
estimand: S | S | native_CS
evaluation_readout: J_common (VCS-N, S-Kernel) | native (CS-K-native); frozen-h linear probe + kNN for all
loss_scale: 1
reference_measure: mixture_equal | mixture_equal | native_base
critic_class: CosineCritic tanh(a<z1,z2>+b) | RFFTanhCritic m=<m> | none
gradient_routing: negative_right_detach | negative_right_detach | native
n_independent_units: 45000 fit images (selection 5000)
n_positive_pairs: <from the step log>
n_negative_pairs: <from the step log; 0 for CS-K-native>
split_manifest_hash: f819026a… (cifar10_dev45k_val5k)
noise_target_kind: none
fit_seconds: <train_seconds>
evaluation_seconds: <eval_seconds>
status: <status>
```

## Disclosed changes and limits
1. New code paths (`rff_tanh` critic, `cs_kernel_native` method, bandwidth calibration, `kernel_sigma` in checkpoints, n_classes plumbing) were added without
   touching the frozen VCS / SimCLR / VICReg numerics: the reference and integration suites pass unchanged, and the gate re-ran the recipe's first step on CPU
   against the P35 run's `first_step_gradients.json` (numbers in the gate section).
2. The S-Kernel bandwidth statistic (pair vectors, positives + one shifted partner) and the CS-K statistic (z_l2 pooled) are the fork's choices; the spec fixes
   only "a fixed multiple of the FIT median distance".  Both are recorded per run; the grid {0.5, 1, 2} brackets them.
3. The dev grid is a selection on the selection split, and the 8× confirmation reads the same split: this is the development endpoint of the programme
   (Spec §1.2, §7.2); the official test set is not touched.
4. Not part of this unit: an exact small-sample kernel or Nyström reference for the S-Kernel (Spec §3.4, "先测单元成本再决定扩展") — the E line's offline
   Nyström reference covers approximation error on the estimator side; a strong-augmentation variant of the kernel methods; ImageNet-100.
