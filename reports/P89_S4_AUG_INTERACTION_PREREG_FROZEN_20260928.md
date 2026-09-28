# Pre-registration — P89: S4, method × augmentation strength at the 8× budget (VCS vs tuned SimCLR; standard vs strong augmentation; Δ_int), 2026-09-28 — FROZEN 2026-09-28T20:14:12Z before GPU compute (S-line CPU gate 1013329 passed: ssl_core 29, integration 40, v2 14; frozen recipe first-step gradients unchanged to 1e-4; CIFAR-100 manifest sha256 96dbd6d4…5430)

Status: FROZEN 2026-09-28T20:14:12Z (main session).  The main session freezes it before any GPU job.  Sources: Server Spec v2 §7.1–7.2 (four main units, mechanism records),
§6.1 (S4 belongs to the full-SSL evidence), §13.2; Plan v2 §6.2; supplement §3.6; `reports/V2_RECONCILIATION_20260928.md` (row P89).  Motivation on
record: the VCS strong-augmentation run (P43, seed 0) is 87.78 selection linear vs 87.01 ± 0.53 for the standard recipe, and P41's tuned SimCLR never received
the same strong-augmentation setting.

## Question
Does VCS gain more from stronger augmentation than SimCLR does, in the same 4-view / 800-epoch / B 256 protocol with each method's frozen hyper-parameters?
Core quantity, per seed s and then averaged:

    Δ_int(s) = (Acc_{V,strong,s} − Acc_{V,std,s}) − (Acc_{C,strong,s} − Acc_{C,std,s}),   Acc = selection-split linear top-1 (kNN reported alongside).

Both methods' absolute results and their own gains are reported next to the difference-in-differences (Spec §7.1).

## The four cells (3 seeds each; reuse where an exact match exists)
| method | standard augmentation (crop scale 0.2–1, jitter 0.4/0.4/0.4/0.1 p 0.8, grey 0.2) | strong augmentation (crop scale 0.08–1, jitter 0.8/0.8/0.8/0.2 p 0.8, grey 0.2) |
|---|---|---|
| VCS (recipe: cosine critic a0 5, K 8, detach, 4 views, AdamW 1e-3, 800 ep) | **reused** P35 seeds 0–2: 86.42 / 87.16 / 87.44 → 87.01 ± 0.53 (kNN 85.60 / 85.36 / 85.42) | seed 0 **reused** P43: 87.78 (kNN 85.58); seeds 1–2 **new** `P89_vcs_a5_views4_800ep_augstrong_seed{1,2}` |
| SimCLR (tuned P41 recipe: τ 0.2, 4 views, control_tuning, 800 ep) | **reused** P41 seeds 0–2: 88.20 / 88.10 / 88.66 → 88.32 ± 0.30 (kNN 87.20 / 87.76 / 87.98) | seeds 0–2 **new** `P89_simclr_views4_800ep_augstrong_seed{0,1,2}` |

Configs (`configs/S_LINE_SHA256.json` → stage `P89_s4_aug_interaction`): the SimCLR strong-aug YAMLs are `cifar10_hpN_simclr_views4_800ep_seed{s}.yaml` with
the **whole `views` block of the P43 config copied verbatim** (`cifar10_hpO_a5_views4_800ep_augstrong_vcs_seed0.yaml`; the generator asserts that this block
differs from the standard one only in `random_resized_crop.scale` and the four `color_jitter` values — operations, probabilities and order are identical);
the VCS strong-aug seeds 1–2 are the P43 YAML with `run.seed` and `run.stage` changed and nothing else.  Temperature (0.2) and a0 (5) are not re-tuned for the
strong setting (Spec §7.2: no unequal re-search per augmentation).  Five new runs.

## Evaluation
Selection split, frozen h linear probe + kNN as in every unit; per-seed values listed.  Per-run cost recorded (the strong block is data-loader bound: P43 took
10.2 h wall clock on node59 at 4 loader workers versus 4.4 h for the standard SimCLR run on the same node class; `num_workers` stays 4 for exactness).

## Reading rules (pre-stated; descriptive with 3 seeds, no significance language)
- Report Δ_int(s) for s = 0, 1, 2, the mean and the range; each method's own gain (mean ± sd); the absolute cells.
- "VCS gains more from strong augmentation than SimCLR" **holds** if mean Δ_int ≥ +1.0 and Δ_int(s) > 0 for every seed; **does not hold** if mean Δ_int ≤ 0;
  **conditional** otherwise (e.g. mean ≥ 1.0 with one seed negative, or 0 < mean < 1.0).
- If both methods gain ≥ 1.0, the effect is a common augmentation effect (Plan §6.2) whatever Δ_int says; if the SimCLR strong cells *lose* accuracy, that is
  reported as such (the tuned temperature was chosen under the standard block).
- The absolute comparison VCS-strong vs SimCLR-strong is reported too but is not the S4 question.

## Mechanism records (Spec §7.2; read-only on checkpoints; do not wait for the new runs where checkpoints exist)
Already logged by the trainer for every run: per step J_raw, R_binary, positive / negative score means and second moments, saturation fractions, the cosine
critic's a and b, gradient norms; at epochs {0, 20, 50, 100, 200, 400, 600, 800}: kNN, effective rank of h and z (spectrum), the held-out J of the critic
(VCS); epoch time / memory.  `scripts/alignment_uniformity.py` gives alignment / uniformity and pair-cosine quantiles at the final checkpoint.
New, `scripts/s4_mechanism_records.py` (GPU minutes; CPU-smoked in the gate on an existing checkpoint) at epochs 100 / 400 / 800 for the four cells, on the
same 2 048 selection images, the same two views (one RNG seed) and the same K = 8 partners, under (a) the run's own augmentation block and (b) the standard
block for every run: the **actual gate** 1 − E_M T² (equal-weight over P and Q; 1 − J is reported next to it, not instead of it), residuals C − T, saturation,
positive / negative score distributions, per-image input-gradient norms ‖∂loss/∂x‖ (both views) and projector-output gradient norms, cosine medians of positive
and shifted pairs (z and h), alignment / uniformity of z and h, effective rank; for SimCLR the native negative softmax weights (negative mass, largest weight,
effective number of negatives) and the same input gradients on the same samples.  Output: `reports/P89_mechanism/*.json` + `reports/P89_mechanism.md`.
Run lines (GPU, one job): `python scripts/s4_mechanism_records.py --runs <8 run ids> --epochs 100,400,800 --n 2048 --aug own --out-dir reports/P89_mechanism`
and the same with `--aug standard`.

## Cost
5 runs × ≈ 10 h (loader-bound strong block; < 23 h on every allowed partition, no chain) ≈ 50 GPU-h; mechanism records ≈ 0.5 GPU-h.

## Launch (after the freeze; the main session submits)
```
sbatch --job-name=p89_babysit --export=ALL,UNITS_FILE=slurm/s_units_P89.txt,STAGE=P89_s4_aug_interaction,OUT=reports/P89_s4_table.md,FINAL_CKPT=epoch_800.pt slurm/babysit_submit.sbatch
```

## Gate (CPU job 1013253)
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
- Mechanism-record script: CPU smoke on existing checkpoints (P35 / P41 / P43 epoch 800, 128 images, own and standard augmentation) runs in gate 2.


## Delivery block (Spec §13.2)
```yaml
experiment_family: full_ssl
protocol_id: P89_s4_aug_interaction
source_commit: <run commit>
source_basis: supplement_af7172c_plus_v2
estimator: vcs_neural | simclr_infonce
estimand: S | InfoNCE_B
evaluation_readout: frozen-h linear probe + kNN (selection split); J_common for VCS
loss_scale: 1
reference_measure: mixture_equal | native_base
critic_class: CosineCritic tanh(a<z1,z2>+b) | none
gradient_routing: negative_right_detach | native
n_independent_units: 45000 fit images (selection 5000)
n_positive_pairs: <step log>
n_negative_pairs: <step log>
split_manifest_hash: f819026a… (cifar10_dev45k_val5k)
noise_target_kind: none
fit_seconds: <train_seconds>
evaluation_seconds: <eval_seconds>
status: <status>
```

## Disclosed limits
1. Reused cells were trained before this pre-registration (P35 2026-09-25/26, P41 2026-09-27/28, P43 2026-09-26); their numbers are on record in the
   technical note and the P41 report and are not re-run.  The new cells are the only ones whose reading rules could still be influenced; the rules above are
   fixed before they run.
2. Seed pairing: Δ_int(s) pairs runs by seed index (same encoder / projector initialisation and loader / pair RNG seeds across methods and augmentations); the
   augmentation draws differ by construction.
3. Cross-dataset replication of the interaction (Spec §7.2 last paragraph) is not part of P91's frozen cells; if S4 holds on CIFAR-10 a CIFAR-100 strong-aug
   addendum (VCS + SimCLR, 3 seeds each) is proposed, not pre-committed.
