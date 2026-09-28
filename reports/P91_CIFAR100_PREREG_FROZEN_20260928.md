# Pre-registration — P91: CIFAR-100 independent visual confirmation of the frozen CIFAR-10 recipes (VCS, tuned SimCLR, tuned VICReg; 2× then 8×; 3 seeds), 2026-09-28 — FROZEN 2026-09-28T20:14:12Z before GPU compute (S-line CPU gate 1013329 passed: ssl_core 29, integration 40, v2 14; frozen recipe first-step gradients unchanged to 1e-4; CIFAR-100 manifest sha256 96dbd6d4…5430)

Status: FROZEN 2026-09-28T20:14:12Z (main session).  The main session freezes it before any GPU job.  Sources: Server Spec v2 §6.3 (CIFAR-100 as the next independent visual
confirmation; from scratch, three seeds, the full pre-set schedule; submit each group with its measured cost), §1.2 / §7.2 (development on fit / selection; the
official test set was seen once and is closed), §13.2; Plan v2 §6.3; `reports/V2_RECONCILIATION_20260928.md` (row P91).

## Question
Do the CIFAR-10 findings replicate on a dataset that took no part in any development decision: (a) the ordering of the three methods at equal budget under
frozen recipes, and (b) the compute-scaling pattern (VCS gains more from 2× → 8× than the tuned controls)?  Nothing is tuned on CIFAR-100 for any method.

## Data (never downloaded; the local copy pre-dates this work)
- Files: `/projects/EEG-foundation-model/yinghao/FMCA-AV/cifar100/{meta,train,test}` (CIFAR-100 python version, 2010 timestamps).  torchvision expects
  `<root>/cifar-100-python/`, so `/home/infres/yinwang/CS_QMI/data/cifar100/cifar-100-python` is a symlink to that directory (created by the gate).
- Verification at every load (`src/vcs_ssl/data/cifar.py::load_cifar100_train`): md5 of `train` (16019d7e…) and `meta` (7973b151…) against the published values,
  index-order sentinel (first ten fine labels 19 29 0 11 1 86 90 28 23 31), 100 classes × 500 images.  The `test` file is hashed for provenance (also by
  torchvision's integrity check) and never read into memory by any code path; `final_official_test` refuses non-CIFAR-10 runs (tested).
- Identity split (`manifests/cifar100_dev45k_val5k.json`, built by `scripts/make_cifar100_manifest.py`): the CIFAR-10 algorithm and seed (numpy default_rng
  20260924, per-class permutation, 50 selection images per class, sorted UIDs) → fit 45 000 / selection 5 000.  Manifest sha256: 96dbd6d427807a0ef94fbcbbff9277bd4c8ebca3f5f7bb44e3246c0a45ad5430.
  Labels are used only by the split construction and the evaluation (unchanged rule).

## Cells (stage `P91_cifar100`; configs in `configs/S_LINE_SHA256.json`; each YAML is the CIFAR-10 recipe with only the `data` block changed: name, root, val_per_class 50, manifest)
| budget | VCS (frozen recipe) | SimCLR (P41 tuned 4v recipe) | VICReg (P41 tuned 4v recipe) | CIFAR-10 reference (selection linear, 3 seeds) |
|---|---|---|---|---|
| 2× = 4 views, B 256, 200 ep | `P91_c100_vcs_a5_views4_200ep_seed{0,1,2}` ← `cifar10_hpK_a5_views4_vcs_seed{s}` | `P91_c100_simclr_views4_200ep_seed{s}` ← `cifar10_hpN_simclr_views4_seed{s}` | `P91_c100_vicreg_views4_200ep_seed{s}` ← `cifar10_hpN_vicreg_views4_seed{s}` | 84.54 ± 0.16 / 87.84 ± 0.42 / 86.73 ± 0.30 |
| 8× = 4 views, B 256, 800 ep | `P91_c100_vcs_a5_views4_800ep_seed{s}` ← `cifar10_hpK_a5_views4_800ep_vcs_seed{s}` | `P91_c100_simclr_views4_800ep_seed{s}` ← `cifar10_hpN_simclr_views4_800ep_seed{s}` | `P91_c100_vicreg_views4_800ep_seed{s}` ← `cifar10_hpN_vicreg_views4_800ep_seed{s}` | 87.01 ± 0.53 / 88.32 ± 0.30 / 87.11 ± 0.50 |

Order: the nine 2× runs first (`slurm/s_units_P91_2x.txt`); the nine 8× runs (`slurm/s_units_P91_8x.txt`) are submitted after the 2× cells are COMPLETED and
their measured cost is recorded in the results file (Spec §6.3).  The 8× cells are pre-committed here (they do not depend on the 2× reading).

## Evaluation
Selection split (5 000 = 100 × 50).  Frozen h before the projector; linear probe with the recipe's hyper-parameters and a 512 → 100 head (SGD 0.1, 100 epochs,
cosine, seed 20260925); kNN k = 200, T = 0.1 on the 45k fit bank with 100-way votes (k kept at 200 for comparability although each class has 450 bank
images — noted); spectrum / effective rank; critic hold-out J (VCS).  Per-run cost as in every unit.

## Reading rules (pre-stated; descriptive; 3 seeds; no significance language)
- Per budget: mean ± sd and per-seed values for the three methods; pairwise differences with the ±1.0 reading rule (|Δ| < 1.0 = tie).
- **(a) Ordering replicates** if the sign pattern of the pairwise differences at each budget matches CIFAR-10's (2×: SimCLR > VICReg > VCS; 8×: SimCLR > {VICReg ≈ VCS}),
  ties counted as compatible with either side; **does not replicate** if any pairwise difference ≥ 1.0 has the opposite sign; **conditional** otherwise.
- **(b) Scaling replicates** if VCS's 2× → 8× gain exceeds each control's gain by ≥ 1.0 (CIFAR-10: +2.5 vs +0.5 / +0.4); **does not** if VCS's gain is not the
  largest; **conditional** otherwise.
- Absolute CIFAR-100 accuracies are reported as such (no CIFAR-10 → CIFAR-100 conversion); a collapse or numerical failure is reported, not re-run with new settings.

## Cost (4 views B 256: RTX6000PRO ≈ 20 s / epoch, H100 ≈ 33 s, A100 ≈ 66 s; CIFAR-100 images have the same shape, so identical speed)
| group | runs | per run | total |
|---|---|---|---|
| 2× | 9 | 1.2 h PRO / 1.9 h H100 / 3.7 h A100 | 11–33 GPU-h |
| 8× | 9 | 4.5 h PRO / 7.5 h H100 / 15 h A100 (< 23 h; no chain except node60, which is avoided) | 40–135 GPU-h |

## Launch (after the freeze; the main session submits)
```
sbatch --job-name=p91_2x_babysit --export=ALL,UNITS_FILE=slurm/s_units_P91_2x.txt,STAGE=P91_cifar100,OUT=reports/P91_cifar100_2x_table.md,FINAL_CKPT=epoch_200.pt slurm/babysit_submit.sbatch
# after the 2x cells are COMPLETED and their cost is recorded:
sbatch --job-name=p91_8x_babysit --export=ALL,UNITS_FILE=slurm/s_units_P91_8x.txt,STAGE=P91_cifar100,OUT=reports/P91_cifar100_8x_table.md,FINAL_CKPT=epoch_800.pt slurm/babysit_submit.sbatch
```
(The stage summariser groups by `run.stage`; the 2× / 8× tables are split by the run-id suffix `_200ep_` / `_800ep_`.)

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


## Delivery block (Spec §13.2)
```yaml
experiment_family: full_ssl
protocol_id: P91_cifar100
source_commit: <run commit>
source_basis: supplement_af7172c_plus_v2
estimator: vcs_neural | simclr_infonce | vicreg
estimand: S | InfoNCE_B | other
evaluation_readout: frozen-h linear probe (100 classes) + kNN, selection split
loss_scale: 1
reference_measure: mixture_equal | native_base | native_base
critic_class: CosineCritic tanh(a<z1,z2>+b) | none | none
gradient_routing: negative_right_detach | native | native
n_independent_units: 45000 fit images (selection 5000, 100 classes x 50)
n_positive_pairs: <step log>
n_negative_pairs: <step log>
split_manifest_hash: 96dbd6d427807a0ef94fbcbbff9277bd4c8ebca3f5f7bb44e3246c0a45ad5430
noise_target_kind: none
fit_seconds: <train_seconds>
evaluation_seconds: <eval_seconds>
status: <status>
```

## Disclosed limits
1. The control recipes were tuned on CIFAR-10 with an equal budget (P41); nothing is tuned on CIFAR-100 for any method — this is a transfer of frozen recipes,
   not a CIFAR-100 tuning study.  Recipe re-tuning, strong-augmentation cells (S4 replication) and ImageNet-100 are not part of this unit.
2. CIFAR-100 uses the coarse-free fine labels (100 classes) in the probe; the coarse labels are not used anywhere.
3. `views`, model, optimizer and schedule blocks are byte-identical to the CIFAR-10 recipes (config sha table in `configs/S_LINE_SHA256.json`); the only
   code difference on the training path is the dataset dispatch (`load_train_partition`) and the class count passed to kNN / probe.
