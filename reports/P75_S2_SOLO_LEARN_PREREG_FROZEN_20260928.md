# Pre-registration DRAFT — S2 (P75): VCS and SimCLR under the solo-learn CIFAR-10 protocol (2026-09-28)

Status: DRAFT (S2 fork). The main session freezes it (renamed `*_FROZEN_*`) before any GPU job. GPU priority: the estimator package
(`VCS_QMI_Estimator_Research_and_Server_v1`) comes first. S2 only uses spare quota (owner, 2026-09-28: "优先压缩包中的，有空余的把别的也提交了").
Sources: `CS_QMI/VCS_QMI_Next_Round_Plan_v2.md` §S2; `reports/NEXT_ROUND_EXECUTION_PLAN_DRAFT_20260928.md` §4 and §7 (owner decisions: S2
exempt from the test-set discipline; claim loosened to gap < 1.5).

## Question
Under a standard, externally used CIFAR-10 protocol (solo-learn: ResNet-18, 1000 epochs, LARS, larger projector, per-method augmentation,
official test set), how far is VCS from SimCLR? The brief's claim is that VCS gains more per compute doubling than SimCLR, so the gap under the
protocol is smaller than the P41 8× gap. This result decides how the SSL section is written, not the paper's claim.

## Code (solo-learn clone `CS_QMI/solo_learn` at 9187ea3 + patch `ssl_pilot/s2_solo_learn/solo_learn_S2.patch`, sha256 prefix ee399078ae0a4c7b)
- `solo/methods/vcs.py`, a new method. Objective and critic replicate the ssl_pilot final recipe:
  - critic tanh(a⟨z1, z2⟩ + b) on L2-normalised projector outputs, with a0 = 5 and b0 = 0 (both learnable);
  - K = 8 distinct nonzero cyclic shifts from a dedicated CPU generator (seed = seed + 100003);
  - the shifted partner in negative pairs is detached;
  - loss = −J, with the P and Q terms averaged separately. No CS log transform, no clipping of J. The critic is computed in fp32 inside the 16-mixed run.
  - **Equivalence check:** loss and gradients match `vcs_ssl.objectives.vcs_pair_loss_negdetach` + `CosineCritic` exactly (|Δ| = 0 in float64).
    Script: `s2_solo_learn/smoke_equivalence.py`, CPU jobs 1013079 / 1013082 / 1013088.
- `main_pretrain.py`:
  - `download=False` for both loaders (local CIFAR copy `CS_QMI/data/cifar10`, never downloaded);
  - `final_metrics.json` written once after epoch 1000 (see "Test-set rule");
  - `S2_NO_TEST=1` removes the val loader (smoke only).
- `solo/args/pretrain.py` accepts an int `devices` (CPU smoke only; the GPU runs keep `devices: [0]`).
- Configs `scripts/pretrain/cifar/s2_{vcs,simclr}.yaml` = solo-learn's `simclr.yaml` plus the following changes:
  - data paths;
  - `num_workers` 8;
  - wandb off (CSV logger in the unit directory);
  - `auto_resume.max_hours` 2000;
  - `check_val_every_n_epoch` 1000 and `num_sanity_val_steps` 0;
  - for VCS only, the method keys.

## Recipe (identical for both methods unless stated)
- **Shared protocol settings:**
  - ResNet-18 CIFAR variant (solo-learn backbone);
  - 2 crops with the protocol's `symmetric.yaml` augmentation (RRC 0.08–1, colour jitter 0.8/0.8/0.8/0.2 p = 0.8, grayscale 0.2, Gaussian blur p = 0.5, flip 0.5, 32 px);
  - batch 256 on 1 GPU;
  - LARS lr 0.4 (× batch/256 = 1), eta 0.02, clip_lr, exclude_bias_n_norm, wd 1e-4;
  - warm-up cosine, 10 warm-up epochs, 1000 epochs;
  - 16-mixed precision, channels_last, sync BN;
  - online linear classifier on detached backbone features (lr 0.1).
- **SimCLR:** solo-learn unchanged — projector Linear(512, 2048)–ReLU–Linear(2048, 256), temperature 0.2.
- **VCS (disclosed deviations, VCS only):**
  1. **Projector** = the ssl_pilot recipe's structure at the protocol's width: Linear(512, 2048, no bias)–BN–ReLU–Linear(2048, 256), no output BN.
     - Reason, found by the CPU smoke: with solo-learn's BN-free SimCLR head, all post-ReLU features share one direction at initialisation. cos ≈ 1 for every pair, t_pos = t_neg = 0.9996, J = −0.999 (job 1013082), and the bounded critic's gradient (1 − T²) vanishes.
     - With the hidden BN (job 1013088), t_pos / t_neg start at 0.86 / 0.84 and J rises from −0.74 to −0.61 within 3 steps.
     - Our P-series showed projector dims and depth to be neutral and output-BN harmful, so the structure is kept and the width follows the protocol.
  2. **Critic scalars** a, b (shape [1]) are excluded from LARS trust-ratio scaling and weight decay by solo-learn's `exclude_bias_n_norm`. They take plain SGD-momentum steps at the scheduled lr (peak 0.4); the recipe used AdamW 1e-3.
     - This is untuned. A per-step analysis puts it in the stable range (|∂J/∂a| ≤ ~1, curvature ≤ ~0.5, so lr·curvature ≪ 2(1 + momentum)).
     - a and b are logged per step (QC below).
  3. **Views:** 2 crops, not the recipe's 4. solo-learn's `main_pretrain` asserts 2 large crops for all but wmse/mae, and it is the protocol's setting. The P41 comparison showed both families gain from 4 views, so 2 vs 2 is the fair comparison here.
  4. **Optimiser:** LARS 0.4 is the SimCLR recipe's value, applied to VCS without tuning. Our P-series found the lr neutral within AdamW; LARS was never tried. If VCS fails QC, this is disclosed as the first suspect; no post-hoc lr search inside P75.
- **Not changed:** BN policy. Each crop is forwarded separately (solo-learn default), unlike ssl_pilot's concatenated 2B forward; this is the same for both methods.

## Units (6)
`s2_solo_learn/units_P75.txt`: {vcs, simclr} × seeds {0, 1, 2}. One GPU per unit. Output: `CS_QMI/outputs/P75_S2_solo/<method>_seed<s>/`.

## Metric
The protocol's own number is **online linear Acc@1 on the official CIFAR-10 test split at the end of epoch 1000** (`val_acc1` in
`final_metrics.json`). This is how the solo-learn CIFAR table is produced (SimCLR ResNet-18 1000 ep: 90.74 in the README at 9187ea3). Also recorded:
Acc@5, `val_loss`, the per-epoch training curves (CSV), and for VCS, J / t_pos / t_neg / saturation / a / b.

## Test-set rule (owner exemption, 2026-09-28: "给S2也豁免")
The official test split is read **once per run, at the end**:
- Lightning validation runs only at epoch 1000 (`check_val_every_n_epoch = 1000`) and there is no sanity-check pass (`num_sanity_val_steps = 0`). No checkpoint selection, early stopping or hyper-parameter choice uses it.
- If a resume lands after the final validation, `_s2_final_metrics` calls `trainer.validate` exactly once.
- A unit whose `final_metrics.json` exists exits without touching data.
- The test loader object is constructed at start-up (it reads the files into memory) but is not evaluated before epoch 1000.
- No offline linear evaluation on the test set is part of P75. solo-learn's `main_linear.py` evaluates every epoch; if an offline probe is wanted, it is run on the ssl_pilot selection split with the frozen P-series probe, as a separate addendum.

## Pre-committed reading (owner-loosened numbers)
- **Holds** if gap = mean_SimCLR − mean_VCS (3 seeds, online Acc@1 at epoch 1000) < 1.5 points.
- **Refuted** if gap ≥ 1.5.
- Reported next to the P41 / P68 8× gaps: selection split 1.3, official test 1.5 (SimCLR 88.13 vs VCS 86.65).
- Seed sd is printed. If |gap − 1.5| < 2 × the pooled SE, the verdict is written "at the threshold" and not rounded into either side.
- **Recipe confirmation (QC, not a claim):** SimCLR's 3-seed mean within 1.0 point of 90.74. Otherwise the protocol is not reproduced, and the gap is reported with that caveat.

## QC sentinels
- Every unit completes 1000 epochs, and `final_metrics.json.source` is recorded.
- The log header records the solo patch sha256 and it equals the frozen value.
- VCS runs:
  - no NaN;
  - critic a stays finite and |a| < 100;
  - the saturation fractions fall below 0.9 by epoch 10 (else: saturation failure, reported as such).
- SimCLR / VCS online train acc rises (sanity).
- Driver / CUDA: env_solo is torch 2.14.0+cu130 (torchvision 0.29, lightning 2.5.6, timm 1.0.30, python 3.11). The first GPU unit's log shows whether the node's driver supports CUDA 13. If not, that partition is excluded and the unit resubmitted.

## Compute
- **Not measured on GPU** (CPU smoke only: 3 steps ≈ 11 s/step on CPU).
- Estimate from the P-series: our 4-view 800-epoch fp32 runs took ~20 s/epoch on A100/H100. Two views with 16-mixed should take ~10–20 s/epoch, and the torchvision data loader with 8 workers is likely the bottleneck. So 1000 epochs ≈ 3–6 h per unit, ≈ 20–35 GPU-h in total, within one 23 h job per unit.
- Chaining: resubmitting the same unit resumes from the per-epoch checkpoint.
- Partitions: H100 / RTX6000PRO first, A100 / L40S allowed, node60 excluded, never P100.

## Submission (after freezing; only into spare quota)
`s2_solo_learn/submit_P75.sh` (all 6), or e.g. `s2_solo_learn/submit_P75.sh vcs 0 simclr 0` for a subset. The script verifies numeric job ids;
ids are then recorded in `reports/job_ids.json`.
