# Pre-registration — P100: momentum-encoder key queue for the 4-view VCS recipe and tuned SimCLR at 8× (3 seeds each), 2026-09-30 — FROZEN 2026-09-30T21:08:04Z before GPU compute (CPU gate 1015868: 106 tests green, recipe first-step gradients unchanged, config hashes of running units unchanged, bit-exact queue/EMA resume)

Status: FROZEN 2026-09-30T21:08:04Z (main session).  The main session freezes it (renamed `*_FROZEN_*`, freeze time in the title) before any GPU job.
Owner go 2026-09-30: "除了imagenet的先不提交，后续都可以提交，你设置好提交程序".  Source: `reports/SECOND_APP_FINAL_SUMMARY_20260928.md` §6 (proposed after
C-T, where the *plain* feature queue collapsed for both VCS and SimCLR: momentum-encoder queue = proposed follow-up).

## Question
Does a MoCo-style momentum-encoder key queue — negatives from a slowly moving copy of the network, drawn from 4 096 keys of previous steps instead of the
current batch — change the 8× CIFAR-10 result of the VCS recipe and of tuned SimCLR, and does it change the VCS–SimCLR gap (1.31 at 8×)?

## What changes (named variant; objective, critic, positives unchanged)
- **Momentum encoder:** a copy of encoder + projector (initialised equal to the online network), not optimised, no gradient; after every optimiser step
  θ_k ← m·θ_k + (1 − m)·θ_q with m = 0.99 (fixed; no cosine ramp).  Its BatchNorm runs in train mode with its **own** running buffers (MoCo convention;
  one GPU, so no shuffle-BN — disclosed).
- **Keys:** the critic-input features (L2-normalised projector output z) of **view 0** of each image in the batch, from the momentum encoder (one key per
  image, B = 256 per step); FIFO queue of Q = 4 096 keys (= 16 steps), enqueued after the optimiser step.
- **VCS:** the J objective, the cosine–tanh critic (a0 = 5), K = 8 and the 6 view pairs are the recipe's.  For every view pair the K = 8 product-of-marginals
  partners of each anchor are drawn from the queue (keys of *other, previous* images) instead of cyclic shifts of the current batch, uniformly **with
  replacement** (a duplicate partner for ≈ 0.7 % of anchors; the without-replacement sampler costs an n × Q argsort per pair — disclosed).  Keys carry no
  gradient by construction, which is the recipe's negative-detach rule.
- **SimCLR:** the tuned 4-view NT-Xent (τ unchanged) with the queue keys appended as *additional* negatives to the 2B − 2 in-batch negatives.
- **First step:** the queue is empty; that step uses each method's own recipe loss (`queue_fallback` logged).
- Everything else (augmentation, B 256, 800 epochs, AdamW 1e-3 / warm-up 10, projector, evaluation) is the frozen 8× recipe of each method.

## Code (this unit)
`src/vcs_ssl/config.py` (pairing.momentum_encoder, pairing.momentum_m: optional, not filled when absent → existing configs keep their resolved dict and
config_hash; policy: queue + 4 views + shared branch + joint mode), `src/vcs_ssl/objectives.py` (`NegativeQueue.sample_indices_fast`,
`vcs_pair_loss_momentum_queue`, `simclr_nt_xent_plus_queue`, `_views_momentum_queue`; `compute_objective_views(..., queue=)`), `src/vcs_ssl/train.py`
(momentum encoder, keys, EMA update, checkpoint/resume of the key encoder + queue, run-manifest record).  Tests `tests/test_vcs_ssl_p100.py` (8).

## Cells (stage `P100_momentum_queue`; `configs/make_p100_configs.py --write`; unit file `slurm/p100_units.txt`)
- `configs/cifar10_hpQ_vcs_mq_views4_800ep_seed0.yaml` f943d018e1992af1 (run `P100_vcs_mq_views4_800ep_seed0`, base `configs/cifar10_hpK_a5_views4_800ep_vcs_seed0.yaml`)
- `configs/cifar10_hpQ_vcs_mq_views4_800ep_seed1.yaml` ff1268afb6213d49 (run `P100_vcs_mq_views4_800ep_seed1`, base `configs/cifar10_hpK_a5_views4_800ep_vcs_seed1.yaml`)
- `configs/cifar10_hpQ_vcs_mq_views4_800ep_seed2.yaml` 5bcbb7cc1fea74d2 (run `P100_vcs_mq_views4_800ep_seed2`, base `configs/cifar10_hpK_a5_views4_800ep_vcs_seed2.yaml`)
- `configs/cifar10_hpQ_simclr_mq_views4_800ep_seed0.yaml` cf17af5c8257edb8 (run `P100_simclr_mq_views4_800ep_seed0`, base `configs/cifar10_hpN_simclr_views4_800ep_seed0.yaml`)
- `configs/cifar10_hpQ_simclr_mq_views4_800ep_seed1.yaml` da1662f5546ea08f (run `P100_simclr_mq_views4_800ep_seed1`, base `configs/cifar10_hpN_simclr_views4_800ep_seed1.yaml`)
- `configs/cifar10_hpQ_simclr_mq_views4_800ep_seed2.yaml` 5669d12984e30a1f (run `P100_simclr_mq_views4_800ep_seed2`, base `configs/cifar10_hpN_simclr_views4_800ep_seed2.yaml`)

## Evaluation
Selection split, frozen-h linear probe + kNN at epoch 800, each method's recipe probe.  Official test closed.

## Reading (pre-stated; 3 seeds, descriptive)
Baselines (same recipe without the queue, 3 seeds): VCS 87.01 ± 0.53 (86.42 / 87.16 / 87.44), SimCLR 88.32 ± 0.30 (88.20 / 88.10 / 88.66).
Per method: **helps** if mean ≥ baseline mean + 2 × pooled SE and every seed ≥ the baseline's worst seed; **hurts** if mean ≤ baseline mean − 2 × pooled SE;
otherwise **on par**.  Descriptive: the VCS − SimCLR gap with the queue vs without (−1.31).  A collapse (collapse flag at two consecutive kNN epochs) is
reported as such and not re-run with other settings.

## Cost
CPU gate step-time ratio vs the recipe: 1.07 (VCS) / 1.09 (SimCLR) on CPU.  8× recipe ≈ 4.5 h on
RTX6000PRO → ≈ 5–6 h per run; 6 runs, normal QOS, RTX6000PRO,H100.  Launch lines: `slurm/p100_lines.txt`.

## Gate
CPU job 1015868: PASS (gate_rc 0). Suites: ssl_core, integration, v2, P95 and P100 (8) green; recipe first-step gradients vs P35 within tolerance; 4-view smokes of both P100 configs + stop/resume COMPLETED; step-time ratio vs own recipe 1.07 (VCS) / 1.09 (SimCLR) on CPU; pilot evaluation of the VCS smoke ran (6 steps, values not meaningful).

## Disclosed limits
Fixed m = 0.99 without a ramp; one key view per image; with-replacement partner sampling; no shuffle-BN; SimCLR's queue is additive while VCS's replaces its
K partners (keeps VCS at the recipe's K = 8 and SimCLR at its tuned in-batch set) — the two methods therefore receive different amounts of extra negatives;
single-seed-free design (3 seeds per method) but no tuning of Q or m.

## Delivery (Spec v2 §13.2)
```yaml
experiment_family: full_ssl
protocol_id: P100_momentum_queue
estimator: vcs_neural | simclr (tuned)
estimand: S (VCS) | InfoNCE_B (SimCLR)
evaluation_readout: frozen-h linear + kNN, selection split
reference_measure: mixture_equal (VCS)
critic_class: cosine_tanh_2 | none
gradient_routing: momentum keys (no gradient)
n_independent_units: 45000 fit images; 3 seeds per method
split_manifest_hash: cifar10_dev45k_val5k (f819026a...)
status: <status>
```
