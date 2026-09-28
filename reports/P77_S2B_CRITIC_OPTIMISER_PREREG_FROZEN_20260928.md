# Pre-registration DRAFT — S2b (P77): the P75 solo-learn VCS unit with the critic scalars on the recipe's optimiser (2026-09-28) — FROZEN 2026-09-28T18:11:12Z before GPU compute (CPU smoke 1013244 passed; seed 0 submitted at --nice=500)

Status: FROZEN 2026-09-28T18:11:12Z (main session).  Filler priority like P75
(`--nice=500`): the package-v2 main units (P85 / P87 / P89 / P91 / P93) keep the queue.  Sources: `reports/P75_S2_SOLO_LEARN_PREREG_FROZEN_20260928.md`
(deviation 2 and its clause "If VCS fails QC, this is disclosed as the first suspect; no post-hoc lr search inside P75"); owner 2026-09-28 (full go for
the v2 programme, experiments + technical docs); Server Spec v2 §6.3 ("P75 标准协议: 保留其已批准框架 … 映射新基线须独立记录" — a new mapping is recorded
as its own unit).  P75 itself is not modified: `solo/methods/vcs.py`, `solo/methods/__init__.py`, `main_pretrain.py` and the P75 configs are byte-identical
(patch sha256 prefix ee399078ae0a4c7b unchanged; verified in the smoke log), and the two pending P75 units (1013113, 1013116) load the same code.

## Observation that motivates the unit (S2 runs in progress, read from the Lightning CSV logs, not from any test metric)
| unit | epoch | critic a | critic b | a + b | J (train) | t_pos / t_neg | sat_pos / sat_neg | online train Acc@1 |
|---|---|---|---|---|---|---|---|---|
| P75 vcs_seed0 | 100 / 400 / 580 | 109.8 / 222.3 / 251.1 | −109.1 / −221.4 / −250.1 | 0.75 / 0.90 / 1.0 | 0.549 / 0.646 / 0.694 | 0.42/−0.43, 0.50/−0.53, 0.54/−0.58 | 0 / 0.07–0.16 | 52.1 / 67.0 / 71.1 |
| P75 vcs_seed1 | 100 / 400 / 580 | 109.7 / 222.2 / 251.0 | −108.9 / −221.3 / −250.1 | same | 0.552 / 0.641 / 0.689 | same | same | 52.0 / 66.8 / 71.1 |
| P75 simclr_seed0 | 100 / 400 | — | — | — | — | — | — | 69.9 / 75.6 |
| recipe `P35_vcs_a5_views4_800ep_seed0` (AdamW 1e-3 for a, b) | 100 / 400 / 800 | 9.9 / 21.6 / 24.4 | −8.0 / −19.4 / −22.0 | 1.9 / 2.2 / 2.4 | 0.951 / 0.981 / 0.987 | 0.86/−0.89, 0.92/−0.95, 0.94/−0.97 | 0.02/0.64, 0.44/0.83, 0.62/0.88 | (4 views, B 256; not comparable in absolute value) |

Under the protocol's LARS (the two scalars are 1-D, so `exclude_bias_n_norm` gives them plain SGD-momentum steps at the scheduled lr, peak 0.4), a and b
run away linearly (≈ +0.8 per epoch early, 10× the recipe's rate per step), b tracks −a so that a + b ≈ 1: the critic becomes a step at cos ≈ 1 − 1/a
and the training objective stalls near 0.69 (recipe: 0.95 at the same epoch).  The online accuracy gap to SimCLR is 18 points at epoch 100 and 9 at
epoch 400.  This is P75's deviation 2; P75's reading is unchanged and its verdict will be written from its own final metrics.

## Question
Under the same protocol, with the critic scalars updated exactly as in the recipe (AdamW, lr 1e-3, betas 0.9/0.999, eps 1e-8, weight decay 0,
linear warm-up over the protocol's 10 epochs from 0, cosine to 1 % of the peak), how far is VCS from SimCLR at epoch 1000?  One pre-stated change,
no grid: the only quantity varied relative to P75 is the optimiser of the two critic scalars.

## Code (new files only; nothing of P75 touched)
- `solo_learn/solo/methods/vcs_s2b.py`: `VCSS2B(VCS)` — removes the critic group from the LARS param groups, adds `torch.optim.AdamW` for
  `critic.parameters()` with the recipe's schedule shape (`LinearWarmupCosineAnnealingLR`, warm-up = `scheduler.warmup_epochs`, eta_min = 0.01 × 1e-3),
  manual optimisation (Lightning 2.5.6 requires it for two optimisers): `manual_backward`, both optimisers stepped through the precision plugin
  (GradScaler unscale/step per optimiser under 16-mixed), both schedulers stepped per step; logs `critic_lr`, `main_lr`.
- `solo_learn/main_pretrain_s2b.py`: registers `vcs_s2b` in the METHODS dict at import and calls the unchanged `main_pretrain.main`.
- `solo_learn/scripts/pretrain/cifar/s2b_vcs.yaml`: `s2_vcs.yaml` + `method: vcs_s2b`, `name`, `method_kwargs.critic_optimizer: adamw`,
  `critic_adamw_lr: 1e-3`, `critic_adamw_min_lr_ratio: 0.01`.  Everything else (augmentation, projector, K, detach, LARS 0.4 for backbone / projector /
  classifier, wd 1e-4, 1000 epochs, 16-mixed, `check_val_every_n_epoch: 1000`, `num_sanity_val_steps: 0`) identical to P75.
- `ssl_pilot/slurm/s2b_solo_unit.sbatch` (SEED env; output `outputs/P77_S2b_solo/vcs_s2b_seed<s>/`; logs the sha256 of the three new files),
  `ssl_pilot/slurm/s2b_solo_smoke_cpu.sbatch` (CPU smoke, see gate).
- Harvest: `scripts/s2_solo_table.py` (P76) is reused with `--root outputs/P77_S2b_solo` semantics added at harvest time (results → `P78_s2b_solo.{md,json}`).

## Units (3) and order (probe gate)
`vcs_s2b` × seeds {0, 1, 2}, one GPU each, `--nice=500`.  Seed 0 first; **probe gate** read at epoch 20 of seed 0 from the CSV log (no test data):
a ∈ [2, 40] and b ∈ [−40, 2] and J ≥ 0.70 (recipe at epoch 20: a 3.5, b −1.4, J 0.88; P75: a 35, b −35, J 0.27).  If the gate fails the unit is
stopped, the failure is reported, and seeds 1–2 are not submitted (no re-tuning inside P77).  If it passes, seeds 1–2 are submitted.

## Metric and test-set rule
Identical to P75: online linear Acc@1 on the official CIFAR-10 test split at the end of epoch 1000 (`final_metrics.json`), read once per run; no
checkpoint selection or early stopping uses it.  The owner's S2 exemption covers this unit because it is the same protocol, same metric, same single
read at epoch 1000; nothing else in the programme gains test access.

## Pre-committed reading
- gap_b = mean SimCLR (P75, 3 seeds) − mean VCS-S2b (3 seeds).  **Holds** if gap_b < 1.5; **refuted** if gap_b ≥ 1.5; "at the threshold" if
  |gap_b − 1.5| < 2 × pooled SE — the P75 numbers, so the two mappings are read with one rule.
- Reported next to P75's gap (same table): if P75 is refuted and P77 holds, the SSL section says "under the protocol, VCS matches SimCLR within 1.5
  when its critic scalars use the recipe's optimiser; with the protocol's LARS step on the scalars it does not" — a mapping statement, not a claim of
  superiority.  If both are refuted, the protocol result stands as refuted and the scalar optimiser is removed from the suspect list.
- QC as P75 (SimCLR 3-seed mean within 1.0 of 90.74) and the sentinels below.

## QC sentinels
1000 epochs completed; `final_metrics.json.source` recorded; a, b, J, t±, sat± per epoch; `critic_lr` follows warm-up/cosine with peak 1e-3;
`main_lr` unchanged from P75; the LARS optimiser's param groups contain no critic group (asserted in the smoke); checkpoint holds two optimiser and two
scheduler states and a resume continues (smoke).  Cost: ≈ 3 h per seed on H100 (11 s / epoch measured for P75 VCS), 9 GPU-h in total, filler priority.

## Gate
CPU smoke job 1013242 failed before training (Hydra resolved the relative config path as a Python module because main_pretrain was imported; fixed by executing it as __main__ through runpy in the wrapper — no change to the method). CPU smoke job 1013244 passed: 3 + 3 steps on CPU, checkpoint with 2 optimizer + 2 scheduler states, LARS groups = backbone/classifier/projector (no critic), critic AdamW lr 2.07e-4 after 6 warm-up steps (= 1e-3 × 6/30), wd 0, betas (0.9, 0.999); a 4.99949, b −0.00050 after 6 steps; resume continued to epoch 1 / 12 optimizer steps; `solo/methods/vcs.py` diff vs 9187ea3 unchanged (sha 595d3856d3ea709d); P75 patch sha ee399078ae0a4c7b unchanged. File hashes: vcs_s2b.py d20a2ad2ffbef418 (before the wrapper fix; final hashes are logged by the unit job), s2b_vcs.yaml 93c97997f0246a1b.

## Spec v2 §13.2 block (filled at results time)
```yaml
experiment_family: full_ssl
protocol_id: P77_S2b_solo_learn_cifar10_1000ep
source_commit: <ssl_pilot commit>; solo-learn 9187ea3 + P75 patch ee399078ae0a4c7b + S2b files (sha in the unit log)
source_basis: supplement_af7172c_plus_v2
estimator: vcs_neural
estimand: S
evaluation_readout: online_linear_acc1_official_test_once
loss_scale: 1
reference_measure: mixture_equal
critic_class: cosine_tanh_2_scalars_AdamW
gradient_routing: negative_right_detach
n_independent_units: 3 seeds
split_manifest_hash: solo-learn CIFAR-10 train/test (torchvision files, local copy)
status: <status>
```
