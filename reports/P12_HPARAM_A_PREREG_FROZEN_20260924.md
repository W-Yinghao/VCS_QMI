# Pre-registration — VCS-QMI A-group hyper-parameter screen (P12–P13), frozen 2026-09-24 before launch

Owner decision: screen every VCS-specific hyper-parameter (group A) with a single seed before any comparison with controls;
"first find the best VCS configuration".  SLURM submit cap 30 jobs → a babysitter feeds the queue in priority order.

## Design
Single factor per run, **seed 0, 200 epochs, warm-up 10**, everything else identical to the frozen `cifar10_confirm200_vcs_seed0.yaml`;
baseline = `P5_vcs_seed0` (K=1, critic [512,512], last gain 0.1, critic lr ×1, critic wd 0, projector 512→128, l2 critic input):
linear-val 74.84 %, kNN 64.64 %, h-rank 12.87, 7 dominant `p_raw` directions, heldout-J 0.8975.  Seed SD at K=1 (P5): 0.46 linear points.

| factor | values (21 runs) | config field |
|---|---|---|
| critic width | 128, 256, 1024, 2048 | `model.critic.hidden_dims=[w,w]` (reference `PairCritic(hidden_dim=w)`) |
| critic depth | 1, 3 hidden layers (width 512) | `model.critic.hidden_dims` (named variant `PairCriticMLP`; identical to the reference for `[w,w]`, tested) |
| critic lr multiplier | 0.1, 0.3, 3, 10 | `optimizer.critic_lr_multiplier` |
| critic weight decay | 1e-4, 1e-3 | `optimizer.critic_weight_decay` |
| critic last-layer gain | 0.01, 1.0 | `model.critic.last_layer_xavier_gain` (variant class; must be > 0) |
| projector output dim (critic input dim) | 64, 256, 512 | `model.projector.output_dim` |
| projector hidden width | 1024, 2048 | `model.projector.hidden_dim` |
| critic input normalization | none (raw `p_raw`) | `model.normalization.vcs_and_simclr` |
| K | 255 (all off-diagonal; 8 and 64 run in P10) | `pairing.k` |

Configs and sha256: `configs/HPARAM_A_SHA256.json`; run ids `P12_vcs_<label>_seed0`; stage `P12_vcs_hparamA`.
The objective, sampler, joint update, data split, augmentations, encoder, B, LR schedule and probe protocol are unchanged.
Code changes (commit noted in `reports/job_ids.json`): config policy relaxed for these fields only; `PairCriticMLP` variant; raw-input
option (critic hold-out diagnostic follows the same input); hyper-parameters recorded in `run_manifest.json` and in the summary table.
Control configs are still pinned to the frozen 512/128 projector and l2 (policy-checked).

## Endpoints
Primary: `linear_val_top1_pct` of frozen `h` at `epoch_200.pt` vs the baseline 74.84 % (single seed, so differences below ≈ 1 point —
two seed SDs — are "within noise").  Secondary: kNN, h-rank, dominant `p_raw` directions, `heldout_J` (own K / own input), saturation,
step time and memory (critic width/K/projector size change compute).

## Pre-committed reading
- A factor "helps" only if linear-val improves by > 1.0 point over the baseline **and** kNN moves in the same direction; ≤ 1 point is
  "no detectable single-seed effect".  Factors that help are candidates for a 3-seed confirmation and for combination one at a time;
  no combined run is launched from this screen without the owner.
- The screen is on the selection set: whatever is chosen is a development choice, and the final comparison must give the controls an
  equal tuning budget (spec §19); nothing here is a VCS-vs-control claim.
- Concentration hypothesis: report h-rank / dominant directions per factor; a factor that raises rank without raising linear-val is
  reported as such (rank is not the goal).
- `heldout_J` is not comparable across K, critic input space or critic capacity as an estimate of anything; it is reported per run only.
- Failures (NaN, OOM — e.g. K=255 or width 2048) are reported with artifacts; no automatic reduction of B or K.

## Not claimed
No test accuracy, no control comparison, no significance from n = 1, no Shannon-MI / convergence language.

## After P13
Neutral table (`reports/P13_hparamA_results_table.md`) → report → wait for the owner.
