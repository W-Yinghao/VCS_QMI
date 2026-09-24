# Pre-registration — CIFAR-10 200-epoch confirmation (P5–P6), frozen 2026-09-24 before launch

Owner decision: "开始" after reading `reports/P4_ssl_pilot_report.md` (proposal E).  Contract unchanged:
`VCS_QMI_SSL_Server_Agent_Spec_v1.md`; plan `configs/next_stage_plan.yaml`.  Code identical to the pilot (`src/`, `reference/`
unchanged since commit `76cebd6`; only `summarize.py` gained seed aggregation and a generic final-epoch lookup).

## Question
Under the full development schedule (200 epochs, warm-up 10, cosine to 1 %), does the VCS-QMI encoder's frozen `h` improve further
than at 20 epochs, does its low-dimensional concentration (h effective rank ≈ 6, `p_raw` ≈ 4 directions) persist, and how large is
seed-to-seed variation for VCS and the two matched controls?

## Units (9 runs, 3 concurrent single-GPU jobs, one chain per method)
`configs/cifar10_confirm200_{vcs,simclr,vicreg}_seed{0,1,2}.yaml`, derived from the frozen pilot configs by exactly these overrides
(`configs/CONFIRM200_SHA256.json` lists sha256 of every file): `run.stage=P5_confirm200`, `run.seed∈{0,1,2}`, `train.epochs=200`,
`train.warmup_epochs=10`, `logging.checkpoint_epochs=[20,50,100,150,200]`, `evaluation.knn_epochs=[0,10,20,50,100,150,200]`,
`evaluation.linear_epochs_of_pretrain=[200]`.  Everything else (B=256, AdamW 1e-3, wd, augmentations, K=1, critic, projector,
FP32, split manifest `c35d7cd3…`) is unchanged.  Seed 0 starts from the same initial weights as the pilot (`bdc3a4e3…c767`);
seeds 1 and 2 have their own initialization, identical across the three methods.  T = 35,000 steps, W = 1,750.

## Primary endpoint
`linear_val_top1_pct` of frozen `h` at `epoch_200.pt` (fixed final checkpoint), pilot probe protocol unchanged (Linear(512,10),
SGD 0.1/0.9/0, 100 epochs cosine→0.001×, seed 20260925, final probe epoch), fit labels → selection labels.  Reported per seed and
as mean ± sample SD over the 3 seeds, together with the same probe on `initial.pt` and the per-seed delta.

## Secondary endpoints / diagnostics
kNN (k=200, T=0.1) at 0/10/20/50/100/150/200; `heldout_J` (VCS, 4 repeats) at the same epochs; `h`/`z` effective rank and the
`p_raw` eigenvalue profile on the first 4,096 sorted selection UIDs; per-step `J_raw`, saturation fractions, gradient norms;
cost (steady images/s, train seconds, peak memory) — GPUs may differ between chains, so cost is compared only within a GPU type.

## Gates
Pilot gates (P0–P2) passed on this code; no new smoke is run.  Any `FAILED_*` or `STOPPED_BUDGET` run is reported as such, may be
resumed from `last.pt` (epoch boundary) once, and is never silently re-seeded or re-tuned.

## Pre-committed reading
- Encoder improvement: per-seed Δ linear (ep200 − ep0); a claim of improvement requires all 3 seeds positive.  Comparison with the
  pilot's 20-epoch value is descriptive only (different schedule).
- Dimensional concentration: if `h` effective rank stays < 10 and `p_raw` keeps ≤ 5 dominant eigenvalues at epoch 200 in all seeds
  → "persistent low-dimensional concentration under the full schedule" (observation; K/critic-capacity untested).  If rank rises
  substantially → "concentration was a short-horizon phenomenon".
- Critic vs features: `heldout_J` and train `J_raw` trajectories reported; a high J with flat linear/kNN is again "pair discrimination
  without proportional transfer".
- Controls: mean ± SD over 3 seeds; no tuned-baseline claim; no test-set number; a VCS–control gap is reported as a gap under the
  matched recipe, not as a verdict on the objective.
- Seed noise: the 3-seed SD is the first noise estimate; no significance test on n = 3.
- Numerical failure / OOM / collapse flag: stop that unit, keep artifacts, report; no method change.

## Not claimed
No official-test accuracy; no Shannon-MI wording; no "converges to S"; no independence claim for K×B pairs; no statement about
CIFAR-100/ImageNet/EEG.

## After P6
Neutral table (`reports/P6_confirm200_results_table.md`) → report → wait for the owner.  Single-factor follow-ups (K, critic capacity)
are decided afterwards, one at a time.
