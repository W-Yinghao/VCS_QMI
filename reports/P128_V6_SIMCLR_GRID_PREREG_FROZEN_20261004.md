# Pre-registration — P128: v6 §8.2 SimCLR tuning budget at the 4-view / 800-epoch protocol (CIFAR-100 and CIFAR-10, seed 0) — FROZEN 2026-10-03T23:56:53Z

Owner 2026-10-03: "再提交几个，占满30个的slot限制" (fill the 30-job queue).  Counterpart of P127 (VCS affine grid) inside the same v6 §8.2 budget package:
"SimCLR 用预定温度 {.1,.2,.5} 与两个学习率倍率组成 6 格，实际中心值由现有调优记录决定".  Centre = existing recipe (τ 0.2, AdamW lr 1e-3; P41 C10 /
P91 C100 seed 0 = 88.20 / 58.25 linear, reused by hash, not re-run).  P41's τ ∈ {0.1, 0.5} search was at 2 views / 200 epochs (84.30 / 86.36 vs
86.09) — a different schedule, prior information only.
Cells: τ ∈ {0.1, 0.2, 0.5} × lr multiplier ∈ {1×, 2×} minus the centre = 5 per dataset, 10 units.  Multiplier choice (main session): **2×** (the centre
plus one step up): the 800-epoch AdamW schedule was never lr-tuned for any method, and 0.5× at a fixed epoch budget is the direction expected to
under-train.  Each config differs from that dataset's SimCLR 4v/800ep seed-0 config only in objective.simclr_temperature, optimizer.lr, run.stage
(verified by diff; the 4-view NT-Xent path reads the temperature from the config).

## Pre-stated reading (dataset-selection table, v6 §8.1)
- Primary: final frozen-h linear (selection split) per dataset; kNN alongside; on CIFAR-100 the P124 coarse / fine / conditional readouts may be added as
  an evaluation-only addendum.  Selection only on development validation labels; official test closed.
- **Dataset-selected SimCLR cell** = highest linear; it replaces the centre only if it beats it by ≥ 0.30 (C10) / ≥ 0.50 (C100) (same thresholds as P127);
  a replacing cell gets seeds 1–2 (addendum).
- **Tuned-vs-tuned statement** (VCS A-P3 grid P127 vs SimCLR grid P128) only after both grids are complete and the selected cells have 3 seeds; until then
  each grid is reported on its own.  JS (matched) has not received a grid — any "fully tuned" statement covers VCS vs SimCLR only.
- No early stopping on any monitor.
## Compute
10 × 800 epochs; RTX6000PRO / H100 / L40S; normal QOS; node60 allowed.  2 submitted now, 8 via a queue feeder (cap 29, so one slot stays free for the
orchestrator's P115 diagnostics job 2).
