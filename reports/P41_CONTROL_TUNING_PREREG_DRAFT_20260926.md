# DRAFT pre-registration — equal-budget tuning of the controls (P41).  NOT FROZEN, NOT LAUNCHED: awaiting the owner's go.

Owner's rule (2026-09-24/25): no comparison with the controls until the VCS recipe is settled; the controls must then receive an equal tuning
budget.  The VCS recipe is now fixed (P36/P40): cosine critic, K = 8, negative detach, [a0 = 5], **4 views**; 1× compute = B 128 / 100 ep
(83.19 ± 0.40), 2× = B 256 / 200 ep (84.54 ± 0.16; a0 = 1: 84.41 ± 0.06); 800 ep 2-view 85.54 (seeds running); 800 ep 4-view running.
The controls so far are the frozen matched recipes from P5 (2 views, B 256, lr 1e-3, 200 ep, SimCLR τ = 0.2, VICReg 25/25/1):
SimCLR 86.09 ± 0.38, VICReg 85.46 ± 0.26.

## What "equal budget" means here
Every recipe knob that was varied for VCS and turned out to matter (views, batch size / steps, schedule length) is offered to each control,
plus each control's own natural temperature/weight knob, at the same compute points.  Code: `run.control_tuning: true` unlocks 4 views for
`simclr_matched` / `vicreg_matched_128` (the control's own pairwise loss averaged over the 6 view pairs — SimCLR: NT-Xent per pair, 2B
anchors each; VICReg: invariance/variance/covariance per pair); tested equal to the standard loss at 2 views.  Everything else about the
controls (projector 512/128, augmentation, AdamW, cosine schedule) stays as frozen in P5.

## Units (single seed unless noted; 3 seeds for the winner of each control)
| control | knob | values | compute |
|---|---|---|---|
| SimCLR | temperature τ | 0.1, 0.5 (0.2 = P5) | 1× (2 views, 200 ep) |
| SimCLR | 4 views | B 256 / 200 ep (2×); B 128 / 100 ep (1×) | as VCS |
| SimCLR | batch | 128 (2 views, 200 ep) | 1× |
| SimCLR | schedule | 800 ep (2 views) | 4× |
| VICReg | variance/covariance weights | (25, 25, 1) = P5; (25, 25, 0.1); (10, 10, 1) | 1× |
| VICReg | 4 views | B 256 / 200 ep; B 128 / 100 ep | as VCS |
| VICReg | batch | 128 (2 views, 200 ep) | 1× |
| VICReg | schedule | 800 ep (2 views) | 4× |
≈ 16 single-seed units (+ 4 seeds), ≈ 1.1× the VCS recipe's search cost on the corresponding knobs (VCS: views 3, batch 4, schedule 2,
critic-scale 5).  Compute-matched comparison points for the paper: 1× (2 v / 200 ep or 4 v / B128 / 100 ep), 2× (4 v / 200 ep), 4× (800 ep).

## Reading (pre-committed)
Per control, the best configuration at each compute point (selection split, as for VCS); the paper reports VCS vs each control at equal compute
with the control's *best* setting, and states the search budgets side by side.  If a control's best exceeds VCS at every budget, that is the
result.  Official test set: once, at the end, for the frozen best of each method.

## Not done until the owner confirms
No control config has been generated or submitted.  The generator (`make_n_control_tuning_configs.py`) is written but not run.
