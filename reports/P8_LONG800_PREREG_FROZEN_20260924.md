# Pre-registration — VCS-QMI 800-epoch runs (P8–P9), frozen 2026-09-24 before launch

Owner decision after `reports/P7_confirm200_report.md`: run the current recipe for 800 epochs, **VCS only, no controls**.

## Units
`configs/cifar10_long800_vcs_seed{0,1,2}.yaml`, derived from the frozen 200-epoch VCS configs by exactly these overrides
(`configs/LONG800_SHA256.json`): `run.stage=P8_long800`, `train.epochs=800`, `logging.checkpoint_epochs=[100,200,400,600,800]`,
`evaluation.knn_epochs=[0,20,50,100,200,400,600,800]`, `evaluation.linear_epochs_of_pretrain=[800]`.  Warm-up stays 10 epochs
(W = 1,750 steps), cosine to 1 % over T = 140,000 steps; same initial weights per seed as P3/P5 (seed 0 `bdc3a4e3…`, seed 1 `94c2cb88…`,
seed 2 `4385db04…`); same split `c35d7cd3…`; K = 1; code unchanged (`src/`, `reference/` as in commit `76cebd6`).
Three parallel single-GPU jobs; each may be resumed once from `last.pt` (epoch boundary) if interrupted; walltime 23 h.

## Question
With 4× the training budget of P5 and the same recipe, does the VCS encoder's frozen `h` keep improving (linear-val, kNN), does the
dimensional concentration change (h effective rank ≈ 13, 7 dominant `p_raw` directions at 200 epochs), and does `heldout_J` move
beyond ≈ 0.90?

## Endpoints
Primary: `linear_val_top1_pct` of frozen `h` at `epoch_800.pt` (pilot probe protocol unchanged), per seed and mean ± SD, with the same
probe on `initial.pt`.  Secondary: kNN / `heldout_J` / h- and z-rank / dominant `p_raw` directions at 0/20/50/100/200/400/600/800;
saturation fractions; cost.

## Pre-committed reading
- The 200-epoch P5 values (74.34 ± 0.46 % linear, 63.93 kNN, h-rank 13.2, J 0.899) are the reference; **no control is run at 800
  epochs, so no VCS-vs-control statement is made from this stage** (the 200-epoch controls are not a fair 800-epoch comparator).
- Continued improvement: linear-val(800) − linear-val(200, same seed) positive in all 3 seeds and larger than the P5 seed SD (0.46).
- Concentration: report h-rank and the number of dominant `p_raw` directions per seed; "persistent" if h-rank < 20 and ≤ 10 directions
  in all seeds; "resolving" if rank rises substantially; anything in between reported as such.
- J: if `heldout_J` stays ≈ 0.90 while linear/kNN move → transfer is decoupled from the critic objective at this stage.
- Failures / resumes are reported; no batch skipping, no method change.

## Not claimed
No test accuracy; no comparison to controls at 800 epochs; no Shannon-MI / convergence-to-S language.

## After P9
Neutral table → report → wait for the owner.
