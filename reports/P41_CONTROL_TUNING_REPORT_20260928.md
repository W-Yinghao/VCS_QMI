# P41 — equal-budget tuning of the controls: report (2026-09-28; complete)

Prereg: `P41_CONTROL_TUNING_PREREG_FROZEN_20260927.md` (+ addendum 1: winners' seeds; addendum 2: the 8× point).  Tables: `P41_control_tuning_table.md`
(30 runs so far; results-only commits cd06514 and later).  Selection split (5 000 held-out images of the frozen dev split), frozen-h linear probe and
kNN (k = 200), 3 seeds unless marked.  Evidence category: completed for 1× / 2× / 4× and SimCLR 8×; all cells have 3 seeds.

## 1. What the controls were given
The knobs that mattered for VCS — 4 views (the control's own loss averaged over the 6 view pairs), batch 128 / 100 epochs at 1×, 800 epochs at 4×,
4 views × 800 epochs at 8× — plus each control's own knob (SimCLR τ ∈ {0.1, 0.5}; VICReg (25, 25, 0.1) and (10, 10, 1)).  12 single-seed units, then
3 seeds for the best per compute point, then the 8× cell.  VCS's own search on the same knobs was 3 (views) + 4 (batch) + 2 (schedule) + 5
(critic scale) units; the control search is ≈ 1.1× of that.

## 2. Single-seed tuning results (linear-val %)
SimCLR: τ 0.1 → 84.30, τ 0.5 → 86.36 (P5 τ 0.2: 86.09 ± 0.38); B128 (2v/200ep) 86.36; 4v/B128/100ep 86.42; 4v/200ep 87.48; 2v/800ep 88.38.
VICReg: cov 0.1 → 83.36; (10, 10, 1) → 86.28 (P5 (25, 25, 1): 85.46 ± 0.26); B128 86.52; 4v/B128/100ep 87.24; 4v/200ep 86.64; 2v/800ep 86.70.
Temperature / weight changes do not help either control; views and schedule do, as they did for VCS.

## 3. Equal-compute comparison (3 seeds; mean ± sd; linear / kNN)
| compute | VCS (frozen recipe, best 3-seed cell) | SimCLR (tuned best) | VICReg (tuned best) | frozen P5 controls (2v/200ep = 1×) |
|---|---|---|---|---|
| 1× | 4v/B128/100ep **83.19 ± 0.40** / 78.43 ± 0.11 | 4v/B128/100ep **86.64 ± 0.33** / 84.45 ± 0.06 | 4v/B128/100ep **87.12 ± 0.10** / 83.91 ± 0.17 | SimCLR 86.09 ± 0.38 / 83.98; VICReg 85.46 ± 0.26 / 81.92 |
| 2× | 4v/B256/200ep **84.54 ± 0.16** / 81.40 ± 0.22 | 4v/200ep **87.84 ± 0.42** / 86.52 ± 0.37 | 4v/200ep **86.73 ± 0.30** / 84.27 ± 0.35 | |
| 4× | 8v/200ep **85.95 ± 0.35** / 83.47 ± 0.38 (2v/800ep 85.30 ± 0.21 / 82.33; 4v/400ep 85.90 single) | 2v/800ep **88.35 ± 0.06** / 87.26 ± 0.41 | 2v/800ep **87.28 ± 0.62** / 84.53 ± 0.18 | |
| 8× | 4v/800ep **87.01 ± 0.53** / 85.46 ± 0.12 (strong-aug 87.78 / 85.58 single; 16v/200ep 86.74; 8v/400ep 86.76) | 4v/800ep **88.32 ± 0.30** / 87.65 ± 0.40 | 4v/800ep **87.11 ± 0.51 / 84.46 ± 0.08** | |
| 16× | 4v/1600ep 87.50 / 85.84; 8v/800ep 87.14 / 86.32 (singles) | — (proposed) | — | |

## 4. Reading (pre-committed: the paper reports VCS vs each control at equal compute with the control's best setting; if a control's best
exceeds VCS at every budget, that is the result)
Both tuned controls exceed VCS at every compute point on the linear probe: SimCLR by 3.5 / 3.3 / 2.4 / 1.3 points at 1× / 2× / 4× / 8×, VICReg
by 3.9 / 2.2 / 1.3 / +0.1 points at 1× / 2× / 4× / 8×.  On kNN the gap to SimCLR is 6.0 / 5.1 / 3.8 / 2.2 and to VICReg
5.5 / 2.9 / 1.1 (8×: VCS ahead of VICReg on kNN).  The gap narrows with compute (VCS gains ≈ +1.3 per doubling on the linear
probe, SimCLR ≈ +0.6 above 2×), and SimCLR at 4× (88.35) is above VCS at 16× (87.50).  The untuned P5 controls already matched or beat VCS's
2× cell; the tuning adds 0.5–2.3 points to them.  **Result: under equal tuning VCS trails both controls at every budget tested**; the earlier
"within 1 point of SimCLR at 8×" statement referred to the untuned control and is withdrawn.  What VCS retains from the mechanism studies is
unchanged (bounded, calibrated critic; h-uniformity ordering; no collapse without batch statistics) and is not an accuracy claim.

## 5. Pending / not run
A 16× control (SimCLR 4v/1600ep) is proposed.  Official test set: once,
for every cell above, after the pending seeds (P67 prereg, to be frozen then).
