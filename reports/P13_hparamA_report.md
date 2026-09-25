# P13 — VCS-QMI A-group (VCS-specific) hyper-parameter screen report

Status: **COMPLETE** (P12 runs 2026-09-24 19:44 → 2026-09-25 04:36 UTC; 21/21 COMPLETED; one infrastructure failure (K=255, epoch-0 hold-out
diagnostic with K > tail batch) fixed by capping K per batch and rerun — archived as `P12_vcs_k255_seed0_FAILED_attempt1_20260924`).
Neutral table: `reports/P13_hparamA_results_table.md`.  Pre-registration: `reports/P12_HPARAM_A_PREREG_FROZEN_20260924.md`.
Single seed 0, 200 epochs; baseline `P5_vcs_seed0`: linear 74.84 %, kNN 64.64 %, h-rank 12.9, 7 dominant `p_raw` directions, heldout-J 0.898.
Rule: HELPS if linear > +1.0, HURTS if < −1.0 (K=1 seed SD 0.46).

## Results (final linear-val / Δ; kNN; h-rank; heldout-J)

| factor | value | linear | Δ | kNN | h-rank | heldout-J | signal |
|---|---|---|---|---|---|---|---|
| **K** | 255 | 76.22 | **+1.38** | 67.62 | 16.7 | 0.933 | HELPS (= K 8/64 level; saturates for K ≥ 8) |
| critic lr × | 10 | 75.08 | +0.24 | 66.56 | 17.7 | 0.918 | neutral (kNN +1.9, rank +4.8) |
| | 3 | 75.06 | +0.22 | 64.08 | 14.9 | 0.907 | neutral |
| | 0.3 | 72.80 | −2.04 | 61.68 | 11.4 | 0.889 | HURTS |
| | 0.1 | 72.86 | −1.98 | 62.66 | 11.4 | 0.888 | HURTS |
| critic width | 2048 | 74.80 | −0.04 | 63.84 | 15.2 | 0.910 | neutral |
| | 1024 | 74.14 | −0.70 | 63.80 | 14.6 | 0.907 | neutral |
| | 256 | 74.40 | −0.44 | 64.30 | 12.5 | 0.896 | neutral |
| | 128 | 73.16 | −1.68 | 62.86 | 11.9 | 0.892 | HURTS |
| critic depth | 3 | 74.00 | −0.84 | 63.70 | 14.3 | 0.903 | neutral |
| | 1 | 73.32 | −1.52 | 62.52 | 11.8 | 0.891 | HURTS |
| critic wd | 1e-3 | 74.36 | −0.48 | 63.48 | 13.7 | 0.898 | neutral |
| | 1e-4 | 73.34 | −1.50 | 63.68 | 13.0 | 0.899 | HURTS (non-monotone vs 1e-3 → noise-level) |
| critic last gain | 1.0 | 74.06 | −0.78 | 63.34 | 14.3 | 0.907 | neutral |
| | 0.01 | 74.44 | −0.40 | 65.04 | 13.0 | 0.902 | neutral |
| projector out | 512 | 73.62 | −1.22 | 64.04 | 13.4 | 0.905 | HURTS |
| | 256 | 74.52 | −0.32 | 63.98 | 13.5 | 0.904 | neutral |
| | 64 | 73.78 | −1.06 | 63.08 | 13.2 | 0.896 | HURTS |
| projector hidden | 2048 | 74.04 | −0.80 | 64.06 | 13.0 | 0.898 | neutral |
| | 1024 | 74.46 | −0.38 | 64.04 | 13.1 | 0.900 | neutral |
| critic input | raw `p_raw` (no L2) | 74.78 | −0.06 | 64.22 | 14.9 | 0.907 | neutral |

## Reading
- **No VCS-specific architectural or optimizer hyper-parameter improves the frozen linear probe by more than the pre-set 1-point margin.**
  The only HELPS is K (already known from P11; 255 ≈ 64 ≈ 8), i.e. the negative pool matters up to K ≈ 8 and then saturates.
- The default critic (512-512, gain 0.1, lr ×1, wd 0) sits on a plateau: making it smaller (width 128, depth 1) or slower (lr ×0.1–0.3)
  hurts by 1.5–2 points; making it larger, deeper, faster, or regularizing it does not help linear-val.  Critic capacity is not the limiter.
- Critic input space (projector output 64–512, hidden 1024–2048, raw vs L2) is not the limiter either; 128 + L2 is at the optimum of the
  tested range.
- Consistent secondary pattern: **faster critic (lr ×3, ×10) and larger K raise h effective rank (13 → 15–18) and kNN (+1.9 for ×10) without a
  matching linear-probe gain**; slower/smaller critics lower rank and kNN.  The representation geometry responds to how well the critic keeps
  up with the encoder, but under this objective the extra dimensions do not carry linearly separable class information at 200 epochs.
- Together with P14 (LR 3e-4 … 1e-2 all ≤ baseline; cosine floor 10 % neutral) and P9 (800 epochs: +4.6 with saturation rising to 85–95 %),
  the evidence supports the working hypothesis stated in P9: the bounded quadratic objective with a tanh critic is nearly satisfied by a
  low-dimensional code early in training, and no hyper-parameter of the critic, projector, K, or LR changes that.  What remains untested
  within the method's rules: harder positives (augmentation), more pairs per step (batch size), matrix weight decay, and combinations
  (K=8 + critic lr ×3/×10, K=8 + strong augmentation, 800 epochs) — all launched as P16.

## Consequence for the "best VCS" set
Current best single change: K = 8 (+1.3 seed 0 / +2.4 mean over seeds).  Candidates to combine: critic lr ×3–×10 (rank/kNN), stronger
augmentation (pending).  Everything else in group A can stay at the frozen default.

**Waiting for the owner** (P14 `const` and the P16 batch continue).
